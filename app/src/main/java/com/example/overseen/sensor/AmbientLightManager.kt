package com.example.overseen.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.nio.ByteBuffer

/**
 * Estados de luminosidade do ambiente conforme RF02.3.
 */
enum class LightState(val spokenPhrase: String, val displayLabel: String) {
    CLARO(spokenPhrase = "Luz acesa", displayLabel = "CLARO (Luz Acesa)"),
    ESCURO(spokenPhrase = "Muito escuro", displayLabel = "ESCURO (Sem Luz)")
}

data class AmbientLightSnapshot(
    val state: LightState = LightState.CLARO,
    val luxValue: Float? = null,
    val luminanceY: Float = 100f,
    val usingHardwareSensor: Boolean = false
)

/**
 * RF02: Detecção de Luminosidade do Ambiente.
 * - RF02.1: Lê Sensor.TYPE_LIGHT se disponível.
 * - RF02.2: Calcula a luminância média Y (0.299R + 0.587G + 0.114B ou plano Y de YUV_420_888) do frame da câmera.
 * - RF02.3:
 *   - CLARO se a luminância da câmera traseira >= 40 OU se o sensor Lux >= 15 lx (evita falso "Muito escuro"
 *     quando o usuário aponta a câmera traseira para uma lâmpada enquanto o sensor frontal está na sombra).
 *   - ESCURO apenas quando a câmera traseira está escura (Luminância < 40) E o sensor de luz também indica escuro (< 15 lx ou ausente).
 * - RF02.4: Histerese temporal de pelo menos 2 segundos (2000ms) para evitar alarmes falsos por sombras rápidas.
 */
class AmbientLightManager(
    context: Context,
    private val hysteresisMs: Long = 2000L,
    private val onStableStateChanged: (LightState) -> Unit
) : SensorEventListener {

    private val sensorManager = context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val lightSensor: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_LIGHT)

    val hasHardwareLightSensor: Boolean = lightSensor != null

    private val _snapshot = MutableStateFlow(
        AmbientLightSnapshot(
            state = LightState.CLARO,
            usingHardwareSensor = hasHardwareLightSensor
        )
    )
    val snapshot: StateFlow<AmbientLightSnapshot> = _snapshot.asStateFlow()

    @Volatile
    private var currentStableState: LightState? = null

    @Volatile
    private var candidateState: LightState? = null

    @Volatile
    private var candidateSinceTimestampMs: Long = 0L

    @Volatile
    private var latestLux: Float? = null

    @Volatile
    private var latestLuminanceY: Float = 100f

    @Volatile
    private var hasReceivedCameraFrame: Boolean = false

    fun startListening() {
        if (lightSensor != null) {
            sensorManager?.registerListener(this, lightSensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    fun stopListening() {
        if (lightSensor != null) {
            sensorManager?.unregisterListener(this)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_LIGHT) {
            val lux = event.values.firstOrNull() ?: return
            latestLux = lux

            // Se a câmera já está recebendo frames, combinamos o sensor Lux com a luminância real da câmera traseira
            val rawState = determineCombinedLightState(
                lux = lux,
                luminanceY = if (hasReceivedCameraFrame) latestLuminanceY else null
            )

            evaluateStateWithHysteresis(
                newRawState = rawState,
                lux = lux,
                luminanceY = latestLuminanceY,
                nowMs = System.currentTimeMillis()
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Não requer ação
    }

    /**
     * Avalia a luminância média a partir do frame da câmera traseira.
     * Como o usuário aponta a câmera traseira para o ambiente/lâmpada/cédula, a luminância Y
     * da câmera tem prioridade para identificar claridade mesmo que o sensor frontal esteja na sombra.
     */
    fun processCameraFrameLuminance(luminanceY: Float, nowMs: Long = System.currentTimeMillis()) {
        latestLuminanceY = luminanceY
        hasReceivedCameraFrame = true
        val lux = latestLux
        val rawState = determineCombinedLightState(lux = lux, luminanceY = luminanceY)

        evaluateStateWithHysteresis(
            newRawState = rawState,
            lux = lux,
            luminanceY = luminanceY,
            nowMs = nowMs
        )
    }

    @Synchronized
    internal fun evaluateStateWithHysteresis(
        newRawState: LightState,
        lux: Float?,
        luminanceY: Float,
        nowMs: Long
    ) {
        val stable = currentStableState
        if (stable == null) {
            // Na primeira leitura, requer confirmação pela janela de histerese se começar escuro,
            // ou define CLARO inicialmente para não falar "Muito escuro" nos primeiros milissegundos de abertura da câmera.
            if (newRawState == LightState.CLARO) {
                currentStableState = LightState.CLARO
                candidateState = LightState.CLARO
                candidateSinceTimestampMs = nowMs
                _snapshot.value = AmbientLightSnapshot(
                    state = LightState.CLARO,
                    luxValue = lux,
                    luminanceY = luminanceY,
                    usingHardwareSensor = hasHardwareLightSensor && lux != null
                )
                return
            } else {
                currentStableState = LightState.CLARO
                candidateState = LightState.ESCURO
                candidateSinceTimestampMs = nowMs
                _snapshot.value = AmbientLightSnapshot(
                    state = LightState.CLARO,
                    luxValue = lux,
                    luminanceY = luminanceY,
                    usingHardwareSensor = hasHardwareLightSensor && lux != null
                )
                return
            }
        }

        if (newRawState == stable) {
            candidateState = stable
            candidateSinceTimestampMs = nowMs
            _snapshot.value = _snapshot.value.copy(
                luxValue = lux,
                luminanceY = luminanceY,
                usingHardwareSensor = hasHardwareLightSensor && lux != null
            )
            return
        }

        // Estado bruto divergiu do estado estável
        if (candidateState != newRawState) {
            candidateState = newRawState
            candidateSinceTimestampMs = nowMs
        } else if (nowMs - candidateSinceTimestampMs >= hysteresisMs) {
            // RF02.4: Transição confirmada após >= 2 segundos de estabilidade
            currentStableState = newRawState
            _snapshot.value = AmbientLightSnapshot(
                state = newRawState,
                luxValue = lux,
                luminanceY = luminanceY,
                usingHardwareSensor = hasHardwareLightSensor && lux != null
            )
            onStableStateChanged(newRawState)
            return
        }

        _snapshot.value = _snapshot.value.copy(
            luxValue = lux,
            luminanceY = luminanceY,
            usingHardwareSensor = hasHardwareLightSensor && lux != null
        )
    }

    companion object {
        const val LUX_THRESHOLD = 15.0f
        const val LUMINANCE_THRESHOLD = 40.0f

        /**
         * Combina a leitura do sensor Lux frontal com a luminância Y da câmera traseira:
         * - Se a câmera traseira está vendo luz (luminanceY >= 40) OU o sensor Lux está vendo luz (lux >= 15),
         *   o ambiente é considerado CLARO (evita falar "Muito escuro" ao apontar a câmera traseira para uma lâmpada).
         * - Só classifica como ESCURO se a câmera traseira estiver escura (< 40) E o sensor Lux também estiver escuro (< 15 ou ausente).
         */
        fun determineCombinedLightState(lux: Float?, luminanceY: Float?): LightState {
            val cameraIsBright = luminanceY != null && luminanceY >= LUMINANCE_THRESHOLD
            val sensorIsBright = lux != null && lux >= LUX_THRESHOLD

            return when {
                cameraIsBright || sensorIsBright -> LightState.CLARO
                luminanceY != null && luminanceY < LUMINANCE_THRESHOLD -> LightState.ESCURO
                lux != null && lux < LUX_THRESHOLD -> LightState.ESCURO
                else -> LightState.CLARO
            }
        }

        /**
         * RF02.2: Calcula a luminância média a partir de um buffer RGB aplicando
         * Y = 0.299R + 0.587G + 0.114B.
         */
        fun computeLuminanceFromRgb(r: Int, g: Int, b: Int): Float {
            return 0.299f * r + 0.587f * g + 0.114f * b
        }

        /**
         * RF02.2: Extrai a luminância média diretamente do ImageProxy (YUV_420_888)
         * usando cópia em bloco ou leitura indexada segura sem depender de limit() mutado.
         */
        fun extractAverageLuminance(image: ImageProxy): Float {
            val planes = image.planes
            if (planes.isEmpty()) return 0f

            val yPlane = planes[0]
            val yBuffer: ByteBuffer = yPlane.buffer.duplicate()
            yBuffer.clear() // Garante position = 0 e limit = capacity

            val rowStride = yPlane.rowStride
            val pixelStride = yPlane.pixelStride
            val width = image.width
            val height = image.height
            val capacity = yBuffer.capacity()

            val step = 4
            var sumY = 0L
            var sampleCount = 0

            var row = 0
            while (row < height) {
                val rowOffset = row * rowStride
                var col = 0
                while (col < width) {
                    val index = rowOffset + col * pixelStride
                    if (index in 0 until capacity) {
                        val yVal = yBuffer.get(index).toInt() and 0xFF
                        sumY += yVal
                        sampleCount++
                    }
                    col += step
                }
                row += step
            }

            return if (sampleCount > 0) sumY.toFloat() / sampleCount else 0f
        }
    }
}
