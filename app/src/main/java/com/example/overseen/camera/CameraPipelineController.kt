package com.example.overseen.camera

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.overseen.ml.BanknoteDetection
import com.example.overseen.ml.BanknoteDetector
import com.example.overseen.ml.ConfirmationBuffer
import com.example.overseen.sensor.AmbientLightManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ThrottledFrameAnalyzer:
 * - Amostragem limitada entre 3 a 5 FPS (~250ms de intervalo mínimo entre frames processados)
 *   para evitar sobreaquecimento e throttling térmico em aparelhos de entrada (Diretriz 1.1.2).
 * - Diretriz 5.1: Toda a análise de luminosidade e inferência TFLite roda em background thread
 *   (Dispatchers.Default).
 * - Diretriz 5.2: Garante que `imageProxy.close()` seja SEMPRE chamado dentro de um bloco `finally`
 *   para jamais travar o pipeline do CameraX.
 */
class VisionFrameAnalyzer(
    private val ambientLightManager: AmbientLightManager,
    private val banknoteDetector: BanknoteDetector,
    private val confirmationBuffer: ConfirmationBuffer,
    private val minFrameIntervalMs: Long = 250L, // 4 FPS (dentro da faixa de 3 a 5 FPS)
    private val onConfirmedBanknote: (BanknoteDetection) -> Unit,
    private val onFrameTelemetry: (luminanceY: Float, candidateDetection: BanknoteDetection?) -> Unit
) : ImageAnalysis.Analyzer {

    private var lastAnalyzedTimestampMs: Long = 0L
    private val isProcessing = AtomicBoolean(false)

    override fun analyze(imageProxy: ImageProxy) {
        try {
            val nowMs = SystemClock.elapsedRealtime()
            if (nowMs - lastAnalyzedTimestampMs < minFrameIntervalMs) {
                return
            }
            if (!isProcessing.compareAndSet(false, true)) {
                return
            }
            lastAnalyzedTimestampMs = nowMs

            // Pipeline 1: Luminosidade do Ambiente (RF02.2 / RF02.3 / RF02.4)
            val avgLuminanceY = AmbientLightManager.extractAverageLuminance(imageProxy)
            ambientLightManager.processCameraFrameLuminance(
                luminanceY = avgLuminanceY,
                nowMs = System.currentTimeMillis()
            )

            // Pipeline 2: Detector TFLite de Cédulas de Real (RF03)
            val detections = banknoteDetector.detect(imageProxy)
            val topDetection = detections.firstOrNull()

            // Buffer de Estabilidade (2 frames consecutivos com score >= 0.75 - RF03.3)
            val confirmed = confirmationBuffer.update(
                detection = topDetection,
                nowMs = System.currentTimeMillis()
            )
            if (confirmed != null) {
                onConfirmedBanknote(confirmed)
            }

            onFrameTelemetry(avgLuminanceY, topDetection)
        } catch (e: Throwable) {
            Log.e(TAG, "Erro ao processar frame da câmera", e)
        } finally {
            isProcessing.set(false)
            // Diretriz 5.2: Fechamento obrigatório dentro do bloco finally
            imageProxy.close()
        }
    }

    companion object {
        private const val TAG = "VisionFrameAnalyzer"
    }
}

/**
 * Gerenciador do CameraX (`Preview` + `ImageAnalysis` + Lanterna/Torch).
 */
class CameraPipelineController(
    private val context: Context,
    private val analyzer: VisionFrameAnalyzer,
    private val onCameraError: (String) -> Unit
) {
    private val analysisScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val backgroundExecutor = Dispatchers.Default.asExecutor()

    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private val _isTorchOn = MutableStateFlow(false)
    val isTorchOn: StateFlow<Boolean> = _isTorchOn.asStateFlow()

    /**
     * RF01: Inicialização automática da câmera traseira em execução.
     */
    fun bindCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder()
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { it.setAnalyzer(backgroundExecutor, analyzer) }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Throwable) {
                Log.e(TAG, "Falha ao inicializar câmera traseira", e)
                onCameraError("Erro ao iniciar a câmera")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * RF04.2: Toque longo na tela alterna a lanterna (flash) ligada/desligada.
     * Retorna o novo estado da lanterna ou null caso o aparelho não possua flash.
     */
    fun toggleTorch(): Boolean? {
        val activeCamera = camera ?: return null
        if (!activeCamera.cameraInfo.hasFlashUnit()) {
            return null
        }
        val targetState = !_isTorchOn.value
        activeCamera.cameraControl.enableTorch(targetState)
        _isTorchOn.value = targetState
        return targetState
    }

    fun release() {
        try {
            cameraProvider?.unbindAll()
        } catch (_: Throwable) {
        }
        analysisScope.cancel()
    }

    companion object {
        private const val TAG = "CameraPipeline"
    }
}
