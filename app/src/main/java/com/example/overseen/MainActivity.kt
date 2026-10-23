package com.example.overseen

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.overseen.audio.AccessibilityFeedbackManager
import com.example.overseen.camera.CameraPipelineController
import com.example.overseen.camera.VisionFrameAnalyzer
import com.example.overseen.ml.BanknoteDetector
import com.example.overseen.ml.ConfirmationBuffer
import com.example.overseen.sensor.AmbientLightManager
import com.example.overseen.sensor.LightState
import com.example.overseen.ui.VisionCashAccessibleScreen
import com.example.overseen.ui.VisionUiState
import com.example.overseen.ui.theme.OverseenTheme
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update

/**
 * Activity Principal do VisionCash & Light (RF01 a RF04):
 * - Abre diretamente na câmera traseira sem splash/onboarding que exija clique visual (RF01).
 * - Orquestra o pipeline de luminosidade (sensor de hardware + fallback YUV com histerese de 2s - RF02).
 * - Orquestra o detector TFLite quantizado INT8 com buffer de estabilidade de 2 frames (RF03).
 * - Gerencia TTS conciso (`QUEUE_FLUSH`), feedback háptico diferencial e gestos de toque duplo e longo (RF04).
 */
class MainActivity : ComponentActivity() {

    private lateinit var feedbackManager: AccessibilityFeedbackManager
    private lateinit var ambientLightManager: AmbientLightManager
    private lateinit var banknoteDetector: BanknoteDetector
    private lateinit var cameraController: CameraPipelineController
    private val confirmationBuffer = ConfirmationBuffer(
        requiredConsecutiveFrames = 2,
        minConfidence = BanknoteDetector.MIN_CONFIDENCE_THRESHOLD
    )

    private val uiStateFlow = MutableStateFlow(VisionUiState())

    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        uiStateFlow.update { it.copy(hasCameraPermission = isGranted) }
        if (isGranted) {
            uiStateFlow.update { it.copy(errorMessage = null) }
            feedbackManager.speak("Câmera ativa", updateLastMessage = false)
        } else {
            val msg = "Permissão de câmera negada"
            uiStateFlow.update { it.copy(errorMessage = msg) }
            feedbackManager.announceError(msg)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // 1. Inicializa camada de áudio (TTS com QUEUE_FLUSH) e Haptics
        feedbackManager = AccessibilityFeedbackManager(this)

        // 2. Inicializa gerenciador de luminosidade com histerese temporal de 2 segundos (RF02.4)
        ambientLightManager = AmbientLightManager(
            context = this,
            hysteresisMs = 2000L,
            onStableStateChanged = { newLightState ->
                val isDark = newLightState == LightState.ESCURO
                feedbackManager.announceLightTransition(
                    isDark = isDark,
                    spokenState = newLightState.spokenPhrase
                )
                uiStateFlow.update { state ->
                    state.copy(
                        lightSnapshot = ambientLightManager.snapshot.value,
                        lastSpokenAnnouncement = newLightState.spokenPhrase
                    )
                }
            }
        )

        // 3. Inicializa detector TFLite quantizado INT8 (RF03 / Diretriz 5.4)
        banknoteDetector = BanknoteDetector(
            context = this,
            onModelError = { errorMsg ->
                uiStateFlow.update { it.copy(errorMessage = errorMsg) }
                feedbackManager.announceError(errorMsg)
            }
        )

        // 4. Inicializa analisador throttled (3 a 5 FPS -> 250ms) e controlador CameraX
        val frameAnalyzer = VisionFrameAnalyzer(
            ambientLightManager = ambientLightManager,
            banknoteDetector = banknoteDetector,
            confirmationBuffer = confirmationBuffer,
            minFrameIntervalMs = 250L,
            onConfirmedBanknote = { confirmedDetection ->
                val spoken = confirmedDetection.banknoteClass.spokenText
                feedbackManager.announceBanknote(spoken)
                uiStateFlow.update { state ->
                    state.copy(
                        lastConfirmedBanknote = confirmedDetection,
                        lastSpokenAnnouncement = spoken,
                        errorMessage = null
                    )
                }
            },
            onFrameTelemetry = { _, candidate ->
                uiStateFlow.update { state ->
                    state.copy(
                        lightSnapshot = ambientLightManager.snapshot.value,
                        currentCandidate = candidate
                    )
                }
            }
        )

        cameraController = CameraPipelineController(
            context = this,
            analyzer = frameAnalyzer,
            onCameraError = { errorMsg ->
                uiStateFlow.update { it.copy(errorMessage = errorMsg) }
                feedbackManager.announceError(errorMsg)
            }
        )

        // 5. Verifica permissão de câmera imediatamente na abertura (RF01)
        val hasPermission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        uiStateFlow.update { it.copy(hasCameraPermission = hasPermission) }
        if (!hasPermission) {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            OverseenTheme {
                val uiState by uiStateFlow.collectAsState()
                VisionCashAccessibleScreen(
                    uiState = uiState,
                    onBindCameraPreview = { lifecycleOwner, previewView ->
                        cameraController.bindCamera(lifecycleOwner, previewView)
                    },
                    onDoubleTapRepeat = {
                        // RF04.1: Toque duplo repete última leitura de cédula ou estado atual da iluminação
                        val currentLightPhrase = ambientLightManager.snapshot.value.state.spokenPhrase
                        feedbackManager.repeatLastMessage(fallbackLightState = currentLightPhrase)
                    },
                    onLongPressToggleTorch = {
                        // RF04.2: Toque longo alterna a lanterna (flash) ligada/desligada
                        val torchNowOn = cameraController.toggleTorch()
                        when (torchNowOn) {
                            true -> {
                                feedbackManager.vibrateLightStateChanged()
                                feedbackManager.speak("Lanterna ligada", updateLastMessage = false)
                                uiStateFlow.update { it.copy(isTorchEnabled = true) }
                            }
                            false -> {
                                feedbackManager.vibrateLightStateChanged()
                                feedbackManager.speak("Lanterna desligada", updateLastMessage = false)
                                uiStateFlow.update { it.copy(isTorchEnabled = false) }
                            }
                            null -> {
                                feedbackManager.vibrateInsufficientLightOrError()
                                feedbackManager.speak("Sem lanterna", updateLastMessage = false)
                            }
                        }
                    },
                    onRequestPermissionClick = {
                        requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    },
                    modifier = Modifier
                        .fillMaxSize()
                        .safeDrawingPadding()
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        ambientLightManager.startListening()
    }

    override fun onPause() {
        ambientLightManager.stopListening()
        super.onPause()
    }

    override fun onDestroy() {
        ambientLightManager.stopListening()
        cameraController.release()
        banknoteDetector.close()
        feedbackManager.shutdown()
        super.onDestroy()
    }
}
