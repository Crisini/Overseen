package com.example.overseen.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.onClick
import androidx.compose.ui.semantics.onLongClick
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.overseen.ml.BanknoteDetection
import com.example.overseen.sensor.AmbientLightSnapshot
import com.example.overseen.sensor.LightState
import com.example.overseen.ui.theme.HighContrastBlack
import com.example.overseen.ui.theme.HighContrastWhite
import com.example.overseen.ui.theme.HighContrastYellow
import java.util.Locale

data class VisionUiState(
    val hasCameraPermission: Boolean = false,
    val lightSnapshot: AmbientLightSnapshot = AmbientLightSnapshot(),
    val lastConfirmedBanknote: BanknoteDetection? = null,
    val currentCandidate: BanknoteDetection? = null,
    val isTorchEnabled: Boolean = false,
    val lastSpokenAnnouncement: String = "Aguardando leitura",
    val errorMessage: String? = null
)

/**
 * Tela única em tela cheia (RF01 / RF04 / RNF04):
 * - Paleta estrita Preto Absoluto (#000000), Amarelo Fluor (#FFFF00) e Branco (#FFFFFF) com contraste >= 7:1 (WCAG AAA).
 * - Tratamento de gestos em toda a tela via `pointerInput`:
 *   - Toque Duplo (RF04.1): Repetir última leitura de cédula ou estado atual de iluminação.
 *   - Toque Longo (RF04.2): Alternar lanterna (flash) ligada/desligada.
 * - Anotações semânticas completas para compatibilidade nativa com TalkBack.
 */
@Composable
fun VisionCashAccessibleScreen(
    uiState: VisionUiState,
    onBindCameraPreview: (androidx.lifecycle.LifecycleOwner, PreviewView) -> Unit,
    onDoubleTapRepeat: () -> Unit,
    onLongPressToggleTorch: () -> Unit,
    onRequestPermissionClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val banknoteSummary = uiState.lastConfirmedBanknote?.banknoteClass?.spokenText ?: "Nenhuma cédula lida ainda"
    val lightSummary = uiState.lightSnapshot.state.spokenPhrase
    val torchSummary = if (uiState.isTorchEnabled) "Lanterna ligada" else "Lanterna desligada"

    val fullScreenAccessibilityDescription = buildString {
        append("VisionCash e Light ativo. ")
        append("Iluminação: $lightSummary. ")
        append("Última cédula: $banknoteSummary. ")
        append("$torchSummary. ")
        append("Toque duplo em qualquer lugar para repetir a leitura. Toque longo para alternar a lanterna.")
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(HighContrastBlack)
            .pointerInput(uiState.hasCameraPermission) {
                detectTapGestures(
                    onDoubleTap = {
                        if (uiState.hasCameraPermission) {
                            onDoubleTapRepeat()
                        } else {
                            onRequestPermissionClick()
                        }
                    },
                    onLongPress = {
                        if (uiState.hasCameraPermission) {
                            onLongPressToggleTorch()
                        } else {
                            onRequestPermissionClick()
                        }
                    },
                    onTap = {
                        if (!uiState.hasCameraPermission) {
                            onRequestPermissionClick()
                        }
                    }
                )
            }
            .semantics(mergeDescendants = false) {
                contentDescription = fullScreenAccessibilityDescription
                stateDescription = "$lightSummary, $banknoteSummary"
                role = Role.Button
                onClick(label = "Repetir última leitura") {
                    if (uiState.hasCameraPermission) {
                        onDoubleTapRepeat()
                    } else {
                        onRequestPermissionClick()
                    }
                    true
                }
                onLongClick(label = "Alternar lanterna") {
                    if (uiState.hasCameraPermission) {
                        onLongPressToggleTorch()
                    } else {
                        onRequestPermissionClick()
                    }
                    true
                }
                customActions = listOf(
                    CustomAccessibilityAction("Repetir última leitura de cédula ou luz") {
                        onDoubleTapRepeat()
                        true
                    },
                    CustomAccessibilityAction("Ligar ou desligar lanterna") {
                        onLongPressToggleTorch()
                        true
                    }
                )
            }
    ) {
        if (uiState.hasCameraPermission) {
            val previewView = remember {
                PreviewView(context).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
            }
            AndroidView(
                factory = {
                    onBindCameraPreview(lifecycleOwner, previewView)
                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
        }

        // Camada de Alto Contraste WCAG AAA (#000000 fundo, #FFFF00 e #FFFFFF texto/bordas)
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Painel Superior: Luminosidade e Lanterna
            HighContrastTopStatusCard(
                lightSnapshot = uiState.lightSnapshot,
                isTorchEnabled = uiState.isTorchEnabled
            )

            // Painel Central: Resultado Principal de Cédula ou Alerta de Permissão/Erro
            HighContrastCenterReadoutCard(
                uiState = uiState
            )

            // Painel Inferior: Guia de Gestos Acessíveis
            HighContrastBottomGesturesCard()
        }
    }
}

@Composable
private fun HighContrastTopStatusCard(
    lightSnapshot: AmbientLightSnapshot,
    isTorchEnabled: Boolean
) {
    val isDark = lightSnapshot.state == LightState.ESCURO
    val borderColor = if (isDark) HighContrastWhite else HighContrastYellow

    Surface(
        color = HighContrastBlack,
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(3.dp, borderColor),
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                heading()
                liveRegion = LiveRegionMode.Polite
                contentDescription = "Ambiente ${lightSnapshot.state.spokenPhrase}. " +
                    if (isTorchEnabled) "Lanterna ligada" else "Lanterna desligada"
            }
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = if (isDark) "AMBIENTE: ESCURO" else "AMBIENTE: CLARO",
                    color = HighContrastYellow,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold
                )
                Text(
                    text = if (isTorchEnabled) "FLASH: ON" else "FLASH: OFF",
                    color = HighContrastWhite,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            val metricText = if (lightSnapshot.usingHardwareSensor && lightSnapshot.luxValue != null) {
                String.format(Locale.US, "Sensor Lux: %.1f lx | Lum Y: %.0f", lightSnapshot.luxValue, lightSnapshot.luminanceY)
            } else {
                String.format(Locale.US, "Luminância Y (Câmera): %.1f / 255", lightSnapshot.luminanceY)
            }
            Text(
                text = metricText,
                color = HighContrastWhite,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
private fun HighContrastCenterReadoutCard(
    uiState: VisionUiState
) {
    val mainTitle: String
    val subtitle: String

    when {
        !uiState.hasCameraPermission -> {
            mainTitle = "PERMISSÃO DE CÂMERA"
            subtitle = "Toque na tela para autorizar a câmera traseira"
        }
        uiState.errorMessage != null -> {
            mainTitle = "ATENÇÃO"
            subtitle = uiState.errorMessage
        }
        uiState.lastConfirmedBanknote != null -> {
            val detection = uiState.lastConfirmedBanknote
            mainTitle = detection.banknoteClass.spokenText.uppercase(Locale.forLanguageTag("pt-BR"))
            val confidencePct = (detection.confidence * 100).toInt()
            subtitle = "${detection.banknoteClass.displayTitle} • Confiança: $confidencePct%"
        }
        else -> {
            mainTitle = "APONTE PARA A CÉDULA"
            subtitle = uiState.lastSpokenAnnouncement
        }
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(HighContrastBlack, RoundedCornerShape(16.dp))
            .border(4.dp, HighContrastYellow, RoundedCornerShape(16.dp))
            .padding(24.dp)
            .semantics {
                liveRegion = LiveRegionMode.Assertive
                contentDescription = "$mainTitle. $subtitle"
            },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = mainTitle,
                color = HighContrastYellow,
                fontSize = 36.sp,
                lineHeight = 42.sp,
                fontWeight = FontWeight.Black,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = subtitle,
                color = HighContrastWhite,
                fontSize = 22.sp,
                lineHeight = 28.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun HighContrastBottomGesturesCard() {
    Surface(
        color = HighContrastBlack,
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(3.dp, HighContrastWhite),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "TOQUE DUPLO: Repetir última leitura",
                color = HighContrastYellow,
                fontSize = 20.sp,
                fontWeight = FontWeight.ExtraBold
            )
            Text(
                text = "TOQUE LONGO: Ligar / Desligar Lanterna",
                color = HighContrastWhite,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
