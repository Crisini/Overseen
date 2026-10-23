package com.example.overseen.audio

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Gerenciador assíncrono de Text-to-Speech (TTS) e Feedback Háptico (RF04.3 / RF04.4).
 * Garante interrupção imediata de falas antigas usando TextToSpeech.QUEUE_FLUSH.
 */
class AccessibilityFeedbackManager(context: Context) : TextToSpeech.OnInitListener {

    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = null
    @Volatile
    private var isTtsReady = false
    private var pendingSpeech: String? = null

    @Volatile
    var lastSpokenMessage: String = "Aguardando leitura"
        private set

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        vibratorManager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    init {
        try {
            tts = TextToSpeech(appContext, this)
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao inicializar motor TTS", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val locale = Locale.forLanguageTag("pt-BR")
            val result = tts?.setLanguage(locale)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                tts?.setLanguage(Locale.getDefault())
            }
            tts?.setSpeechRate(1.15f)
            isTtsReady = true
            pendingSpeech?.let {
                speak(it, updateLastMessage = true)
                pendingSpeech = null
            }
        } else {
            Log.e(TAG, "Inicialização do TTS falhou com status: $status")
        }
    }

    /**
     * RF04.4: Síntese concisa de voz com QUEUE_FLUSH para interromper o áudio anterior
     * caso uma nova leitura relevante ocorra.
     */
    fun speak(message: String, updateLastMessage: Boolean = true) {
        if (message.isBlank()) return
        if (updateLastMessage) {
            lastSpokenMessage = message
        }
        if (!isTtsReady) {
            pendingSpeech = message
            return
        }
        tts?.speak(message, TextToSpeech.QUEUE_FLUSH, null, "visioncash_utterance_${System.currentTimeMillis()}")
    }

    /**
     * RF04.1: Toque duplo na tela repete a última leitura de cédula ou o estado atual da iluminação.
     */
    fun repeatLastMessage(fallbackLightState: String) {
        val messageToSpeak = if (lastSpokenMessage.isNotBlank() && lastSpokenMessage != "Aguardando leitura") {
            lastSpokenMessage
        } else {
            fallbackLightState
        }
        vibrateBanknoteDetected()
        speak(messageToSpeak, updateLastMessage = false)
    }

    /**
     * RF04.3: 1 vibração curta (50ms) - Cédula reconhecida.
     */
    fun vibrateBanknoteDetected() {
        vibrateOneShot(50L, VibrationEffect.DEFAULT_AMPLITUDE)
    }

    /**
     * RF04.3: 2 vibrações médias (100ms cada) - Estado de luz alterado.
     */
    fun vibrateLightStateChanged() {
        val timings = longArrayOf(0L, 100L, 80L, 100L)
        val amplitudes = intArrayOf(0, 200, 0, 200)
        vibrateWaveform(timings, amplitudes)
    }

    /**
     * RF04.3: 1 vibração longa (400ms) - Iluminação insuficiente para leitura ou erro crítico.
     */
    fun vibrateInsufficientLightOrError() {
        vibrateOneShot(400L, 255)
    }

    /**
     * Anuncia uma cédula reconhecida (TTS + 1 vibração curta de 50ms).
     */
    fun announceBanknote(spokenLabel: String) {
        vibrateBanknoteDetected()
        speak(spokenLabel, updateLastMessage = true)
    }

    /**
     * Anuncia transição de estado de luminosidade (TTS + vibração correspondente).
     */
    fun announceLightTransition(isDark: Boolean, spokenState: String) {
        if (isDark) {
            vibrateInsufficientLightOrError()
        } else {
            vibrateLightStateChanged()
        }
        speak(spokenState, updateLastMessage = true)
    }

    /**
     * Notifica erro crítico (falha de câmera ou modelo TFLite) via TTS e vibração.
     */
    fun announceError(errorMessage: String) {
        vibrateInsufficientLightOrError()
        speak(errorMessage, updateLastMessage = true)
    }

    private fun vibrateOneShot(durationMs: Long, amplitude: Int) {
        val vib = vibrator ?: return
        if (!vib.hasVibrator()) return
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vib.vibrate(VibrationEffect.createOneShot(durationMs, amplitude))
            } else {
                @Suppress("DEPRECATION")
                vib.vibrate(durationMs)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Erro ao acionar vibração curta", e)
        }
    }

    private fun vibrateWaveform(timings: LongArray, amplitudes: IntArray) {
        val vib = vibrator ?: return
        if (!vib.hasVibrator()) return
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vib.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
            } else {
                @Suppress("DEPRECATION")
                vib.vibrate(timings, -1)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Erro ao acionar padrão de vibração", e)
        }
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (e: Exception) {
            Log.w(TAG, "Erro ao encerrar TTS", e)
        } finally {
            tts = null
            isTtsReady = false
        }
    }

    companion object {
        private const val TAG = "AccessibilityFeedback"
    }
}
