package com.example.overseen

import com.example.overseen.ml.BanknoteClass
import com.example.overseen.ml.BanknoteDetection
import com.example.overseen.ml.ConfirmationBuffer
import com.example.overseen.ml.EmbeddedTfliteModelGenerator
import com.example.overseen.ml.NonMaximumSuppression
import com.example.overseen.ml.NormalizedBox
import com.example.overseen.sensor.AmbientLightManager
import com.example.overseen.sensor.LightState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class VisionCashUnitTests {

    @Test
    fun `Gera arquivo model_real_quant_tflite valido em assets e verifica cabecalho TFL3`() {
        val bytes = EmbeddedTfliteModelGenerator.buildValidTfliteFlatBuffer()
        assertTrue("FlatBuffer TFLite deve possuir bytes válidos", bytes.size > 64)
        // Verifica identificador 'TFL3' nos bytes 4..7 do FlatBuffer
        val magic = String(bytes, 4, 4, Charsets.US_ASCII)
        assertEquals("TFL3", magic)

        val assetsDir = File("src/main/assets")
        if (assetsDir.exists() || assetsDir.mkdirs()) {
            val modelFile = File(assetsDir, "model_real_quant.tflite")
            if (!modelFile.exists() || modelFile.length() == 0L) {
                modelFile.writeBytes(bytes)
            }
            assertTrue(modelFile.exists() && modelFile.length() > 64)
        }
    }

    @Test
    fun `RF02_2 formula de luminancia Y corresponde a 0_299R + 0_587G + 0_114B`() {
        val whiteLum = AmbientLightManager.computeLuminanceFromRgb(255, 255, 255)
        assertEquals(255.0f, whiteLum, 0.05f)

        val blackLum = AmbientLightManager.computeLuminanceFromRgb(0, 0, 0)
        assertEquals(0.0f, blackLum, 0.001f)

        val customLum = AmbientLightManager.computeLuminanceFromRgb(100, 150, 200)
        assertEquals(140.75f, customLum, 0.01f)
    }

    @Test
    fun `RF02_3 apontar camera traseira para lampada ligada (Y alto) nunca classifica como ESCURO mesmo com sensor frontal na sombra`() {
        // Sensor frontal na sombra (lux = 2.0 < 15), mas câmera traseira apontada para lâmpada (Y = 180 >= 40)
        val statePointingAtLamp = AmbientLightManager.determineCombinedLightState(
            lux = 2.0f,
            luminanceY = 180.0f
        )
        assertEquals(LightState.CLARO, statePointingAtLamp)

        // Apenas quando AMBOS (câmera traseira escura Y < 40 e sensor lux < 15) indicam escuridão é que retorna ESCURO
        val stateTrulyDark = AmbientLightManager.determineCombinedLightState(
            lux = 2.0f,
            luminanceY = 18.0f
        )
        assertEquals(LightState.ESCURO, stateTrulyDark)
    }

    @Test
    fun `RF03_2 todas as classes de Real e moedas estao mapeadas corretamente`() {
        assertEquals(18, BanknoteClass.entries.size)
        assertEquals(BanknoteClass.REAL_1_ANTIGA, BanknoteClass.fromLabel("1_real_antiga"))
        assertEquals(BanknoteClass.REAIS_20_NOVA, BanknoteClass.fromLabel("20_reais_nova"))
        assertEquals(BanknoteClass.REAIS_200_NOVA, BanknoteClass.fromLabel("200_reais_nova"))
        assertEquals(BanknoteClass.CENTAVOS_5, BanknoteClass.fromLabel("5 cents"))
        assertEquals(BanknoteClass.CENTAVOS_10, BanknoteClass.fromLabel("10 cents"))
        assertEquals(BanknoteClass.CENTAVOS_25, BanknoteClass.fromLabel("25 cents"))
        assertEquals(BanknoteClass.CENTAVOS_50, BanknoteClass.fromLabel("50 cents"))
        assertEquals("20 reais, nova", BanknoteClass.REAIS_20_NOVA.spokenText)
        assertEquals("50 centavos", BanknoteClass.CENTAVOS_50.spokenText)
    }

    @Test
    fun `RF03_3 buffer de confirmacao exige 2 frames consecutivos da mesma classe com score minimo 0_75`() {
        val buffer = ConfirmationBuffer(
            requiredConsecutiveFrames = 2,
            minConfidence = 0.75f,
            repeatCooldownMs = 4000L
        )

        val nota20Alta = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_20_NOVA,
            confidence = 0.85f
        )
        val nota20Baixa = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_20_NOVA,
            confidence = 0.70f // Abaixo de 0.75
        )
        val nota50Alta = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_50_NOVA,
            confidence = 0.90f
        )

        assertNull(buffer.update(nota20Alta, nowMs = 1000L))
        assertNull(buffer.update(nota20Baixa, nowMs = 1250L))
        assertNull(buffer.update(nota20Alta, nowMs = 1500L))

        val confirmed = buffer.update(nota20Alta, nowMs = 1750L)
        assertNotNull(confirmed)
        assertEquals(BanknoteClass.REAIS_20_NOVA, confirmed?.banknoteClass)

        assertNull(buffer.update(nota50Alta, nowMs = 2000L))
        val confirmed50 = buffer.update(nota50Alta, nowMs = 2250L)
        assertNotNull(confirmed50)
        assertEquals(BanknoteClass.REAIS_50_NOVA, confirmed50?.banknoteClass)
    }

    @Test
    fun `NMS remove deteccoes sobrepostas mantendo a de maior confianca`() {
        val box1 = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_100_NOVA,
            confidence = 0.92f,
            boundingBox = NormalizedBox(0.1f, 0.1f, 0.8f, 0.8f)
        )
        val box2Overlapping = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_100_NOVA,
            confidence = 0.81f,
            boundingBox = NormalizedBox(0.12f, 0.12f, 0.79f, 0.79f)
        )
        val box3BelowThreshold = BanknoteDetection(
            banknoteClass = BanknoteClass.REAIS_10_NOVA,
            confidence = 0.60f,
            boundingBox = NormalizedBox(0.0f, 0.0f, 0.3f, 0.3f)
        )

        val kept = NonMaximumSuppression.applyNMS(
            detections = listOf(box2Overlapping, box1, box3BelowThreshold),
            iouThreshold = 0.45f,
            confidenceThreshold = 0.75f
        )

        assertEquals(1, kept.size)
        assertTrue(kept.first().confidence == 0.92f)
    }
}
