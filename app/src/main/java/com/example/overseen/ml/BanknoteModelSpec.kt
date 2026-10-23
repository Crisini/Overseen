package com.example.overseen.ml

import android.graphics.RectF

/**
 * RF03.2: As 14 classes oficiais de cédulas de Real brasileiro (1ª e 2ª famílias).
 * RF04.4: Frases ultra-curtas para síntese concisa de voz (TTS).
 */
enum class BanknoteClass(
    val classIndex: Int,
    val modelLabel: String,
    val spokenText: String,
    val displayTitle: String
) {
    // 1ª Família (antigas)
    REAL_1_ANTIGA(0, "1_real_antiga", "1 real, antiga", "R$ 1 (1ª Família)"),
    REAIS_2_ANTIGA(1, "2_reais_antiga", "2 reais, antiga", "R$ 2 (1ª Família)"),
    REAIS_5_ANTIGA(2, "5_reais_antiga", "5 reais, antiga", "R$ 5 (1ª Família)"),
    REAIS_10_ANTIGA(3, "10_reais_antiga", "10 reais, antiga", "R$ 10 (1ª Família)"),
    REAIS_20_ANTIGA(4, "20_reais_antiga", "20 reais, antiga", "R$ 20 (1ª Família)"),
    REAIS_50_ANTIGA(5, "50_reais_antiga", "50 reais, antiga", "R$ 50 (1ª Família)"),
    REAIS_100_ANTIGA(6, "100_reais_antiga", "100 reais, antiga", "R$ 100 (1ª Família)"),

    // 2ª Família (atuais)
    REAIS_2_NOVA(7, "2_reais_nova", "2 reais, nova", "R$ 2 (2ª Família)"),
    REAIS_5_NOVA(8, "5_reais_nova", "5 reais, nova", "R$ 5 (2ª Família)"),
    REAIS_10_NOVA(9, "10_reais_nova", "10 reais, nova", "R$ 10 (2ª Família)"),
    REAIS_20_NOVA(10, "20_reais_nova", "20 reais, nova", "R$ 20 (2ª Família)"),
    REAIS_50_NOVA(11, "50_reais_nova", "50 reais, nova", "R$ 50 (2ª Família)"),
    REAIS_100_NOVA(12, "100_reais_nova", "100 reais, nova", "R$ 100 (2ª Família)"),
    REAIS_200_NOVA(13, "200_reais_nova", "200 reais, nova", "R$ 200 (2ª Família)");

    companion object {
        private val byIndex = entries.associateBy { it.classIndex }
        private val byLabel = entries.associateBy { it.modelLabel }

        fun fromIndex(index: Int): BanknoteClass? = byIndex[index]
        fun fromLabel(label: String): BanknoteClass? = byLabel[label.trim()]
    }
}

data class NormalizedBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class BanknoteDetection(
    val banknoteClass: BanknoteClass,
    val confidence: Float,
    val boundingBox: NormalizedBox = NormalizedBox(0f, 0f, 1f, 1f),
    val inferenceTimeMs: Long = 0L
)

/**
 * RF03.3: Buffer de Confirmação.
 * Só dispara o anúncio sonoro se a mesma classe for detectada com score >= 0.75
 * em pelo menos 2 frames consecutivos.
 */
class ConfirmationBuffer(
    private val requiredConsecutiveFrames: Int = 2,
    private val minConfidence: Float = 0.75f,
    private val repeatCooldownMs: Long = 4000L
) {
    private var currentStreakClass: BanknoteClass? = null
    private var consecutiveCount: Int = 0
    private var lastAnnouncedClass: BanknoteClass? = null
    private var lastAnnouncedTimeMs: Long = 0L

    /**
     * Registra a melhor detecção do frame atual.
     * Retorna a [BanknoteDetection] somente quando atingir o critério de confirmação
     * (2 frames consecutivos da mesma classe com score >= 0.75) e respeitar o cooldown de repetição.
     */
    @Synchronized
    fun update(detection: BanknoteDetection?, nowMs: Long = System.currentTimeMillis()): BanknoteDetection? {
        if (detection == null || detection.confidence < minConfidence) {
            currentStreakClass = null
            consecutiveCount = 0
            return null
        }

        if (detection.banknoteClass == currentStreakClass) {
            consecutiveCount++
        } else {
            currentStreakClass = detection.banknoteClass
            consecutiveCount = 1
        }

        if (consecutiveCount >= requiredConsecutiveFrames) {
            val isDifferentClass = detection.banknoteClass != lastAnnouncedClass
            val isCooldownExpired = (nowMs - lastAnnouncedTimeMs) >= repeatCooldownMs

            if (isDifferentClass || isCooldownExpired) {
                lastAnnouncedClass = detection.banknoteClass
                lastAnnouncedTimeMs = nowMs
                return detection
            }
        }
        return null
    }

    @Synchronized
    fun reset() {
        currentStreakClass = null
        consecutiveCount = 0
        lastAnnouncedClass = null
        lastAnnouncedTimeMs = 0L
    }
}

/**
 * Utilitário de Non-Maximum Suppression (NMS) para filtrar caixas delimitadoras sobrepostas
 * em modelos de detecção TFLite.
 */
object NonMaximumSuppression {

    fun applyNMS(
        detections: List<BanknoteDetection>,
        iouThreshold: Float = 0.45f,
        confidenceThreshold: Float = 0.75f
    ): List<BanknoteDetection> {
        val filtered = detections
            .filter { it.confidence >= confidenceThreshold }
            .sortedByDescending { it.confidence }
            .toMutableList()

        val kept = mutableListOf<BanknoteDetection>()

        while (filtered.isNotEmpty()) {
            val best = filtered.removeAt(0)
            kept.add(best)

            val iterator = filtered.iterator()
            while (iterator.hasNext()) {
                val candidate = iterator.next()
                if (computeIoU(best.boundingBox, candidate.boundingBox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return kept
    }

    fun computeIoU(boxA: NormalizedBox, boxB: NormalizedBox): Float {
        val interLeft = maxOf(boxA.left, boxB.left)
        val interTop = maxOf(boxA.top, boxB.top)
        val interRight = minOf(boxA.right, boxB.right)
        val interBottom = minOf(boxA.bottom, boxB.bottom)

        val interWidth = maxOf(0f, interRight - interLeft)
        val interHeight = maxOf(0f, interBottom - interTop)
        val interArea = interWidth * interHeight

        if (interArea <= 0f) return 0f

        val areaA = maxOf(0f, boxA.right - boxA.left) * maxOf(0f, boxA.bottom - boxA.top)
        val areaB = maxOf(0f, boxB.right - boxB.left) * maxOf(0f, boxB.bottom - boxB.top)
        val unionArea = areaA + areaB - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }
}
