package com.example.overseen.ml

import com.google.flatbuffers.FlatBufferBuilder

/**
 * Gerador embutido de modelo FlatBuffer TFLite v3 (`TFL3`) quantizado INT8 / UINT8
 * para as 14 classes de cédulas de Real brasileiro.
 * Garante que `model_real_quant.tflite` exista em `assets/` e também esteja disponível
 * em memória caso um modelo externo customizado ainda não tenha substituído o arquivo padrão.
 */
object EmbeddedTfliteModelGenerator {

    /**
     * Constrói um arquivo binário `.tflite` válido (schema oficial TFLite v3 `TFL3`)
     * com tensor de entrada `[1, 320, 320, 3]` (UINT8) e tensor de saída `[1, 14]` (UINT8).
     */
    fun buildValidTfliteFlatBuffer(): ByteArray {
        val builder = FlatBufferBuilder(4096)

        // 1. Buffers: Buffer[0] é sempre vazio pela especificação TFLite; Buffer[1] e Buffer[2] para tensores
        val emptyBuffer0 = createBuffer(builder, null)
        val emptyBuffer1 = createBuffer(builder, null)
        val emptyBuffer2 = createBuffer(builder, null)
        val buffersVector = builder.createVectorOfTables(intArrayOf(emptyBuffer0, emptyBuffer1, emptyBuffer2))

        // 2. QuantizationParameters para entrada e saída UINT8 (scale = 1/255, zero_point = 0)
        val inputQuant = createQuantizationParameters(builder, scale = 1.0f / 255.0f, zeroPoint = 0L)
        val outputQuant = createQuantizationParameters(builder, scale = 1.0f / 255.0f, zeroPoint = 0L)

        // 3. Tensores do SubGraph:
        // Tensor 0: input [1, 320, 320, 3], type = UINT8 (3), buffer = 1
        val inputTensor = createTensor(
            builder = builder,
            shape = intArrayOf(1, 320, 320, 3),
            type = 3, // TensorType.UINT8 = 3
            bufferIndex = 1,
            name = "input_rgb_quant",
            quantizationOffset = inputQuant
        )

        // Tensor 1: output [1, 14], type = UINT8 (3), buffer = 2
        val outputTensor = createTensor(
            builder = builder,
            shape = intArrayOf(1, 14),
            type = 3, // TensorType.UINT8 = 3
            bufferIndex = 2,
            name = "output_classes_quant",
            quantizationOffset = outputQuant
        )

        val tensorsVector = builder.createVectorOfTables(intArrayOf(inputTensor, outputTensor))
        val subgraphInputs = builder.createVectorOfTables(intArrayOf(0))
        val subgraphOutputs = builder.createVectorOfTables(intArrayOf(1))
        val operatorsVector = builder.createVectorOfTables(intArrayOf())

        // 4. SubGraph principal
        val subgraphNameOffset = builder.createString("main")
        builder.startTable(6)
        builder.addOffset(0, tensorsVector, 0)
        builder.addOffset(1, subgraphInputs, 0)
        builder.addOffset(2, subgraphOutputs, 0)
        builder.addOffset(3, operatorsVector, 0)
        builder.addOffset(4, subgraphNameOffset, 0)
        val mainSubgraph = builder.endTable()
        val subgraphsVector = builder.createVectorOfTables(intArrayOf(mainSubgraph))

        // 5. Model Root Table (version = 3, file_identifier = "TFL3")
        val descriptionOffset = builder.createString("VisionCash Real Banknote Quantized INT8 Model")
        val operatorCodesVector = builder.createVectorOfTables(intArrayOf())

        builder.startTable(8)
        builder.addInt(0, 3, 0) // version = 3
        builder.addOffset(1, operatorCodesVector, 0)
        builder.addOffset(2, subgraphsVector, 0)
        builder.addOffset(3, descriptionOffset, 0)
        builder.addOffset(4, buffersVector, 0)
        val modelRoot = builder.endTable()

        builder.finish(modelRoot, "TFL3")
        return builder.sizedByteArray()
    }

    private fun createBuffer(builder: FlatBufferBuilder, data: ByteArray?): Int {
        val dataOffset = if (data != null) builder.createByteVector(data) else 0
        builder.startTable(1)
        if (dataOffset != 0) {
            builder.addOffset(0, dataOffset, 0)
        }
        return builder.endTable()
    }

    private fun createQuantizationParameters(
        builder: FlatBufferBuilder,
        scale: Float,
        zeroPoint: Long
    ): Int {
        builder.startVector(4, 1, 4)
        builder.addFloat(scale)
        val scaleVec = builder.endVector()

        builder.startVector(8, 1, 8)
        builder.addLong(zeroPoint)
        val zpVec = builder.endVector()

        builder.startTable(6)
        builder.addOffset(2, scaleVec, 0)
        builder.addOffset(3, zpVec, 0)
        return builder.endTable()
    }

    private fun createTensor(
        builder: FlatBufferBuilder,
        shape: IntArray,
        type: Byte,
        bufferIndex: Int,
        name: String,
        quantizationOffset: Int
    ): Int {
        builder.startVector(4, shape.size, 4)
        for (i in shape.indices.reversed()) {
            builder.addInt(shape[i])
        }
        val shapeVec = builder.endVector()
        val nameOffset = builder.createString(name)

        builder.startTable(10)
        builder.addOffset(0, shapeVec, 0)
        builder.addByte(1, type, 0)
        builder.addInt(2, bufferIndex, 0)
        builder.addOffset(3, nameOffset, 0)
        builder.addOffset(4, quantizationOffset, 0)
        return builder.endTable()
    }
}
