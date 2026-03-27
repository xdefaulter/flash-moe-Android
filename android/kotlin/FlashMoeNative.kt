package com.flashmoe

object FlashMoeNative {
    init {
        System.loadLibrary("flashmoe_android")
    }

    data class Stats(
        val expertReads: Long,
        val expertBytes: Long,
        val ioMs: Double,
        val generateMs: Double,
    )

    external fun init(modelDir: String, modelConfigPath: String? = null): Int
    external fun generate(inputTokens: IntArray, maxNewTokens: Int): IntArray?
    external fun getStats(): Stats?
    external fun shutdown()
}
