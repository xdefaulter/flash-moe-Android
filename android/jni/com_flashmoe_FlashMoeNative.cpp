#include <jni.h>
#include <vector>

#include "infer_android.h"

extern "C" JNIEXPORT jint JNICALL
Java_com_flashmoe_FlashMoeNative_init(JNIEnv* env, jclass, jstring modelDir, jstring modelConfigPath) {
    const char* model_dir = env->GetStringUTFChars(modelDir, nullptr);
    const char* cfg = nullptr;
    if (modelConfigPath != nullptr) {
        cfg = env->GetStringUTFChars(modelConfigPath, nullptr);
    }

    int rc = fm_android_init(model_dir, cfg);

    env->ReleaseStringUTFChars(modelDir, model_dir);
    if (modelConfigPath != nullptr && cfg != nullptr) {
        env->ReleaseStringUTFChars(modelConfigPath, cfg);
    }
    return rc;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_flashmoe_FlashMoeNative_generate(JNIEnv* env, jclass,
                                          jintArray inputTokens,
                                          jint maxNewTokens) {
    if (inputTokens == nullptr || maxNewTokens <= 0) {
        return nullptr;
    }

    const jsize in_len = env->GetArrayLength(inputTokens);
    if (in_len <= 0) return nullptr;

    jint* in_ptr = env->GetIntArrayElements(inputTokens, nullptr);
    jintArray out = env->NewIntArray(maxNewTokens);
    if (out == nullptr) {
        env->ReleaseIntArrayElements(inputTokens, in_ptr, JNI_ABORT);
        return nullptr;
    }

    std::vector<jint> out_vec(static_cast<size_t>(maxNewTokens));
    int produced = fm_android_generate(
        reinterpret_cast<const int*>(in_ptr),
        static_cast<int>(in_len),
        static_cast<int>(maxNewTokens),
        reinterpret_cast<int*>(out_vec.data()),
        static_cast<int>(maxNewTokens));

    env->ReleaseIntArrayElements(inputTokens, in_ptr, JNI_ABORT);
    if (produced < 0) {
        return nullptr;
    }

    env->SetIntArrayRegion(out, 0, produced, out_vec.data());
    return out;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_flashmoe_FlashMoeNative_getStats(JNIEnv* env, jclass) {
    fm_android_stats_t stats{};
    if (fm_android_get_stats(&stats) != 0) return nullptr;

    jclass cls = env->FindClass("com/flashmoe/FlashMoeNative$Stats");
    if (!cls) return nullptr;

    jmethodID ctor = env->GetMethodID(cls, "<init>", "(JJDD)V");
    if (!ctor) return nullptr;

    return env->NewObject(cls, ctor,
                          static_cast<jlong>(stats.expert_reads),
                          static_cast<jlong>(stats.expert_bytes),
                          static_cast<jdouble>(stats.io_ms),
                          static_cast<jdouble>(stats.generate_ms));
}

extern "C" JNIEXPORT void JNICALL
Java_com_flashmoe_FlashMoeNative_shutdown(JNIEnv*, jclass) {
    fm_android_shutdown();
}
