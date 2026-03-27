#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned long long expert_reads;
    unsigned long long expert_bytes;
    double io_ms;
    double generate_ms;
} fm_android_stats_t;

// Android inference API (JNI-friendly C ABI)
int fm_android_init(const char* model_dir, const char* model_config_path);
int fm_android_generate(const int* input_tokens, int input_len, int max_new_tokens,
                        int* output_tokens, int output_cap);
int fm_android_get_stats(fm_android_stats_t* out_stats);
void fm_android_shutdown(void);

#ifdef __cplusplus
}
#endif
