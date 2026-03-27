#include "vulkan_backend.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace {

typedef bool (*FnDequantMatVec)(const uint8_t*, size_t, const float*, size_t, float*, size_t);
typedef bool (*FnSwiGLU)(const float*, const float*, size_t, float*);
typedef bool (*FnMoeCombine)(const float*, const float*, size_t, size_t, float*);
typedef bool (*FnRmsNorm)(float*, size_t, float);
typedef bool (*FnAttention)(const float*, const float*, const float*, size_t, size_t, size_t, float*);
typedef bool (*FnSoftmax)(float*, size_t);

void assign_null_functions(VulkanBackend* backend) {
    backend->fn_dequant_matvec = nullptr;
    backend->fn_swiglu = nullptr;
    backend->fn_moe_combine = nullptr;
    backend->fn_rmsnorm = nullptr;
    backend->fn_attention = nullptr;
    backend->fn_softmax = nullptr;
}

bool try_init_plugin_backend(VulkanBackend* backend) {
#if defined(__linux__) || defined(__ANDROID__)
    const char* disabled = std::getenv("FLASHMOE_DISABLE_VULKAN_PLUGIN");
    if (disabled && disabled[0] == '1') return false;

    const char* lib_name = std::getenv("FLASHMOE_VULKAN_PLUGIN");
    if (!lib_name || !lib_name[0]) lib_name = "libflashmoe_vulkan_kernels.so";

    void* handle = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return false;

    backend->plugin_handle = handle;
    backend->fn_dequant_matvec = dlsym(handle, "fm_vk_dequant_matvec");
    backend->fn_swiglu = dlsym(handle, "fm_vk_swiglu");
    backend->fn_moe_combine = dlsym(handle, "fm_vk_moe_combine");
    backend->fn_rmsnorm = dlsym(handle, "fm_vk_rmsnorm");
    backend->fn_attention = dlsym(handle, "fm_vk_attention");
    backend->fn_softmax = dlsym(handle, "fm_vk_softmax");

    const bool full = backend->fn_dequant_matvec && backend->fn_swiglu && backend->fn_moe_combine &&
                      backend->fn_rmsnorm && backend->fn_attention && backend->fn_softmax;
    if (!full) {
        dlclose(handle);
        backend->plugin_handle = nullptr;
        assign_null_functions(backend);
        return false;
    }
    return true;
#else
    (void)backend;
    return false;
#endif
}

void shutdown_plugin_backend(VulkanBackend* backend) {
#if defined(__linux__) || defined(__ANDROID__)
    if (backend->plugin_handle) dlclose(backend->plugin_handle);
#else
    (void)backend;
#endif
    backend->plugin_handle = nullptr;
    assign_null_functions(backend);
}

float nibble_to_weight(uint8_t q) {
    return (static_cast<int>(q) - 8) / 8.0f;
}

bool cpu_dequant_matvec(const uint8_t* packed_weights,
                        size_t packed_len,
                        const float* x,
                        size_t in_dim,
                        float* out,
                        size_t out_len) {
    if (!packed_weights || !x || !out || packed_len == 0 || in_dim == 0 || out_len == 0) return false;
    const size_t nibbles = packed_len * 2;
    for (size_t o = 0; o < out_len; ++o) {
        float acc = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) {
            const size_t idx = (o * in_dim + i) % nibbles;
            const uint8_t byte = packed_weights[idx >> 1];
            const uint8_t q = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
            acc += nibble_to_weight(q) * x[i];
        }
        out[o] = acc;
    }
    return true;
}

bool cpu_swiglu(const float* gate, const float* up, size_t n, float* out) {
    if (!gate || !up || !out) return false;
    for (size_t i = 0; i < n; ++i) {
        const float g = gate[i];
        const float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
    return true;
}

bool cpu_moe_combine(const float* expert_outputs,
                     const float* weights,
                     size_t num_experts,
                     size_t hidden_dim,
                     float* out) {
    if (!expert_outputs || !weights || !out || num_experts == 0 || hidden_dim == 0) return false;
    std::memset(out, 0, hidden_dim * sizeof(float));
    for (size_t e = 0; e < num_experts; ++e) {
        const float w = weights[e];
        const float* src = expert_outputs + e * hidden_dim;
        for (size_t i = 0; i < hidden_dim; ++i) out[i] += w * src[i];
    }
    return true;
}

bool cpu_rmsnorm(float* x, size_t n, float eps) {
    if (!x || n == 0) return false;
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
    return true;
}

bool cpu_softmax(float* x, size_t n) {
    if (!x || n == 0) return false;
    float maxv = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) maxv = x[i] > maxv ? x[i] : maxv;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxv);
        s += x[i];
    }
    if (s <= 0.0) return false;
    const float inv = 1.0f / static_cast<float>(s);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
    return true;
}

bool cpu_attention(const float* q,
                   const float* k_cache,
                   const float* v_cache,
                   size_t seq_len,
                   size_t num_heads,
                   size_t head_dim,
                   float* out) {
    if (!q || !k_cache || !v_cache || !out || seq_len == 0 || num_heads == 0 || head_dim == 0) return false;
    std::memset(out, 0, num_heads * head_dim * sizeof(float));

    std::vector<float> scores(seq_len, 0.0f);
    std::vector<float> probs(seq_len, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t h = 0; h < num_heads; ++h) {
        const float* qh = q + h * head_dim;
        for (size_t t = 0; t < seq_len; ++t) {
            const float* kh = k_cache + (t * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
            scores[t] = dot * scale;
        }
        std::memcpy(probs.data(), scores.data(), seq_len * sizeof(float));
        if (!cpu_softmax(probs.data(), seq_len)) return false;

        float* oh = out + h * head_dim;
        for (size_t t = 0; t < seq_len; ++t) {
            const float p = probs[t];
            const float* vh = v_cache + (t * num_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) oh[d] += p * vh[d];
        }
    }
    return true;
}

}  // namespace

bool vk_backend_init(VulkanBackend* backend) {
    if (!backend) return false;
    backend->initialized = true;
    backend->has_vulkan_compute = try_init_plugin_backend(backend);
    backend->using_cpu_fallback = !backend->has_vulkan_compute;
    return true;
}

bool vk_backend_dequant_matvec(VulkanBackend* backend,
                               const uint8_t* packed_weights,
                               size_t packed_len,
                               const float* x,
                               size_t in_dim,
                               float* out,
                               size_t out_len) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnDequantMatVec>(backend->fn_dequant_matvec)(packed_weights, packed_len, x, in_dim,
                                                                              out, out_len);
    }
    return cpu_dequant_matvec(packed_weights, packed_len, x, in_dim, out, out_len);
}

bool vk_backend_swiglu(VulkanBackend* backend,
                       const float* gate,
                       const float* up,
                       size_t n,
                       float* out) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnSwiGLU>(backend->fn_swiglu)(gate, up, n, out);
    }
    return cpu_swiglu(gate, up, n, out);
}

bool vk_backend_moe_combine(VulkanBackend* backend,
                            const float* expert_outputs,
                            const float* weights,
                            size_t num_experts,
                            size_t hidden_dim,
                            float* out) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnMoeCombine>(backend->fn_moe_combine)(expert_outputs, weights, num_experts, hidden_dim,
                                                                        out);
    }
    return cpu_moe_combine(expert_outputs, weights, num_experts, hidden_dim, out);
}

bool vk_backend_rmsnorm(VulkanBackend* backend,
                        float* x,
                        size_t n,
                        float eps) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnRmsNorm>(backend->fn_rmsnorm)(x, n, eps);
    }
    return cpu_rmsnorm(x, n, eps);
}

bool vk_backend_attention(VulkanBackend* backend,
                          const float* q,
                          const float* k_cache,
                          const float* v_cache,
                          size_t seq_len,
                          size_t num_heads,
                          size_t head_dim,
                          float* out) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnAttention>(backend->fn_attention)(q, k_cache, v_cache, seq_len, num_heads, head_dim,
                                                                    out);
    }
    return cpu_attention(q, k_cache, v_cache, seq_len, num_heads, head_dim, out);
}

bool vk_backend_softmax(VulkanBackend* backend,
                        float* x,
                        size_t n) {
    if (!backend || !backend->initialized) return false;
    if (backend->has_vulkan_compute) {
        return reinterpret_cast<FnSoftmax>(backend->fn_softmax)(x, n);
    }
    return cpu_softmax(x, n);
}

void vk_backend_shutdown(VulkanBackend* backend) {
    if (!backend) return;
    shutdown_plugin_backend(backend);
    backend->initialized = false;
    backend->has_vulkan_compute = false;
    backend->using_cpu_fallback = true;
}
