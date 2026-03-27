#pragma once

#include <cstddef>
#include <cstdint>

struct VulkanBackend {
    bool initialized;
    bool using_cpu_fallback;
    bool has_vulkan_compute;
    void* plugin_handle;
    void* fn_dequant_matvec;
    void* fn_swiglu;
    void* fn_moe_combine;
    void* fn_rmsnorm;
    void* fn_attention;
    void* fn_softmax;
};

// Initializes Vulkan backend when available.
// Returns true on success; false means caller should use CPU/surrogate path.
bool vk_backend_init(VulkanBackend* backend);

// CPU fallback kernels with Vulkan-compatible signatures.
bool vk_backend_dequant_matvec(VulkanBackend* backend,
                               const uint8_t* packed_weights,
                               size_t packed_len,
                               const float* x,
                               size_t in_dim,
                               float* out,
                               size_t out_dim);

bool vk_backend_swiglu(VulkanBackend* backend,
                       const float* gate,
                       const float* up,
                       size_t n,
                       float* out);

bool vk_backend_moe_combine(VulkanBackend* backend,
                            const float* expert_outputs,
                            const float* weights,
                            size_t num_experts,
                            size_t hidden_dim,
                            float* out);

bool vk_backend_rmsnorm(VulkanBackend* backend,
                        float* x,
                        size_t n,
                        float eps);

bool vk_backend_attention(VulkanBackend* backend,
                          const float* q,
                          const float* k_cache,
                          const float* v_cache,
                          size_t seq_len,
                          size_t num_heads,
                          size_t head_dim,
                          float* out);

bool vk_backend_softmax(VulkanBackend* backend,
                        float* x,
                        size_t n);

void vk_backend_shutdown(VulkanBackend* backend);
