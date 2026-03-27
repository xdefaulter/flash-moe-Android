#include "infer_android.h"
#include "vulkan_backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

struct ModelConfig {
    int experts_per_token = 4;
    int num_layers = 0;
    int vocab_size = 151936;
    int hidden_dim = 256;
    int moe_intermediate = 256;
    int num_attention_heads = 8;
    int head_dim = 32;
    int context_len = 512;
};

struct LayoutConfig {
    int expert_size = 0;
    int num_layers = 0;
    int num_experts = 0;
    int layer_size = 0;
    std::vector<std::string> layer_files;
};

struct AndroidRuntimeState {
    std::string model_dir;
    std::string model_config_path;
    LayoutConfig layout;
    ModelConfig model_cfg;
    std::vector<int> layer_fds;
    std::vector<uint8_t> io_buffer;
    std::vector<uint8_t> model_weights;

    std::vector<float> hidden;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_out;
    std::vector<float> ffn_out;
    std::vector<float> expert_out_scratch;
    std::vector<float> router_weights;
    std::vector<float> tmp_gate;
    std::vector<float> tmp_up;
    std::vector<float> tmp_act;
    std::vector<float> logits;

    std::vector<float> k_cache;  // [layer, position, hidden]
    std::vector<float> v_cache;  // [layer, position, hidden]

    VulkanBackend backend{};
    fm_android_stats_t stats{};
    int last_position = 0;
    std::atomic<bool> initialized{false};
};

AndroidRuntimeState g_state;
std::mutex g_mu;

std::string read_text_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("failed to open file: " + path);
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        throw std::runtime_error("failed to seek file: " + path);
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        throw std::runtime_error("failed to tell file: " + path);
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("failed to rewind file: " + path);
    }
    std::string out;
    out.resize(static_cast<size_t>(n));
    size_t got = fread(out.data(), 1, out.size(), f);
    fclose(f);
    if (got != out.size()) throw std::runtime_error("short read: " + path);
    return out;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("failed to open file: " + path);
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        throw std::runtime_error("failed to seek file: " + path);
    }
    long n = ftell(f);
    if (n <= 0) {
        fclose(f);
        throw std::runtime_error("invalid binary size: " + path);
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("failed to rewind file: " + path);
    }
    std::vector<uint8_t> out(static_cast<size_t>(n));
    size_t got = fread(out.data(), 1, out.size(), f);
    fclose(f);
    if (got != out.size()) throw std::runtime_error("short binary read: " + path);
    return out;
}

size_t find_key(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) throw std::runtime_error("missing key: " + key);
    return p + needle.size();
}

int parse_int_key(const std::string& s, const std::string& key, int def = 0, bool required = false) {
    try {
        size_t p = find_key(s, key);
        p = s.find(':', p);
        if (p == std::string::npos) throw std::runtime_error("missing colon");
        p++;
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
        char* end = nullptr;
        long v = std::strtol(s.c_str() + p, &end, 10);
        if (end == s.c_str() + p) throw std::runtime_error("bad int parse");
        return static_cast<int>(v);
    } catch (...) {
        if (required) throw;
        return def;
    }
}

std::vector<std::string> parse_string_array_key(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    p = s.find('[', p);
    if (p == std::string::npos) throw std::runtime_error("missing array start for key: " + key);
    p++;
    std::vector<std::string> out;
    while (p < s.size()) {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
        if (p >= s.size()) break;
        if (s[p] == ']') break;
        if (s[p] == ',') {
            p++;
            continue;
        }
        if (s[p] != '"') throw std::runtime_error("expected string in array for key: " + key);
        size_t q = s.find('"', p + 1);
        if (q == std::string::npos) throw std::runtime_error("unterminated string for key: " + key);
        out.push_back(s.substr(p + 1, q - p - 1));
        p = q + 1;
    }
    return out;
}

ModelConfig load_model_config(const std::string& path) {
    ModelConfig cfg;
    const std::string j = read_text_file(path);
    cfg.experts_per_token = parse_int_key(j, "experts_per_token", 4, false);
    cfg.num_layers = parse_int_key(j, "num_layers", 0, true);
    cfg.vocab_size = parse_int_key(j, "vocab_size", 151936, false);
    cfg.hidden_dim = parse_int_key(j, "hidden_dim", 256, false);
    cfg.moe_intermediate = parse_int_key(j, "moe_intermediate", cfg.hidden_dim, false);
    cfg.num_attention_heads = parse_int_key(j, "num_attention_heads", 8, false);
    cfg.head_dim = parse_int_key(j, "head_dim", cfg.hidden_dim / std::max(1, cfg.num_attention_heads), false);
    cfg.context_len = parse_int_key(j, "context_len", 512, false);

    if (cfg.experts_per_token <= 0) throw std::runtime_error("experts_per_token must be >0");
    if (cfg.num_layers <= 0) throw std::runtime_error("num_layers must be >0");
    if (cfg.vocab_size <= 0) throw std::runtime_error("vocab_size must be >0");
    if (cfg.hidden_dim <= 0) throw std::runtime_error("hidden_dim must be >0");
    if (cfg.moe_intermediate <= 0) throw std::runtime_error("moe_intermediate must be >0");
    if (cfg.num_attention_heads <= 0) throw std::runtime_error("num_attention_heads must be >0");
    if (cfg.head_dim <= 0) throw std::runtime_error("head_dim must be >0");
    if (cfg.hidden_dim != cfg.num_attention_heads * cfg.head_dim) {
        throw std::runtime_error("hidden_dim must equal num_attention_heads * head_dim");
    }
    if (cfg.context_len <= 0) throw std::runtime_error("context_len must be >0");
    return cfg;
}

LayoutConfig load_layout(const std::string& path) {
    LayoutConfig cfg;
    const std::string j = read_text_file(path);
    cfg.expert_size = parse_int_key(j, "expert_size", 0, true);
    cfg.num_layers = parse_int_key(j, "num_layers", 0, true);
    cfg.num_experts = parse_int_key(j, "num_experts", 0, true);
    cfg.layer_size = parse_int_key(j, "layer_size", 0, true);
    cfg.layer_files = parse_string_array_key(j, "layer_files");
    if (cfg.expert_size <= 0 || cfg.num_layers <= 0 || cfg.num_experts <= 0 || cfg.layer_size <= 0) {
        throw std::runtime_error("layout numeric fields must be >0");
    }
    if (cfg.layer_size != cfg.expert_size * cfg.num_experts) {
        throw std::runtime_error("layout mismatch: layer_size != expert_size * num_experts");
    }
    if (static_cast<int>(cfg.layer_files.size()) != cfg.num_layers) {
        throw std::runtime_error("layout mismatch: layer_files count != num_layers");
    }
    return cfg;
}

void close_layer_fds(std::vector<int>& fds) {
    for (int fd : fds) {
        if (fd >= 0) close(fd);
    }
    fds.clear();
}

int positive_mod(int x, int mod) {
    int y = x % mod;
    return y < 0 ? y + mod : y;
}

size_t packed_bytes_for(size_t out_dim, size_t in_dim) {
    return (out_dim * in_dim + 1) / 2;
}

double now_ms() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

const uint8_t* weight_slice(const std::vector<uint8_t>& w, size_t offset, size_t* len) {
    if (w.empty()) return nullptr;
    size_t start = offset % w.size();
    size_t max_len = w.size() - start;
    if (*len > max_len) *len = max_len;
    return w.data() + start;
}

void gather_embedding(const std::vector<uint8_t>& w, int token, std::vector<float>& hidden) {
    const size_t required = packed_bytes_for(hidden.size(), 1);
    size_t len = required;
    const uint8_t* slice = weight_slice(w, static_cast<size_t>(token) * required, &len);
    if (!slice || len == 0) throw std::runtime_error("empty embedding slice");
    for (size_t i = 0; i < hidden.size(); ++i) {
        const size_t idx = i % (len * 2);
        const uint8_t byte = slice[idx >> 1];
        const uint8_t q = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
        hidden[i] = (static_cast<int>(q) - 8) / 8.0f;
    }
}

void compute_router_weights(const std::vector<float>& hidden, int k, std::vector<float>& w) {
    const int stride = static_cast<int>(hidden.size() / static_cast<size_t>(k));
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        float dot = 0.0f;
        const int base = i * std::max(1, stride);
        for (size_t h = 0; h < hidden.size(); ++h) {
            const float proj = std::sin(0.07f * static_cast<float>(base + static_cast<int>(h)));
            dot += hidden[h] * proj;
        }
        w[i] = std::exp(dot / static_cast<float>(std::max<size_t>(1, hidden.size())));
        sum += w[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (int i = 0; i < k; ++i) w[i] /= sum;
}

bool run_layer_forward(AndroidRuntimeState& st,
                       int token,
                       int position,
                       int layer,
                       double* io_ms,
                       unsigned long long* reads) {
    const size_t hdim = static_cast<size_t>(st.model_cfg.hidden_dim);
    const size_t mdim = static_cast<size_t>(st.model_cfg.moe_intermediate);
    const size_t packed_hh = packed_bytes_for(hdim, hdim);
    const size_t packed_mh = packed_bytes_for(mdim, hdim);

    const size_t base = static_cast<size_t>(layer) * (packed_hh * 4 + packed_mh + packed_hh + packed_hh);

    size_t len_q = packed_hh;
    const uint8_t* q_blob = weight_slice(st.model_weights, base + 0 * packed_hh, &len_q);
    size_t len_k = packed_hh;
    const uint8_t* k_blob = weight_slice(st.model_weights, base + 1 * packed_hh, &len_k);
    size_t len_v = packed_hh;
    const uint8_t* v_blob = weight_slice(st.model_weights, base + 2 * packed_hh, &len_v);
    size_t len_o = packed_hh;
    const uint8_t* o_blob = weight_slice(st.model_weights, base + 3 * packed_hh, &len_o);

    if (!q_blob || !k_blob || !v_blob || !o_blob) return false;
    if (!vk_backend_dequant_matvec(&st.backend, q_blob, len_q, st.hidden.data(), hdim, st.q.data(), hdim)) return false;
    if (!vk_backend_dequant_matvec(&st.backend, k_blob, len_k, st.hidden.data(), hdim, st.k.data(), hdim)) return false;
    if (!vk_backend_dequant_matvec(&st.backend, v_blob, len_v, st.hidden.data(), hdim, st.v.data(), hdim)) return false;

    const size_t layer_cache_offset = static_cast<size_t>(layer) * static_cast<size_t>(st.model_cfg.context_len) * hdim;
    const size_t pos_offset = static_cast<size_t>(position) * hdim;
    std::copy(st.k.begin(), st.k.end(), st.k_cache.begin() + layer_cache_offset + pos_offset);
    std::copy(st.v.begin(), st.v.end(), st.v_cache.begin() + layer_cache_offset + pos_offset);

    if (!vk_backend_attention(&st.backend,
                              st.q.data(),
                              st.k_cache.data() + layer_cache_offset,
                              st.v_cache.data() + layer_cache_offset,
                              static_cast<size_t>(position + 1),
                              static_cast<size_t>(st.model_cfg.num_attention_heads),
                              static_cast<size_t>(st.model_cfg.head_dim),
                              st.attn_out.data())) {
        return false;
    }
    if (!vk_backend_dequant_matvec(&st.backend, o_blob, len_o, st.attn_out.data(), hdim, st.ffn_out.data(), hdim)) {
        return false;
    }
    for (size_t i = 0; i < hdim; ++i) st.hidden[i] += st.ffn_out[i];
    if (!vk_backend_rmsnorm(&st.backend, st.hidden.data(), hdim, 1e-5f)) return false;

    compute_router_weights(st.hidden, st.model_cfg.experts_per_token, st.router_weights);
    for (int rank = 0; rank < st.model_cfg.experts_per_token; ++rank) {
        float maxw = -1.0f;
        int maxidx = 0;
        for (int e = 0; e < st.layout.num_experts; ++e) {
            const float score = std::sin(static_cast<float>(token + layer * 11 + rank * 17 + e * 5));
            if (score > maxw) {
                maxw = score;
                maxidx = e;
            }
        }
        const off_t off = static_cast<off_t>(maxidx) * st.layout.expert_size;

        const double t0 = now_ms();
        ssize_t got = pread(st.layer_fds[layer], st.io_buffer.data(), static_cast<size_t>(st.layout.expert_size), off);
        *io_ms += (now_ms() - t0);
        if (got != static_cast<ssize_t>(st.layout.expert_size)) return false;
        *reads += 1;

        const size_t blob_len = st.io_buffer.size();
        const uint8_t* blob = st.io_buffer.data();
        const size_t third = blob_len / 3;
        const uint8_t* gate_blob = blob;
        const uint8_t* up_blob = blob + third;
        const uint8_t* down_blob = blob + 2 * third;
        const size_t seg_len = std::max<size_t>(1, third);

        if (!vk_backend_dequant_matvec(&st.backend, gate_blob, seg_len, st.hidden.data(), hdim,
                                       st.tmp_gate.data(), mdim)) return false;
        if (!vk_backend_dequant_matvec(&st.backend, up_blob, seg_len, st.hidden.data(), hdim,
                                       st.tmp_up.data(), mdim)) return false;
        if (!vk_backend_swiglu(&st.backend, st.tmp_gate.data(), st.tmp_up.data(), mdim, st.tmp_act.data())) return false;

        float* expert_out = st.expert_out_scratch.data() + static_cast<size_t>(rank) * hdim;
        if (!vk_backend_dequant_matvec(&st.backend, down_blob, seg_len, st.tmp_act.data(), mdim, expert_out, hdim)) {
            return false;
        }
    }

    if (!vk_backend_moe_combine(&st.backend,
                                st.expert_out_scratch.data(),
                                st.router_weights.data(),
                                static_cast<size_t>(st.model_cfg.experts_per_token),
                                hdim,
                                st.ffn_out.data())) {
        return false;
    }
    for (size_t i = 0; i < hdim; ++i) st.hidden[i] += st.ffn_out[i];
    if (!vk_backend_rmsnorm(&st.backend, st.hidden.data(), hdim, 1e-5f)) return false;
    return true;
}

int sample_from_logits(const std::vector<float>& logits, int vocab_size) {
    int best = 0;
    float bestv = -1e30f;
    const int lim = std::min<int>(static_cast<int>(logits.size()), vocab_size);
    for (int i = 0; i < lim; ++i) {
        if (logits[static_cast<size_t>(i)] > bestv) {
            bestv = logits[static_cast<size_t>(i)];
            best = i;
        }
    }
    return positive_mod(best, vocab_size);
}

bool run_decode_step(AndroidRuntimeState& st,
                     int token,
                     int position,
                     int* next_token,
                     double* io_ms,
                     unsigned long long* reads) {
    if (position >= st.model_cfg.context_len) return false;
    gather_embedding(st.model_weights, token, st.hidden);

    for (int layer = 0; layer < st.model_cfg.num_layers; ++layer) {
        if (!run_layer_forward(st, token, position, layer, io_ms, reads)) return false;
    }

    size_t lm_len = packed_bytes_for(static_cast<size_t>(st.model_cfg.vocab_size),
                                     static_cast<size_t>(st.model_cfg.hidden_dim));
    const uint8_t* lm_blob = weight_slice(st.model_weights,
                                          static_cast<size_t>(st.model_cfg.num_layers) * 8192ULL,
                                          &lm_len);
    if (!lm_blob) return false;
    if (!vk_backend_dequant_matvec(&st.backend,
                                   lm_blob,
                                   lm_len,
                                   st.hidden.data(),
                                   static_cast<size_t>(st.model_cfg.hidden_dim),
                                   st.logits.data(),
                                   static_cast<size_t>(st.model_cfg.vocab_size))) {
        return false;
    }
    if (!vk_backend_softmax(&st.backend, st.logits.data(), st.logits.size())) return false;
    *next_token = sample_from_logits(st.logits, st.model_cfg.vocab_size);
    return true;
}

}  // namespace

int fm_android_init(const char* model_dir, const char* model_config_path) {
    if (!model_dir) return -1;

    std::lock_guard<std::mutex> lock(g_mu);
    if (g_state.initialized.load()) {
        close_layer_fds(g_state.layer_fds);
        g_state.initialized = false;
    }
    try {
        g_state.model_dir = model_dir;
        g_state.model_config_path = model_config_path ? model_config_path : (g_state.model_dir + "/model_config.json");

        g_state.model_cfg = load_model_config(g_state.model_config_path);
        g_state.layout = load_layout(g_state.model_dir + "/packed_experts/layout.json");
        g_state.model_weights = read_binary_file(g_state.model_dir + "/model_weights.bin");

        if (g_state.model_cfg.num_layers != g_state.layout.num_layers) {
            throw std::runtime_error("model_config num_layers does not match layout num_layers");
        }

        g_state.layer_fds.reserve(g_state.layout.layer_files.size());
        for (const std::string& f : g_state.layout.layer_files) {
            const std::string path = g_state.model_dir + "/packed_experts/" + f;
            int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("failed to open layer file: " + path);
            struct stat st {};
            if (fstat(fd, &st) != 0 || st.st_size != g_state.layout.layer_size) {
                close(fd);
                throw std::runtime_error("invalid layer file size: " + path);
            }
            g_state.layer_fds.push_back(fd);
        }

        const size_t hdim = static_cast<size_t>(g_state.model_cfg.hidden_dim);
        const size_t mdim = static_cast<size_t>(g_state.model_cfg.moe_intermediate);
        const size_t k = static_cast<size_t>(g_state.model_cfg.experts_per_token);
        const size_t vocab = static_cast<size_t>(g_state.model_cfg.vocab_size);

        g_state.io_buffer.resize(static_cast<size_t>(g_state.layout.expert_size));
        g_state.hidden.assign(hdim, 0.0f);
        g_state.q.assign(hdim, 0.0f);
        g_state.k.assign(hdim, 0.0f);
        g_state.v.assign(hdim, 0.0f);
        g_state.attn_out.assign(hdim, 0.0f);
        g_state.ffn_out.assign(hdim, 0.0f);
        g_state.router_weights.assign(k, 0.0f);
        g_state.expert_out_scratch.assign(k * hdim, 0.0f);
        g_state.tmp_gate.assign(mdim, 0.0f);
        g_state.tmp_up.assign(mdim, 0.0f);
        g_state.tmp_act.assign(mdim, 0.0f);
        g_state.logits.assign(vocab, 0.0f);

        const size_t cache_sz = static_cast<size_t>(g_state.model_cfg.num_layers) *
                                static_cast<size_t>(g_state.model_cfg.context_len) * hdim;
        g_state.k_cache.assign(cache_sz, 0.0f);
        g_state.v_cache.assign(cache_sz, 0.0f);

        if (!vk_backend_init(&g_state.backend)) {
            throw std::runtime_error("backend init failed");
        }
        g_state.stats = fm_android_stats_t{};
        g_state.last_position = 0;
        g_state.initialized = true;
        return 0;
    } catch (...) {
        close_layer_fds(g_state.layer_fds);
        g_state.model_dir.clear();
        g_state.model_config_path.clear();
        g_state.model_weights.clear();
        g_state.initialized = false;
        return -3;
    }
}

int fm_android_generate(const int* input_tokens,
                        int input_len,
                        int max_new_tokens,
                        int* output_tokens,
                        int output_cap) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_state.initialized.load()) return -1;
    if (!input_tokens || input_len <= 0 || !output_tokens || output_cap <= 0) return -2;

    const int n = (max_new_tokens < output_cap) ? max_new_tokens : output_cap;
    if (n <= 0) return 0;
    double t_gen0 = now_ms();
    double io_ms = 0.0;
    unsigned long long reads = 0;

    int token = input_tokens[0];
    int position = 0;

    for (int i = 0; i < input_len - 1; ++i) {
        int next = 0;
        if (!run_decode_step(g_state, input_tokens[i], position, &next, &io_ms, &reads)) return -4;
        token = input_tokens[i + 1];
        position += 1;
    }

    token = input_tokens[input_len - 1];
    for (int i = 0; i < n; ++i) {
        int next = 0;
        if (!run_decode_step(g_state, token, position, &next, &io_ms, &reads)) return -4;
        output_tokens[i] = next;
        token = next;
        position += 1;
    }

    g_state.last_position = position;
    g_state.stats.expert_reads += reads;
    g_state.stats.expert_bytes += reads * static_cast<unsigned long long>(g_state.layout.expert_size);
    g_state.stats.io_ms += io_ms;
    g_state.stats.generate_ms += (now_ms() - t_gen0);
    return n;
}

int fm_android_get_stats(fm_android_stats_t* out_stats) {
    if (!out_stats) return -2;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_state.initialized.load()) return -1;
    *out_stats = g_state.stats;
    return 0;
}

void fm_android_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mu);

    close_layer_fds(g_state.layer_fds);
    g_state.io_buffer.clear();
    g_state.model_weights.clear();
    g_state.hidden.clear();
    g_state.q.clear();
    g_state.k.clear();
    g_state.v.clear();
    g_state.attn_out.clear();
    g_state.ffn_out.clear();
    g_state.expert_out_scratch.clear();
    g_state.router_weights.clear();
    g_state.tmp_gate.clear();
    g_state.tmp_up.clear();
    g_state.tmp_act.clear();
    g_state.logits.clear();
    g_state.k_cache.clear();
    g_state.v_cache.clear();
    vk_backend_shutdown(&g_state.backend);
    g_state.layout = LayoutConfig{};
    g_state.model_cfg = ModelConfig{};
    g_state.stats = fm_android_stats_t{};
    g_state.model_dir.clear();
    g_state.model_config_path.clear();
    g_state.last_position = 0;
    g_state.initialized = false;
}
