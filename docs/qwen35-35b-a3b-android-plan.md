# Qwen3.5-35B-A3B + Android Port Plan

## Goal

Add support for **Qwen3.5-35B-A3B** in this repository first on macOS (current C/Objective-C/Metal pipeline), then extend the same streaming-MoE design to Android as an experiment.

This plan is intentionally execution-oriented: each section maps to specific code files in this repo so implementation can proceed in measurable checkpoints.

---

## 0) Current-State Constraints (from this codebase)

The current engine is tightly specialized to `Qwen3.5-397B-A17B`:

- Compile-time architecture constants (`NUM_LAYERS`, `NUM_EXPERTS`, `HIDDEN_DIM`, attention shape, etc.) are hard-coded in `metal_infer/infer.m`.
- Expert binary layout is hard-coded (`EXPERT_SIZE`, offsets, quant format assumptions).
- Preprocessing scripts are wired for the existing tensor names/layout and pack to a fixed format.
- Runtime expects one canonical model directory layout and manifest conventions.

Result: adding 35B-A3B cleanly requires separating **model architecture metadata** from **runtime kernels + streaming scheduler**.

---

## 1) Phase 1 — Make the Mac Engine Multi-Model

### 1.1 Add runtime model config

Create a small runtime model descriptor loaded from JSON (for example `model_config.json`) with at least:

- `model_name`
- `num_layers`
- `hidden_dim`
- `num_attention_heads`
- `num_kv_heads`
- `head_dim`
- `vocab_size`
- `num_experts`
- `experts_per_token` (router top-k used by checkpoint)
- `moe_intermediate`
- `shared_intermediate`
- `full_attn_interval` or explicit `full_attn_layers[]`
- quant/packing fields: `bits`, `group_size`, `expert_blob_layout`

Then replace compile-time assumptions in `infer.m` with runtime values where feasible. Keep kernel tile sizes compile-time; move only architecture dimensions and per-model routing layout runtime.

### 1.2 Formalize weight manifests

Introduce a strict manifest contract for:

- non-expert tensors (offset, shape, dtype, transpose flags)
- expert blobs (per-layer file path, per-expert offset table)
- optional tokenizer metadata

This avoids model-specific if/else logic in inference and lets repackers emit manifests for both 397B and 35B variants.

### 1.3 Extend repack pipeline

Update Python preprocessing to support `Qwen3.5-35B-A3B` tensor schemas:

- add a model-adapter layer in repacker scripts keyed by architecture family
- verify expert ordering and gate/up/down tensor naming for the A3B checkpoint
- emit exact same binary contract consumed by runtime (or versioned contract)

Deliverable: one command that can produce a runnable packed directory for either model family.

### 1.4 Validation harness

Add parity checks before performance tuning:

- single-token forward checksum vs a trusted reference (PyTorch/HF) for a fixed prompt
- top-k token agreement for a short decode window
- router top-k agreement sampling for random tokens

Only after functional parity, tune scheduling.

---

## 2) Phase 2 — Optimize 35B-A3B on macOS

35B-A3B should shift bottlenecks versus 397B:

- less total expert data per token
- potentially different expert fanout and MLP shapes
- lower I/O pressure relative to compute

Optimization checklist:

- Retune `K` (runtime top-k override only if quality acceptable)
- Retune per-layer read parallelism for smaller expert blobs
- Re-evaluate whether `F_NOCACHE`/warm-cache split is still beneficial for this size
- Re-profile command-buffer boundaries (`CMD1/CMD2/CMD3`) for new ratio of GPU vs I/O time

Define success metrics:

- tokens/sec
- p50/p95 per-layer latency
- quality gate (JSON/tool-calling validity, if used)
- energy/thermal stability over long decode

---

## 3) Phase 3 — Android Experiment Architecture

The Android port should preserve the same high-level idea:

- router selects experts
- stream only selected expert weights from storage
- execute fused MoE math on mobile GPU/accelerator

But the backend stack differs from Metal/macOS.

### 3.1 Recommended backend path

Use **Vulkan compute** as the primary target for Android experiment:

- maps best to current custom-kernel design
- explicit buffer + synchronization control
- broad device coverage (vs vendor-specific NN APIs)

Optional later: compare against NNAPI backends where available.

### 3.2 Android runtime split

Proposed modules:

- `android/engine-core/` (C++): model config, manifests, router math, scheduler, file I/O
- `android/backend-vulkan/` (C++): shader compilation, buffers, dispatch, synchronization
- `android/app/` (Kotlin): JNI glue, prompt loop, telemetry UI

Keep the on-disk packed model format shared with desktop as much as possible.

### 3.3 Storage + paging strategy on Android

Equivalent goals to macOS page-cache strategy:

- use positional reads (`pread`) on packed expert files
- avoid user-space full-model caching
- aggressively reuse small staging buffers
- gather device-specific traces for random read throughput and thermal throttling

Key risk: mobile UFS/NVMe throughput and latency variance across devices is much larger than Apple Silicon laptops.

### 3.4 Shader migration map (Metal → Vulkan)

Port kernels in this order:

1. dequant matvec (4-bit path first)
2. fused SwiGLU
3. RMSNorm
4. MoE combine + residual
5. attention kernels

Use a correctness-first approach:

- CPU fallback for each op
- per-kernel numerical diff tests (small tensors)
- then perf tuning (subgroup ops, shared memory tiling)

---

## 4) Phase 4 — First Android Milestone (Practical)

Instead of full parity immediately, define a narrow milestone:

- single-model support: Qwen3.5-35B-A3B only
- short context window
- batch size = 1
- no tool-calling features initially
- minimal UI: prompt + stream tokens + perf counters

Milestone definition of done:

- generates coherent text for a fixed prompt set
- no OOM on target test device
- stable run for N tokens without backend reset
- publishes per-token timing breakdown (I/O, router, GPU compute)

---

## 5) Risks and Mitigations

### Risk A: Model schema mismatch across checkpoints

Mitigation: schema adapters in repacker + strict manifest validation tool.

### Risk B: Router or expert ordering mismatch causes silent quality failure

Mitigation: add router top-k diff tests and per-layer activation sanity checks.

### Risk C: Android thermal throttling destroys throughput

Mitigation: include sustained-performance mode and mandatory long-run profiling (10-20 minute decode).

### Risk D: Vulkan portability variance

Mitigation: define one reference device first (e.g., Snapdragon flagship) before broad compatibility.

---

## 6) Suggested Near-Term Work Items (next commits)

1. Add runtime model-config parser and remove hard-coded architecture constants from the hottest path where practical.
2. Version the packed-format manifest and add a validator utility. ✅ `layout.json` now carries `format/version/layer_files`, and `validate_packed_experts_layout.py` checks schema + optional file-size integrity.
3. Extend repacker scripts with a second adapter for Qwen3.5-35B-A3B.
4. Add functional parity tests (single-step + short decode agreement). ✅ Added `tools/parity_check.py` for token exact-match and router top-k agreement (exact + Jaccard) over JSONL traces.
5. Create Android experimental scaffold (`android/` with JNI + Vulkan backend skeleton). ✅ Added `android/infer_android.h` + `android/infer_android.cpp` with working init/generate/shutdown flow, real decode math path (QKV attention + KV cache + MoE + RMSNorm + logits projection), per-layer expert streaming with `pread`, and backend runtime selection (`Vulkan plugin -> CPU fallback`) for dequant/SwiGLU/combine/RMSNorm/attention/softmax, plus JNI/Kotlin bridge for Pixel/Samsung integration.

If we execute these five in order, we can bring up 35B-A3B quickly on desktop and avoid rewriting preprocessing/runtime again when moving to Android.
