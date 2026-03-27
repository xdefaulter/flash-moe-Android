#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file packs all experts contiguously:
    [expert0][expert1]...[expertN-1]

Within each expert block, components are packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight, down_proj.scales, down_proj.biases

Unlike the original script, sizes and counts are inferred from expert_index.json so
this tool can repack different Qwen MoE variants as long as they expose the same
component names.

Usage:
    python repack_experts.py                    # repack all layers
    python repack_experts.py --layers 0-4       # repack layers 0-4
    python repack_experts.py --layers 0,5,10    # repack specific layers
    python repack_experts.py --dry-run          # verify without writing
    python repack_experts.py --verify-only 0    # verify layer 0 against originals
"""

import argparse
import json
import os
import sys
import time

PACKED_LAYOUT_VERSION = 1
PACKED_LAYOUT_NAME = "flash_moe.packed_experts"

COMPONENT_NAMES = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
]


def parse_layers(spec, num_layers):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == "all":
        return list(range(num_layers))
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict + model_path."""
    with open(index_path) as f:
        idx = json.load(f)
    return idx["expert_reads"], idx["model_path"]


def _infer_component_dtype(component_name):
    return "U32" if component_name.endswith(".weight") else "BF16"


def build_layout(expert_reads):
    """Infer packed expert layout dynamically from expert index metadata."""
    if not expert_reads:
        raise ValueError("expert_reads is empty")

    first_layer_key = sorted(expert_reads.keys(), key=int)[0]
    first_layer = expert_reads[first_layer_key]

    missing = [name for name in COMPONENT_NAMES if name not in first_layer]
    if missing:
        raise ValueError(f"layer {first_layer_key} missing required components: {missing}")

    components = []
    offset = 0
    num_experts = None

    for name in COMPONENT_NAMES:
        info = first_layer[name]
        expert_size = int(info["expert_size"])
        shape = info.get("shape")

        if num_experts is None:
            if not shape:
                raise ValueError(f"missing shape metadata for {name}")
            num_experts = int(shape[0])

        components.append(
            {
                "name": name,
                "offset": offset,
                "size": expert_size,
                "dtype": _infer_component_dtype(name),
                "shape": shape[1:] if shape and len(shape) > 1 else None,
            }
        )
        offset += expert_size

    expert_size = offset
    num_layers = len(expert_reads)
    layer_size = num_experts * expert_size

    return {
        "components": components,
        "expert_size": expert_size,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "layer_size": layer_size,
    }


def verify_component_sizes(expert_reads, components, num_experts):
    """Verify component sizes and expert counts in index match inferred layout."""
    expected = {c["name"]: c["size"] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name in COMPONENT_NAMES:
            if comp_name not in comps:
                print(f"MISMATCH: layer {layer_key} missing component {comp_name}")
                return False
            info = comps[comp_name]
            if info["expert_size"] != expected[comp_name]:
                print(
                    f"MISMATCH: layer {layer_key}, {comp_name}: "
                    f"index says {info['expert_size']}, expected {expected[comp_name]}"
                )
                return False
            shape = info.get("shape")
            if not shape or int(shape[0]) != num_experts:
                print(
                    f"MISMATCH: layer {layer_key}, {comp_name}: "
                    f"shape[0]={shape[0] if shape else 'None'}, expected {num_experts}"
                )
                return False

    print("Component sizes verified: all match inferred layout")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info["file"])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def repack_layer(layer_idx, expert_reads, fds, output_dir, layout, dry_run=False):
    """Repack all experts for one layer into a contiguous binary file."""
    components = layout["components"]
    expert_size = layout["expert_size"]
    num_experts = layout["num_experts"]
    layer_size = layout["layer_size"]

    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(num_experts):
            for comp in components:
                info = layer_info[comp["name"]]
                _src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
                _dst_offset = expert_idx * expert_size + comp["offset"]
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()

    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0
    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp["name"]]
            src_fd = fds[info["file"]]
            src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
            dst_offset = expert_idx * expert_size + comp["offset"]
            read_plan.append((src_fd, src_offset, dst_offset, comp["size"]))

    read_plan.sort(key=lambda x: (x[0], x[1]))

    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, fds, output_dir, layout):
    """Read back spot-check experts from packed file and compare to originals."""
    components = layout["components"]
    expert_size = layout["expert_size"]
    num_experts = layout["num_experts"]

    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    sample_experts = sorted({0, 1, max(0, num_experts // 2), num_experts - 1})
    mismatches = 0
    for expert_idx in sample_experts:
        for comp in components:
            info = layer_info[comp["name"]]
            src_fd = fds[info["file"]]
            src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
            dst_offset = expert_idx * expert_size + comp["offset"]

            original = os.pread(src_fd, comp["size"], src_offset)
            packed = os.pread(fd_packed, comp["size"], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts {sample_experts})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir, layout):
    """Write layout.json describing packed expert format."""
    layer_files = [f"layer_{layer_idx:02d}.bin" for layer_idx in range(layout["num_layers"])]
    payload = {
        "format": PACKED_LAYOUT_NAME,
        "version": PACKED_LAYOUT_VERSION,
        "expert_size": layout["expert_size"],
        "num_layers": layout["num_layers"],
        "num_experts": layout["num_experts"],
        "layer_size": layout["layer_size"],
        "components": layout["components"],
        "layer_files": layer_files,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument(
        "--index",
        default="/Users/danielwoods/Workspace/ane-research/expert_index.json",
        help="Path to expert_index.json",
    )
    parser.add_argument("--layers", default=None, help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument("--dry-run", action="store_true", help="Verify offsets without writing")
    parser.add_argument("--verify-only", type=int, default=None, metavar="LAYER", help="Verify a specific layer")
    args = parser.parse_args()

    print("Loading expert index...")
    expert_reads, model_path = load_index(args.index)
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    layout = build_layout(expert_reads)
    print(
        f"Inferred layout: layers={layout['num_layers']}, experts/layer={layout['num_experts']}, "
        f"expert_size={layout['expert_size']:,} bytes, layer_size={layout['layer_size']:,} bytes"
    )

    if not verify_component_sizes(expert_reads, layout["components"], layout["num_experts"]):
        print("ABORTING: component size mismatch")
        sys.exit(1)

    output_dir = os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers, layout["num_layers"])
    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layout["layer_size"]
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            per_layer_gb = layout["layer_size"] / (1024**3)
            suggested_last = max(0, int(free_gb / max(per_layer_gb, 1e-9)) - 1)
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{suggested_last}")
            sys.exit(1)

    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, fds, output_dir, layout)
        for fd in fds.values():
            os.close(fd)
        return

    write_layout(output_dir, layout)

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        bytes_written, elapsed = repack_layer(layer_idx, expert_reads, fds, output_dir, layout, dry_run=args.dry_run)
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float("inf")
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(
                f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                f"({throughput:.1f} GB/s) | "
                f"Total: {total_written/1024**3:.1f}/{len(layers)*layout['layer_size']/1024**3:.1f} GB "
                f"({overall_throughput:.1f} GB/s avg) | "
                f"ETA: {eta:.0f}s"
            )

            if not verify_layer(layer_idx, expert_reads, fds, output_dir, layout):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    for fd in fds.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == "__main__":
    main()
