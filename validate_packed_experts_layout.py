#!/usr/bin/env python3
"""Validate packed_experts/layout.json emitted by repack_experts.py."""

import argparse
import json
import os
import sys

REQUIRED_TOP_LEVEL = {
    "format",
    "version",
    "expert_size",
    "num_layers",
    "num_experts",
    "layer_size",
    "components",
    "layer_files",
}

REQUIRED_COMPONENT_FIELDS = {"name", "offset", "size", "dtype"}


class ValidationError(Exception):
    pass


def _require(condition, msg):
    if not condition:
        raise ValidationError(msg)


def validate_structure(payload):
    missing = sorted(REQUIRED_TOP_LEVEL - set(payload.keys()))
    _require(not missing, f"missing required top-level fields: {missing}")

    _require(payload["format"] == "flash_moe.packed_experts", "unexpected format field")
    _require(isinstance(payload["version"], int) and payload["version"] >= 1, "version must be >= 1")

    num_layers = payload["num_layers"]
    num_experts = payload["num_experts"]
    expert_size = payload["expert_size"]
    layer_size = payload["layer_size"]

    _require(isinstance(num_layers, int) and num_layers > 0, "num_layers must be positive int")
    _require(isinstance(num_experts, int) and num_experts > 0, "num_experts must be positive int")
    _require(isinstance(expert_size, int) and expert_size > 0, "expert_size must be positive int")
    _require(isinstance(layer_size, int) and layer_size > 0, "layer_size must be positive int")

    _require(layer_size == num_experts * expert_size, "layer_size must equal num_experts * expert_size")

    components = payload["components"]
    _require(isinstance(components, list) and components, "components must be a non-empty list")

    running_offset = 0
    for idx, comp in enumerate(components):
        missing_comp = sorted(REQUIRED_COMPONENT_FIELDS - set(comp.keys()))
        _require(not missing_comp, f"component[{idx}] missing fields: {missing_comp}")
        _require(isinstance(comp["name"], str) and comp["name"], f"component[{idx}] invalid name")
        _require(isinstance(comp["offset"], int) and comp["offset"] >= 0, f"component[{idx}] invalid offset")
        _require(isinstance(comp["size"], int) and comp["size"] > 0, f"component[{idx}] invalid size")
        _require(comp["offset"] == running_offset, f"component[{idx}] offset is not contiguous")
        running_offset += comp["size"]

    _require(running_offset == expert_size, "sum(component sizes) must equal expert_size")

    layer_files = payload["layer_files"]
    _require(isinstance(layer_files, list), "layer_files must be a list")
    _require(len(layer_files) == num_layers, "layer_files length must match num_layers")

    for i, layer_file in enumerate(layer_files):
        expected = f"layer_{i:02d}.bin"
        _require(layer_file == expected, f"layer_files[{i}] must be {expected}")


def validate_files(payload, base_dir):
    for layer_file in payload["layer_files"]:
        path = os.path.join(base_dir, layer_file)
        _require(os.path.exists(path), f"missing packed layer file: {path}")
        size = os.path.getsize(path)
        _require(
            size == payload["layer_size"],
            f"{layer_file} has size {size}, expected {payload['layer_size']}",
        )


def main():
    parser = argparse.ArgumentParser(description="Validate packed_experts/layout.json")
    parser.add_argument(
        "layout",
        nargs="?",
        default="packed_experts/layout.json",
        help="Path to layout.json (default: packed_experts/layout.json)",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also validate packed layer files exist and match layer_size",
    )
    args = parser.parse_args()

    with open(args.layout) as f:
        payload = json.load(f)

    validate_structure(payload)

    if args.check_files:
        base_dir = os.path.dirname(os.path.abspath(args.layout))
        validate_files(payload, base_dir)

    print(f"OK: {args.layout} is valid")
    if args.check_files:
        print("OK: all layer files are present and sized correctly")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
