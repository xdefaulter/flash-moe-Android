#!/usr/bin/env python3
"""Parity checker for token and router top-k traces.

Input format (JSONL):
  {"step": 0, "token": 123}
  {"step": 1, "token": 456}

Router format (JSONL):
  {"step": 0, "layer": 0, "topk": [12, 44, 51, 60]}
"""

import argparse
import json


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid JSON: {e}") from e
    return rows


def token_parity(ref_rows, test_rows):
    ref = {int(r["step"]): int(r["token"]) for r in ref_rows}
    tst = {int(r["step"]): int(r["token"]) for r in test_rows}
    common = sorted(set(ref) & set(tst))
    if not common:
        raise ValueError("no overlapping token steps")
    matches = sum(1 for s in common if ref[s] == tst[s])
    print(f"token_exact_match: {matches}/{len(common)} ({matches/len(common):.2%})")


def router_parity(ref_rows, test_rows):
    def index(rows):
        out = {}
        for r in rows:
            k = (int(r["step"]), int(r["layer"]))
            out[k] = list(map(int, r["topk"]))
        return out

    ref = index(ref_rows)
    tst = index(test_rows)
    common = sorted(set(ref) & set(tst))
    if not common:
        raise ValueError("no overlapping (step, layer) pairs")

    exact = 0
    jacc_sum = 0.0
    for k in common:
        a = ref[k]
        b = tst[k]
        if a == b:
            exact += 1
        sa, sb = set(a), set(b)
        jacc = len(sa & sb) / max(1, len(sa | sb))
        jacc_sum += jacc

    print(f"router_exact_match: {exact}/{len(common)} ({exact/len(common):.2%})")
    print(f"router_mean_jaccard: {jacc_sum/len(common):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare token/router parity traces")
    p.add_argument("--mode", choices=["tokens", "router"], required=True)
    p.add_argument("--ref", required=True, help="reference JSONL")
    p.add_argument("--test", required=True, help="candidate JSONL")
    args = p.parse_args()

    ref_rows = load_jsonl(args.ref)
    test_rows = load_jsonl(args.test)
    if args.mode == "tokens":
        token_parity(ref_rows, test_rows)
    else:
        router_parity(ref_rows, test_rows)
