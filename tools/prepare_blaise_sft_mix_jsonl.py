#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize the Blaise SFT mix configs into one local JSONL file.

The training dataloader path is intentionally file-backed so the actual run
does not depend on HF streaming or parquet decode in the hot path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset


DEFAULT_REPO_ID = "BlaiseAI/blaise-sft-training-mix"
DEFAULT_CONFIGS = ("nemotron-full-family", "nemotron-mixed-sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument(
        "--output",
        default=str(
            Path.home()
            / "data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full.jsonl"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_offsets(path: Path) -> np.ndarray:
    offsets = []
    with path.open("rb") as dataset_file:
        while True:
            offset = dataset_file.tell()
            line = dataset_file.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    arr = np.asarray(offsets, dtype=np.int64)
    tmp = path.with_suffix(path.suffix + f".offsets.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, f"{path}.offsets.npy")
    return arr


def main() -> None:
    args = parse_args()
    configs = tuple(args.config or DEFAULT_CONFIGS)
    output = Path(args.output)
    offsets_path = Path(f"{output}.offsets.npy")
    if output.exists() and offsets_path.exists() and not args.force:
        offsets = np.load(offsets_path, mmap_mode="r")
        print(f"{output} already exists with {len(offsets)} rows; use --force to rebuild.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    rows = 0
    with tmp.open("w", encoding="utf-8") as writer:
        for config in configs:
            dataset = load_dataset(args.repo_id, config, split="train")
            print(f"Writing {args.repo_id}/{config}: {len(dataset)} rows", flush=True)
            for item in dataset:
                writer.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                writer.write("\n")
                rows += 1
    os.replace(tmp, output)
    offsets = build_offsets(output)
    print(f"Wrote {output} with {rows} rows; offsets={len(offsets)}")


if __name__ == "__main__":
    main()
