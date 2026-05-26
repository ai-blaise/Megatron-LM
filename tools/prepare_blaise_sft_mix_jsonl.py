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
import time

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


DEFAULT_REPO_ID = "BlaiseAI/blaise-sft-training-mix"
DEFAULT_CONFIGS = ("nemotron-full-family", "nemotron-mixed-sample")
DEFAULT_TRAJECTORY_REPO_ID = "BlaiseAI/prod-distillation-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument(
        "--base-jsonl",
        default=None,
        help="Existing local JSONL to copy before appending requested additions.",
    )
    parser.add_argument("--include-trajectories", action="store_true")
    parser.add_argument("--trajectory-repo-id", default=DEFAULT_TRAJECTORY_REPO_ID)
    parser.add_argument("--trajectory-config", default="trajectories")
    parser.add_argument("--trajectory-split", default="train")
    parser.add_argument("--trajectory-max-samples", type=int, default=None)
    parser.add_argument(
        "--trajectory-streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream trajectory parquet rows instead of predownloading all shards.",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("files", "datasets"),
        default="files",
        help="Use direct file downloads or datasets loading for trajectory rows.",
    )
    parser.add_argument("--trajectory-skip-bad-shards", action=argparse.BooleanOptionalAction, default=True)
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


def _trajectory_filenames(repo_id: str) -> list[str]:
    api = HfApi()
    info = api.dataset_info(repo_id)
    return sorted(
        sibling.rfilename
        for sibling in (info.siblings or [])
        if sibling.rfilename.startswith("trajectories/")
        and sibling.rfilename.endswith(".parquet")
    )


def _write_trajectory_record(writer, item: dict, repo_id: str, config: str) -> None:
    writer.write(
        json.dumps(
            {
                "text": item.get("e2e_trajectory"),
                "source_dataset": f"{repo_id}/{config}",
                "source": item.get("source"),
                "harness": item.get("harness"),
                "row_id": item.get("row_id"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    writer.write("\n")


def _append_trajectories_from_files(args: argparse.Namespace, writer) -> int:
    filenames = _trajectory_filenames(args.trajectory_repo_id)
    print(
        f"Writing {args.trajectory_repo_id}/{args.trajectory_config}: "
        f"{len(filenames)} parquet shards via direct download",
        flush=True,
    )
    written = 0
    skipped = []
    for shard_idx, filename in enumerate(filenames, start=1):
        try:
            local_path = hf_hub_download(
                repo_id=args.trajectory_repo_id,
                repo_type="dataset",
                filename=filename,
            )
            table = pq.read_table(
                local_path,
                columns=["row_id", "source", "harness", "e2e_trajectory"],
            )
            batch_rows = table.to_pylist()
        except Exception as exc:  # noqa: BLE001
            if not args.trajectory_skip_bad_shards:
                raise
            skipped.append((filename, type(exc).__name__, str(exc).splitlines()[0]))
            print(
                f"WARNING: skipping trajectory shard {filename}: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                flush=True,
            )
            continue

        shard_written = 0
        for item in batch_rows:
            text = item.get("e2e_trajectory")
            if not isinstance(text, str) or not text:
                continue
            _write_trajectory_record(
                writer, item, args.trajectory_repo_id, args.trajectory_config
            )
            written += 1
            shard_written += 1
            if args.trajectory_max_samples is not None and written >= args.trajectory_max_samples:
                break
        print(
            f"  shard {shard_idx}/{len(filenames)} {filename}: "
            f"rows={shard_written} total_trajectories={written}",
            flush=True,
        )
        if args.trajectory_max_samples is not None and written >= args.trajectory_max_samples:
            break

    if skipped:
        print("Skipped trajectory shards:", flush=True)
        for filename, error_type, message in skipped:
            print(f"  {filename}: {error_type}: {message}", flush=True)
    return written


def _append_trajectories_from_datasets(args: argparse.Namespace, writer) -> int:
    dataset = load_dataset(
        args.trajectory_repo_id,
        args.trajectory_config,
        split=args.trajectory_split,
        streaming=args.trajectory_streaming,
    )
    dataset_len = "streaming"
    if not args.trajectory_streaming and not args.trajectory_max_samples:
        try:
            dataset_len = str(len(dataset))
        except TypeError:
            pass
    print(
        f"Writing {args.trajectory_repo_id}/{args.trajectory_config}: "
        f"{dataset_len} rows",
        flush=True,
    )
    written = 0
    last_report = time.monotonic()
    for item in dataset:
        text = item.get("e2e_trajectory")
        if not isinstance(text, str) or not text:
            continue
        _write_trajectory_record(writer, item, args.trajectory_repo_id, args.trajectory_config)
        written += 1
        now = time.monotonic()
        if now - last_report > 30:
            print(f"  trajectories written={written}", flush=True)
            last_report = now
        if args.trajectory_max_samples is not None and written >= args.trajectory_max_samples:
            break
    return written


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
    trajectory_rows = 0
    with tmp.open("w", encoding="utf-8") as writer:
        if args.base_jsonl:
            base_path = Path(args.base_jsonl)
            print(f"Copying local base JSONL: {base_path}", flush=True)
            with base_path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if line.strip():
                        writer.write(line)
                        rows += 1
        else:
            for config in configs:
                dataset = load_dataset(args.repo_id, config, split="train")
                print(f"Writing {args.repo_id}/{config}: {len(dataset)} rows", flush=True)
                for item in dataset:
                    writer.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                    writer.write("\n")
                    rows += 1

        if args.include_trajectories:
            if args.trajectory_mode == "files":
                trajectory_rows = _append_trajectories_from_files(args, writer)
            else:
                trajectory_rows = _append_trajectories_from_datasets(args, writer)
            rows += trajectory_rows
    os.replace(tmp, output)
    offsets = build_offsets(output)
    print(
        f"Wrote {output} with {rows} rows "
        f"(trajectories={trajectory_rows}); offsets={len(offsets)}"
    )


if __name__ == "__main__":
    main()
