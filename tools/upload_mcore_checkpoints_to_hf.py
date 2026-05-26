#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Upload completed Megatron/MCore checkpoint directories to a HF model repo.

This is a sidecar: it watches local checkpoint directories and uploads the
files present on this node. For a multinode torch_dist checkpoint, run one
sidecar per node. The retention owner promotes/deletes remote checkpoints only
after all expected nodes have written upload markers for an iteration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
import re
import time
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi, create_repo


ITER_RE = re.compile(r"iter_(\d{7})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder-prefix", default="corsaire-1-research-preview")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--node-name", default=os.uname().nodename)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--expected-nodes", type=int, default=int(os.environ.get("NNODES", "1")))
    parser.add_argument("--shard-start", type=int, default=None)
    parser.add_argument("--shard-end", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-seconds", type=int, default=180)
    parser.add_argument("--finalize-timeout-seconds", type=int, default=7200)
    parser.add_argument("--upload-interval", type=int, default=50)
    parser.add_argument("--retain", type=int, default=1)
    parser.add_argument("--delete-before-upload", action="store_true")
    parser.add_argument("--retention-owner", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-file", default=None)
    return parser.parse_args()


def iteration_from_dir(path: Path) -> int | None:
    match = ITER_RE.match(path.name)
    return int(match.group(1)) if match else None


def available_iterations(root: Path) -> list[int]:
    if not root.exists():
        return []
    iterations = []
    for child in root.iterdir():
        if child.is_dir():
            iteration = iteration_from_dir(child)
            if iteration is not None:
                iterations.append(iteration)
    return sorted(iterations)


def load_state(state_file: Path) -> tuple[set[int], set[int]]:
    if not state_file.exists():
        return set(), set()
    try:
        state = json.loads(state_file.read_text())
        return set(state.get("uploaded", [])), set(state.get("finalized", []))
    except Exception:
        return set(), set()


def save_state(state_file: Path, uploaded: set[int], finalized: set[int]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(state_file.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "uploaded": sorted(uploaded),
                "finalized": sorted(finalized),
            },
            indent=2,
        )
    )
    os.replace(tmp, state_file)


def rank_shard_patterns(iter_name: str, shard_start: int | None, shard_end: int | None) -> list[str]:
    if shard_start is None or shard_end is None:
        return [f"{iter_name}/__*_0.distcp"]
    return [f"{iter_name}/__{rank}_0.distcp" for rank in range(shard_start, shard_end + 1)]


def upload_patterns(root: Path, iteration: int, args: argparse.Namespace) -> list[str]:
    iter_name = f"iter_{iteration:07d}"
    patterns = [
        "latest_checkpointed_iteration.txt",
        "latest_train_state.pt",
        f"{iter_name}/.metadata",
        f"{iter_name}/common.pt",
        f"{iter_name}/metadata.json",
        f"{iter_name}/run_config.yaml",
        f"{iter_name}/train_state.pt",
        f"{iter_name}/tokenizer/**/*",
    ]
    patterns.extend(rank_shard_patterns(iter_name, args.shard_start, args.shard_end))
    existing_patterns = []
    for pattern in patterns:
        if any(root.glob(pattern)):
            existing_patterns.append(pattern)
    return existing_patterns


def newest_mtime(paths: list[Path]) -> float:
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def matched_files(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def wait_until_stable(root: Path, patterns: list[str], stable_seconds: int) -> list[Path]:
    while True:
        files = matched_files(root, patterns)
        if files and time.time() - newest_mtime(files) >= stable_seconds:
            return files
        time.sleep(min(30, max(5, stable_seconds // 6)))


def should_upload_iteration(iteration: int, args: argparse.Namespace) -> bool:
    return not (args.upload_interval > 1 and iteration % args.upload_interval != 0)


def remote_step_folder(prefix: str, iteration: int) -> str:
    return f"{prefix}-step-{iteration:07d}"


def node_marker_name(node_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", node_name)


def upload_bytes(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    data: bytes,
    commit_message: str,
) -> None:
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=BytesIO(data),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )


def upload_json(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    payload: dict,
    commit_message: str,
) -> None:
    upload_bytes(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        data=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=commit_message,
    )


def remote_node_markers(api: HfApi, repo_id: str, repo_type: str, folder: str) -> list[str]:
    marker_prefix = f"{folder}/_hf_upload_nodes/"
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return []
    return sorted(path for path in files if path.startswith(marker_prefix) and path.endswith(".json"))


def remote_step_folders(api: HfApi, repo_id: str, repo_type: str, prefix: str) -> set[tuple[int, str]]:
    folders = set()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return folders
    regex = re.compile(rf"^{re.escape(prefix)}-step-(\d{{7}})(?:/|$)")
    for path in files:
        match = regex.match(path)
        if match:
            folders.add((int(match.group(1)), match.group(0).rstrip("/")))
    return folders


def delete_old_remote_steps(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    prefix: str,
    retain: int,
    protect_iteration: int,
    dry_run: bool,
) -> None:
    if retain < 0:
        return
    folders = sorted(remote_step_folders(api, repo_id, repo_type, prefix), reverse=True)
    protected = {protect_iteration}
    keep = set(step for step, _ in folders[:retain]) if retain > 0 else set()
    delete = [(step, folder) for step, folder in folders if step not in keep and step not in protected]
    for step, folder in delete:
        print(f"[hf-upload] deleting remote old checkpoint step={step}: {folder}", flush=True)
        if dry_run:
            continue
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[CommitOperationDelete(path_in_repo=folder)],
            commit_message=f"Delete old MCore checkpoint {folder}",
        )


def write_node_marker(
    api: HfApi,
    root: Path,
    files: list[Path],
    iteration: int,
    folder: str,
    args: argparse.Namespace,
) -> None:
    payload = {
        "iteration": iteration,
        "node_name": args.node_name,
        "node_rank": args.node_rank,
        "expected_nodes": args.expected_nodes,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [str(path.relative_to(root)) for path in files],
    }
    marker = f"{folder}/_hf_upload_nodes/{args.node_rank:04d}-{node_marker_name(args.node_name)}.json"
    upload_json(
        api,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=marker,
        payload=payload,
        commit_message=f"Mark MCore checkpoint step {iteration:07d} uploaded from {args.node_name}",
    )


def finalize_iteration(api: HfApi, iteration: int, args: argparse.Namespace) -> bool:
    folder = remote_step_folder(args.folder_prefix, iteration)
    deadline = time.time() + args.finalize_timeout_seconds
    while True:
        markers = remote_node_markers(api, args.repo_id, args.repo_type, folder)
        if len(markers) >= args.expected_nodes:
            payload = {
                "iteration": iteration,
                "folder": folder,
                "expected_nodes": args.expected_nodes,
                "uploaded_nodes": len(markers),
                "node_markers": markers,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
            }
            if args.dry_run:
                print(f"[hf-upload] would finalize step={iteration} markers={len(markers)}", flush=True)
                return True
            upload_json(
                api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo=f"{folder}/_hf_upload_complete.json",
                payload=payload,
                commit_message=f"Finalize MCore checkpoint step {iteration:07d}",
            )
            upload_json(
                api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo="latest_mcore_checkpoint.json",
                payload=payload,
                commit_message=f"Promote latest MCore checkpoint step {iteration:07d}",
            )
            upload_bytes(
                api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo="latest_checkpointed_iteration.txt",
                data=f"{iteration}\n".encode(),
                commit_message=f"Promote latest checkpoint iteration {iteration:07d}",
            )
            delete_old_remote_steps(
                api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                prefix=args.folder_prefix,
                retain=args.retain,
                protect_iteration=iteration,
                dry_run=False,
            )
            print(
                f"[hf-upload] finalized step={iteration} markers={len(markers)}/{args.expected_nodes}",
                flush=True,
            )
            return True
        if args.finalize_timeout_seconds <= 0 or time.time() >= deadline:
            print(
                f"[hf-upload] step={iteration} not finalized: "
                f"markers={len(markers)}/{args.expected_nodes}",
                flush=True,
            )
            return False
        time.sleep(args.poll_seconds)


def upload_iteration(api: HfApi, root: Path, iteration: int, args: argparse.Namespace) -> bool:
    if not should_upload_iteration(iteration, args):
        return False
    patterns = upload_patterns(root, iteration, args)
    if not patterns:
        print(f"[hf-upload] no local files found for step {iteration}", flush=True)
        return False
    files = wait_until_stable(root, patterns, args.stable_seconds)
    total_gib = sum(path.stat().st_size for path in files) / (1024**3)
    folder = remote_step_folder(args.folder_prefix, iteration)
    print(
        f"[hf-upload] uploading step={iteration} files={len(files)} size={total_gib:.2f}GiB "
        f"to {args.repo_id}/{folder} from {args.node_name}",
        flush=True,
    )
    if args.dry_run:
        for path in files:
            print(f"  {path.relative_to(root)}", flush=True)
        return True
    if args.delete_before_upload and args.retention_owner:
        delete_old_remote_steps(
            api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            prefix=args.folder_prefix,
            retain=max(args.retain - 1, 0),
            protect_iteration=iteration,
            dry_run=False,
        )
    elif args.delete_before_upload:
        print("[hf-upload] ignoring --delete-before-upload on non-retention owner", flush=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(root),
        path_in_repo=folder,
        allow_patterns=patterns,
        commit_message=f"Upload MCore checkpoint step {iteration:07d} from {args.node_name}",
    )
    write_node_marker(api, root, files, iteration, folder, args)
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.checkpoint_root)
    state_file = Path(
        args.state_file
        or root / ".hf_upload_state" / f"{args.folder_prefix}-{args.node_name}.json"
    )
    api = HfApi()
    if not args.dry_run:
        create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    uploaded, finalized = load_state(state_file)
    print(
        f"[hf-upload] watching {root} -> {args.repo_id} "
        f"prefix={args.folder_prefix} interval={args.upload_interval} retain={args.retain} "
        f"node={args.node_rank}/{args.expected_nodes} retention_owner={args.retention_owner}",
        flush=True,
    )
    while True:
        for iteration in available_iterations(root):
            if not should_upload_iteration(iteration, args):
                continue
            if iteration in uploaded:
                continue
            try:
                if not upload_iteration(api, root, iteration, args):
                    continue
            except Exception as exc:
                print(f"[hf-upload] step={iteration} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not args.dry_run:
                uploaded.add(iteration)
                save_state(state_file, uploaded, finalized)
        if args.retention_owner:
            for iteration in sorted(uploaded):
                if iteration in finalized or not should_upload_iteration(iteration, args):
                    continue
                try:
                    if not finalize_iteration(api, iteration, args):
                        continue
                except Exception as exc:
                    print(f"[hf-upload] step={iteration} finalize failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if not args.dry_run:
                    finalized.add(iteration)
                    save_state(state_file, uploaded, finalized)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
