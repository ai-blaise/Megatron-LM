#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare BlaiseAI DeepSeek-V3.2 SFT conversations for Megatron SFTDataset."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _json_loads_maybe(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _default_tool_name(tools: Any) -> str:
    parsed_tools = _json_loads_maybe(tools, [])
    if isinstance(parsed_tools, list) and parsed_tools:
        first = parsed_tools[0]
        if isinstance(first, dict):
            function = first.get("function", first)
            if isinstance(function, dict) and function.get("name"):
                return function["name"]
    return "web-search"


def _tool_query(tool_content: Any) -> str:
    parsed = _json_loads_maybe(tool_content)
    if isinstance(parsed, dict):
        query = parsed.get("query")
        if query is not None:
            return str(query)
    return ""


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads_maybe(tools, [])
    return parsed if isinstance(parsed, list) else []


def _tool_call(tool_name: str, query: str, row_idx: int, call_idx: int) -> Dict[str, Any]:
    return {
        "id": f"call_{row_idx}_{call_idx}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": _json_dumps({"query": query}),
        },
    }


def normalize_conversation(row: Dict[str, Any], row_idx: int) -> Tuple[Optional[List[Dict]], Dict[str, int]]:
    """Convert empty assistant placeholders into assistant tool_calls."""
    stats = {
        "tool_calls": 0,
        "empty_assistant_without_tool": 0,
    }
    conversations = row.get("messages") or row.get("conversations")
    if not isinstance(conversations, list):
        return None, stats

    tools = _normalize_tools(row.get("tools"))
    tool_name = _default_tool_name(tools)
    output: List[Dict[str, Any]] = []
    call_idx = 0

    for index, message in enumerate(conversations):
        role = (message.get("role") or "").lower()
        content = message.get("content") or ""

        if role == "system":
            normalized = {"role": "system", "content": content}
            if tools:
                normalized["tools"] = tools
            output.append(normalized)
            continue

        if role == "assistant" and not content and not message.get("tool_calls"):
            next_message = conversations[index + 1] if index + 1 < len(conversations) else {}
            if (next_message.get("role") or "").lower() != "tool":
                stats["empty_assistant_without_tool"] += 1
                return None, stats
            call_idx += 1
            output.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            tool_name=tool_name,
                            query=_tool_query(next_message.get("content") or ""),
                            row_idx=row_idx,
                            call_idx=call_idx,
                        )
                    ],
                    "_synthetic_tool_calls": True,
                    "_synthetic_tool_call_count": 1,
                }
            )
            stats["tool_calls"] += 1
            continue

        if role == "tool":
            output.append({"role": "tool", "content": content})
            continue

        if role in ("user", "assistant", "developer"):
            normalized = {"role": role, "content": content}
            if message.get("tool_calls"):
                normalized["tool_calls"] = message["tool_calls"]
            output.append(normalized)
            continue

        return None, stats

    if output and output[0]["role"] != "system":
        output.insert(0, {"role": "system", "content": "", "tools": tools})

    return output, stats


def _load_dataset(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("prepare_blaise_sft.py requires the datasets package") from exc

    split = args.split
    if args.max_samples is not None and not args.streaming and "[" not in split:
        split = f"{split}[:{args.max_samples}]"

    if args.data_files:
        data_files = args.data_files
        return load_dataset(
            args.dataset,
            data_files=data_files,
            split=split,
            streaming=args.streaming,
        )

    return load_dataset(
        args.dataset,
        args.config,
        split=split,
        streaming=args.streaming,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BlaiseAI/blaise-sft-training-mix")
    parser.add_argument("--config", default="nemotron-full-family")
    parser.add_argument(
        "--data-files",
        default=None,
        help="Optional dataset repo parquet file, e.g. full_mix_all_sources.parquet.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    counts = {
        "read": 0,
        "written": 0,
        "skipped": 0,
        "tool_calls": 0,
        "empty_assistant_without_tool": 0,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        for row_idx, row in enumerate(dataset):
            counts["read"] += 1
            messages, row_stats = normalize_conversation(row, row_idx)
            counts["tool_calls"] += row_stats["tool_calls"]
            counts["empty_assistant_without_tool"] += row_stats["empty_assistant_without_tool"]

            if messages is None:
                counts["skipped"] += 1
                continue

            record = {
                "messages": messages,
                "enable_thinking": row.get("enable_thinking"),
                "source_dataset": row.get("source_dataset"),
                "task": row.get("task"),
                "episode": row.get("episode"),
                "run_id": row.get("run_id"),
            }
            output_file.write(_json_dumps(record) + "\n")
            counts["written"] += 1

            if args.max_samples is not None and counts["written"] >= args.max_samples:
                break

    print(
        "Prepared {written} records from {read} rows at {path}. "
        "Synthesized {tool_calls} assistant tool calls; skipped {skipped} rows "
        "({empty_assistant_without_tool} empty assistant turns without following tool output).".format(
            path=output_path,
            **counts,
        )
    )


if __name__ == "__main__":
    main()
