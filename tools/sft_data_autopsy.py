#!/usr/bin/env python3
"""CPU autopsy for DeepSeek SFT JSONL rendering, packing, and supervision.

This intentionally uses the live Megatron SFT normalizer/tokenizer path instead
of a simplified parser. The goal is to answer: what exact text becomes
supervised labels, how dense are those labels after 32k packing, and how badly
are labels distributed across CP chunks?
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megatron.core.tokenizers.text.libraries.sft_tokenizer import (
    DEEPSEEK_BOS_TOKEN,
    IGNORE_INDEX,
    SFTTokenizer,
)
from megatron.training.datasets.sft_dataset import (
    _load_or_build_sft_row_shuffle_index,
    _normalize_sft_messages,
)


TOKENIZER: SFTTokenizer | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=os.path.expanduser(
            "~/data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full.jsonl"
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        default=os.path.expanduser(
            "~/models/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4"
        ),
    )
    parser.add_argument("--prompt-format", default="deepseek-v3.2")
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=3)
    parser.add_argument("--head", type=int, default=64)
    parser.add_argument("--random", type=int, default=512)
    parser.add_argument("--even", type=int, default=128)
    parser.add_argument("--detail", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shuffle-rows",
        action="store_true",
        help="Analyze logical SFT samples through the deterministic full-row shuffle sidecar.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=1234,
        help="Seed for the SFT row shuffle sidecar.",
    )
    parser.add_argument(
        "--shuffle-index-path",
        default=None,
        help="Optional explicit .npy shuffle sidecar path.",
    )
    parser.add_argument("--output-dir", default="logs/sft-data-autopsy")
    parser.add_argument(
        "--max-span-decode-tokens",
        type=int,
        default=256,
        help="Max tokens to decode for each supervised span preview.",
    )
    parser.add_argument(
        "--max-render-preview-chars",
        type=int,
        default=600,
        help="Max chars per rendered segment preview in detail output.",
    )
    return parser.parse_args()


def init_worker(tokenizer_model: str, prompt_format: str) -> None:
    global TOKENIZER
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    TOKENIZER = SFTTokenizer(tokenizer_model, prompt_format)


def read_offsets(data_path: Path) -> np.ndarray:
    offsets_path = Path(f"{data_path}.offsets.npy")
    if not offsets_path.exists():
        raise FileNotFoundError(f"missing offsets sidecar: {offsets_path}")
    return np.load(offsets_path, mmap_mode="r")


def read_jsonl_at(data_path: str, offset: int) -> dict[str, Any]:
    with open(data_path, "rb") as handle:
        handle.seek(int(offset))
        return json.loads(handle.readline())


def split_conversations(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    split: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            if current:
                split.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        split.append(current)
    return split


def padding_divisor(cp: int, tp: int, hybrid_cp: bool = False, dp: int = 1) -> int:
    cp_pad = dp * cp * 2 if hybrid_cp else (cp * 2 if cp > 1 else 1)
    tp_pad = tp if tp > 1 else 1
    return cp_pad * tp_pad


def active_spans(mask: list[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def preview_text(text: str, limit: int) -> str:
    text = text.replace("\x00", "").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[snip]...\n" + text[-half:]


def safe_decode(tokenizer: SFTTokenizer, ids: list[int]) -> str:
    ids = [int(item) for item in ids if int(item) >= 0]
    if not ids:
        return ""
    return tokenizer._tokenizer.decode(ids, skip_special_tokens=False)


def token_top_counts(tokenizer: SFTTokenizer, ids: list[int], topn: int = 16) -> list[dict[str, Any]]:
    counts = collections.Counter(int(item) for item in ids if int(item) >= 0)
    out = []
    for token_id, count in counts.most_common(topn):
        out.append(
            {
                "id": token_id,
                "count": count,
                "text": safe_decode(tokenizer, [token_id]),
            }
        )
    return out


def render_segments(
    tokenizer: SFTTokenizer, conversation: list[dict[str, Any]], max_preview_chars: int
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    thinking_mode = tokenizer._deepseek_thinking_mode(conversation)
    tokens: list[int] = []
    targets: list[int] = []

    bos_tokens = tokenizer._encode_deepseek_segment(DEEPSEEK_BOS_TOKEN)
    tokens.extend(int(item) for item in bos_tokens)
    targets.extend([IGNORE_INDEX] * len(bos_tokens))

    segments = [
        {
            "role": "bos",
            "masked": True,
            "token_count": len(bos_tokens),
            "target_count": 0,
            "thinking_mode": thinking_mode,
            "preview": DEEPSEEK_BOS_TOKEN,
        }
    ]

    for index, message in enumerate(conversation):
        rendered = tokenizer._deepseek_render_message(index, conversation, thinking_mode)
        segment_tokens = [int(item) for item in tokenizer._encode_deepseek_segment(rendered)]
        masked = tokenizer._mask_deepseek_message_from_loss(message)
        tokens.extend(segment_tokens)
        if masked:
            targets.extend([IGNORE_INDEX] * len(segment_tokens))
        else:
            targets.extend(segment_tokens)
        segments.append(
            {
                "role": message.get("role"),
                "masked": bool(masked),
                "synthetic_tool_calls": bool(message.get("_synthetic_tool_calls")),
                "tool_call_count": len(message.get("tool_calls") or []),
                "token_count": len(segment_tokens),
                "target_count": 0 if masked else len(segment_tokens),
                "thinking_mode": thinking_mode,
                "has_reasoning_content": bool(message.get("reasoning_content")),
                "has_tool_calls": bool(message.get("tool_calls")),
                "content_len": len(message.get("content") or ""),
                "preview": preview_text(rendered, max_preview_chars),
            }
        )
    return segments, tokens, targets


def pack_like_training(
    tokenizer: SFTTokenizer,
    conversations: list[list[dict[str, Any]]],
    seq_len: int,
    pad_div: int,
    max_span_decode_tokens: int,
    max_render_preview_chars: int,
) -> dict[str, Any]:
    pad = tokenizer.pad_id
    pack_tokens: list[int] = []
    pack_targets: list[int] = []
    pack_padding_mask: list[bool] = []
    all_segments: list[dict[str, Any]] = []

    full_tokens: list[int] = []
    full_targets: list[int] = []
    full_padding: list[bool] = []

    def extend_padding(
        tokens: list[int], targets: list[int], padding_mask: list[bool], pad_len: int
    ) -> None:
        tokens.extend([pad] * pad_len)
        targets.extend([pad] * pad_len)
        padding_mask.extend([True] * pad_len)

    for conv_idx, conversation in enumerate(conversations):
        segments, tokens, targets = render_segments(
            tokenizer, conversation, max_render_preview_chars
        )
        for segment in segments:
            segment["conversation_index"] = conv_idx
        all_segments.extend(segments)

        full_tokens.extend(tokens)
        full_targets.extend(targets)
        full_padding.extend([False] * len(tokens))
        full_mod = len(full_tokens) % pad_div
        if full_mod:
            extend_padding(full_tokens, full_targets, full_padding, pad_div - full_mod)

        if len(pack_tokens) < seq_len + 1:
            pack_tokens.extend(tokens)
            pack_targets.extend(targets)
            pack_padding_mask.extend([False] * len(tokens))
            mod_token_count = len(pack_tokens) % pad_div
            if mod_token_count:
                extend_padding(
                    pack_tokens, pack_targets, pack_padding_mask, pad_div - mod_token_count
                )
            if len(pack_tokens) >= seq_len + 1:
                pack_tokens = pack_tokens[:seq_len]
                pack_targets = pack_targets[:seq_len]
                pack_padding_mask = pack_padding_mask[:seq_len]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_padding_mask.append(True)

    if len(pack_tokens) < seq_len + 1:
        extend_padding(pack_tokens, pack_targets, pack_padding_mask, seq_len + 1 - len(pack_tokens))

    if len(pack_tokens) != seq_len + 1:
        raise RuntimeError(f"bad pack length: {len(pack_tokens)} != {seq_len + 1}")

    labels = pack_targets[1:]
    target_padding = pack_padding_mask[1:]
    loss_mask = [
        (int(label) != IGNORE_INDEX) and (not bool(is_pad))
        for label, is_pad in zip(labels, target_padding)
    ]
    active_indices = [idx for idx, active in enumerate(loss_mask) if active]
    active_labels = [int(labels[idx]) for idx in active_indices]

    full_labels = full_targets[1:]
    full_target_padding = full_padding[1:]
    full_active = [
        (int(label) != IGNORE_INDEX) and (not bool(is_pad))
        for label, is_pad in zip(full_labels, full_target_padding)
    ]

    chunk = seq_len // 4
    cp_counts = [
        int(sum(loss_mask[start : start + chunk]))
        for start in range(0, seq_len, chunk)
    ]

    spans = active_spans(loss_mask)
    decoded_spans = []
    for start, end in spans[:8]:
        ids = [int(labels[pos]) for pos in range(start, min(end + 1, start + max_span_decode_tokens))]
        decoded_spans.append(
            {
                "start": start,
                "end": end,
                "length": end - start + 1,
                "text": preview_text(safe_decode(tokenizer, ids), 1200),
            }
        )

    first_active = active_indices[0] if active_indices else None
    first_window = None
    if first_active is not None:
        start = max(0, first_active - 16)
        end = min(seq_len, first_active + 64)
        first_window = {
            "start": start,
            "end": end,
            "input_text": safe_decode(tokenizer, [int(item) for item in pack_tokens[start:end]]),
            "label_text": safe_decode(
                tokenizer, [int(item) for item in labels[start:end] if int(item) != IGNORE_INDEX]
            ),
            "mask": [1 if item else 0 for item in loss_mask[start:end]],
            "input_ids": [int(item) for item in pack_tokens[start:end]],
            "label_ids": [int(item) for item in labels[start:end]],
        }

    return {
        "full_token_count_padded": len(full_tokens),
        "full_active_labels": int(sum(full_active)),
        "pack_active_labels": len(active_indices),
        "truncated_active_labels": max(0, int(sum(full_active)) - len(active_indices)),
        "density": len(active_indices) / seq_len,
        "cp_counts": cp_counts,
        "cp_zero_count": sum(1 for item in cp_counts if item == 0),
        "span_count": len(spans),
        "spans": [{"start": start, "end": end, "length": end - start + 1} for start, end in spans],
        "decoded_spans": decoded_spans,
        "first_active_window": first_window,
        "active_label_top_ids": token_top_counts(tokenizer, active_labels),
        "segments": all_segments,
    }


def try_hf_template(tokenizer: SFTTokenizer, conversation: list[dict[str, Any]]) -> dict[str, Any]:
    hf_tokenizer = tokenizer._tokenizer
    if not getattr(hf_tokenizer, "chat_template", None):
        return {"available": False, "reason": "tokenizer.chat_template is None"}
    try:
        rendered = hf_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic report wants the reason.
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "char_len": len(rendered),
        "preview": preview_text(rendered, 1200),
    }


def message_flags(messages: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = collections.Counter((msg.get("role") or "<missing>") for msg in messages)
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    return {
        "role_counts": dict(role_counts),
        "assistant_count": len(assistant_messages),
        "assistant_empty_count": sum(1 for msg in assistant_messages if not (msg.get("content") or "")),
        "assistant_tool_call_count": sum(1 for msg in assistant_messages if msg.get("tool_calls")),
        "synthetic_assistant_tool_call_count": sum(
            1 for msg in assistant_messages if msg.get("_synthetic_tool_calls")
        ),
        "reasoning_content_count": sum(1 for msg in assistant_messages if msg.get("reasoning_content")),
        "assistant_content_thought_prefix_count": sum(
            1
            for msg in assistant_messages
            if (msg.get("content") or "").lstrip().lower().startswith(("thought:", "reasoning:", "- thought:", "- reasoning:"))
        ),
        "assistant_content_final_answer_count": sum(
            1 for msg in assistant_messages if "final answer" in (msg.get("content") or "").lower()
        ),
        "tool_count": role_counts.get("tool", 0),
        "system_count": role_counts.get("system", 0),
        "user_count": role_counts.get("user", 0),
    }


def analyze_row(task: tuple[str, int, int, int, int, int, int, int]) -> dict[str, Any]:
    data_path, sample_idx, row_idx, offset, seq_len, cp, tp, dp = task
    if TOKENIZER is None:
        raise RuntimeError("worker tokenizer was not initialized")
    tokenizer = TOKENIZER
    raw = read_jsonl_at(data_path, offset)
    raw_messages = raw.get("messages", raw.get("conversations"))
    normalized = _normalize_sft_messages(raw, row_idx)
    if not isinstance(normalized, list):
        return {
            "sample_idx": sample_idx,
            "row_idx": row_idx,
            "offset": int(offset),
            "error": "normalized messages are not a list",
            "raw_keys": sorted(raw.keys()),
        }

    conversations = split_conversations(normalized)
    packed = pack_like_training(
        tokenizer,
        conversations,
        seq_len,
        padding_divisor(cp=cp, tp=tp, dp=dp),
        max_span_decode_tokens=256,
        max_render_preview_chars=600,
    )
    hf_template = try_hf_template(tokenizer, conversations[0] if conversations else normalized)
    flags = message_flags(normalized)
    raw_flags = message_flags(raw_messages if isinstance(raw_messages, list) else [])

    return {
        "sample_idx": sample_idx,
        "row_idx": row_idx,
        "offset": int(offset),
        "raw_keys": sorted(raw.keys()),
        "raw_enable_thinking": raw.get("enable_thinking"),
        "raw_flags": raw_flags,
        "flags": flags,
        "conversation_count": len(conversations),
        "packed": packed,
        "hf_template": hf_template,
    }


def choose_indices(total: int, head: int, random_count: int, even: int, seed: int) -> list[int]:
    selected: set[int] = set(range(min(head, total)))
    if even > 0:
        if even == 1:
            selected.add(0)
        else:
            for item in np.linspace(0, total - 1, even, dtype=np.int64).tolist():
                selected.add(int(item))
    if random_count > 0:
        rng = random.Random(seed)
        selected.update(rng.sample(range(total), min(random_count, total)))
    return sorted(selected)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize(results: list[dict[str, Any]], seq_len: int, cp: int, gbs: int = 12) -> dict[str, Any]:
    good = [item for item in results if "packed" in item]
    active = [item["packed"]["pack_active_labels"] for item in good]
    density = [item["packed"]["density"] for item in good]
    full_active = [item["packed"]["full_active_labels"] for item in good]
    truncated = [item["packed"]["truncated_active_labels"] for item in good]
    zero_rows = sum(1 for item in active if item == 0)
    cp_counts = [count for item in good for count in item["packed"]["cp_counts"]]
    cp_zero = sum(1 for count in cp_counts if count == 0)
    roles = collections.Counter()
    flags_total = collections.Counter()
    top_ids = collections.Counter()
    for item in good:
        roles.update(item["flags"].get("role_counts", {}))
        for key, value in item["flags"].items():
            if key != "role_counts" and isinstance(value, int):
                flags_total[key] += value
        for token in item["packed"]["active_label_top_ids"]:
            top_ids[(token["id"], token["text"])] += int(token["count"])

    per_step_expected = statistics.mean(active) * gbs if active else 0.0
    random_ce = math.log(129280)

    return {
        "rows_ok": len(good),
        "rows_error": len(results) - len(good),
        "seq_len": seq_len,
        "cp": cp,
        "gbs_reference": gbs,
        "active_labels_per_row": {
            "mean": statistics.mean(active) if active else 0.0,
            "median": statistics.median(active) if active else 0.0,
            "min": min(active) if active else 0,
            "max": max(active) if active else 0,
            "p10": percentile(active, 0.10),
            "p25": percentile(active, 0.25),
            "p75": percentile(active, 0.75),
            "p90": percentile(active, 0.90),
            "p99": percentile(active, 0.99),
        },
        "density": {
            "mean": statistics.mean(density) if density else 0.0,
            "median": statistics.median(density) if density else 0.0,
            "p90": percentile(density, 0.90),
            "p99": percentile(density, 0.99),
        },
        "full_active_labels_per_row": {
            "mean": statistics.mean(full_active) if full_active else 0.0,
            "median": statistics.median(full_active) if full_active else 0.0,
            "max": max(full_active) if full_active else 0,
        },
        "truncated_active_labels": {
            "rows_with_truncation": sum(1 for value in truncated if value > 0),
            "total_truncated": int(sum(truncated)),
            "max": max(truncated) if truncated else 0,
        },
        "zero_supervision_rows": zero_rows,
        "zero_supervision_row_frac": zero_rows / len(good) if good else 0.0,
        "cp_chunks": {
            "total": len(cp_counts),
            "zero": cp_zero,
            "zero_frac": cp_zero / len(cp_counts) if cp_counts else 0.0,
            "mean_active": statistics.mean(cp_counts) if cp_counts else 0.0,
            "median_active": statistics.median(cp_counts) if cp_counts else 0.0,
            "p90_active": percentile(cp_counts, 0.90),
            "max_active": max(cp_counts) if cp_counts else 0,
        },
        "expected_active_labels_per_gbs12_step_from_sample_mean": per_step_expected,
        "random_ce_for_padded_vocab_129280": random_ce,
        "role_counts": dict(roles),
        "flag_totals": dict(flags_total),
        "top_active_label_ids": [
            {"id": token_id, "text": text, "count": count}
            for (token_id, text), count in top_ids.most_common(24)
        ],
        "hf_template_available_rows": sum(
            1 for item in good if item.get("hf_template", {}).get("available")
        ),
        "hf_template_unavailable_reasons": dict(
            collections.Counter(
                item.get("hf_template", {}).get("reason", "<none>")
                for item in good
                if not item.get("hf_template", {}).get("available")
            )
        ),
    }


def select_detail_rows(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    good = [item for item in results if "packed" in item]
    picks: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(items: list[dict[str, Any]]) -> None:
        for item in items:
            if len(picks) >= limit:
                return
            sample_idx = int(item.get("sample_idx", item["row_idx"]))
            if sample_idx not in seen:
                seen.add(sample_idx)
                picks.append(item)

    add(sorted(good, key=lambda item: item["packed"]["pack_active_labels"])[: limit // 4 + 1])
    add(sorted(good, key=lambda item: item["packed"]["pack_active_labels"], reverse=True)[: limit // 4 + 1])
    add(
        sorted(
            [item for item in good if item["flags"].get("synthetic_assistant_tool_call_count", 0) > 0],
            key=lambda item: item["packed"]["pack_active_labels"],
        )[: limit // 4 + 1]
    )
    add(
        sorted(
            [item for item in good if item["packed"].get("truncated_active_labels", 0) > 0],
            key=lambda item: item["packed"]["truncated_active_labels"],
            reverse=True,
        )[: limit // 4 + 1]
    )
    add(good)
    return picks[:limit]


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# SFT Data Autopsy")
    lines.append("")
    lines.append(f"- Data path: `{args.data_path}`")
    lines.append(f"- Tokenizer model: `{args.tokenizer_model}`")
    lines.append(f"- Prompt format: `{args.prompt_format}`")
    lines.append(
        f"- Row shuffle: `{args.shuffle_rows}` seed=`{args.shuffle_seed}` "
        f"path=`{args.shuffle_index_path or '<default>'}`"
    )
    lines.append(f"- Rows analyzed: `{summary['rows_ok']}` ok, `{summary['rows_error']}` errors")
    lines.append(f"- Seq len / CP / TP / DP: `{args.seq_len}` / `{args.cp}` / `{args.tp}` / `{args.dp}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "active_labels_per_row",
        "density",
        "full_active_labels_per_row",
        "truncated_active_labels",
        "cp_chunks",
    ):
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary[key], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    lines.append(f"- Zero-supervision rows: `{summary['zero_supervision_rows']}` (`{summary['zero_supervision_row_frac']:.4f}`)")
    lines.append(
        "- Expected active labels per GBS=12 step from sample mean: "
        f"`{summary['expected_active_labels_per_gbs12_step_from_sample_mean']:.1f}`"
    )
    lines.append(f"- Random CE for padded vocab 129280: `{summary['random_ce_for_padded_vocab_129280']:.4f}`")
    lines.append("")
    lines.append("### Role Counts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["role_counts"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("### Flag Totals")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["flag_totals"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("### Top Active Label IDs")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["top_active_label_ids"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("### HF Template Check")
    lines.append("")
    lines.append(f"- HF template available rows: `{summary['hf_template_available_rows']}`")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["hf_template_unavailable_reasons"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Detail Rows")
    lines.append("")
    for item in detail_rows:
        packed = item["packed"]
        lines.append(f"### Sample {item.get('sample_idx', item['row_idx'])} -> Row {item['row_idx']}")
        lines.append("")
        lines.append(
            "- active="
            f"`{packed['pack_active_labels']}`, density=`{packed['density']:.6f}`, "
            f"cp_counts=`{packed['cp_counts']}`, truncated_active=`{packed['truncated_active_labels']}`"
        )
        lines.append(f"- flags: `{json.dumps(item['flags'], ensure_ascii=False)}`")
        lines.append(f"- raw_flags: `{json.dumps(item['raw_flags'], ensure_ascii=False)}`")
        lines.append(f"- hf_template: `{json.dumps(item['hf_template'], ensure_ascii=False)}`")
        lines.append("")
        if packed.get("decoded_spans"):
            lines.append("Supervised span previews:")
            lines.append("")
            for span in packed["decoded_spans"][:4]:
                lines.append(f"- span `{span['start']}..{span['end']}` len `{span['length']}`:")
                lines.append("")
                lines.append("```text")
                lines.append(span["text"])
                lines.append("```")
                lines.append("")
        if packed.get("first_active_window"):
            window = packed["first_active_window"]
            lines.append("First active window:")
            lines.append("")
            lines.append("```text")
            lines.append(preview_text(window["input_text"], 1200))
            lines.append("```")
            lines.append("")
        lines.append("Rendered segment sketch:")
        lines.append("")
        for segment in packed["segments"][:12]:
            lines.append(
                f"- conv={segment.get('conversation_index')} role={segment.get('role')} "
                f"masked={segment.get('masked')} tokens={segment.get('token_count')} "
                f"targets={segment.get('target_count')} synthetic={segment.get('synthetic_tool_calls')} "
                f"tool_calls={segment.get('tool_call_count')} reasoning={segment.get('has_reasoning_content')}"
            )
            preview = segment.get("preview") or ""
            if preview:
                lines.append("")
                lines.append("```text")
                lines.append(preview_text(preview, 900))
                lines.append("```")
                lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    tokenizer_model = str(Path(args.tokenizer_model).expanduser())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    offsets = read_offsets(data_path)
    shuffle_index = None
    if args.shuffle_rows:
        os.environ["MEGATRON_SFT_SHUFFLE_ROWS"] = "1"
        os.environ["MEGATRON_SFT_SHUFFLE_SEED"] = str(args.shuffle_seed)
        if args.shuffle_index_path:
            os.environ["MEGATRON_SFT_SHUFFLE_INDEX_PATH"] = str(
                Path(args.shuffle_index_path).expanduser()
            )
        shuffle_index = _load_or_build_sft_row_shuffle_index(
            dataset_path=str(data_path),
            total_rows=len(offsets),
            seed=args.shuffle_seed,
        )
    indices = choose_indices(
        total=len(offsets),
        head=args.head,
        random_count=args.random,
        even=args.even,
        seed=args.seed,
    )
    tasks = []
    for idx in indices:
        sample_idx = int(idx)
        row_idx = sample_idx
        if shuffle_index is not None:
            row_idx = int(shuffle_index[sample_idx % len(shuffle_index)])
        tasks.append(
            (
                str(data_path),
                sample_idx,
                row_idx,
                int(offsets[row_idx]),
                args.seq_len,
                args.cp,
                args.tp,
                args.dp,
            )
        )

    print(
        f"Analyzing {len(tasks)} logical samples from {len(offsets)} total rows "
        f"with {args.workers} workers shuffle_rows={args.shuffle_rows}",
        flush=True,
    )

    if args.workers <= 1:
        init_worker(tokenizer_model, args.prompt_format)
        results = [analyze_row(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(tokenizer_model, args.prompt_format),
        ) as pool:
            results = list(pool.imap_unordered(analyze_row, tasks, chunksize=4))
    results.sort(key=lambda item: int(item.get("sample_idx", item.get("row_idx", -1))))

    summary = summarize(results, seq_len=args.seq_len, cp=args.cp)
    detail_rows = select_detail_rows(results, args.detail)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    json_path = out_dir / f"sft_data_autopsy_{timestamp}.json"
    md_path = out_dir / f"sft_data_autopsy_{timestamp}.md"
    latest_json = out_dir / "latest.json"
    latest_md = out_dir / "latest.md"

    payload = {
        "args": vars(args),
        "elapsed_sec": time.time() - started,
        "dataset_total_rows": int(len(offsets)),
        "sampled_indices": indices,
        "shuffle_rows": bool(args.shuffle_rows),
        "shuffle_seed": int(args.shuffle_seed),
        "shuffle_index_path": (
            str(Path(args.shuffle_index_path).expanduser())
            if args.shuffle_index_path
            else (f"{data_path}.shuffle.seed{args.shuffle_seed}.npy" if args.shuffle_rows else None)
        ),
        "summary": summary,
        "detail_rows": detail_rows,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, summary, detail_rows, args)
    write_markdown(latest_md, summary, detail_rows, args)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
