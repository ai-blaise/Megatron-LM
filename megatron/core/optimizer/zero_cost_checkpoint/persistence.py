# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from typing import Any

import torch


MAGIC = b"MZCC0001"


def atomic_torch_save(
    path: str,
    payload: dict[str, Any],
    *,
    compression: str = "none",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            encoded, codec = _compress(data, compression)
            envelope = pickle.dumps(
                {"version": 1, "codec": codec, "payload": encoded},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            digest = hashlib.sha256(envelope).digest()
            f.write(MAGIC)
            f.write(digest)
            f.write(envelope)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(os.path.dirname(path), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_atomic_torch_save(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a ZCC snapshot")
        digest = f.read(32)
        envelope_data = f.read()
    if hashlib.sha256(envelope_data).digest() != digest:
        raise ValueError(f"{path} failed ZCC checksum validation")
    envelope = pickle.loads(envelope_data)
    if (
        not isinstance(envelope, dict)
        or "codec" not in envelope
        or "payload" not in envelope
    ):
        return envelope
    return pickle.loads(_decompress(envelope["payload"], envelope["codec"]))


def tensor_payload(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data": tensor.detach().cpu(),
    }


def _compress(data: bytes, compression: str) -> tuple[bytes, str]:
    if compression == "none":
        return data, "none"
    if compression.startswith("zstd:"):
        try:
            import zstandard as zstd
        except ImportError:
            return data, "none"
        level = int(compression.split(":", 1)[1])
        return zstd.ZstdCompressor(level=level).compress(data), compression
    raise ValueError(f"unsupported ZCC compression mode: {compression}")


def _decompress(data: bytes, codec: str) -> bytes:
    if codec == "none":
        return data
    if codec.startswith("zstd:"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "zstandard is required to read this compressed ZCC snapshot"
            ) from exc
        return zstd.ZstdDecompressor().decompress(data)
    raise ValueError(f"unsupported ZCC snapshot codec: {codec}")
