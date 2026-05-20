#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}"
cd "$REPO_DIR"

if [[ -z "${CUDA_HOME:-}" && -x "$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" ]]; then
  export CUDA_HOME="$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

uv run --no-sync python - <<'PY'
import importlib

modules = [
    ("HISA", "megatron.core.extensions.hisa_indexer.kernels.build"),
    ("IndexCache", "megatron.core.quantization.indexcache.kernels.build"),
    ("HIGGS", "megatron.core.quantization.higgs.kernels.build"),
]

for label, module_name in modules:
    print(f"prebuilding {label}: {module_name}", flush=True)
    module = importlib.import_module(module_name)
    module.get_ext()
    print(f"{label} ready", flush=True)

print("custom extension prebuild complete", flush=True)
PY

if [[ "${TURBOQUANT:-0}" == "1" ]]; then
  uv run --no-sync python - <<'PY'
import importlib
module = importlib.import_module("megatron.core.quantization.turboquant.kernels.build")
module.get_ext()
print("TurboQuant ready", flush=True)
PY
fi

