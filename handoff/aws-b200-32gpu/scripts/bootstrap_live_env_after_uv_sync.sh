#!/usr/bin/env bash
set -euo pipefail

# This is intentionally separate from `uv sync`.
# Current repo metadata does not fully reproduce the live environment because
# DeepEP/DeepGEMM were installed from local patched source trees and torchcomms
# is a local wheel path. Run this after `uv sync --locked --extra dev --extra mlm`
# until pyproject.toml/uv.lock are made authoritative.

REPO_DIR="${REPO_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}"
cd "$REPO_DIR"

export CUDA_HOME="${CUDA_HOME:-$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$HOME/.cache/deep_gemm/deepseek_v32_reap_sft}"

TORCHCOMMS_WHEEL="${TORCHCOMMS_WHEEL:-$HOME/wheelhouse/torchcomms-0.2.0-cp312-cp312-linux_x86_64.whl}"
if [[ -f "$TORCHCOMMS_WHEEL" ]]; then
  uv pip install "$TORCHCOMMS_WHEEL"
else
  echo "WARNING: torchcomms wheel not found at $TORCHCOMMS_WHEEL" >&2
fi

MEGATRON_BRIDGE_COMMIT="${MEGATRON_BRIDGE_COMMIT:-9c9dd848966322fc3ed7706747ec13219ef49dda}"
uv pip install "omegaconf>=2.3.0"
uv pip install --no-deps "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@$MEGATRON_BRIDGE_COMMIT"

DEEP_EP_WHEEL="${DEEP_EP_WHEEL:-$REPO_DIR/handoff/gcp-a4/wheelhouse/deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl}"
REQUIRE_DEEPEP_EXPERT_MAJOR="${REQUIRE_DEEPEP_EXPERT_MAJOR:-1}"
if [[ -f "$DEEP_EP_WHEEL" ]]; then
  uv pip install --force-reinstall "$DEEP_EP_WHEEL"
elif [[ "$REQUIRE_DEEPEP_EXPERT_MAJOR" == "1" ]]; then
  cat >&2 <<EOF
Missing patched DeepEP wheel:
  $DEEP_EP_WHEEL

The current MoE fast path requires Buffer.dispatch_expert_major and
Buffer.combine_expert_major. Refusing to build unpatched DeepEP because that
would silently remove the dispatch-side megakernel path.
EOF
  exit 2
else
  DEEP_EP_DIR="${DEEP_EP_DIR:-/tmp/DeepEP-v1.2.1}"
  if [[ ! -d "$DEEP_EP_DIR/.git" ]]; then
    rm -rf "$DEEP_EP_DIR"
    git clone https://github.com/deepseek-ai/DeepEP.git "$DEEP_EP_DIR"
  fi
  git -C "$DEEP_EP_DIR" checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee

  uv run --no-sync python - "$DEEP_EP_DIR/setup.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "    include_dirs = ['csrc/']\n"
new = (
    "    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')\n"
    "    include_dirs = ['csrc/', f'{cuda_home}/include/cccl']\n"
)
if old in text:
    path.write_text(text.replace(old, new))
elif "include/cccl" not in text:
    raise SystemExit("DeepEP setup.py include patch did not apply")
PY
  uv pip install --no-build-isolation "$DEEP_EP_DIR"
fi

uv run --no-sync python - <<'PY'
from deep_ep import Buffer

missing = [
    name
    for name in ("dispatch_expert_major", "combine_expert_major")
    if not hasattr(Buffer, name)
]
if missing:
    raise SystemExit(f"patched DeepEP APIs missing: {', '.join(missing)}")
print("patched DeepEP expert-major APIs available", flush=True)
PY

DEEP_GEMM_DIR="${DEEP_GEMM_DIR:-/tmp/DeepGEMM.install}"
if [[ ! -d "$DEEP_GEMM_DIR/.git" ]]; then
  rm -rf "$DEEP_GEMM_DIR"
  git clone https://github.com/deepseek-ai/DeepGEMM.git "$DEEP_GEMM_DIR"
fi
git -C "$DEEP_GEMM_DIR" checkout 714dd1a4a980f7937a74343d19a8eba4fe321480
git -C "$DEEP_GEMM_DIR" submodule update --init --recursive
uv pip install --no-build-isolation "$DEEP_GEMM_DIR"

uv run --no-sync python handoff/aws-b200-32gpu/scripts/verify_env.py --expected-gpus "${EXPECTED_GPUS:-8}"
