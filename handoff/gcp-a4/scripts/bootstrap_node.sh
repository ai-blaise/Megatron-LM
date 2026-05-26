#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ai-blaise/Megatron-LM.git}"
REPO_DIR="${REPO_DIR:-/home/sjpat/Megatron-LM}"
BRANCH="${BRANCH:-flashtraining}"
TORCHCOMMS_WHEEL="${TORCHCOMMS_WHEEL:-/home/sjpat/wheelhouse/torchcomms-0.2.0-cp312-cp312-linux_x86_64.whl}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
INSTALL_LIVE_EXTRAS="${INSTALL_LIVE_EXTRAS:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
PREBUILD_EXTENSIONS="${PREBUILD_EXTENSIONS:-1}"

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  missing_packages=()
  command -v git >/dev/null 2>&1 || missing_packages+=(git)
  command -v rsync >/dev/null 2>&1 || missing_packages+=(rsync)
  command -v tar >/dev/null 2>&1 || missing_packages+=(tar)
  command -v gcc >/dev/null 2>&1 || missing_packages+=(gcc)
  command -v g++ >/dev/null 2>&1 || missing_packages+=(gcc-c++)
  command -v make >/dev/null 2>&1 || missing_packages+=(make)
  command -v patch >/dev/null 2>&1 || missing_packages+=(patch)
  [[ -e /usr/include/infiniband/mlx5dv.h ]] || missing_packages+=(rdma-core-devel)

  if ((${#missing_packages[@]})); then
    if command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y "${missing_packages[@]}"
    else
      echo "Missing system packages: ${missing_packages[*]}" >&2
      echo "Install them before running bootstrap_node.sh." >&2
      exit 3
    fi
  fi
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv install completed but uv is still not on PATH" >&2
  exit 127
fi

if [[ ! -f "$TORCHCOMMS_WHEEL" ]]; then
  cat >&2 <<EOF
Missing torchcomms wheel:
  $TORCHCOMMS_WHEEL

uv.lock and pyproject.toml reference this local path. Put the wheel at that
path on every node, or update the source path and regenerate uv.lock before
running the locked sync.
EOF
  exit 2
fi

uv sync --locked --only-group build

NVIDIA_SITE="$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia"
export CUDA_HOME="${CUDA_HOME:-$NVIDIA_SITE/cu13}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
CUDNN_HOME="$NVIDIA_SITE/cudnn"
NCCL_HOME="$NVIDIA_SITE/nccl"
NVSHMEM_HOME="$NVIDIA_SITE/nvshmem"
CUSPARSELT_HOME="$NVIDIA_SITE/cusparselt"

prepare_cuda_python_layout() {
  local dir so base link shim

  for dir in \
    "$CUDA_HOME/lib" \
    "$CUDNN_HOME/lib" \
    "$NCCL_HOME/lib" \
    "$NVSHMEM_HOME/lib" \
    "$CUSPARSELT_HOME/lib"; do
    [[ -d "$dir" ]] || continue
    for so in "$dir"/lib*.so.*; do
      [[ -e "$so" ]] || continue
      base="$(basename "$so")"
      link="${base%%.so.*}.so"
      [[ -e "$dir/$link" ]] || ln -s "$base" "$dir/$link"
    done
  done

  mkdir -p "$CUDA_HOME/include"
  shim="$CUDA_HOME/include/cuda_profiler_api.h"
  if [[ ! -e "$shim" ]]; then
    cat >"$shim" <<'EOF'
#pragma once

#include <cuda_runtime_api.h>
EOF
  fi
}

prepare_cuda_python_layout

export PATH="$REPO_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$NCCL_HOME/lib:$CUDNN_HOME/lib:$NVSHMEM_HOME/lib:$CUSPARSELT_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$NCCL_HOME/lib:$CUDNN_HOME/lib:$NVSHMEM_HOME/lib:$CUSPARSELT_HOME/lib:${LIBRARY_PATH:-}"
export CPATH="$CUDNN_HOME/include:$CUDA_HOME/include:$NCCL_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CPATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100a}"
export NVTE_CMAKE_BUILD_DIR="${NVTE_CMAKE_BUILD_DIR:-/tmp/te-cmake-build-$USER}"
export NVTE_CMAKE_EXTRA_ARGS="${NVTE_CMAKE_EXTRA_ARGS:--DCMAKE_CUDA_FLAGS=-I$NCCL_HOME/include -DCMAKE_CXX_FLAGS=-I$NCCL_HOME/include -DCUDAToolkit_LIBRARY_DIR=$CUDA_HOME/lib -DCUDA_CUDART=$CUDA_HOME/lib/libcudart.so.13 -DCUDA_cudart_LIBRARY=$CUDA_HOME/lib/libcudart.so.13 -DCUDA_cudart_static_LIBRARY=$CUDA_HOME/lib/libcudart_static.a -DCUDA_cublas_LIBRARY=$CUDA_HOME/lib/libcublas.so.13 -DCUDA_cublasLt_LIBRARY=$CUDA_HOME/lib/libcublasLt.so.13}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

uv sync --locked --extra dev --extra mlm

if [[ "$INSTALL_LIVE_EXTRAS" == "1" ]]; then
  EXPECTED_GPUS="$EXPECTED_GPUS" TORCHCOMMS_WHEEL="$TORCHCOMMS_WHEEL" \
    handoff/aws-b200-32gpu/scripts/bootstrap_live_env_after_uv_sync.sh
fi

uv run --no-sync python handoff/aws-b200-32gpu/scripts/verify_env.py --expected-gpus "$EXPECTED_GPUS"

if [[ "$PREBUILD_EXTENSIONS" == "1" ]]; then
  handoff/aws-b200-32gpu/scripts/prebuild_custom_extensions.sh
fi
