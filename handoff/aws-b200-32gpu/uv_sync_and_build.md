# UV Sync And Build Reproducibility

## Current State

The current live `.venv` is not exactly represented by `pyproject.toml` and
`uv.lock`.

Observed command:

```bash
uv sync --locked --extra dev --extra mlm --check
```

Result: environment is outdated. A clean sync would uninstall packages that the
current runtime has been relying on.

Most important removals:

- `deep-ep==1.2.1+9af0e0d`
- `deep-gemm==2.5.0+714dd1a`
- `nv-one-logger-core`
- `nv-one-logger-training-telemetry`
- installed TE wheel packages

It would also install the git-sourced TransformerEngine dependency from
`pyproject.toml`, which is not the same as the live wheel state:

- live: `transformer-engine==2.11.0`,
  `transformer-engine-torch==2.11.0`,
  `transformer-engine-cu13==2.11.0`
- pyproject source:
  `https://github.com/NVIDIA/TransformerEngine.git@c6853b65b7177ab3785c48c130166ec3f9324c46`

## Current Live Key Packages

From the current `.venv`:

```text
torch==2.11.0
triton==3.6.0
transformer-engine==2.11.0
transformer-engine-torch==2.11.0
deep_ep==1.2.1+9af0e0d
deep_gemm==2.5.0+714dd1a
torchcomms==0.2.0
nvidia-nccl-cu13==2.28.9
nvidia-nvshmem-cu13==3.4.5
nvidia-cutlass-dsl==4.4.0
nvidia-mathdx==25.6.0
fast-hadamard-transform==1.0.4.post1
wandb==0.25.0
datasets==4.6.1
```

`nv-grouped-gemm` was not installed in the live `.venv` when checked, although
it appears in the `dev`/`lts` extras. The model has been using TE grouped MLP
paths.

## TorchComms Local Wheel

`pyproject.toml` has:

```toml
torchcomms = { path = "/home/sjpat/wheelhouse/torchcomms-0.2.0-cp312-cp312-linux_x86_64.whl" }
```

AWS options:

1. Copy that wheel to the same path on every node.
2. Change the path and update `uv.lock`.
3. Host the wheel somewhere accessible and update the source mapping.

Without this, `uv sync` will not reproduce the current environment.

## DeepEP

Live install source:

```text
deep-ep @ file:///tmp/DeepEP-v1.2.1
remote: https://github.com/deepseek-ai/DeepEP.git
commit: 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
```

The local source has this uncommitted build patch:

```diff
- include_dirs = ['csrc/']
+ cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
+ include_dirs = ['csrc/', f'{cuda_home}/include/cccl']
```

AWS build sketch:

```bash
git clone https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP-v1.2.1
git -C /tmp/DeepEP-v1.2.1 checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
# apply the CUDA_HOME/include/cccl patch above unless upstream already has it

export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=10.0

uv pip install --no-build-isolation /tmp/DeepEP-v1.2.1
```

Make sure `nvidia-nvshmem-cu13` is installed before building DeepEP. Otherwise
internode/low-latency DeepEP features can be disabled at build time.

## DeepGEMM

Live install source:

```text
deep-gemm @ file:///tmp/DeepGEMM.install
remote: https://github.com/deepseek-ai/DeepGEMM.git
commit: 714dd1a4a980f7937a74343d19a8eba4fe321480
```

AWS build sketch:

```bash
git clone https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM.install
git -C /tmp/DeepGEMM.install checkout 714dd1a4a980f7937a74343d19a8eba4fe321480

export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=10.0
export DG_JIT_CACHE_DIR="$HOME/.cache/deep_gemm/deepseek_v32_reap_sft"

uv pip install --no-build-isolation /tmp/DeepGEMM.install
```

DeepGEMM was not the current active HISA selector backend in the launcher
(`MEGATRON_HISA_SELECTOR_BACKEND=bmm`), but the package exists in the live env
and may be useful later.

## Recommended AWS Bootstrap Flow

Until `pyproject.toml` and `uv.lock` are made fully authoritative:

```bash
uv sync --locked --extra dev --extra mlm

# Then install the current live-only packages:
handoff/aws-b200-32gpu/scripts/bootstrap_live_env_after_uv_sync.sh

uv run --no-sync python handoff/aws-b200-32gpu/scripts/verify_env.py --expected-gpus 8
handoff/aws-b200-32gpu/scripts/prebuild_custom_extensions.sh
```

Longer-term better fix:

- add portable DeepEP/DeepGEMM sources to `pyproject.toml`
- update `uv.lock`
- replace the local torchcomms wheel path with a portable source
- make `uv sync --locked --extra dev --extra mlm --check` pass on a clean node

## Custom CUDA Extensions

The launcher prebuilds:

- HISA extension:
  `megatron.core.extensions.hisa_indexer.kernels.build`
- HIGGS extension:
  `megatron.core.quantization.higgs.kernels.build`
- IndexCache extension:
  `megatron.core.quantization.indexcache.kernels.build`
- TurboQuant extension if `TURBOQUANT=1`

HISA build uses CUDA 13 from the `.venv` if present and hard-codes
`sm_100` gencode. It also uses nvidia MathDx/CUTLASS include paths from
the `.venv`.

Run `scripts/prebuild_custom_extensions.sh` on each node before a real
multi-node run to avoid one rank blocking others while JIT compiling.
