# RDEP Preparation Scaffold

This branch carries a default-off RDEP preparation scaffold only. It records the
route-row identity and pooled-row accounting needed to prepare a Megatron
runtime integration, but it does not hook into `TransformerBlock`, MoE dispatch,
pipeline schedules, peer-to-peer transport, or GPU kernels.

The scaffold follows the public RDEP contract from Noumena's writeup, paper, and
`Noumena-Network/nmoe` reference implementation:

- Dense replicas inside one NVLink domain share the same sparse expert pool.
- Each selected token-slot becomes a route row with identity
  `((rank * tokens_per_rank) + token) * top_k + slot`.
- Contiguous expert placement maps `expert_id` to
  `(expert_id // num_local_experts, expert_id % num_local_experts)`.
- Pooled route rows raise expected rows per expert to
  `world_size * tokens_per_rank * top_k / num_experts`.
- Routing-context memory is treated as accepted-route metadata, not as a
  sequence-chunk or StreamBP artifact.

```python
from megatron.core.transformer.rdep import (
    RDEPPreparationConfig,
    build_rdep_preparation_plan,
)

plan = build_rdep_preparation_plan(
    tokens_per_rank=4096,
    config=RDEPPreparationConfig(
        enabled=True,
        world_size=8,
        num_experts=64,
        top_k=6,
        num_moe_layers=27,
    ),
)
```

For the public 8xB200 RDEP shape (`world_size=8`, `tokens_per_rank=4096`,
`top_k=6`, `num_experts=64`), this produces `3072` expected rows per expert and
`42,467,328` compact routing-context bytes for 27 MoE layers.

## Local Validation

```bash
python -m pytest tests/unit_tests/transformer/test_rdep.py
python -m py_compile megatron/core/transformer/rdep.py
```

## Production Blockers

- Define the Megatron runtime contract for route-row dispatch, return placement,
  backward replay, failure handling, and optimizer-step ordering.
- Prove parity against a sequential MoE implementation for forward, backward,
  router gradients, expert gradients, and capacity handling.
- Validate interaction with StreamBP, MoE routing, sequence parallelism,
  pipeline parallelism, DSA, checkpointing, and mixed precision.
- Port or reimplement transport and owner-side kernels with CuTe/CZS where
  custom kernels are required, then validate on the target B200/GB300 hardware.
- Add end-to-end production A/B runs before enabling RDEP in training hot paths.
