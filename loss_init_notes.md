# Loss Initialization Notes

Active run under investigation: job 285 (`corsaire-1-mix-lr10x-ckptfix`).

## Live Run Snapshot

- Shape: TP8 / CP4 / PP1 / EP8 / DP3, world size 96.
- Batch: MBS 1, GBS 12, 32k context.
- LR: constant `5e-5`.
- Checkpoint load: `/home/sjpat/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8`.
- Data: `/home/sjpat/data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full-plus-trajectories.jsonl`.
- Tokenizer: `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4`, prompt format `deepseek-v3.2`.
- Padded vocab: `129280`; uniform random CE baseline is `ln(129280) ~= 11.77`.

## Current Evidence

- Job 284:
  - iter 1: lm_loss `14.39935`, grad_norm `4242.416`
  - iter 2: lm_loss `15.33984`, grad_norm `7142.010`
  - iter 3: lm_loss `13.14731`, grad_norm `1232.449`
  - iter 4: lm_loss `13.88242`, grad_norm `1338.481`
  - iter 5: lm_loss `12.88985`, grad_norm `276.871`; checkpoint save started at `2026-05-25 19:41:06 UTC`
- Job 284 was stopped because checkpoint save stayed in
  `preprocess_state_dict_for_uneven_dtensor -> all_gather_object` for several
  minutes and emitted no checkpoint files. The code now skips the custom uneven
  metadata collective for ordinary evenly sharded DTensors and keeps it for
  save-time SWiGLU W/V split tensors or truly uneven shard shapes.
- Relaunched as job 285 with the same shape/env and patched checkpoint code.
- Loss 14-15 is worse than uniform over the active padded vocab. It should not be called "random"; it means the checkpoint assigns very low probability to current supervised targets.
- Historical fresh-start runs also began around 14-15, including LR=0/no-update diagnostics, so the cold high loss is not solely caused by optimizer updates.
- No current run has been launched from iter45. Jobs 284 and 285 are intentionally fresh base-checkpoint runs over the new mixed data.
- The active base checkpoint metadata contains `embedding.word_embeddings.weight` and `output_layer.weight`, both shaped `[129280, 7168]`, so the obvious "missing LM head" failure is not supported by metadata.
- The active run does load with checkpoint args missing and the FSDP adapter uses `strict=False` due to fp8 configuration. This remains a risk because missing runtime-key diagnostics may be hidden, even though the checkpoint metadata has the expected embedding/head tensors.

## Data Autopsy: First 96 Shuffled Rows

Command output saved under `logs/sft-data-autopsy/job284-init-loss/`.

- 84 structured-message rows were analyzed by the existing autopsy tool. The 12 "errors" are raw rendered trajectory rows that the autopsy tool reports as `normalized messages are not a list`; the training dataset path handles those with `tokenize_rendered_deepseek_text`, so this is an autopsy-tool limitation, not direct evidence of a training failure.
- Mean active supervised labels per row: `10479`; median `9369`; zero-supervision rows: `0`.
- Expected active supervised labels per GBS=12 step: about `125751`.
- CP supervision is imbalanced: `40.5%` of CP chunks have zero active labels in this sample. This affects signal distribution, but does not explain worse-than-uniform cold CE by itself.
- The first 12 shuffled logical rows are dominated by `Nemotron-Terminal-Corpus` with long thinking/terminal JSON traces, plus one search-agent row and one raw trajectory row. This can plausibly make base-checkpoint cold CE much higher than generic chat, especially if the base checkpoint was not trained on this exact rendering/distribution.
- A wider 8192-row shuffled sample was still dominated by terminal traces:
  `6780` Nemotron terminal rows, `1199` raw prod-distillation trajectories,
  `138` search rows, and `75` tool-calling rows.

## Investigation Threads

- Verify the active tokenizer vocab and special-token IDs match the checkpoint.
- Inspect actual first shuffled rows for row type, role mix, rendered format, and supervised span density.
- Confirm `labels = targets[1:]` aligns with `tokens[:-1]` for both structured conversations and raw rendered text.
- Check whether supervised spans are dominated by text formats the base checkpoint likely did not see: tool-call XML/DSML blocks, trajectory traces, or raw rendered content.
