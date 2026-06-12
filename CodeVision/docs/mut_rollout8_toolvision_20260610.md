# ToolVision MUT Rollout8 Runbook

## Inputs

Pass16 final pool:

`/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/all_valid_pass16_v3.parquet`

ToolVision eval parquets:

- MUT candidates, no-tool pass16 `0-8/16`:
  `/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_candidates_0_8_toolvision_eval.parquet`
- Non-MUT hard regular, no-tool pass16 `9-15/16`:
  `/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/regular_9_15_toolvision_eval.parquet`
- Per-source shards:
  `/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/by_source/{mut_0_8,regular_9_15}/*.parquet`
- Smoke:
  `/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/smoke_mut_0_8_128_toolvision_eval.parquet`

Counts:

- `0-8/16` MUT candidates: 51,773
- `9-15/16` regular hard: 13,151

## Meaning

- `0-8/16`: hard under no-tool lmms rollout. Run ToolVision agent rollout8. If any of 8 tool trajectories is correct after rescoring, mark as MUT-positive.
- `9-15/16`: no MUT label. Keep as regular hard RL data candidate.
- `16/16`: easy, not included in this ToolVision rollout stage.

## Environment

Use the same ToolVision eval stack as the current-prompt eval:

- conda env: `cvtool`
- model: `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03`
- system prompt: `recipe/codevision/config/sp3.txt`
- tool config: `recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml`
- rollout8 sampling: `temperature=0.7`, `top_p=0.95`

The wrapper is:

`recipe/codevision/run_toolvision_mut_rollout8.sh`

It calls the existing `recipe/codevision/eval_vstar_tools_a100_4gpu.sh`.

## Smoke

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/d/conda/envs/cvtool

nohup bash -lc '
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EVAL_PARQUET=/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/smoke_mut_0_8_128_toolvision_eval.parquet
export EXP_NAME=mut_rollout8_smoke_128_t0p7
export VAL_N_RESP_PER_PROMPT=8
export VAL_TEMPERATURE=0.7
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export NGPUS_PER_NODE=8
export INFER_TP_SIZE=4
export SAVE_EVAL_METADATA=1
export SAVE_VAL_GENERATIONS=1
export SAVE_FULL_TRAJECTORY_ALL=1
bash recipe/codevision/run_toolvision_mut_rollout8.sh
' > /mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_rollout8_smoke_128_t0p7.log 2>&1 &
```

Check:

```bash
tail -f /mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_rollout8_smoke_128_t0p7.log
ls -lh /mnt/cpfs/delinmao/ToolVision/CodeVision/saves/CodeVision/mut_rollout8_smoke_128_t0p7/
```

## Full MUT Candidate Rollout

Full `0-8/16` is 51,773 prompts and 414,184 tool-agent trajectories. Prefer source shards if storage or wall time is a concern.

Single full job:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/d/conda/envs/cvtool

nohup bash -lc '
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EVAL_PARQUET=/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_candidates_0_8_toolvision_eval.parquet
export EXP_NAME=mut_rollout8_all_0_8_t0p7
export VAL_N_RESP_PER_PROMPT=8
export VAL_TEMPERATURE=0.7
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export NGPUS_PER_NODE=8
export INFER_TP_SIZE=4
export SAVE_EVAL_METADATA=1
export SAVE_VAL_GENERATIONS=1
export SAVE_FULL_TRAJECTORY_ALL=1
bash recipe/codevision/run_toolvision_mut_rollout8.sh
' > /mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_rollout8_all_0_8_t0p7.log 2>&1 &
```

Example source shard:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/d/conda/envs/cvtool

SOURCE=chartqa
nohup bash -lc "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EVAL_PARQUET=/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/by_source/mut_0_8/${SOURCE}.parquet
export EXP_NAME=mut_rollout8_${SOURCE}_0_8_t0p7
export VAL_N_RESP_PER_PROMPT=8
export VAL_TEMPERATURE=0.7
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export NGPUS_PER_NODE=8
export INFER_TP_SIZE=4
export SAVE_EVAL_METADATA=1
export SAVE_VAL_GENERATIONS=1
export SAVE_FULL_TRAJECTORY_ALL=1
bash recipe/codevision/run_toolvision_mut_rollout8.sh
" > /mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/mut_rollout8_${SOURCE}_0_8_t0p7.log 2>&1 &
```

## Monitoring

Log:

`/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/*.log`

Eval output:

`/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/CodeVision/$EXP_NAME/`

Useful files:

- `metrics.json`
- `diagnostics/metadata.jsonl`
- `diagnostics/sampled_traces.jsonl`
- `generations/` when `SAVE_VAL_GENERATIONS=1`

Do not use `metadata.jsonl:is_correct` directly as the final MUT label. The eval reward can include format/tool components. Postprocess final answers with the ToolVision scorer and then mark MUT-positive when agent rollout8 has at least one correct trajectory.
