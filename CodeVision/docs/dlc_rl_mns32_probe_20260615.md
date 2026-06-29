# DLC RL mns32 Probes, 2026-06-15

Purpose: test whether `MAX_NUM_SEQS=32` is rollout-format safe with the DLC
tool service, and compare the current screened 26k RL data against the original
40k RL data that contains the previously removed harder/OOD sources.

Tool service source:

- Runbook: `docs/dlc_tool_services_runbook_20260615.md`
- Tool DLC job: `dlcwv66tm4r5zxyp`
- Tool pod IP: `172.17.2.38`

## Scripts

### 26k data, replica 0

```bash
bash scripts/submit_dlc_gspo_probe_mns32_20step.sh
```

Defaults:

- `EXP_NAME=qwen3vl8b_gspo_probe_mns32_26k_20step_0615`
- `TRAIN_FILES=/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet`
- `TOOL_DLC_REPLICA=0`
- tools: `172.17.2.38:18080-18083`
- `MAX_NUM_SEQS=32`
- `TOTAL_TRAINING_STEPS=20`
- `SAVE_FREQ=-1`

### Original 40k data, replica 1

```bash
bash scripts/submit_dlc_gspo_probe_mns32_40k_20step.sh
```

Defaults:

- `EXP_NAME=qwen3vl8b_gspo_probe_mns32_40k_20step_0615`
- `TRAIN_FILES=/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train.parquet`
- `TOOL_DLC_REPLICA=1`
- tools: `172.17.2.38:18090-18093`
- `MAX_NUM_SEQS=32`
- `TOTAL_TRAINING_STEPS=20`
- `SAVE_FREQ=-1`

Original 40k top sources:

- `virl39k`: 5000
- `chartqa`: 4810
- `pixmo_count`: 4000
- `WaltonFuture`: 3000
- `virgorlsa`: 3000
- `refl4`: 3000
- `arxivqa`: 2500
- `sat2`: 2500
- `tqa`: 2000
- `thinklite_vl_hard`: 2000
- `mmk12`: 1500
- `wemath_standard`: 1500
- `ocrbench`: 1252
- `docvqa`: 1242
- `infographicvqa`: 1000
- `puzzlevqa`: 1000

Scorer alignment:

- All original 40k rows have `extra_info.reward_family`.
- Current router supports all families present in 40k: `judge`,
  `multiple_choice`, `chartqa_relaxed`, `numeric_exact`, `math_verify`,
  `bbox_iou`, `ocr_levenshtein`, `ocr_inclusion`, `exact`, and `boolean`.
- Source-forced current standards still apply. In particular, `pixmo_count`
  is scored as `fsc147_relative` now, even though old rows carry
  `reward_family=numeric_exact`.
- `judge` family rows will call the configured LLM judge when rule exact match
  is insufficient.

## Submit

Export keys first:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

export WANDB_API_KEY='<wandb_api_key>'
export LLM_JUDGE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_JUDGE_MODEL_NAME='qwen3.6-plus'
export LLM_JUDGE_API_KEY='<dashscope_api_key>'
export LLM_JUDGE_ENABLE_THINKING=0
```

Submit both:

```bash
bash scripts/submit_dlc_gspo_probe_mns32_20step.sh
bash scripts/submit_dlc_gspo_probe_mns32_40k_20step.sh
```

## Check

```bash
ls saves/ToolVisionRL/qwen3vl8b_gspo_probe_mns32_26k_20step_0615/rollout_generations
ls saves/ToolVisionRL/qwen3vl8b_gspo_probe_mns32_40k_20step_0615/rollout_generations
```

First-step files:

```text
saves/ToolVisionRL/qwen3vl8b_gspo_probe_mns32_26k_20step_0615/rollout_generations/1.jsonl
saves/ToolVisionRL/qwen3vl8b_gspo_probe_mns32_40k_20step_0615/rollout_generations/1.jsonl
```

Success signal:

- `format_reward` near `0.95+`
- `invalid_tool_call` near `0`
- `malformed_tool_call_count` near `0`
- no repeated `<answer>` spam or leaked schema/task text

Interpretation:

- 26k good, 40k bad: removed OOD/harder sources still hurt rollout or reward.
- 26k bad, 40k good: 26k mixture/source composition is unexpectedly worse.
- both bad: `MAX_NUM_SEQS=32` is too high; stay at `16` or test lower.
- both good: use `32` as the next speed/quality candidate.
