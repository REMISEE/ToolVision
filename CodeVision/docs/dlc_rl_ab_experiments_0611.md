# DLC RL A/B Experiments, 2026-06-11

## Goal

Diagnose why RL rollout starts with low format reward while eval with the same SFT model and prompt reaches near-perfect format.

The current hypothesis is serving-side generation corruption under the RL vLLM load: malformed tool-call JSON, repeated answer tags, and occasional drift into unrelated task/schema text. These experiments separate data/OOD effects from vLLM prefix-cache effects.

## Common Baseline

Both experiments use the same current RL baseline:

- Model: `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03`
- System prompt: `recipe/codevision/config/sp3.txt`
- Tool config: `recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml`
- Rollout temperature: `0.7`
- Rollout top-p: `0.95`
- Per-turn response cap: `2048`
- Responses per prompt: `8`
- Train batch size: `64`
- Max turns: `12`
- Save frequency: `50`
- Reward mode: `rnec_with_clean`
- Invalid tool-call reward weight: `0.02`

## Code Changes Used

- `recipe/codevision/qwen3_vl_gspo_direct.sh`
  - Adds Hydra passthrough for `actor_rollout_ref.rollout.enable_prefix_caching`.
  - Adds Hydra passthrough for `actor_rollout_ref.rollout.max_num_seqs`.

- `scripts/submit_dlc_gspo_direct_full.sh`
  - Passes `ROLLOUT_ENABLE_PREFIX_CACHING` and `MAX_NUM_SEQS` into DLC jobs.

- `scripts/submit_dlc_gspo_expA_evalbench_full.sh`
  - Experiment A submit wrapper.

- `scripts/submit_dlc_gspo_expB_26k_nopc_full.sh`
  - Experiment B submit wrapper.

- `scripts/submit_dlc_gspo_expC_26k_prefix_mns16_full.sh`
  - Experiment C submit wrapper.

## Experiment A: Data Control

Purpose: test whether low RL format is caused by the screened 26k RL data being OOD.

Only changed variable:

- `TRAIN_FILES` is replaced with eval-style benchmark parquets.

Kept unchanged:

- vLLM prefix caching remains enabled.
- `MAX_NUM_SEQS=1024`.
- Same model, prompt, tool config, reward mode, temperature, top-p, batch size, and n=8.

Train files:

- `/mnt/cpfs/delinmao/Benchmarks/CountQA/countqa_codevision_eval.parquet`
- `/mnt/cpfs/delinmao/Benchmarks/FSC147/fsc147_val_codevision_eval.parquet`
- `/mnt/cpfs/delinmao/Benchmarks/FSC147/fsc147_test_codevision_eval.parquet`
- `/mnt/cpfs/delinmao/Benchmarks/HR-Bench/hr_bench_4k_codevision_eval.parquet`
- `/mnt/cpfs/delinmao/Benchmarks/HR-Bench/hr_bench_8k_codevision_eval.parquet`
- `/mnt/cpfs/delinmao/Benchmarks/vstar-bench/vstar_codevision_eval.parquet`

Run name:

- Job: `codevision_gspo_expA_evalbench_full_0611`
- Experiment: `qwen3vl8b_gspo_expA_evalbench_full_0611`

## Experiment B: Infra Control

Purpose: test whether low RL format is caused by vLLM prefix-cache serving under high-concurrency shared-prefix rollout.

Only changed variable:

- `ROLLOUT_ENABLE_PREFIX_CACHING=False`

Kept unchanged:

- Same screened 26k train data.
- `MAX_NUM_SEQS=1024`.
- Same model, prompt, tool config, reward mode, temperature, top-p, batch size, and n=8.

Train file:

- `/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet`

Run name:

- Job: `codevision_gspo_expB_26k_nopc_full_0611`
- Experiment: `qwen3vl8b_gspo_expB_26k_nopc_full_0611`

## Experiment C: Scheduler-Pressure Control

Purpose: test whether low RL format is caused by high `max_num_seqs` scheduler pressure rather than prefix caching itself.

Only changed variable:

- `MAX_NUM_SEQS=16`

Kept unchanged:

- Same screened 26k train data.
- `ROLLOUT_ENABLE_PREFIX_CACHING=True`
- Same model, prompt, tool config, reward mode, temperature, top-p, batch size, and n=8.

Run name:

- Job: `codevision_gspo_expC_26k_prefix_mns16_full_0611`
- Experiment: `qwen3vl8b_gspo_expC_26k_prefix_mns16_full_0611`

## Submit Prerequisites

Run from the DSW that can reach the tool services.

Do not paste real keys into this document. Export them in the shell before submitting:

```bash
export WANDB_API_KEY='<wandb_api_key>'
export LLM_JUDGE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_JUDGE_MODEL_NAME='qwen3.6-plus'
export LLM_JUDGE_API_KEY='<dashscope_api_key>'
export LLM_JUDGE_ENABLE_THINKING=0
```

Set tool URLs from the current tool-hosting DSW:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
scripts/dsw_tool_urls.sh
```

If submitting from a different DSW than the tool host, set:

```bash
export DSW_TOOL_HOST='<tool_dsw_ip>'
scripts/dsw_tool_urls.sh
```

The submit scripts will run a tool-port preflight unless `SKIP_TOOL_PORT_CHECK=1` is set.

## Dry Run

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
DRY_RUN=1 bash scripts/submit_dlc_gspo_expA_evalbench_full.sh
DRY_RUN=1 bash scripts/submit_dlc_gspo_expB_26k_nopc_full.sh
DRY_RUN=1 bash scripts/submit_dlc_gspo_expC_26k_prefix_mns16_full.sh
```

Check that the dry-run output shows:

- A: `ROLLOUT_ENABLE_PREFIX_CACHING=True`, `MAX_NUM_SEQS=1024`, and the six benchmark train files.
- B: `ROLLOUT_ENABLE_PREFIX_CACHING=False`, `MAX_NUM_SEQS=1024`, and the screened 26k train file.
- C: `ROLLOUT_ENABLE_PREFIX_CACHING=True`, `MAX_NUM_SEQS=16`, and the screened 26k train file.

## Submit

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
bash scripts/submit_dlc_gspo_expA_evalbench_full.sh
bash scripts/submit_dlc_gspo_expB_26k_nopc_full.sh
bash scripts/submit_dlc_gspo_expC_26k_prefix_mns16_full.sh
```

Both are full runs. Stop them manually after the first useful rollout/checkpoint if the format diagnosis is already clear.

## First-Step Inspection

After rollout starts, inspect:

```bash
ls saves/ToolVisionRL/qwen3vl8b_gspo_expA_evalbench_full_0611/rollout_generations
ls saves/ToolVisionRL/qwen3vl8b_gspo_expB_26k_nopc_full_0611/rollout_generations
ls saves/ToolVisionRL/qwen3vl8b_gspo_expC_26k_prefix_mns16_full_0611/rollout_generations
```

The first-step file is expected around:

```bash
saves/ToolVisionRL/<EXP_NAME>/rollout_generations/1.jsonl
```

Key signals:

- Format reward should recover toward eval-like behavior if the tested variable is the cause.
- Malformed tool-call JSON should drop sharply if prefix caching is the cause.
- Repeated `<answer>` tags or leaked schema/task text should disappear if serving corruption is fixed.

## Decision Matrix

- A bad, B good: infra/vLLM prefix-cache path is the likely cause. Continue from B or upgrade vLLM before restoring prefix caching.
- A good, B bad: data/OOD is the likely cause. Build RL data closer to eval/MUT distributions before full training.
- B bad, C good: high `max_num_seqs` scheduler pressure is the likely cause. Use a smaller value, then tune upward for throughput.
- B good, C bad: prefix caching itself is the likely cause. Keep prefix caching disabled or upgrade vLLM before restoring it.
- A bad, B bad, C bad: run Experiment D with prefix caching disabled and `MAX_NUM_SEQS=16`, then consider vLLM upgrade.
- A good, B good: both data and prefix cache contributed; use B-style infra and review 26k source mix.

## Optional Experiment D

If A/B/C still have low format, combine the two infra mitigations:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
JOB_NAME=codevision_gspo_expD_26k_nopc_mns16_0611 \
EXP_NAME=qwen3vl8b_gspo_expD_26k_nopc_mns16_0611 \
ROLLOUT_ENABLE_PREFIX_CACHING=False \
MAX_NUM_SEQS=16 \
bash scripts/submit_dlc_gspo_expB_26k_nopc_full.sh
```

## Max-Num-Seqs Speed Probes

After Experiment C showed `MAX_NUM_SEQS=16` fixes format but is slow, use short 20-step probes to find a faster safe upper bound.

Both probes use:

- screened 26k data
- prefix caching enabled
- `TOTAL_TRAINING_STEPS=20`
- `SAVE_FREQ=-1`
- `LOG_TRAIN_FREQ=2`

Dry run:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
DRY_RUN=1 bash scripts/submit_dlc_gspo_probe_mns64_20step.sh
DRY_RUN=1 bash scripts/submit_dlc_gspo_probe_mns256_20step.sh
```

Submit:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
bash scripts/submit_dlc_gspo_probe_mns64_20step.sh
bash scripts/submit_dlc_gspo_probe_mns256_20step.sh
```

Decision:

- 256 good: use 256 for the next full run.
- 256 bad, 64 good: use 64.
- both bad: stay at 16 or test 32/48.
