# ArxivQA Holdout Eval Notes - 2026-06-22

## Dataset

Default eval parquet:

```bash
/mnt/cpfs/delinmao/Benchmarks/ArxivQA/arxivqa_codevision_eval.parquet
```

This is a local 2,000-example holdout sampled from:

```bash
/mnt/cpfs/delinmao/data/raw/arxivqa/arxivqa.jsonl
/mnt/cpfs/delinmao/data/raw/arxivqa/images
```

It excludes known ArxivQA raw indices from:

- final_v3 pass16 all-valid pool
- mut_v1 and mut_v2 train parquet
- old 40k/26k RL parquet variants
- existing pass16 full ArxivQA shards

The previous train-overlap eval file was moved aside:

```bash
/mnt/cpfs/delinmao/Benchmarks/ArxivQA/arxivqa_final_v3_train_overlap_codevision_eval.parquet
```

## Validation

Validation output:

```bash
/mnt/cpfs/delinmao/Benchmarks/ArxivQA/arxivqa_holdout2000_validation.json
/mnt/cpfs/delinmao/Benchmarks/ArxivQA/arxivqa_holdout2000_summary.json
```

Current checks:

- rows: 2,000
- data_source: arxivqa
- prompt/images/reward_model/extra_info/ground_truth nulls: 0
- unique raw indices: 2,000
- answer counts: A=433, B=634, C=609, D=324
- overlap with checked train/pass16 pools: 0

Scorer route:

- `extra_info.source_dataset=arxivqa`
- `reward_model.ground_truth` is A/B/C/D
- `recipe/codevision/rewards/router.py` maps arxivqa to `multiple_choice`

## Eval Command: V04 SFT

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

MODEL_PATH=/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04 \
JOB_NAME_PREFIX=cv-v04-arxivqa-holdout \
EXP_PREFIX=v04_arxivqa_holdout \
GROUP1_DATASETS=arxivqa \
GROUP2_DATASETS= \
TOOL_DLC_HOST=172.17.1.140 \
TOOL_DLC_BASE_PORT=18150 \
ENABLE_LLM_JUDGE=0 \
SAVE_EVAL_METADATA=1 \
SAVE_VAL_GENERATIONS=1 \
PRIORITY=8 \
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

## Eval Command: MUTV1 128bs Step 60

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

MODEL_PATH=/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03 \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/mutv1_128bs_0618/global_step_60 \
JOB_NAME_PREFIX=cv-mutv1-128bs-s60-arxivqa \
EXP_PREFIX=mutv1_128bs_s60_arxivqa \
GROUP1_DATASETS=arxivqa \
GROUP2_DATASETS= \
TOOL_DLC_HOST=172.17.1.140 \
TOOL_DLC_BASE_PORT=18150 \
ENABLE_LLM_JUDGE=0 \
SAVE_EVAL_METADATA=1 \
SAVE_VAL_GENERATIONS=1 \
PRIORITY=8 \
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

## Eval Command: MUTV1 128bs Step 140

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

MODEL_PATH=/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03 \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/mutv1_128bs_0618/global_step_140 \
JOB_NAME_PREFIX=cv-mutv1-128bs-s140-arxivqa \
EXP_PREFIX=mutv1_128bs_s140_arxivqa \
GROUP1_DATASETS=arxivqa \
GROUP2_DATASETS= \
TOOL_DLC_HOST=172.17.1.140 \
TOOL_DLC_BASE_PORT=18150 \
ENABLE_LLM_JUDGE=0 \
SAVE_EVAL_METADATA=1 \
SAVE_VAL_GENERATIONS=1 \
PRIORITY=8 \
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

## Notes

- `TOOL_DLC_BASE_PORT=18150` uses replica 7: ports 18150-18153.
- Set `SKIP_TOOL_PORT_CHECK=1` only if the submit DSW cannot reach tool DLC ports but DLC workers can.
- `ENABLE_LLM_JUDGE=0` means no offline SFT Qwen/OpenAI key is required for this ArxivQA MC eval.
