# RL Scoring, Rollout, Trajectory, and Checkpoint Runbook

Date: 2026-06-08

## Current Findings

### 1. RL trajectory visibility

The 2026-06-02 RL run did not keep local full training trajectories.

Evidence from `outputs/2026-06-02/09-56-08/.hydra/config.yaml`:

- `trainer.rollout_data_dir: null`
- `trainer.validation_data_dir: null`
- `trainer.test_freq: -1`
- `trainer.log_train_freq: 10`
- `trainer.log_train_generations: 32`

So that run only had sampled training generations sent to configured loggers every 10 steps. It did not dump every rollout to local JSONL, and it did not run validation during training.

This is not limited by code to every 20 steps. There are three separate knobs:

- `LOG_TRAIN_FREQ`: sampled train examples to console/wandb. Old default was 20; 2026-06-02 used 10.
- `ROLLOUT_DATA_DIR`: local JSONL dump for train rollouts. If set, the trainer writes one `{step}.jsonl` per step.
- `TEST_FREQ` + `VALIDATION_DATA_DIR`: validation generations. `TEST_FREQ=-1` disables periodic validation.

Change made: `qwen3_vl_gspo_direct.sh` and `qwen3_vl_gspo.sh` now default `ROLLOUT_DATA_DIR=${SAVE_DIR}/rollout_generations`, so future runs keep local step-level rollout JSONL.

### 2. RL checkpoints

The 2026-06-02 run did not leave a usable RL checkpoint under `saves/ToolVisionRL`.

Evidence:

- `saves/ToolVisionRL/*` only contains `tool_config.runtime.yaml`.
- No `global_step_*` checkpoint directories were found under `saves/ToolVisionRL`.
- 2026-06-02 Hydra config had `trainer.save_freq: 400`.

If a run stops at around 100-200 steps and `SAVE_FREQ=400`, no periodic checkpoint is expected. The trainer would save at the final step only if it exits normally through the training loop.

Change made: both RL launchers now default to:

- `SAVE_FREQ=50`
- `MAX_ACTOR_CKPT_TO_KEEP=5`
- `MAX_CRITIC_CKPT_TO_KEEP=5`

This keeps steps 50/100/150/... while limiting disk growth.

### 3. Why SFT eval format is high but RL step-1 format can be low

The most likely cause is decoding mode, not model path.

The SFT eval runs and the 2026-06-02 RL run use the same model path:

`/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03`

But they use different generation regimes:

- SFT eval / validation path: `val_kwargs.temperature=0`, `do_sample=false`, greedy decoding.
- RL train rollout path: `temperature=1.0`, `top_p=1`, `do_sample=true`, `n=8`.

The format reward check is strict: after removing `<tool_response>`, output must start with `<think>`, have balanced `<think></think>`, exactly one final `<answer>...</answer>`, and the answer must be at the end. A checkpoint can score 0.8-0.9 under greedy eval but much lower under sampled RL rollouts if sampled responses omit tags, emit malformed tool calls, or generate extra text after `</answer>`.

Because 2026-06-02 had `rollout_data_dir=null`, we cannot inspect the exact bad step-1 generations. The next run should start with a 1-step diagnostic that dumps local rollouts.

## Implemented Scorer v2 Changes

ToolVision scorer now includes the missing source-specific routes:

- `docvqa` / `infographicvqa`: forced to ANLS-style `ocr_levenshtein`.
- `textvqa`: EvalAI/VQA soft accuracy using repeated annotator answers.
- `gqa`: normalized exact route.
- `fsc147`: raw relative count score.
- `refl4`: raw IoU remains available; downstream primary success is IoU >= 0.5.

The CPU rescore script is:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
PYTHONPATH=. python3 recipe/codevision/tools/rescore_pass16_v2.py \
  --input /path/to/pass16_by_sample.parquet \
  --output /mnt/cpfs/delinmao/data/toolvision_pass16_v2/rescored.parquet \
  --summary /mnt/cpfs/delinmao/data/toolvision_pass16_v2/rescored_by_source.csv
```

Output columns added:

- `score_raw_16`
- `success_strict_16`
- `success_lenient_16`
- `success_primary_16`
- `correct_count_strict`
- `correct_count_lenient`
- `correct_count_primary`
- `mean_score_v2`
- `bucket_v2`
- `metric_family_v2`

First real rescore output has been generated:

- Parquet: `/mnt/cpfs/delinmao/data/toolvision_pass16_v2/pass16_by_sample_v2.parquet`
- Summary: `/mnt/cpfs/delinmao/data/toolvision_pass16_v2/rescore_old_sources_by_source.csv`
- Old-vs-v2 comparison: `/mnt/cpfs/delinmao/data/toolvision_pass16_v2/old_vs_v2_by_source.csv`

Key old-vs-v2 primary zero-rate changes on `toolvision_pass16_full/reports_new_sources/pass16_by_sample.parquet`:

| Source | Old 0/16 | v2 primary 0/16 | Note |
|---|---:|---:|---|
| chartqa | 90.40% | 76.75% | relaxed scorer rescues some numeric/format cases |
| docvqa | 93.58% | 73.89% | ANLS route rescues partial OCR matches |
| infographicvqa | 78.38% | 51.65% | ANLS route has large effect |
| ocrbench | 88.17% | 86.01% | inclusion scorer only slightly changes buckets |
| refl4 | 100.00% | 85.23% | IoU>=0.5 primary rescues some; lenient IoU>0 is much higher |
| pixmo_count | 57.96% | 57.96% | unchanged under numeric exact |

Primary success thresholds:

- `gqa`, MC, exact, ChartQA relaxed, OCRBench inclusion, numeric exact: `score == 1.0`
- `textvqa`: soft VQA `score == 1.0`
- `docvqa` / `infographicvqa`: ANLS `score >= 0.5`
- `fsc147`: relative score `>= 0.9`
- `refl4`: IoU `>= 0.5`

## Recommended Execution Plan

### A. Immediate RL diagnostic, no long training

Run one training step with rollout dumps and checkpoint settings. This answers the format-score question with real local evidence.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

PROJECT_NAME=ToolVisionRL \
EXP_NAME=qwen3vl8b_format_probe_1step \
SAVE_DIR=./saves/ToolVisionRL/qwen3vl8b_format_probe_1step \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=1 \
MAX_ACTOR_CKPT_TO_KEEP=1 \
ROLLOUT_DATA_DIR=./saves/ToolVisionRL/qwen3vl8b_format_probe_1step/rollout_generations \
LOG_TRAIN_FREQ=1 \
LOG_TRAIN_GENERATIONS=32 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
N_RESP_PER_PROMPT=8 \
bash recipe/codevision/qwen3_vl_gspo_direct.sh
```

Inspect:

```bash
head -20 saves/ToolVisionRL/qwen3vl8b_format_probe_1step/rollout_generations/0.jsonl
find saves/ToolVisionRL/qwen3vl8b_format_probe_1step -maxdepth 3 -type d -name 'global_step_*'
```

Pass condition:

- `rollout_generations/0.jsonl` exists.
- `global_step_0/actor` exists because `SAVE_FREQ=1`.
- Bad `format_reward` rows show whether the issue is missing `<think>`, missing `</think>`, missing `<answer>`, multiple answers, or text after answer.

### B. CPU rescore old pass16 data

Use the existing pass16 parquet if present:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
mkdir -p /mnt/cpfs/delinmao/data/toolvision_pass16_v2

PYTHONPATH=. python3 recipe/codevision/tools/rescore_pass16_v2.py \
  --input /mnt/cpfs/delinmao/data/toolvision_pass16_full/reports_new_sources/pass16_by_sample.parquet \
  --output /mnt/cpfs/delinmao/data/toolvision_pass16_v2/pass16_by_sample_v2.parquet \
  --summary /mnt/cpfs/delinmao/data/toolvision_pass16_v2/rescore_old_sources_by_source.csv
```

If there are additional old-source parquets, run the same script per parquet and concatenate summaries.

### C. lmms 06-02 rerun

Use lmms-eval for inference only, then rescore with ToolVision scorer v2.

Locked decisions:

- Model: `/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct`
- GPUs: 8
- Samples: `--repeats 16`
- Prompt first pass: lmms default prompts for same-sample GQA/TextVQA/FSC147.
- Scoring: never use lmms metric as final bucket; convert raw outputs to `pred_texts_json` and run `rescore_pass16_v2.py`.

Gate order:

1. Ask/record colleague setup: model path, prompt, GQA split.
2. Run built-in `gqa_lite` 200-example baseline.
3. Build same-sample external tasks for:
   - GQA 20k
   - TextVQA 10k
   - FSC147 1286
4. Smoke: 20 rows/source, `--repeats 2`; require `len(resps)==2`.
5. Control: 300 rows/source, `--repeats 4`, real/blank/shuffled.
6. Full: all rows/source, `--repeats 16`, 8 GPUs.

Control pass thresholds:

- GQA real primary acc >= blank primary acc + 15 percentage points.
- TextVQA real primary acc >= blank primary acc + 20 percentage points.
- FSC147 real mean relative score >= blank mean relative score + 20 percentage points.

Full-run command shape after tasks exist:

```bash
cd /mnt/cpfs/delinmao/lmms-eval
source /opt/conda/etc/profile.d/conda.sh
conda activate lmms-eval

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=12361 -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct,max_pixels=6422528,attn_implementation=sdpa,interleave_visuals=False \
  --include_path /mnt/cpfs/delinmao/ToolVision/CodeVision/lmms_tasks \
  --tasks tv_pass16_gqa,tv_pass16_textvqa,tv_pass16_fsc147 \
  --batch_size 8 \
  --repeats 16 \
  --gen_kwargs temperature=1.0,top_p=1.0,max_new_tokens=64,num_beams=1 \
  --log_samples \
  --output_path /mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun/full_qwen3vl8b_instruct
```

### D. Next normal RL run

Use the launcher defaults added here, or make them explicit:

```bash
SAVE_FREQ=50 \
MAX_ACTOR_CKPT_TO_KEEP=5 \
ROLLOUT_DATA_DIR=./saves/ToolVisionRL/<exp>/rollout_generations \
LOG_TRAIN_FREQ=10 \
LOG_TRAIN_GENERATIONS=32 \
bash recipe/codevision/qwen3_vl_gspo_direct.sh
```

Do not start another long RL run without `ROLLOUT_DATA_DIR` and `SAVE_FREQ<=50`.
