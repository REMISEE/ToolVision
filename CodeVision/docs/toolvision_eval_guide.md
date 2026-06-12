# ToolVision Eval Notes

Updated: 2026-04-23

## What changed today

- Added a 2-model-GPU eval wrapper:
  - `recipe/codevision/eval_vstar_tools_a100_2gpu.sh`
- Added a 3-GPU Slurm entrypoint:
  - `scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch`
  - GPU 0-1: model inference
  - GPU 2: external tools
- Switched tool config default to the SFT-aligned v02 config:
  - `recipe/codevision/config/code_image_tool_config_v02.yaml`
- Made eval data configurable with `EVAL_PARQUET`.
- Added benchmark adapter:
  - `recipe/codevision/prepare_benchmarks.py`
- Added DashScope/OpenAI-compatible LLM judge env support:
  - `LLM_JUDGE_BASE_URL`
  - `LLM_JUDGE_MODEL_NAME`
  - `LLM_JUDGE_API_KEY`
  - `LLM_JUDGE_TRUST_ENV`

The result-sensitive eval parameters are unchanged:

```bash
max_prompt_len=$((1024 * 16))
max_resp_len=$((1024 * 16))
max_tool_resp_len=$((1024 * 10))
max_image_resolution=$((1024 * 8 * 28 * 28))
N_RESP_PER_PROMPT=8
MAX_TURNS=12
```

## Available eval parquet files

```bash
/mnt/users/maodelin-20251119/Benchmarks/vstar-bench/vstar_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/ChartQA/chartqa_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/OCRBench/ocrbench_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/countbench/countbench_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/HR-Bench/hr_bench_4k_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/HR-Bench/hr_bench_8k_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/FSC147/fsc147_val_codevision_eval.parquet
/mnt/users/maodelin-20251119/Benchmarks/FSC147/fsc147_test_codevision_eval.parquet
```

Current counts:

```text
V*Bench:    191
ChartQA:   2500
OCRBench:  1000
countbench: 510
HRBench4K: 800
HRBench8K: 800
FSC147 val:  1286
FSC147 test: 1190
```

Notes:

- `countbench` original local parquet has 540 rows.
- 491 rows had embedded images.
- 19 missing images were recovered from `image_url`.
- 30 URLs are still unavailable or blocked; failures are listed in:

```bash
/mnt/users/maodelin-20251119/Benchmarks/countbench/countbench_failed_downloads.tsv
```

## Prepare benchmark adapters

Run this only when source benchmark files change or missing remote images need retrying:

```bash
cd /mnt/users/maodelin-20251119/ToolVision/CodeVision
python3 recipe/codevision/prepare_benchmarks.py --datasets chartqa ocrbench countbench hrbench
```

For countbench without remote download:

```bash
python3 recipe/codevision/prepare_benchmarks.py --datasets countbench --no-download-missing-images
```

## LLM judge setup

DashScope OpenAI-compatible settings:

```bash
export LLM_JUDGE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_JUDGE_MODEL_NAME='qwen3.6-plus'
export LLM_JUDGE_TRUST_ENV=1
read -s -p "DashScope API key: " LLM_JUDGE_API_KEY; echo
export LLM_JUDGE_API_KEY
```

Check that the key is set without printing it:

```bash
test -n "$LLM_JUDGE_API_KEY" && echo "key is set" || echo "key is missing"
```

Quick judge test:

```bash
python3 -c 'from verl.utils.reward_score.llmjudge import LLMJudgeClient; import os,time; c=LLMJudgeClient(base_url=os.environ["LLM_JUDGE_BASE_URL"], model_name=os.environ["LLM_JUDGE_MODEL_NAME"], api_key=os.environ["LLM_JUDGE_API_KEY"], timeout=120, max_retries=1); t=time.time(); s,r=c.verify(answer="14", ground_truth="14", question="How many food items are shown?", task="chartqa"); print("score =", s, "dt =", round(time.time()-t, 2), "reason =", r)'
```

Judge behavior:

- Rule reward runs first.
- LLM judge is called only when rule reward mismatches.
- V*Bench/ChartQA/countbench usually rely mostly on rule reward.
- OCRBench may call judge often, so it can be much slower.

## Submit V*Bench eval

Without LLM judge:

```bash
cd /mnt/users/maodelin-20251119/ToolVision/CodeVision

EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/vstar-bench/vstar_codevision_eval.parquet \
EXP_NAME=vstar_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

With LLM judge:

```bash
cd /mnt/users/maodelin-20251119/ToolVision/CodeVision

export LLM_JUDGE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_JUDGE_MODEL_NAME='qwen3.6-plus'
export LLM_JUDGE_TRUST_ENV=1
read -s -p "DashScope API key: " LLM_JUDGE_API_KEY; echo
export LLM_JUDGE_API_KEY

EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/vstar-bench/vstar_codevision_eval.parquet \
EXP_NAME=vstar_tools_a100_2gpu_judge \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

Smoke run:

```bash
sbatch -t 01:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

## Submit other benchmarks

ChartQA:

```bash
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/ChartQA/chartqa_codevision_eval.parquet \
EXP_NAME=chartqa_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

OCRBench:

```bash
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/OCRBench/ocrbench_codevision_eval.parquet \
EXP_NAME=ocrbench_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

countbench:

```bash
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/countbench/countbench_codevision_eval.parquet \
EXP_NAME=countbench_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

HRBench4K:

```bash
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/HR-Bench/hr_bench_4k_codevision_eval.parquet \
EXP_NAME=hrbench4k_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

HRBench8K:

```bash
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/HR-Bench/hr_bench_8k_codevision_eval.parquet \
EXP_NAME=hrbench8k_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch
```

FSC147 val/test official MAE/RMSE:

```bash
N_RESP_PER_PROMPT=1 \
EVAL_PARQUET=/mnt/users/maodelin-20251119/Benchmarks/FSC147/fsc147_test_codevision_eval.parquet \
EXP_NAME=fsc147_test_tools_a100_2gpu \
sbatch -t 04:00:00 scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch

python recipe/codevision/tools/aggregate_fsc147_metrics.py \
  saves/CodeVision/fsc147_test_tools_a100_2gpu/metrics.json --splits test
```

Use `N_RESP_PER_PROMPT=1` for official FSC147-style single-prediction MAE/RMSE. Higher `N_RESP_PER_PROMPT`
is useful for exploration but changes the aggregation semantics.

## Logs and outputs

Slurm logs:

```bash
/mnt/users/maodelin-20251119/logs/
```

Eval run outputs:

```bash
/mnt/users/maodelin-20251119/ToolVision/CodeVision/outputs/slurm_vstar_tools_eval_3gpu/<job_id>/
```

The eval script prints:

- `EVAL_PARQUET`
- model path
- model/tool GPU split
- tool service URLs
- whether LLM judge API key is set
