# SFT v04 Rotate Cleanup Runbook

Goal: build `sft-mix200-simple-notool-sp3-v04` with much less orientation prior than v03, while keeping the RL-time prompt/tool setup aligned with `sp3.txt` and `code_image_tool_config_v03_sftclean.yaml`.

## Data Recipe

- Start from v03: `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/data/codevision_sft_mix200_simple_notool_sp3_v03.json`.
- Drop old CodeVision rows where `metadata.source_dataset` is missing.
- Drop retained v03 rows that already call rotate/flip, so orientation examples are controlled by the new quotas.
- Add from original CodeVision-SFT base:
  - rotate: 100
  - flip: 50
  - crop: 1000
- Rewrite every row system to `sp3.txt` + `code_image_tool_config_v03_sftclean.yaml`.

Important: `/mnt/cpfs/delinmao/ToolVision/codevision_sft.json` is not image-aligned with the current LLaMA-Factory `data/codevision_images` directory. The builder requires `--base-image-root` pointing to the real original CodeVision-SFT dataset root that contains `codevision_images/`.

Current downloaded base dataset:

```text
/mnt/cpfs/delinmao/data/CodeVision-SFT/codevision_sft.json
/mnt/cpfs/delinmao/data/CodeVision-SFT/codevision_images/
/mnt/cpfs/delinmao/data/CodeVision-SFT/codevision-images.zip
```

## Build Dataset

Plan/check only:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
python scripts/build_sft_v04_dataset.py --plan-only --allow-missing-base-images
```

Materialize once the original base image root is available:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
python scripts/build_sft_v04_dataset.py \
  --base-input /mnt/cpfs/delinmao/data/CodeVision-SFT/codevision_sft.json \
  --base-image-root /mnt/cpfs/delinmao/data/CodeVision-SFT \
  --rotate-quota 100 \
  --flip-quota 50 \
  --crop-quota 1000
```

Output:

```text
/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/data/codevision_sft_mix200_simple_notool_sp3_v04.json
/mnt/cpfs/delinmao/ToolVision/CodeVision/outputs/analysis/sft_v04_dataset_stats.json
```

## Train

DLC dry run:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
DRY_RUN=1 bash scripts/submit_dlc_sft_v04.sh
```

DLC submit:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
bash scripts/submit_dlc_sft_v04.sh
```

SFT defaults to `report_to: none`. For W&B:

```bash
read -s -p "W&B API key: " WANDB_API_KEY; echo
export WANDB_API_KEY
ENABLE_WANDB=1 bash scripts/submit_dlc_sft_v04.sh
```

Expected model:

```text
/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04
```

## Eval

Quick local/DSW check on ChartQA + OCRBench, pinned to the RL-time prompt/tool setup:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
export LLM_JUDGE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_JUDGE_MODEL_NAME='qwen3.6-plus'
read -s -p "DashScope API key: " LLM_JUDGE_API_KEY; echo
export LLM_JUDGE_API_KEY

bash scripts/run_mix200_v04_chartqa_ocrbench_llmjudge_nohup.sh
```

DLC full eval for the default 8 historical benches. This uses the deployed DLC
tool-service replica 3 at `172.17.0.142:18110-18113` and submits two parallel
4-GPU eval jobs:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

Dry run:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
DRY_RUN=1 bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

Default DLC eval:

```text
MODEL_PATH=/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04
SYSTEM_PROMPT_PATH=recipe/codevision/config/sp3.txt
TOOL_CFG_TEMPLATE_PATH=recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml
BENCHMARKS=vstar,chartqa,ocrbench,countbench,hrbench4k,hrbench8k,fsc147_val,fsc147_test
EXP_PREFIX=mix200_sft_sp3_v04
CODEVISION_ENV=/mnt/cpfs/delinmao/envs/codevision_new
WORKER_GPU=4 per job
TEMPERATURES=0
N_RESP_PER_PROMPT=1
VAL_N_RESP_PER_PROMPT=1
ROLLOUT_AGENT_NUM_WORKERS=8
MAX_NUM_SEQS=16
```

The split script uses `/mnt/cpfs/delinmao/bin/dlc_pai` by default to avoid the
`AccessKeyId is mandatory` path from the raw DSW `dlc` binary. It does not start
tools inside the eval worker; both jobs call:

```text
OCR_BASE_URL=http://172.17.0.142:18110
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18111
DEPTH_BASE_URL=http://172.17.0.142:18112
COUNTGD_BASE_URL=http://172.17.0.142:18113
```

To mirror the expanded full-eval queue used in some v03 runs, override the benchmark list:

```bash
GROUP1_DATASETS='vstar chartqa ocrbench countbench mvtoolbench cvbench pixmo_count_lmms' \
GROUP2_DATASETS='hrbench4k hrbench8k fsc147_val fsc147_test countqa spatialmqa ocrbench_v2' \
  bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

For full trace debugging of tool overuse, submit with:

```bash
SAVE_FULL_TRAJECTORY_ALL=1 SAVE_VAL_GENERATIONS=1 bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
```

For the first comparison, check both accuracy and operation rates on ChartQA/OCRBench:

- ChartQA overall accuracy.
- OCRBench accuracy.
- `rotate` call rate on chart/text samples.
- `crop` and OCR call rate.
- Sampled traces where rotate was used and final answer was wrong.
