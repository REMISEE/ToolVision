# BigBench Eval Runbook 2026-06-24

## 1. CountBench Base Sanity Eval

目的：补跑一组真正的 no-tool direct-answer `Qwen3-VL-8B-Thinking + CountBenchQA-491`，不要只引用 report 的 `91.5`。

注意：不要用 ToolVision agent 框架测 base Thinking 的 CountBench baseline。agent prompt 会包含 tool list / JSON 协议，不能代表官方 direct-answer 能力。之前误跑的 agent 输出已隔离到：

```text
/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/CodeVision/quarantine_wrong_eval_20260624/base_thinking_countbench_agent_8gpu_g1_countbench_t0
```

本轮使用 lmms-eval external task，本地数据和 task：

```text
/mnt/cpfs/delinmao/Benchmarks/countbench/lmms_local/countbench_local.jsonl
/mnt/cpfs/delinmao/ToolVision/CodeVision/lmms_tasks/countbench_local/tv_countbench_local.yaml
/mnt/cpfs/delinmao/ToolVision/CodeVision/lmms_tasks/countbench_local/utils.py
```

提交命令：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

MODEL_PATH=/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking \
MODEL_TAG=qwen3vl8b_thinking \
PRIORITY=8 \
bash scripts/submit_dlc_lmms_countbench_base.sh
```

CountBench scorer 口径：
- 数据：`/mnt/cpfs/delinmao/Benchmarks/countbench/countbench_codevision_eval.parquet`
- 行数：491
- 数据源：`vikhyatk/CountBenchQA`, split `test`
- metric：numeric exact accuracy
- LLM judge：关掉；这组不需要 judge

## 2. Prepare Expanded Benchmarks Locally

先把 parquet 全部物化到 `/mnt/cpfs/delinmao/Benchmarks`：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

bash scripts/prepare_big_eval_benchmarks_20260624.sh
```

如需 smoke：

```bash
bash scripts/prepare_big_eval_benchmarks_20260624.sh --inspect --limit 1 --inspect-limit 1
```

MMVet-Hard 当前仓库没有标准 lmms task；如果后面拿到 hard 数据文件：

```bash
MMVET_HARD_DATA_PATH=/path/to/mmvet_hard.jsonl \
bash scripts/prepare_big_eval_benchmarks_20260624.sh --datasets mmvet_hard
```

## 3. Submit Expanded Big Eval

默认按 4 个工具 replica 分组提交：

| Group | Replica | Port base | Datasets |
|---|---:|---:|---|
| gA | 4 | 18120 | `realworldqa mmstar cvbench spatialmqa` |
| gB | 5 | 18130 | `docvqa_val infovqa_val ocrbench_v2` |
| gC | 6 | 18140 | `mme_realworld_lite mme_realworld mme_realworld_cn` |
| gD | 7 | 18150 | `pixmo_count pixmo_count_lmms countqa mvtoolbench mmvet` |

命令：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

export OFFLINE_SFT_QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export OFFLINE_SFT_QWEN_MODEL='qwen3.6-plus'
export OFFLINE_SFT_QWEN_API_KEY='...'

MODEL_PATH=/path/to/model-or-merged-checkpoint \
MODEL_TAG=my_model_tag \
TOOL_DLC_HOST=172.17.1.140 \
ENABLE_LLM_JUDGE=1 \
PRIORITY=8 \
bash scripts/submit_dlc_bigbench_eval_20260624.sh
```

如只测 base Thinking：

```bash
MODEL_PATH=/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking \
MODEL_TAG=base_thinking \
TOOL_DLC_HOST=172.17.1.140 \
ENABLE_LLM_JUDGE=1 \
PRIORITY=8 \
bash scripts/submit_dlc_bigbench_eval_20260624.sh
```

## 4. Scorer / Aggregation Contract

不要混用训练 reward 总分做 benchmark 分。报告时用：

| Benchmark | Primary scorer | Aggregation |
|---|---|---|
| CountBenchQA-491 | numeric exact | sample mean |
| RealWorldQA | lmms exact / MC extraction | sample mean |
| MME-RealWorld / Lite / CN | A-E MC extraction | sample-weighted overall |
| MMStar | MC extraction | macro average over `l2_category` |
| DocVQA / InfoVQA | ANLS | sample mean |
| OCRBench v2 | lmms task scorer | official task aggregate |
| Pixmo-Count / CountQA | numeric exact plus MAE/relative diagnostics | sample mean for exact; report MAE separately where needed |
| SpatialMQA / CVBench | MC extraction | official dataset aggregate; CVBench uses 2D/3D combined if comparing to report |
| MMVet / MMVet-Hard | LLM judge fallback | mean judge score |

旧 9 个 summary 的固定口径：
- V*：`weighted191 = (direct_attributes * 115 + relative_position * 76) / 191`
- OCRBench：weighted over 1000 samples，不用 macro10
- FSC147：主表 MAE/RMSE；normalized 表可以用 relative score

## 5. Current Local Data Status

已确认本地存在并通过基本 schema 校验：
- CountBenchQA-491: `/mnt/cpfs/delinmao/Benchmarks/countbench/countbench_codevision_eval.parquet`
- Existing panel: V*, ChartQA, OCRBench, HRBench, FSC147, ArxivQA, CVBench, Pixmo-Count, SpatialMQA, CountQA, OCRBench v2
- MME-RealWorld: `/mnt/cpfs/delinmao/Benchmarks/MME-RealWorld/mme_realworld_codevision_eval.parquet` rows=23609
- MME-RealWorld-Lite: `/mnt/cpfs/delinmao/Benchmarks/MME-RealWorld-Lite/mme_realworld_lite_codevision_eval.parquet` rows=1919
- MME-RealWorld-CN: `/mnt/cpfs/delinmao/Benchmarks/MME-RealWorld-CN/mme_realworld_cn_codevision_eval.parquet` rows=5917
- RealWorldQA: `/mnt/cpfs/delinmao/Benchmarks/RealWorldQA/realworldqa_codevision_eval.parquet` rows=765
- MMStar: `/mnt/cpfs/delinmao/Benchmarks/MMStar/mmstar_codevision_eval.parquet` rows=1500
- DocVQA val: `/mnt/cpfs/delinmao/Benchmarks/DocVQA/docvqa_val_codevision_eval.parquet` rows=5349
- InfoVQA val: `/mnt/cpfs/delinmao/Benchmarks/InfoVQA/infovqa_val_codevision_eval.parquet` rows=2801
- MMVet: `/mnt/cpfs/delinmao/Benchmarks/MMVet/mmvet_codevision_eval.parquet` rows=218

`prepare_doc_info_mmvet_20260624` completed with exit code 0. All checked rows have non-empty `prompt`, `images`, and `reward_model.ground_truth`.
