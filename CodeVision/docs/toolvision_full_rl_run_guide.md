# ToolVision 全量 RL 运行指南

这份文档用于在 DLC 上提交 CodeVision 的 ToolVision 26k GSPO 主 RL 训练。

## 当前默认配置

- W&B project：`ToolVisionRL`
- W&B run / experiment：`qwen3vl8b_gspo_full26k`
- 训练数据：`/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet`
- 初始模型：`/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03`
- tool schema：`recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml`
- system prompt：`recipe/codevision/config/sp3.txt`
- DLC 资源：2 个 worker，每个 worker 8 张 GPU
- RL 参数：`TRAIN_BSZ=64`，`N_RESP_PER_PROMPT=8`，`MAX_TURNS=12`，`TOTAL_EPOCHS=1`
- eval：默认关闭，`VAL_BEFORE_TRAIN=False`，`TEST_FREQ=-1`
- judge fallback：默认开启，`ENABLE_LLM_JUDGE=1`
- 非 judge family 的额外 judge：默认关闭，`TOOLVISION_RL_USE_LLM_JUDGE=0`

训练数据默认使用 medium-clean 26k parquet。旧 40k parquet 没有被覆盖，只有显式覆盖 `TRAIN_FILES` 时才会使用。

## 1. 进入环境

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
source /mnt/cpfs/delinmao/use_tools.sh
```

## 2. 配置 W&B 和 DashScope Judge

正式全量训练默认需要两类 key：

- `WANDB_API_KEY`：只负责记录训练日志。
- `OFFLINE_SFT_QWEN_API_KEY` 或 `DASHSCOPE_API_KEY`：给 judge fallback 调用 DashScope-compatible API。

推荐使用下面这组变量名：

```bash
export WANDB_API_KEY='YOUR_WANDB_KEY'

export OFFLINE_SFT_QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export OFFLINE_SFT_QWEN_MODEL='qwen3.6-plus'
export OFFLINE_SFT_QWEN_API_KEY='YOUR_DASHSCOPE_KEY'
```

`scripts/submit_dlc_gspo_direct_full.sh` 会自动把 `OFFLINE_SFT_QWEN_*` 映射到 reward 代码实际读取的 `LLM_JUDGE_*` 变量。

## 3. 检查 DashScope API 连通性

提交全量训练前先跑一次：

```bash
python scripts/check_dashscope_compatible_api.py
```

成功输出应类似：

```text
base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
model=qwen3.6-plus
response='ok'
```

如果出现 API key 编码错误，通常是 key 环境变量里混入了非 ASCII 字符或坏字节。重新清理并导出：

```bash
unset OFFLINE_SFT_QWEN_API_KEY DASHSCOPE_API_KEY LLM_JUDGE_API_KEY OPENAI_API_KEY
export OFFLINE_SFT_QWEN_API_KEY='YOUR_DASHSCOPE_KEY'
```

`check_dashscope_compatible_api.py` 默认不读取代理环境变量，避免 DSW 里的 SOCKS proxy 引发 `socksio` 依赖错误。如果确实必须走代理，再设置：

```bash
export API_CHECK_TRUST_ENV=1
```

## 4. 提交全量 RL

```bash
WORKER_IMAGE='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04' \
bash scripts/submit_dlc_gspo_direct_full.sh
```

如果要区分多次实验，覆盖 `JOB_NAME` 和 `EXP_NAME`：

```bash
JOB_NAME=codevision_gspo_full26k_try2 \
EXP_NAME=qwen3vl8b_gspo_full26k_try2 \
WORKER_IMAGE='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04' \
bash scripts/submit_dlc_gspo_direct_full.sh
```

脚本提交前会打印关键配置。至少确认这些行：

```text
PROJECT_NAME=ToolVisionRL
EXP_NAME=qwen3vl8b_gspo_full26k
TRAIN_FILES=['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet']
ENABLE_WANDB=1
ENABLE_LLM_JUDGE=1
LLM_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_JUDGE_MODEL_NAME=qwen3.6-plus
TRAIN_BSZ=64
N_RESP_PER_PROMPT=8
TOTAL_EPOCHS=1
```

## 5. Judge fallback 到底会判哪些样本

默认 `ENABLE_LLM_JUDGE=1`，但 `TOOLVISION_RL_USE_LLM_JUDGE=0`。

含义是：

- `reward_family=judge/html_code/svg_code/general_code`：规则分不满分时，会调用 judge fallback。
- 其他 rule family，例如 `multiple_choice`、`numeric_exact`、`math_verify`、`bbox_iou`：默认不会因为 rule mismatch 就调用 judge。

当前 26k 数据里仍有 judge-family 样本。这些样本如果不启用 judge，会退化成较硬的规则/精确匹配分数；因此正式训练默认开启 judge fallback。

不建议默认设置：

```bash
export TOOLVISION_RL_USE_LLM_JUDGE=1
```

这个会让更多非 judge family 的 rule-mismatch 样本也走 judge，速度和 API 成本都会上升。

## 6. W&B 记录内容

脚本会传：

```bash
TRAINER_LOGGER='["console","wandb"]'
```

W&B 会记录：

- `training/global_step`
- reward 指标，例如 `reward/R_acc`、`reward/R_fmt`、`reward/R_total`
- actor loss、KL、entropy、grad norm
- prompt / response 长度统计
- tool call 和 generation timing
- throughput 和显存指标
- 每 `LOG_TRAIN_FREQ` step 记录若干 train generations

默认：

```bash
LOG_TRAIN_FREQ=20
LOG_TRAIN_GENERATIONS=8
```

## 7. 查看 DLC Job

列 job：

```bash
dlc_pai get job --workspace_id 245264 --page_size 10
```

看某个 job 详情：

```bash
dlc_pai get job JOB_ID --show_detail
```

看 master 日志：

```bash
dlc_pai logs JOB_ID JOB_ID-master-0 -n 2000
```

停 job：

```bash
dlc_pai stop job JOB_ID
```

## 8. Smoke Run

smoke 脚本现在默认只跑 2 个 training step：

```bash
export WANDB_API_KEY='YOUR_WANDB_KEY'

WORKER_IMAGE='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04' \
TRAIN_FILES="['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet']" \
bash scripts/submit_dlc_gspo_direct_smoke.sh
```

如需改步数：

```bash
TOTAL_TRAINING_STEPS=5 bash scripts/submit_dlc_gspo_direct_smoke.sh
```

## 9. 常见问题

`IndexError: index 1 is out of bounds for dimension 0 with size 1`

使用 clean train parquet。原始 parquet 有 605 条样本包含两个 `<image>` 标记但只有一张图。

`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`

API 检查脚本默认已经绕开 proxy。若其他地方出现，先检查并清理 `http_proxy`、`https_proxy`、`all_proxy` 等环境变量，或者安装 `httpx[socks]`。

`UnicodeEncodeError` while calling API

API key 环境变量里有非 ASCII 字符或坏字节。`unset` 后重新从干净字符串导出。

没有 W&B run

确认提交前导出了 `WANDB_API_KEY`，并且脚本打印了 `ENABLE_WANDB=1`。

提交脚本提示缺少 judge 变量

正式脚本默认 `ENABLE_LLM_JUDGE=1`，所以需要提供 `OFFLINE_SFT_QWEN_BASE_URL`、`OFFLINE_SFT_QWEN_MODEL` 和 `OFFLINE_SFT_QWEN_API_KEY`，或者等价的 `LLM_JUDGE_*` 变量。
