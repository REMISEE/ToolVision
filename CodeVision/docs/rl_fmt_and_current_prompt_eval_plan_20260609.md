# RL Format 低分诊断与当前 Prompt Eval 计划

## 1-step RL 用的数据

这次 1-step run 用的是：

```text
/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_no_ood.parquet
```

它不是“去掉所有数学/学科题后的纯工具数据”。这个 parquet 只去掉了几个明显 OOD source：

```text
mmk12
wemath_standard
thinklite_vl_hard
puzzlevqa
```

仍然包含 `arxivqa`、`sat2`、`WaltonFuture`、`virl39k`、`tqa` 等偏学科/非纯工具 source。它能用于定位 RL rollout 的 format 问题，但不能代表只在小工具任务上的初始表现。

1-step 采样里 source 分布也偏混合，包含 `chartqa/refl4/virl39k/pixmo_count/WaltonFuture/arxivqa/docvqa/...`，所以后续如果要做更干净的工具 RL，应该单独构造 tool-heavy 子集。

## 为什么 eval fmt 能 0.8-0.99，而 RL step0 只有 0.2-0.3

当前最可能的原因不是 reward 传错，而是 generation 条件不一致：

```text
eval validation path:
  do_sample=False
  temperature=0
  val n=1
  输出更短、更稳定

RL training rollout:
  do_sample=True
  temperature=1.0
  rollout n=8
  多工具轮次后更容易长输出/截断/多 final answer
```

1-step 的 rollout 里 format_reward 约 0.258，重新用 strict format 规则离线复算也是同一个量级，说明低分不是字段传播 bug。更关键的现象是：

```text
no-tool trajectory fmt 约 86%
tool trajectory fmt 约 8%
```

也就是说模型不是完全不会按格式答，而是进入工具轮次后很容易跑飞。之前高 format 的 eval 同样有很高工具调用率，所以“用 tool 必然低 format”不成立；更像是 RL sampling 温度、逐轮生成长度、prompt/tool schema 与数据分布共同造成。

已做的保护改动：

```text
1. agent_loop format_reward 字段改成 union 收集，避免只看第一条样本的 key。
2. 如果缺 format_reward，按 UVTR strict format 离线复算。
3. tool_agent_loop 支持 max_response_tokens_per_turn。
4. vllm_async_server 支持每次 generate request 级 max_tokens。
5. RL launcher 默认 ROLLOUT_MAX_TOKENS_PER_TURN=2048。
6. RL launcher 默认 SAVE_FREQ=50，避免只保留很晚 ckpt。
```

## 当前 prompt/tool eval 对照

新增脚本：

```text
scripts/submit_dlc_current_prompt_tool_eval.sh
```

默认提交 4 个 job：

```text
chartqa, temperature=0
chartqa, temperature=0.7
fsc147,  temperature=0
fsc147,  temperature=0.7
```

统一条件：

```text
model:  /mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03
prompt: recipe/codevision/config/sp3.txt
tool:   recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml
eval n: 1
per-turn response cap: 2048
```

这个 eval 的目标是隔离变量：

```text
temperature=0:
  对齐旧 eval 的 greedy validation，看当前 prompt/tool schema 是否仍能保持高 fmt。

temperature=0.7:
  更接近 RL 初始采样，但比 1.0 保守，用来看 temperature 是否显著拉低 fmt。
```

如果 `temperature=0` format 仍高、`0.7` 明显下降，temperature 是主变量之一。  
如果两者都高，而 RL step0 仍低，问题更可能在 training rollout path、数据分布、n=8 多样采样或 RL prompt 构造。  
如果两者都低，当前 prompt/tool schema 本身已经不稳，需要先修 SFT/eval prompt 对齐。

## 建议的下一步

先跑当前 prompt/tool eval，不直接开 full RL：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
scripts/submit_dlc_current_prompt_tool_eval.sh
```

如果资源紧张，可以先只跑 FSC147：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
DATASETS='fsc147' TEMPERATURES='0 0.7' scripts/submit_dlc_current_prompt_tool_eval.sh
```

读结果时重点看：

```text
saves/CodeVision/current_prompt_tool_eval_*/metrics.json
saves/CodeVision/current_prompt_tool_eval_*/diagnostics/sampled_traces.jsonl
```

如果 `temperature=0.7` 比 `0` 只轻微下降，可以启动修复后的 GSPO 26k run：

```text
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.95
ROLLOUT_DO_SAMPLE=True
ROLLOUT_MAX_TOKENS_PER_TURN=2048
SAVE_FREQ=50
TRAIN_FILES=['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet']
```

温度 warm-up 可以做，但当前 launcher 还不是按 step 自动调度；最稳妥的第一版是整段先用 0.7，确认 format/tool 不崩后再考虑第二阶段升到 1.0。
