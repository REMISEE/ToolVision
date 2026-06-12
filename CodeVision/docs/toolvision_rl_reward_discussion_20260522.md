# ToolVision RL Reward 讨论归档 2026-05-22

## 背景

当前全量 RL 跑到约 90 step 后，工具调用基本消失：

- `reward/NumTurns=0.0`
- `timing_s/agent_loop/tool_calls/mean=0.0`
- `reward/R_acc` 和 `reward/R_fmt` 有提升
- 但模型明显学到了“不用工具更安全”

这说明现有 reward 设计没有给 tool use 正向信号，反而对工具探索有负向选择。

## 当前 simple_penalty reward 的问题

当前运行使用：

```bash
TOOL_REWARD_MODE=simple_penalty
```

实际公式近似为：

```text
R_total =
  R_acc
  + w_fmt * R_fmt
  - w_over * max(0, tool_count - overuse_threshold)
  - w_tool_error * I(tool_error)
  - w_invalid * I(invalid_call)
```

默认配置：

```text
w_fmt = 0.2
w_over = 0.05
w_tool_error = 0.2
w_invalid = 0.2
overuse_threshold = 4
```

因此：

- 不用工具但答对、格式对：`1.2`
- 用工具但答对、格式对、没报错：也是 `1.2`
- 用工具一旦 error 或 invalid：会被扣分

结论：用工具没有额外收益，只有潜在风险。模型 collapse 到 no-tool 是符合 reward 激励的。

## CodeVision legacy reward 里存在的 R_nec

代码里已有一套 legacy tool-aware reward，其中包含同题多 rollout 的 tool/no-tool 正确率差：

```text
acc_tool = correct_with_tool / total_with_tool
acc_no_tool = correct_without_tool / total_without_tool
R_nec = max(0, acc_tool - acc_no_tool)
```

样本级使用条件：

```text
当前 rollout 答对
且 used_tool=True
且 correct_without_tool < 2
```

注意：

- 只要同题里至少有 1 条 no-tool rollout，就能计算 `R_nec`。
- 如果同题所有 rollout 都用了工具，没有 no-tool rollout，则当前实现里 `R_nec=0`。
- `correct_without_tool < 2` 不是要求 no-tool 至少 2 条，而是 no-tool 已经答对 2 条或更多时，不再认为工具有必要性。

原版 `qwen3_vl.sh` 的参数：

```text
alpha = 1.0
beta = 0.0
gamma = 0.5
delta = 0.5
format_reward_weight = 0.1
exec_reward_weight = 0.0
emerge_reward_weight = 0.2
```

所以原版虽然有 `R_nec` 代码，但默认 `beta=0.0`，并没有真正给 `R_nec` 权重。

## 为什么暂时不直接用 legacy reward

legacy reward 还包含：

```text
R_mandatory
R_exact
R_emerge
R_spurious
C_usage
```

其中 `R_mandatory` 和 `R_exact` 依赖 `required_transforms`，更适合 CodeVision 里有明确 crop/rotate/flip 监督的任务。我们的通用 ToolVision RL 数据没有“这题必须用哪个具体工具”的标签，因此直接启用 legacy 会混入不适合当前数据的过程奖励。

本轮目标只是验证：给 tool/no-tool 正确率差一个正向信号，能否把工具调用从 0 拉回来。

## 策略 A：R_nec-only

策略 A 只加入 `R_nec`，不加入 `R_mandatory/R_exact/R_emerge`，并合并 tool error 和 invalid call 的惩罚。

公式：

```text
R_total =
  R_acc
  + 0.2 * R_fmt
  + 0.3 * R_nec
  - 0.1 * I(tool_error or invalid_call)
  - 0.05 * max(0, tool_count - 4)
```

参数：

```text
w_fmt = 0.2
beta = 0.3
w_bad = 0.1
w_over = 0.05
overuse_threshold = 4
```

目的：

- 只奖励“工具轨迹比 no-tool 轨迹更容易正确”的情况。
- 不给“只要用工具”奖励。
- tool error 和 invalid call 合并成一次轻惩罚，避免探索期被双重压制。

冷启动风险：

- 如果一开始几乎没有 tool 且 correct 的轨迹，`R_nec` 不会产生有效正反馈。
- 如果同题所有 rollout 都是 tool，没有 no-tool 对照，当前实现里 `R_nec=0`。

因此策略 A 是干净但可能偏保守的 ablation。

## 策略 B：R_nec + correct-tool fallback

如果策略 A 不能把 `NumTurns` 拉起来，再考虑策略 B。

公式：

```text
R_total =
  R_acc
  + 0.2 * R_fmt
  + 0.3 * R_nec
  + 0.05 * I(used_tool and correct and R_nec == 0)
  - 0.1 * I(tool_error or invalid_call)
  - 0.05 * max(0, tool_count - 4)
```

`lambda_tool_correct=0.05` 很小，只用于冷启动，不希望把模型推成无脑调工具。

策略 B 的意义：

- 当还没有形成有效 tool/no-tool 对比时，给 “用了工具且答对” 一点弱正反馈。
- 用 `R_nec == 0` 避免同一条轨迹同时吃 `R_nec` 和 fallback bonus。

风险：

- 模型可能在本来能直接回答的问题上多调用工具。
- 需要依靠 `overuse`、`tool_error/invalid` 和小权重控制。

## 当前执行决策

本轮先跑策略 A。

原因：

- 不引入新数据。
- 不改训练形态和大部分超参。
- 只验证 `R_nec` 能否把工具调用拉回来。

新增脚本：

```bash
scripts/submit_dlc_gspo_direct_rnec.sh
```

该脚本复用 full run 的训练参数，只覆盖 reward 和实验名：

```bash
TOOL_REWARD_MODE=rnec_only
FORMAT_REWARD_WEIGHT=0.2
TOOL_REWARD_BETA=0.3
TOOL_REWARD_OVERUSE_WEIGHT=0.05
TOOL_REWARD_TOOL_ERROR_WEIGHT=0.1
TOOL_REWARD_INVALID_CALL_WEIGHT=0.1
TOOL_REWARD_OVERUSE_THRESHOLD=4
```

建议先跑短步数验证：

```bash
TOTAL_TRAINING_STEPS=50 bash scripts/submit_dlc_gspo_direct_rnec.sh
```

观察指标：

- `reward/NumTurns`
- `timing_s/agent_loop/tool_calls/mean`
- `reward/R_nec`
- `reward/R_acc`
- `reward/R_fmt`
- `reward/P_bad_tool`
- `reward/P_overuse`
- `timing_s/step`

如果策略 A 下 `NumTurns` 仍接近 0，再上策略 B。
