# 明天接手 - 2026-05-28（更新版）

如果你忘了昨天讲到哪儿，看这一份就够了。

---

## ★ 最新结论（2026-05-28 傍晚）

**promptfix SFT 实验失败了**：

| Run | R_acc step 22 | R_fmt step 22 | NumTurns 归零时机 |
|---|---|---|---|
| rnec (原 SFT, R_nec reward) | **0.525** | **0.840** | step 50 |
| legacy (原 SFT, 普通 reward) | 0.510 | **0.890** | step 50 |
| rnec_clean (原 SFT, clean reward) | 0.416 | 0.455 | step 30 |
| **promptfix (新 SFT 占位符)** | **0.375** | **0.365** | step 25 ← **最差** |

**两次 reward/SFT 干预都失败了**。原始 rnec 是目前最好的 baseline。

**核心 finding**：
1. NumTurns→0 是所有 run 的共同结局，不是 SFT prompt 的问题，**是 EV 计算的必然结果**：
   - 不调 tool + 答对 EV = 0.7
   - 调 tool (current 10% JSON 成功率) EV ≈ 0.1
   - 差 7 倍，模型理性放弃 tool
2. **reward shaping 救不了**：要让调 tool EV > 不调，clean_tool 要 ≥ 8.0（不现实）
3. **真正的瓶颈是 P(success of JSON tool call)** ≈ 10%，这个数取决于 SFT 质量
4. R_fmt 和 bad_tool 都不可有意义地 hack（彻查过，详见聊天记录）

## ★ 今天最后一发 RL（造新数据前）

**配置文件**: `scripts/submit_dlc_gspo_direct_final_before_newdata.sh`
**4 个 lever 同时拉**：
1. 回退到**原始 SFT 模型**（promptfix 是 regression）
2. **SFT-aligned 10-tool config** with 抽象 description example（`v03_sftclean.yaml`）
3. **clean_tool_weight = 0.15**（×3，逼模型探索 tool）
4. **清洗后 34k 数据**（砍 mmk12+wemath+thinklite+puzzlevqa = 6000）

**判定标准**:
- **PASS** (NumTurns≥0.5 + R_acc≥0.55) → reward shaping 还能救，可以继续迭代
- **FAIL** (NumTurns→0 within 30 step) → 闭环，必须造新 SFT 数据，**别再调 reward**

---

## 0. 当前正在跑的 run

- **WandB run name**: `qwen3vl8b_gspo_full40k_rnec_clean_promptfix`
- **配置**:
  - SFT model: `sft-mix200-simple-notool-sp3-v03-promptfix`（昨天新训的）
  - Reward: rnec_with_clean，参数全不动（clean_tool=0.05, bad=0.02, beta=0.3, fmt=0.2, overuse=0.05/threshold=4）
  - 数据: 原 40k `train.parquet`（**没清洗**）
  - 单变量实验：**只换 SFT 模型**，其他全不变
- **预计完成时间**: T+24h（按你重启时间）

## 0a. 第一次 kick off 失败的教训（重要）

第一次 5-27 15:38 的 run 用了 `REWARD_LAUNCH_ASYNC=True`，结果跑完第一 step 就崩了：
```
ray.exceptions.RaySystemError: No module named 'custom_module'
```

原因：Ray serialize `recipe/codevision/reward.py` 时把它标记成 `custom_module`，async worker deserialize 失败。**已修**——`submit_dlc_gspo_direct_rnec_clean_promptfix.sh` 现在默认 `REWARD_LAUNCH_ASYNC=False`。

判断 run 真在跑的信号：
- log 里能看到 `Training Progress: X%|... [XX:XX<XX:XX, X.XXs/it]` 数字在变
- step 1 完成后 wandb 上有完整指标点（不是只有 step 0）

判断 run 崩了的信号：
- log 出现 `Error executing job` + `ray.exceptions.RayTaskError`
- `Training Progress: 0%|... [XX:XX<?, ?it/s]` 进度停住不动
- log 末尾出现完整 Python traceback

## 1. 第一件事：看曲线决定下一步

WandB 上对比这 4 条 run：
- `qwen3vl8b_gspo_full40k`（原始 legacy）
- `qwen3vl8b_gspo_full40k_rnec`（rnec_only）
- `qwen3vl8b_gspo_full40k_rnec_clean`（旧 SFT + clean_tool reward）
- **`qwen3vl8b_gspo_full40k_rnec_clean_promptfix`（新 SFT + 同 reward）** ← 这就是判断的关键

**判断准则**（看 step 30+ 的稳态值）：

| 指标 | 通过线 | 决策 |
|---|---|---|
| `reward/NumTurns` | ≥ 0.3 | promptfix 起作用 → **进入 Step 3 数据清洗** |
| `reward/NumTurns` | 还是 ≈ 0 | promptfix 不够 → **进入 Step 2 涨 reward** |
| `reward/R_acc` | 比 rnec_clean 持平或更高 | 没退化 |
| `reward/R_acc` | 明显下降 | 出问题，需要诊断（不太可能） |

## 2. 三个分支的执行计划

### Branch A: NumTurns 起来了（≥0.3）
说明 prompt drift 是首要根因，reward 没问题。下一步：

1. **数据清洗**：从 `train.parquet` 砍掉 SFT 完全没见过的强 OOD 类别
   - 砍 mmk12 (1500) + wemath_standard (1500) = 3000 条（**保守版**）
   - 或加砍 thinklite_vl_hard (2000) + puzzlevqa (1000) = 6000 条（**激进版**）
   - 输出 `train_no_math.parquet`
2. **新 RL run**：相同 promptfix SFT + rnec_with_clean + 清洗后数据
3. **新 submit 脚本**：基于 `submit_dlc_gspo_direct_rnec_clean_promptfix.sh`，加 `TRAIN_FILES` 指向新 parquet

### Branch B: NumTurns 还是 0
说明 prompt 对齐了但 reward 不够强压不住"调 tool 怕亏"的心理。涨 reward：

1. **新 submit 脚本** `submit_dlc_gspo_direct_rnec_clean_promptfix_boostreward.sh`
2. 改动：
   ```bash
   export TOOL_REWARD_CLEAN_TOOL_WEIGHT=0.10  # 0.05 → 0.10
   # 其他不动
   ```
3. ratio 从 2.5 → 5（clean 0.10 vs bad 0.02），把"尝试 tool 期望收益"从 −0.006（20% 成功率）拉到 +0.004
4. 如果还不行再涨到 0.15

### Branch C: 两条都要试
直接并行起两个 RL run（如果有 DLC quota），24h 后看哪个赢。

## 3. 独立任务：pass@16 数据生成（让另一个 Codex 窗口做）

详细 spec 在我们昨天对话里（搜 "Thread B: pass@16 数据生成 spec"）。要点：

- **模型**: `Qwen3-VL-8B-Instruct`（原始 instruct，不是 SFT 后的）
- **输入**: 40k `train.parquet`
- **不带 tool prompt**（让模型纯文本直答）
- **n=16 rollouts per sample**
- **不要筛**（`--min-reward -1 --max-reward 2`），保留 pred_rewards/pred_accs 16 元素数组
- **8 GPU 并行**，预计 20-40h
- 需要新写一个 `parquet_generic` dispatch engine 给 Innovator-VL（详细步骤看那份 spec）

这条线跑出来后才能做 MUT 标注。

## 4. 中期 (本周 / 下周): 重做 SFT trajectory

依赖 pass@16 结果。pass@16 完成后：

**SFT 数据组成（目标 4-5k 条）**：
| 难度 | 占比 |
|---|---|
| pass@16 ∈ [0.2, 0.5] | 60% |
| pass@16 ∈ [0.5, 0.8] | 25% |
| pass@16 ∈ [0.8, 1.0] | 15% |

**RL 数据组成（目标 30-40k）**：
| 难度 | 占比 |
|---|---|
| pass@16 ∈ [0, 0.2] + verified MUT | 30% |
| pass@16 ∈ [0.2, 0.6] | 50% |
| pass@16 ∈ [0.6, 1.0] | 20% |

verified MUT = pass@16 低 + 套上 ToolVision agent 用 tool 能解 → 真正"必须用工具"的题。RL 给强 mut bonus。

## 5. 当前所有已部署的代码改动（同步到 ssh）

| 文件 | 改动 |
|---|---|
| `verl/experimental/agent_loop/agent_loop.py` | +rnec_with_clean mode + 两个方法 + clean_tool_weight 参数 |
| `verl/trainer/ppo/metric_utils.py` | reward_component_keys 加了 R_nec, R_clean_tool, P_overuse, P_bad_tool, P_tool_error, P_invalid_call |
| `recipe/codevision/qwen3_vl_gspo_direct.sh` | +TOOL_REWARD_CLEAN_TOOL_WEIGHT 透传 + reward_launch_async 透传 |
| `scripts/submit_dlc_gspo_direct_full.sh` | +5 行 append_env（CLEAN_TOOL_WEIGHT / REWARD_LAUNCH_ASYNC / LOG_TRAIN_GENERATIONS / LOG_VAL_GENERATIONS / LOG_TRAIN_FREQ） |
| `scripts/submit_dlc_gspo_direct_rnec_clean.sh` | rnec_with_clean mode 入口（旧 SFT） |
| `scripts/submit_dlc_gspo_direct_rnec_clean_promptfix.sh` | **当前正在跑的入口**：新 SFT + rnec_with_clean + LOG_TRAIN=32/freq=10 |

**本地有一个未同步的修改**：`submit_dlc_gspo_direct_rnec_clean_promptfix.sh` 里 `REWARD_LAUNCH_ASYNC` 默认值已经改回 False（昨天 async bug 后的修复）。当前 run 跑完后再同步这个文件。

## 6. 已知 bug / gap（不紧急）

1. **REWARD_LAUNCH_ASYNC=True 不工作** —— Ray serialization 找不到 `custom_module`。要修需要 Ray runtime_env 配 PYTHONPATH。目前禁用。
2. **tool_parser.py Type B malformation gap** —— `<tool_call>` 块有 `"name"` 但 args 烂的情况 silent drop，不计 invalid。要修需要 parser 返回 invalid_count 写回 extra_fields。
3. **LLM_JUDGE_TIMEOUT=30 / MAX_RETRIES=2** —— 这次顺手收紧了，没看出副作用。

## 7. 一句话状态

> 等当前 promptfix run（24h）跑出 wandb 曲线，看 NumTurns 起没起来。起来了 → 走 Branch A 清数据；没起来 → 走 Branch B 涨 reward。pass@16 数据生成由另一个 Codex 窗口独立推进，不阻塞。
