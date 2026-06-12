# 25 Judge Backend v1 与十项落地记录

日期：2026-04-10  
状态：已落地主干，剩余 1 个主闭环项未完成  
范围：记录这轮围绕 `offline_sft_pipeline` 做的 10 项规划落地情况，重点覆盖：

- 预算语义重定义
- forced-answer final round
- `question` / `answer_instruction` 拆分
- token usage 回传
- committee judge backend
- scorer 分发
- judge prompt / request / orchestrator 接线
- 当前剩余项

---

## 1. 一句话结论

这轮主干已经从“fake judge + 三步后直接 max_step_reached”推进到：

1. `remaining_exec_steps` 语义替代旧的 `remaining_rounds`
2. 最后一轮支持 forced final answer，而不是执行完最后一个 step 立刻终止
3. `RootSample.question` 已清洗为纯题目正文
4. `answer_instruction` 已进入 schema、prompt、judge request
5. planner / executor / judge 的 token usage 都能回收到 sample/trajectory 汇总
6. judge 已切到真实可接的 `CommitteeJudgeBackend`
7. scorer 已按 `source_dataset` 分发

当前**唯一还没真正闭环**的主项是：

- 用真实 `judge_models.json` endpoint + 真实 API key 跑一次 committee online smoke run

如果按“严格对齐官方评测”来算，还存在若干**次级未完成项**：

- `fsc147` 目前是整数 exact fallback，不是官方 MAE/RMSE 风格代理分
- `we_math_pro` / `we_math_standard` 目前是保守 fallback exact matcher
- `textvqa` 只有在 `answer` 提供 `list[str]` 时才会走 soft-vqa；当前很多 unified export 还是单参考

---

## 2. 十项规划落地总表

### 2.1 规划 1：提高最大轨迹长度，超时调大

状态：已完成

落地结果：

- `Budget` 已从旧语义切到 `remaining_exec_steps`
- dataset pipeline 默认预算已提高到 `remaining_exec_steps=6`
- planner/executor API timeout 默认已提高到 `200s`
- runtime service timeout 默认已提高到 `200s`

主要文件：

- `offline_sft_pipeline/core/models.py`
- `offline_sft_pipeline/scripts/run_dataset_pipeline.py`
- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/runtime/code_image_runtime_wrapper.py`

说明：

- judge 单模型 timeout 已进入 `judge_models.json`，按模型单独配置，不再绑死在统一超时上。

### 2.2 规划 2：明确 remaining rounds 概念，让最后一轮直接 answer

状态：已完成

落地结果：

- 旧 `remaining_rounds` 已改为 `remaining_exec_steps`
- planner request 新增 `must_answer_now`
- 当 `remaining_exec_steps <= 0` 时，不再直接标 `max_step_reached`
- orchestrator 会进入 forced final-answer round
- 只有 forced final-answer round 仍不答时，才标 `max_step_reached`

主要文件：

- `offline_sft_pipeline/core/models.py`
- `offline_sft_pipeline/pipelines/request_models.py`
- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/pipelines/api_text_multimodal.py`

### 2.3 规划 3：明确改动路径和改哪些文件

状态：已完成

本轮实际改动集中在：

- schema / model：
  - `offline_sft_pipeline/core/models.py`
  - `offline_sft_pipeline/schemas/trajectory_schema.json`
  - `offline_sft_pipeline/schemas/judge_record_schema.json`
- sample normalize：
  - `offline_sft_pipeline/core/sample_normalization.py`
- prompt / request / client：
  - `offline_sft_pipeline/pipelines/request_models.py`
  - `offline_sft_pipeline/pipelines/api_text_multimodal.py`
  - `offline_sft_pipeline/pipelines/planner_client.py`
  - `offline_sft_pipeline/prompts/planner_user_v01.txt`
  - `offline_sft_pipeline/prompts/judge_system_v01.txt`
  - `offline_sft_pipeline/prompts/judge_user_v01.txt`
- orchestrator / store：
  - `offline_sft_pipeline/pipelines/orchestrator_v01.py`
  - `offline_sft_pipeline/core/store.py`
- judge backend：
  - `offline_sft_pipeline/pipelines/backends.py`
  - `offline_sft_pipeline/judge_models.json`
  - `offline_sft_pipeline/eval/scorers.py`
- script 接线：
  - `offline_sft_pipeline/scripts/run_dataset_pipeline.py`

### 2.4 规划 4：token 消耗返回

状态：已完成

落地结果：

- planner / executor / judge 都记录原始 token usage
- dataset pipeline 会按：
  - 每次调用
  - 每条 trajectory
  - 每个 sample
  三级汇总

主要文件：

- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/scripts/run_dataset_pipeline.py`

当前策略：

- 底层按“每次模型调用”记录
- 上层自动聚合为 trajectory total / sample total

### 2.5 规划 5：judge model 自己做题，再和 GT 比

状态：已完成主干

落地结果：

- committee judge 不再走“fake overall score”
- 每个 judge model 独立读取：
  - question
  - answer instruction
  - conversation history
  - visible images
- 每个 judge model 只输出最终答案文本
- 本地 scorer 用 GT `answer` 计算单模型 score
- committee backend 聚合为 `overall_score`

主要文件：

- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/request_models.py`
- `offline_sft_pipeline/prompts/judge_system_v01.txt`
- `offline_sft_pipeline/prompts/judge_user_v01.txt`

### 2.6 规划 6：模糊匹配 / 非 0-1 计分

状态：部分完成

已完成部分：

- `textvqa` 支持 `answer: str | list[str]`
- 当 `answer` 为 `list[str]` 时，已支持 VQA 风格 `min(1, matches / 3)` soft score
- `gqa` 已做 exact match，当前对齐 `ignore_case + ignore_punctuation`
- 多选题已做 option-letter normalize

当前实现：

- `arxivqa`：option letter exact
- `cavqa_multichoice`：option letter exact
- `gqa`：ignore-case / ignore-punctuation exact
- `textvqa`：single-ref exact 或 multi-ref soft-vqa
- `fsc147`：relative-error proxy score
- `we_math_pro`：fallback normalized exact
- `we_math_standard`：fallback normalized exact

未完成部分：

- `we_math_*` 尚未做专门的 math normalizer / symbolic matcher

### 2.7 规划 7：停止条件顺序

状态：已完成 v1

当前停止顺序如下：

1. planner / executor / runtime / judge 异常：`error`
2. runtime 失败：`failed`
3. runtime 空结果：`pruned`
4. judge 总分未过 keep threshold：`pruned`
5. 若仍可继续：`running`
6. 若 `remaining_exec_steps == 0`：进入 forced final-answer planner round
7. forced final-answer 仍不答：`max_step_reached`

说明：

- `delta_drop_stop`
- `patience_no_gain`
- 更复杂的“多轮无增益早停”

这批策略这轮**没有进入主干**，当前仍使用最小可运行版 stop policy。

### 2.8 规划 8：多模型 judge backend 的接入方式

状态：已完成

当前方案：

- 不做两层 provider 配置抽象
- 只做 `judge_models.json`
- 所有 judge model 都假设是 OpenAI-compatible `/v1/chat/completions`
- backend 读取 enabled 模型并发调用

配置文件：

- `offline_sft_pipeline/judge_models.json`

字段：

- `name`
- `model`
- `base_url`
- `api_key_env`
- `timeout_s`
- `enabled`

代码内常量：

- `DEFAULT_JUDGE_MAX_CONCURRENCY = 3`
- `DEFAULT_JUDGE_MAX_RETRIES = 2`

### 2.9 规划 9：answer instruction 与问题正文分离，避免干扰 planner

状态：已完成

已落地规则：

- `RootSample.question` 现在承载“纯题目正文”
- 新增 `answer_instruction`
- normalize 阶段按 `source_dataset`：
  - 从题目尾句剥离已有 instruction
  - 或补默认 instruction

当前数据集规则：

- `arxivqa`：`Answer with the option letter only.`
- `cavqa_multichoice`：剥离尾句后统一为 `Answer with the option letter only.`
- `fsc147`：剥离尾句后统一为 `Answer with only an integer.`
- `gqa`：`Answer the question using a single word or phrase.`
- `textvqa`：`Answer the question using a single word or phrase.`
- `we_math_*`：暂留空

planner prompt 当前语义：

- 一直看到 `answer_instruction`
- 但明确写了：
  - 它只是 output-format constraint
  - 只有在进入 answer mode 时才必须遵守

### 2.10 规划 10：并行

状态：部分完成

已完成部分：

- judge 内部 committee 调多个模型时已经并发
- 外层 orchestrator 仍保持单轨迹串行，不会引入 store 写冲突

未完成部分：

- dataset pipeline 的 sample 级多进程并行还没做
- `--workers` 之类的批量并发入口还没加

这项不是本轮 judge 主干阻塞项，暂未推进。

---

## 3. 本轮 judge v1 实际落地内容

## 3.1 新增 `judge_models.json`

新增文件：

- `offline_sft_pipeline/judge_models.json`

用途：

- 配置 enabled judge model
- 为不同 provider / endpoint 单独设置模型名、base_url、api key env、timeout

当前默认状态：

- DashScope Qwen 三个 API 项默认 enabled
- 其余 Qwen-VL / Gemini 项默认 disabled

说明：

- 文件中的 Gemini / 自部署项仍是占位 endpoint，需要按实际环境改。

## 3.2 新增 scorer 分发

新增文件：

- `offline_sft_pipeline/eval/scorers.py`
- `offline_sft_pipeline/eval/__init__.py`

职责：

- 统一按 `source_dataset` 分发 scorer
- 统一做 normalize
- 支持 `answer: str | list[str]`

## 3.3 CommitteeJudgeBackend 真正落地

主要文件：

- `offline_sft_pipeline/pipelines/backends.py`

职责：

1. 读取 `judge_models.json`
2. 加载 judge system prompt
3. 构造 judge OpenAI-style multimodal messages
4. 并发调用 enabled judge model
5. 清理输出答案文本
6. 调 scorer 与 GT 比分
7. 聚合：
   - `per_model_scores`
   - `overall_score`
   - `token_usage`
   - `model_results`

## 3.4 judge request 接线

本轮新增的关键 request 字段：

- `sample_dir`
- `trajectory_dir`
- `answer_instruction`
- `answer`

原因：

- judge 需要像 planner 一样回放消息里的图片
- judge scorer 需要拿到 GT `answer`

## 3.5 orchestrator 已切到可喂新 judge 的输入

`orchestrator_v01.py` 当前在 step judge 时会传入：

- sample/trajectory 路径
- 根样本 `answer`
- 根样本 `metadata.source_dataset`
- 轨迹 `answer_instruction`

所以新 committee backend 已经不是“文件存在但没接线”，而是已经接到主路径上。

## 3.6 dataset pipeline 脚本已支持切 judge backend

`run_dataset_pipeline.py` 新增：

- `--judge-backend committee|fake`
- `--judge-models-file`

当前默认：

- `--judge-backend committee`

也就是说，批跑脚本现在默认会走新 judge backend，而不是 fake judge。

---

## 4. 这轮没有做的内容

## 4.1 唯一未闭环主项

当前唯一没真正完成的主闭环项是：

- 用真实可用的 `judge_models.json`
- 配好所有 `api_key_env`
- 跑一次 committee online smoke run

换句话说：

- 代码已经可跑
- 单测和本地 patch 测试已过
- 但还没有用你自己的真实 endpoint 做一轮线上联调验收

这就是“还有一个没实现”的那一项。

## 4.2 次级待补项

这几项不是主干阻塞，但仍然算后续增强项：

1. `fsc147` scorer 改成更合理的误差分，而不是 integer exact
2. `we_math_*` 增加专门的数学 normalize / 比对逻辑
3. `textvqa` 在 unified export 里补多参考答案，充分发挥 soft-vqa scorer
4. dataset pipeline 样本级并行
5. 更复杂 early-stop policy：
   - delta drop
   - patience no gain
   - top-k keep

---

## 5. 本轮测试与验证

已跑通过的测试：

- `python -m unittest offline_sft_pipeline.tests.test_judge_backend`
- `python -m unittest offline_sft_pipeline.tests.test_sample_normalization`
- `python -m unittest offline_sft_pipeline.tests.test_pipelines`
- `python -m unittest offline_sft_pipeline.tests.test_orchestrator_v01`

当前结论：

- scorer 分发能跑
- committee backend 并发聚合能跑
- token usage 聚合没炸
- orchestrator judge request 改造没有回归坏主流程

---

## 6. 建议的下一步

建议顺序如下：

1. 先把 `judge_models.json` 里的真实 endpoint 和 env key 补齐
2. 用 `--judge-backend committee` 跑 2 到 3 个 sample 做 online smoke
3. 观察：
   - 单模型 latency
   - 总墙钟
   - 429 / timeout
   - 各 scorer 输出分布
4. 再决定是否：
   - 调整 `DEFAULT_JUDGE_MAX_CONCURRENCY`
   - 调整 `timeout_s`
   - 改进 `fsc147` / `we_math_*` scorer

---

## 7. 对“还有一个没实现”的最终说明

如果按“代码主干有没有落地”来判断：

- 是，judge v1 主干已经落地了。

如果按“有没有真正线上跑过真实 committee judge”来判断：

- 还差一个，确实还没完成。

这个剩余项不是代码结构问题，而是联调验收问题：

- 真实 endpoint
- 真实 API key
- 真实 smoke run

这也是当前最该做的下一步。
