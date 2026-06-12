# 29 TextVQA 官方 Scorer 与 Planner 收口优化说明 2026-04-15

这份文档记录 2026-04-15 在 `image_index` 协议改造之后，继续落地的四类收口优化：

1. planner JSON 脏输出兼容补救
2. planner 默认 suggestion 数从 `3` 收到 `2`
3. TextVQA 接入官方 EvalAI scorer，并补 question / answer instruction 解耦
4. reference-backed 数据集在高分与满分时的收口逻辑调整

这份文档不重复解释 `image_index` 协议本身；相关背景见：

- [28_image_index统一协议改造与验收说明_20260415.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/28_image_index统一协议改造与验收说明_20260415.md)

---

## 1. 本轮改动结论

当前已经落地的行为是：

1. planner 输出如果是 fenced JSON 或带有常见非法转义，后端会先尝试补救，不再轻易掉回旧 `<think>` 解析报错。
2. 默认 `planner_suggestion_count` 从 `3` 改成 `2`，与 `max_child_trajectories=2` 对齐，避免先跑 3 条再只保留 2 条的额外成本。
3. TextVQA scorer 从项目内简化版 soft score 升级为对齐官方 `EvalAIAnswerProcessor` + leave-one-out soft score。
4. TextVQA 导出数据里的 `question` 与 `answer_instruction` 已在归一化入口解耦。
5. `arxivqa / cavqa_multichoice / gqa / textvqa` 在 `overall_score >= 0.9` 时，会进入 forced final answer 路径，把 judge 提供的稳定答案传给 planner 收尾。
6. 同一批 reference-backed 数据集在 judge 满分时，不再继续 rollout 同题的其他分支；但 winning branch 仍会保留并继续跑一轮 planner，由 planner 自己输出最终 CoT 与 answer。
7. `fsc147` 不走 reference-backed 的这套逻辑，仍然保留 count 专项处理。
8. stop policy 的 `patience` 本轮没有改。

---

## 2. 改动文件

### 2.1 parser / planner 收口

- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/scripts/run_dataset_pipeline.py`
- `offline_sft_pipeline/scripts/run_example_real_pipeline.py`

### 2.2 TextVQA

- `offline_sft_pipeline/eval/vqa_eval_metric.py`
- `offline_sft_pipeline/eval/scorers.py`
- `offline_sft_pipeline/core/sample_normalization.py`
- `offline_sft_pipeline/core/dataset_names.py`

### 2.3 测试

- `offline_sft_pipeline/tests/test_pipelines.py`
- `offline_sft_pipeline/tests/test_judge_backend.py`
- `offline_sft_pipeline/tests/test_sample_normalization.py`
- `offline_sft_pipeline/tests/test_orchestrator_v01.py`

---

## 3. planner JSON 脏输出兼容

### 3.1 问题

planner 已切到 JSON 协议，但线上模型仍会输出两类常见脏格式：

1. inline fenced JSON

```text
```json {"mode":"suggestions", ...} ```
```

2. 非法 JSON 转义

```json
{"think": "The graph\'s peak is clear.", ...}
```

这两类文本在旧逻辑里会导致：

1. JSON 预解析失败
2. 掉回旧 `<think>/<answer>/<suggestions>` 标签解析
3. 最终抛出 `Required <think> block not found`

### 3.2 当前补救

在 `parsing.py` 里新增了两层补救：

1. `strip_markdown_json_fence()` 能处理 inline fenced JSON，不再要求 fence 必须是多行格式。
2. `_repair_common_json_issues()` 会在 JSON 预解析前修复少量常见脏输出，目前包括把无效的 `\'` 修成 `'`。

这意味着：

- 对常见 fenced JSON / `\'` 问题，后端现在可以直接 salvage
- salvage 失败时，才继续走旧标签协议 fallback

---

## 4. planner suggestion 数从 3 收到 2

### 4.1 之前的问题

原来默认配置是：

- `planner_suggestion_count = 3`
- `max_child_trajectories = 2`

但 orchestrator 的行为不是“planner 产 3 条，只保留文本 top-2”，而是：

1. planner 产出前 3 条 suggestion
2. 这 3 条都会进入 child 初始化、executor、runtime、judge
3. 最后 `_select_next_frontier()` 只保留 top-2

所以旧配置的真实成本是：

- 同一轮多跑了一整条无必要分支

### 4.2 当前配置

默认改成：

- `planner_suggestion_count = 2`
- `max_child_trajectories = 2`

脚本入口也同步对齐：

- `run_dataset_pipeline.py`
- `run_example_real_pipeline.py`

当前判断是：

- 对大多数 `arxivqa / cavqa / gqa / textvqa` 样本，2 条 suggestion 已经足够
- 如果后面确实有需要更强探索的数据集，再按数据集单独放宽，不再全局默认 `3`

---

## 5. TextVQA 官方 scorer

### 5.1 问题

项目里原先已有 `textvqa` scorer，但只是简化版：

- 预测先做轻量 VQA 归一化
- 对 10 个答案直接数匹配数
- 分数近似写成 `min(1, matches / 3)`

这与官方 TextVQA 口径不完全一致。

### 5.2 当前实现

本轮把 TextVQA scorer 升级成与 `lmms-eval` 对齐的实现：

1. vendoring 一份最小 `EvalAIAnswerProcessor`
2. 对预测答案和 10 个 GT 答案都用同一套 EvalAI 规则归一化
3. 用 leave-one-out soft score：
   - 每个 GT 答案轮流拿其余 9 个作为对照
   - 单轮分数 `min(1, matches / 3)`
   - 最终对 10 轮求平均

匹配器名字改为：

- `textvqa_evalai_soft_vqa`

### 5.3 效果

这样以后 TextVQA 常见分数台阶会更符合官方行为：

- `0.0`
- `0.3`
- `0.6`
- `0.9`
- `1.0`

这也是为什么本轮没有立刻去改 TextVQA 的 `patience`：

- 先让官方 scorer 上线
- 再看真实 rollout 的分布和平台期

---

## 6. TextVQA question / answer instruction 解耦

### 6.1 问题

导出的 TextVQA `samples.jsonl` 当前是这种形态：

```text
what does the ad on the left of the coca cola say?
Answer the question using a single word or phrase.
```

也就是：

- `question` 仍然尾带 instruction
- `answer_instruction` 为空

### 6.2 当前归一化规则

在 `sample_normalization.py` 中新增了 TextVQA 尾缀剥离：

```text
Answer the question using a single word or phrase.
```

归一化后：

- `question` 只保留问题本体
- `answer_instruction` 单独填成 `Answer the question using a single word or phrase.`

这条规则在当前导出的 TextVQA 样本上是稳定成立的。

---

## 7. `0.9` forced answer 与满分 sample-level 收束

### 7.1 `0.9` 以上

对以下 reference-backed 数据集：

- `arxivqa`
- `cavqa_multichoice`
- `gqa`
- `textvqa`

当最新 judge `overall_score >= 0.9`，并且 judge metadata 中存在稳定一致的 `candidate_answer` 时：

- orchestrator 会把该答案作为 `forced_final_answer` 注入下一轮 planner request
- 下一轮 planner 进入 `must_answer`
- planner 需要围绕这个答案自己写最后一轮 CoT 和 answer

这条逻辑不用于 `fsc147`。

### 7.2 满分时

本轮新增了一条 sample-level 收束逻辑：

对同样这批 reference-backed 数据集，如果某条分支在 step judge 后达到：

- `overall_score >= 0.999`
- judge 能给出稳定一致的 `candidate_answer`

那么：

1. 同一个 sample 的其他 `running` 分支会被标成 `stopped_early`
2. 这条 winning branch 会成为唯一保留 frontier
3. orchestrator 不再 rollout 这个 sample 的其他路线
4. 但 winning branch 仍然会继续跑一轮 planner
5. planner 依然要自己输出最终 CoT 与 answer

也就是说：

- 它不是 “judge 直接写 final answer”
- 而是 “judge 满分后，整个 sample 收束成单一路径，再由 planner 收尾”

这正是当前想要的行为：

- 节省同题其他路线成本
- 保留一条完整的 assistant final answer turn

---

## 8. 本轮没有改的内容

以下内容本轮明确保持不变：

1. `must_suggest_score_threshold = 0.6`
2. `must_answer_score_threshold = 0.9`
3. stop policy 的 `patience`
4. `fsc147` 的 count 专项 forced-final-answer 逻辑

原因是：

- 当前优先级是先让 TextVQA scorer 和 reference-backed 收口逻辑稳定
- `patience` 是否需要再收，需要基于新 scorer 上线后的真实分数分布来决定

---

## 9. 验收建议

### 9.1 parser / planner / orchestrator

```bash
cd /data/home/suchenghao/ToolVision

python -m unittest \
  offline_sft_pipeline/tests/test_pipelines.py \
  offline_sft_pipeline/tests/test_orchestrator_v01.py
```

### 9.2 TextVQA scorer / normalization

```bash
cd /data/home/suchenghao/ToolVision

python -m unittest \
  offline_sft_pipeline/tests/test_sample_normalization.py \
  offline_sft_pipeline/tests/test_judge_backend.py
```

本轮在当前环境已跑过：

- `test_pipelines.py + test_orchestrator_v01.py`：`OK (skipped=2)`
- `test_sample_normalization.py + test_judge_backend.py + test_orchestrator_v01.py`：`23 tests OK`

---

## 10. 后续建议

下一步最值得做的是两件事：

1. 跑一小批真实 TextVQA sample，观察官方 scorer 上线后的分数分布是否主要集中在 `0.3 / 0.6 / 0.9 / 1.0`
2. 再决定 TextVQA 是否要单独把 `patience` 收到 `1`

也就是说：

- 当前先上官方 scorer
- 再用真实 rollout 行为决定 stop policy
- 不要在 scorer 还没稳定时提前动 `patience`
