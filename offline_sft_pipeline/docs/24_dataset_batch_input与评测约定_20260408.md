# 24 Dataset Batch Input 与评测约定

日期：2026-04-08  
状态：当前执行约定  
目的：冻结当前数据集批处理入口的最小输入格式、批处理脚本职责、日志约定，以及后续评测脚本的 TODO。

---

## 1. 当前结论

当前把数据集接入 offline pipeline，先拆成两层文件：

1. `root_samples.jsonl`
   - 只负责喂给 pipeline 跑生成
2. `eval_annotations.jsonl`
   - 只负责后续离线评测

也就是说：

- 生成和评测分开
- `RootSample.answer` 允许保留一个单字符串答案，便于简单数据集兼容
- 多参考答案、共识分数、count 的 MAE 规则，不放进主 pipeline 输入对象里

---

## 2. `root_samples.jsonl` 约定

每行一个 `RootSample` JSON object。

最小推荐格式：

```json
{
  "sample_id": "qga__val__000001",
  "question": "How many apples are there?",
  "images": [
    {
      "image_id": "img_0",
      "path": "/abs/path/to/000001.jpg"
    }
  ],
  "metadata": {
    "source_dataset": "qga",
    "source_split": "val",
    "source_sample_id": "000001"
  },
  "answer": "3"
}
```

说明：

- `sample_id`
  - 推荐格式：`<dataset>__<split>__<raw_id>`
- `question`
  - 保留原问题文本
- `images`
  - 当前统一用数组，即使只有一张图
- `images[].path`
  - 当前推荐直接写绝对路径，避免 cwd 差异
- `metadata`
  - 当前只保留轻量来源追踪信息
- `answer`
  - 当前允许保留一个单字符串答案
  - 如果数据集是多参考答案任务，也可以先写 `null`

当前主 pipeline 运行时真正依赖的是：

- `sample_id`
- `question`
- `images`

`metadata` 和 `answer` 当前不进入 planner / executor / runtime 的核心决策链路。

---

## 3. `eval_annotations.jsonl` 约定

这个文件留给后续 `evaluate_pipeline_run.py` 使用。

推荐格式：

```json
{
  "sample_id": "qga__val__000001",
  "metric": "count_mae",
  "references": ["3"]
}
```

如果是多参考答案任务：

```json
{
  "sample_id": "vstar__val__000001",
  "metric": "vqa_consensus",
  "references": ["cat", "a cat", "kitty", "..."]
}
```

当前先冻结 3 种候选 metric：

- `exact_match`
- `vqa_consensus`
- `count_mae`

---

## 4. `run_dataset_pipeline.py` 的职责

脚本位置：

- `offline_sft_pipeline/scripts/run_dataset_pipeline.py`

输入：

- `root_samples.jsonl`

输出：

- `<run_root>/store/`
- `<run_root>/logs/<sample_id>.log`
- `<run_root>/sample_results.jsonl`
- `<run_root>/run_summary.json`

行为约定：

1. 逐行读取 `root_samples.jsonl`
2. 每行校验为 `RootSample`
3. 调用同一套 real planner / real executor / real runtime wiring
4. 每个 sample 独立落盘到：
   - `store/samples/<sample_id>/...`
5. debug 输出写入：
   - `logs/<sample_id>.log`
6. 终端只打印最终 run summary，不刷每条 sample 的 debug

---

## 5. Resume 约定

当前 resume 规则固定为：

- 若 `store/samples/<sample_id>/trajectories/traj__<sample_id>__root/trajectory.json` 已存在
- 则认为这个 sample 已经启动过，`--resume` 模式下直接跳过

这条规则当前足够简单，也与现有 `OfflineTrajectoryStore.init_root_trajectory(...)` 的唯一 root trajectory 语义一致。

---

## 6. 当前不做的部分

当前先不做：

- shard 切分
- 多机调度
- 自动 evaluator
- GT-aware online judge
- committee judge hot path 集成

这些都留到后续单独实现。

---

## 7. `evaluate_pipeline_run.py` TODO

当前先不实现 evaluator 脚本，只冻结目标职责：

1. 读取 `eval_annotations.jsonl`
2. 读取某个 `run_root/store`
3. 对每个 `sample_id` 找 terminal answered trajectory
4. 取 `final_answer`
5. 按 `metric` 计算分数
6. 输出：
   - sample-level 结果
   - run-level 汇总指标

第一版建议最少输出：

- `total_samples`
- `answered_samples`
- `answer_rate`
- `end_to_end_accuracy`
- `accuracy_on_answered`
- `count_mae`（当任务是 count）

---

## 8. 文档边界

这份文档只记录：

- 数据集批处理输入格式
- 批处理脚本职责
- 日志和 resume 约定
- evaluator TODO

它不替代：

- `0_root_sample_最小输入定义_20260326.md`
- 后续真正的 evaluator 设计文档
