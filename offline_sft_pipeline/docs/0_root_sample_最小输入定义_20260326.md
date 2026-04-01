# 0 Root Sample 最小输入定义

日期：2026-03-26  
状态：v0.1 对齐稿  
目的：冻结 offline pipeline 的入口样本最小字段，避免后续实现时 `trajectory` 不知道从哪里创建。

配套旧文：

- [pipeline_schema_explainer_20260326.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/pipeline_schema_explainer_20260326.md)
- [alignment_notes_20260326.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/alignment_notes_20260326.md)

---

## 1. 先讲结论

`root sample` 不是训练样本。

它只是 offline 生成 pipeline 的入口对象，用来回答：

- 这条样本的唯一 ID 是什么
- 问题是什么
- 初始可见图像有哪些
- 可选的标注信息是什么

它不应该直接带：

- `trajectory_id`
- `planner_history`
- `steps`
- `judge_records`
- `budget`

这些都属于后续 runtime / orchestrator 创建的状态对象。

---

## 2. 推荐的最小结构

V0.1 推荐把入口对象统一成下面这个形状：

```json
{
  "sample_id": "textvqa__train__000001",
  "question": "价格标签上写的数字是多少？",
  "images": [
    {
      "image_id": "img_0",
      "path": "inputs/sample_000001.png"
    }
  ],
  "metadata": {},
  "answer": null
}
```

---

## 3. 字段说明

### 3.1 必需字段

#### `sample_id: string`

要求：

- 在当前混合数据仓里全局唯一
- 稳定，不随重跑变化
- 最好能看出样本来源

用途：

- 创建输出目录
- 和 trajectory、judge、export 建立关联

推荐做法：

- 如果后续会混合多个 VQA / OCR / grounding 数据集，建议直接带数据集命名空间
- 推荐格式：

```text
<dataset_name>__<split>__<raw_sample_id>
```

例如：

- `textvqa__train__000001`
- `chartqa__val__128`
- `docvqa__test__abc123`

如果你不想把来源写进 `sample_id`，那至少也要在 `metadata` 里带：

- `source_dataset`
- `source_split`
- `source_sample_id`

#### `question: string`

要求：

- 保留原始问题文本
- 不要在这里掺入 system prompt 或 tool 描述

用途：

- 后续进入 `messages.json` 的 user turn

#### `images: list`

V0.1 即使大部分任务只有一张图，也建议统一成数组。

每个元素最小要求：

```json
{
  "image_id": "img_0",
  "path": "inputs/sample_000001.png"
}
```

原因：

- 以后更容易支持多图任务
- 避免单图、多图两套入口格式

其中：

- `image_id` 是 root sample 内部引用 ID
- `path` 是图片路径

### 3.2 可选字段

#### `metadata: object`

用于保存：

- 数据源
- 数据集名 / split / 原始样本 ID
- 原始任务类型
- bbox hint
- transform hint
- 其他追踪信息

#### `answer: string | null`

这不是生成时必须字段。

但如果数据源自带标准答案，建议保留，方便：

- 线下评估
- judge 对照
- exporter 打标签

---

## 4. 为什么不用更简单的 `image_path`

如果只写：

```json
{
  "sample_id": "...",
  "question": "...",
  "image_path": "..."
}
```

虽然更短，但会带来两个问题：

1. 后面支持多图时要改入口格式。
2. 后面 root image artifact 的引用关系不够清楚。

所以建议从一开始就统一成：

- `images: [...]`

即使只有一张图，也放单元素数组。

---

## 5. root sample 到 trajectory 的转换

收到 root sample 后，orchestrator 要做的不是直接运行 planner，而是先做初始化。

初始化后的产物通常包括：

1. `trajectory.json`
2. `messages.json`
3. root image artifact

例如：

```json
[
  {
    "message_id": "m_sys",
    "role": "system",
    "content": "You are a helpful vision tool-use assistant.",
    "image_artifact_ids": [],
    "metadata": {}
  },
  {
    "message_id": "m_user",
    "role": "user",
    "content": "价格标签上写的数字是多少？",
    "image_artifact_ids": ["img_root_0"],
    "metadata": {}
  }
]
```

此时才进入第 0 轮 planner。

---

## 6. V0.1 不建议放进 root sample 的字段

先不要放：

- `trajectory_id`
- `round_idx`
- `step_idx`
- `status`
- `pending_execution`
- `planner_output`
- `judge`
- `tools`
- `messages`

原因：

- root sample 是输入对象
- 这些都是运行态状态对象

如果一开始把它们混在一起，后续 store 和 exporter 会很乱。

---

## 7. 推荐的目录内落盘方式

如果后续做成 JSONL 输入，推荐每行一个 root sample：

```json
{"sample_id":"textvqa__train__000001","question":"价格标签上写的数字是多少？","images":[{"image_id":"img_0","path":"inputs/sample_000001.png"}],"metadata":{"source_dataset":"textvqa","source_split":"train","source_sample_id":"000001"},"answer":null}
```

如果先手工测试，也可以用单文件 JSON。

---

## 8. 当前建议冻结的最小字段

建议现在就冻结：

1. `sample_id`
2. `question`
3. `images`
4. `metadata`
5. `answer`

其中前 3 个必需，后 2 个可选。

如果后续确认一定是多数据集混跑，建议把 `sample_id` 的命名空间规则也一并冻结。

---

## 9. 一句话版本

`root sample` 只负责描述“这道题和它的初始图片”，不要提前混入 trajectory、planner、judge 的运行态字段。
