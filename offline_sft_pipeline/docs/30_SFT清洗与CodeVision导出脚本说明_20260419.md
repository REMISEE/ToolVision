# SFT 清洗与 CodeVision 导出脚本说明

本文档记录当前用于复杂题 SFT 数据准备的 4 个脚本，它们的作用、输入输出，以及推荐执行顺序。

适用目录：

- 脚本目录：[scripts](/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts)
- 产物目录：[outputs/sft_prep](/data/home/suchenghao/ToolVision/offline_sft_pipeline/outputs/sft_prep)

## 总体流程

当前流程分成 4 步：

1. 从 pipeline run 中筛出答对的 answered trajectory
2. 对这些 trajectory 做 leak 标注
3. 对命中 leak 的 CoT 进行单独改写
4. 把清理后的 trajectory 导出成 CodeVision-SFT 风格目录

推荐顺序：

1. `collect_correct_answered_trajectories.py`
2. `annotate_complex_leak_signals.py`
3. `run_cot_leak_rewrite.py`
4. `export_codevision_sft_dataset.py`

说明：

- 第 1 步和第 2 步只是筛选和标注，不改原始 store。
- 第 3 步把需要改的 CoT 单独拿出来给模型改写，不改原始 store。
- 第 4 步在“导出层”应用 rewrite，生成训练用 json 和 images 目录，不回写原始 `messages.json`。

## 1. collect_correct_answered_trajectories.py

脚本：

- [collect_correct_answered_trajectories.py](/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/collect_correct_answered_trajectories.py)

作用：

- 从一个或多个 `dataset_pipeline_runs/...` 目录中，筛出“答对的 answered trajectory”
- 输出一个 jsonl，作为后续 leak 标注的输入

输入：

- 一个或多个 `--run-root`

输出：

- `<output-prefix>.jsonl`
- `<output-prefix>.summary.json`

主要字段：

- `run_root`
- `run_name`
- `dataset`
- `sample_id`
- `trajectory_id`
- `pred`
- `answer`
- `gt_score`
- `judge_score`

当前约定：

- `gqa` 走 exact match
- `textvqa` 走官方 soft scorer，通常使用 `--textvqa-min-score 0.9`
- 输出的是 answered trajectory，不是 root trajectory

## 2. annotate_complex_leak_signals.py

脚本：

- [annotate_complex_leak_signals.py](/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/annotate_complex_leak_signals.py)

作用：

- 对第 1 步产出的正确 answered trajectory 做 leak 标注
- 只标注，不删除样本，不修改原始 store

输入：

- `collect_correct_answered_trajectories.py` 产出的 `*.jsonl`

输出：

- `<output-prefix>.jsonl`
- `<output-prefix>.summary.json`
- `<output-prefix>.needs_edit.jsonl`

当前标注口径：

- `executor_step`
  - 不应提及 `planner`
  - 不应回显 hidden planning 字段名，例如 `Global CoT`、`Suggestion CoT`、`Step Goal`
- `final_answer`
  - 不应提及 `judge`
  - 不应提及 supplied answer source，例如 `reference answer`、`ground truth`
  - 不应提及 forced-answer / prompt policy 约束

输出关键字段：

- `detector_version`
- `leak_signal`
- `leak_scope`
- `executor_leak_signal`
- `final_answer_leak_signal`
- `leak_categories`
- `edit_targets`
- `assistant_checks`

其中：

- `edit_targets` 是人工修改时最常看的字段
- `.needs_edit.jsonl` 只保留需要修改的样本

## 3. run_cot_leak_rewrite.py

脚本：

- [run_cot_leak_rewrite.py](/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/run_cot_leak_rewrite.py)

配套 prompt：

- [cot_leak_rewrite_system_v01.txt](/data/home/suchenghao/ToolVision/offline_sft_pipeline/prompts/cot_leak_rewrite_system_v01.txt)

作用：

- 读取 `*.needs_edit.jsonl`
- 把每个需要改的 message 单独整理成一个 rewrite work item
- 可选择只准备输入，不调用模型
- 或调用 `qwen3.6-plus` 做“纯文字改写，不给题目上下文”

输入：

- `annotate_complex_leak_signals.py` 产出的 `*.needs_edit.jsonl`

输出目录结构：

- `<rewrite-dir>/items/<dataset>/<sample_id>/<trajectory_id>/<message_id>/`

每个 work item 下的文件：

- `metadata.json`
- `source_raw.md`
- `source_marked.md`
- `prompt_user.txt`
- `rewrite_text.md`
- `rewrite_result.json`
- `model_response.txt`
- `model_raw_payload.json`

其中：

- `source_marked.md` 会把需要修改的句子用明显标记包起来，便于人工检查
- `rewrite_text.md` 是模型给出的最终改写文本

顶层汇总文件：

- `<rewrite-dir>/results.jsonl`
- `<rewrite-dir>/summary.json`

说明：

- 这个脚本不修改原始 `messages.json`
- 真正应用 rewrite 的位置在第 4 步导出脚本

## 4. export_codevision_sft_dataset.py

脚本：

- [export_codevision_sft_dataset.py](/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/export_codevision_sft_dataset.py)

作用：

- 读取 annotated 样本
- 如果找到 rewrite 结果，就在导出时替换对应 assistant message 的 `<think>`
- 把 ToolVision 的多轮 trajectory 导出成 CodeVision-SFT 风格目录

输入：

- 一个或多个 `--input-jsonl`
- 零个或多个 `--rewrite-dir`

输出目录：

- `<output-dir>/codevision_sft.json`
- `<output-dir>/codevision_images/`
- `<output-dir>/dataset_info.snippet.json`
- `<output-dir>/export_summary.json`
- `<output-dir>/export_report.json`
- `<output-dir>/skipped_rows.json`

导出后的目录结构与 `CodeVision-SFT` 保持一致：

```text
your_export_dir/
  codevision_sft.json
  codevision_images/
    sample0_0.png
    sample0_1.png
    ...
```

当前导出规则：

- `conversations`
  - `user -> human`
  - `assistant -> gpt`
  - `tool -> tool`
- `images`
  - 写成相对路径，例如 `codevision_images/sample0_0.png`
- `metadata`
  - 当前只保留最小字段：
    - `sample_id`
    - `transform`
    - `question`
    - `answer`
    - `source_dataset`
    - `source_sample_id`
- `system`
  - 复用 CodeVision-SFT 第一条样本中的 system prompt

关于 rewrite 的应用：

- 如果某条样本命中了 leak，但 rewrite 目录里存在对应 `rewrite_text.md`
- 则导出时会替换该 message 的 `<think>...</think>`
- 原始 store 不会被修改

关于 tool message：

- tool 返回里如果带图，导出时会补回 CodeVision 原有的 follow-up 提示文本

## LLaMA-Factory 如何读取图片

当前确认的底层逻辑：

- `dataset_info.json` 放在 `dataset_dir` 目录下
- `file_name` 是相对于 `dataset_dir` 的 json 文件路径
- `media_dir` 默认等于 `dataset_dir`
- 所以 `images` 里写相对路径即可

例如：

```json
{
  "toolvision_codevision_sft": {
    "file_name": "codevision_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images",
      "system": "system"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "observation_tag": "tool"
    }
  }
}
```

也就是说，如果训练时：

- `dataset_dir=/path/to/your_export_dir`

那么：

- `codevision_sft.json` 会从 `/path/to/your_export_dir/codevision_sft.json` 读取
- `images` 里的 `codevision_images/sample0_0.png` 会被解析成：
  `/path/to/your_export_dir/codevision_images/sample0_0.png`

## 推荐实践

当前推荐做法：

1. 先分别导出各个数据源，不打乱
2. 等 rewrite 跑完之后，再重新导出对应数据源
3. 最后再做 merge
4. merge 完成之后再 shuffle

原因：

- 不打乱时更容易人工检查和定位问题
- rewrite 还没跑完时，可以先导出局部数据，不影响后续增量补齐
- merge 和 shuffle 放到最后一步最稳

## 当前不做的事

当前这些脚本都不会做下面的事：

- 不修改原始 `store/.../messages.json`
- 不修改原始 `executor_cot.md`
- 不直接合并你同学的 `CodeVision-SFT`
- 不在导出阶段自动 shuffle

这些动作应当在后续单独脚本中完成。
