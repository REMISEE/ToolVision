# 1 单步 Runtime Wrapper 接口定义

日期：2026-03-26  
状态：v0.1 对齐稿  
目的：定义“executor 一步代码如何被执行并保存结果”这条最小闭环。

---

## 1. 先讲结论

这里的 `runtime wrapper` 只做一件事：

> 接收一段 executor 代码和当前可见图片，执行一次，然后产出一个标准化的 `runtime_result.json`。

更准确地说，它接收的是：

- “已经推进到当前这一步时”的真实输入状态
- 包括当前可见图片、当前 step 代码、当前 step 输出目录

然后真实调用 `CodeImageTool` 跑一次。

它不是：

- planner
- executor
- orchestrator
- exporter

它只负责“单步执行”。

它也不是 mock。

如果这层跑通，并且输入输出接口冻结下来，它就是最终 pipeline 里的真实执行组件，不是后面还要推倒重写的临时模拟器。

---

## 2. 责任边界

### 2.1 该做的事

1. 加载当前可见图片
2. 调 `CodeImageTool` 执行一次代码
3. 采集返回的 image / text / meta
4. 保存中间文件
5. 生成符合 `executor_runtime_result_schema.json` 的结果对象

### 2.2 不该做的事

1. 不负责调用 planner
2. 不负责调用 executor 模型
3. 不负责 frontier 更新
4. 不负责决定是否停掉 trajectory
5. 不负责直接改写 `trajectory.json`

这些应该由上层 orchestrator / store 处理。

---

## 3. 推荐输入接口

建议在 Python 层把请求对象收敛成下面这个结构：

```json
{
  "sample_id": "sample_000001",
  "trajectory_id": "traj_root",
  "round_idx": 0,
  "step_idx": 1,
  "executor_cot_path": "steps/step_001/executor_cot.md",
  "executor_code_path": "steps/step_001/executor_code.py",
  "visible_images": [
    {
      "artifact_id": "img_root_0",
      "path": "artifacts/root_0.png"
    }
  ],
  "image_index": 0,
  "step_output_dir": "steps/step_001"
}
```

---

## 4. 输入字段说明

### `sample_id`

用于跨系统追踪样本。

### `trajectory_id`

标识当前执行的是哪条 trajectory。

### `round_idx`

标识当前属于第几轮 planner 之后的执行。

### `step_idx`

标识当前是这条 trajectory 已执行的第几步。

### `executor_cot_path`

指向 executor 为当前 step 生成的局部 thought 文件。

V0.1 不是必须读取它，但建议保留路径，便于回放。

### `executor_code_path`

要执行的代码文件路径。

### `visible_images`

当前这一步能看到的图像列表。

这些图像的顺序就是后续 `image_index` 的索引顺序。

### `image_index`

默认从哪张图开始执行。

V0.1 建议保留这个字段，因为它直接对应 `CodeImageTool` 当前协议。

### `step_output_dir`

当前 step 的专属输出目录。

---

## 5. 推荐输出接口

runtime wrapper 的返回值建议只返回两类信息：

1. `runtime_result`
2. `saved_artifacts`

其中 `runtime_result` 本体必须符合：

- [executor_runtime_result_schema.json](/data/home/suchenghao/ToolVision/offline_sft_pipeline/schemas/executor_runtime_result_schema.json)

最小示例如下：

```json
{
  "schema_version": "0.1.0",
  "sample_id": "sample_000001",
  "trajectory_id": "traj_root",
  "round_idx": 0,
  "step_idx": 1,
  "created_at": "2026-03-26T10:00:00Z",
  "success": true,
  "images": [
    {
      "artifact_id": "img_step_001_0",
      "path": "steps/step_001/output_0.png",
      "media_type": "image/png",
      "width": 512,
      "height": 512
    }
  ],
  "text": "detected 1 region and returned 1 crop",
  "meta": {
    "model": "grounded_sam2",
    "operation": "dino_crop"
  },
  "observed_helper_call_count": 2,
  "observed_helper_calls": [
    {
      "order": 1,
      "name": "_call_ground_box",
      "status": "ok"
    },
    {
      "order": 2,
      "name": "_call_dino_crop",
      "status": "ok"
    }
  ],
  "code_execution": {
    "code_path": "steps/step_001/executor_code.py",
    "exit_code": 0,
    "started_at": "2026-03-26T10:00:00Z",
    "finished_at": "2026-03-26T10:00:02Z",
    "elapsed_seconds": 2.0,
    "stdout_path": "steps/step_001/stdout.txt",
    "stderr_path": "steps/step_001/stderr.txt"
  },
  "error": null
}
```

这里的 `text`、`meta`、`images` 都应该来自真实执行后的结果，而不是手工模拟值。

---

## 6. 当前 step 推荐落盘结构

建议固定为：

```text
steps/
`- step_001/
   |- executor_cot.md
   |- executor_code.py
   |- stdout.txt
   |- stderr.txt
   |- runtime_result.json
   |- output_0.png
   `- output_1.png
```

如果以后需要 traceback，也可以加：

- `traceback.txt`

---

## 7. 这个模块和 `messages.json` 的关系

runtime wrapper 自己不直接写 `messages.json`。

更合理的分工是：

1. runtime wrapper 只产出 `runtime_result.json`
2. 上层 orchestrator / store 根据它再追加：
   - assistant message
   - tool message

原因：

- runtime 层只关心执行
- message 线性化属于上层逻辑

---

## 8. helper 调用观测怎么做

V0.1 建议目标是“运行时观测”，不是纯代码静态解析。

理想做法：

- helper 被调用时记录
  - 顺序
  - 名称
  - 是否成功

如果短期还没接上这层埋点，可以先做最小版：

- 至少记录 helper 名和调用顺序

但字段先不要删，因为它对 judge 和回放都很有价值。

---

## 9. 为什么这一层是第一优先级

因为没有它，下面这些模块都拿不到真实执行结果：

- store
- judge
- exporter
- replay

它是第一条真实闭环：

`executor code -> runtime execute -> save image/text/meta -> runtime_result.json`

所以它一旦跑通，就不是“演示版”，而是可以直接接进最终 pipeline 的执行引擎。

---

## 10. 建议现在冻结的接口点

建议冻结：

1. 输入里一定有 `visible_images`
2. 输出里一定有 `runtime_result.json`
3. 输出里一定保存新生成图片
4. runtime wrapper 不直接改 `trajectory.json`
5. runtime wrapper 不直接决定 stop / keep

---

## 11. 一句话版本

单步 runtime wrapper 是整个 pipeline 的“单步执行引擎”，只管把一步真实跑出来，不管规划、不管分叉、不管导出。
