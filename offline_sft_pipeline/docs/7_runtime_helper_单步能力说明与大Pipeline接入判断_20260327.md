# 7 Runtime Helper 单步能力说明与大 Pipeline 接入判断

日期：2026-03-27  
状态：v0.1 说明文档  
目的：把当前已经跑通的 runtime-helper 功能讲清楚，明确它在大 pipeline 里的位置、当前 smoke 输出应该怎么理解、哪些部分已经可接入、哪些部分还没完成。

---

## 1. 一句话结论

当前已经跑通的是：

> “给一段 executor 单步代码和当前可见图片，用 `CodeImageTool` 真执行一次，并把真实 `image / text / meta / helper trace` 落成 `runtime_result.json`” 这条链。

这意味着：

- 它已经可以作为大 pipeline 里的 **runtime / sandbox execution 底座**
- 但它 **不等于整个 offline branching pipeline 已经完成**

当前还没有在这次 smoke 中证明的部分包括：

- planner 真调用
- executor 模型真调用
- trajectory / messages / frontier / judge 的完整更新
- planner -> executor -> runtime -> judge/frontier -> planner 的 rolling replanning 闭环

---

## 2. 这份文档要解释什么

这份文档主要解释 4 件事：

1. 当前“这个功能”到底是什么
2. 它和大 pipeline 的关系是什么
3. 这次 `runtime_helper_smoke` 里 3 个 step 应该怎么理解
4. planner / executor 在真实使用时应该长什么样

---

## 3. 当前功能到底是什么

当前功能的核心组件是：

- `offline_sft_pipeline/runtime/code_image_runtime_wrapper.py`
- `CodeVision/verl/tools/code_image_tool.py`
- `CodeVision/verl/external_services/groundedsam2/*`
- OCR / GroundedSAM2 的 service adapter

它负责做的事情是：

1. 读取当前 step 的 executor 代码
2. 读取当前 step 可见图片
3. 在安全执行环境中运行代码
4. 允许代码调用 helper，例如：
   - `_call_ocr_assist(...)`
   - `_call_sam_mask(...)`
   - `_call_dino_crop(...)`
5. 收集真实执行结果：
   - 输出图片
   - 输出文本
   - 输出结构化 `meta`
   - helper 调用顺序和状态
   - stdout / stderr
6. 落盘为标准 `runtime_result.json`

它 **不负责**：

- 规划下一步
- 选择分支
- 判断 trajectory 是否停止
- 直接改写 `trajectory.json`
- 直接改写 `messages.json`

也就是说，它就是单步执行引擎，不是 planner，不是 orchestrator。

---

## 4. 它在大 Pipeline 里的位置

大 pipeline 的目标循环是：

`planner -> executor -> runtime -> judge/frontier -> planner`

其中当前已跑通的是 `runtime` 这一格。

各层职责如下：

### 4.1 planner

planner 负责：

- 读取当前 trajectory 完整历史
- 判断现在能不能直接回答
- 如果还不能，提出 2 到 3 条候选路线
- 尽量把路线分歧提前解决

planner 不写代码。

### 4.2 executor

executor 负责：

- 读取当前被选中的 suggestion
- 只处理当前一步
- 输出：
  - 当前 step 的局部 thought
  - 当前 step 的可执行 Python 代码

executor 不负责把整条 suggestion 一路机械跑完。

### 4.3 runtime

runtime 负责：

- 执行 executor 生成的当前一步代码
- 跑 helper
- 产出真实图像 / 文本 / meta / trace
- 落盘 `runtime_result.json`

### 4.4 judge / frontier

judge/frontier 负责：

- 判断这条轨迹是否值得继续扩展
- 或是否应该剪枝 / 提前停止 / 标记可导出

所以现在的正确说法是：

- runtime 层已经能接入大 pipeline
- 整个大 pipeline 还没有端到端接完

---

## 5. 当前 helper 的语义

当前 executor 代码里可直接调用的 helper，本质上是稳定能力接口，不应该被底层模型实现绑死。

这次 smoke 涉及了 3 个 helper：

### 5.1 `_call_ocr_assist(...)`

作用：

- 对输入图像做 OCR
- 返回 OCR 可视化图、OCR 文本、OCR 结构化结果

典型返回：

```python
{
    "image": PIL.Image,
    "images": [PIL.Image, ...],
    "text": "18",
    "meta": {
        "model": "paddleocr_http",
        "ocr_result": ...,
        "ocr_pages": ...,
        "num_ocr_items": 1
    }
}
```

### 5.2 `_call_sam_mask(...)`

作用：

- 通过 GroundedSAM2 根据文本提示词做定位 + mask
- 返回半透明高亮图

典型返回：

```python
{
    "image": PIL.Image,
    "images": [PIL.Image],
    "text": "GroundedSAM2(mask) generated 1 masks.",
    "meta": {
        "model": "grounded_sam2",
        "operation": "mask",
        "annotations": ...,
        "mask_scores": ...
    }
}
```

### 5.3 `_call_dino_crop(...)`

作用：

- 通过 GroundedSAM2 先定位目标，再按 `box` 或 `mask` 返回局部 crop

典型返回：

```python
{
    "image": PIL.Image,
    "images": [PIL.Image, ...],
    "text": "GroundedSAM2(dino_crop) returned 1 crop images.",
    "meta": {
        "model": "grounded_sam2",
        "operation": "dino_crop",
        "crop_boxes": ...
    }
}
```

---

## 6. `runtime_helper_smoke` 的真实含义

这次 smoke 脚本是：

- `offline_sft_pipeline/scripts/run_runtime_helper_smoke.py`

它做的事不是“跑一条真实多轮 trajectory”，而是：

> 对同一张 root 图，手工构造多个单步 case，逐个执行，验证 runtime + helper + 落盘链路。

也就是说，这个 smoke 的 3 个 case 是 **并列单步例子**，不是连续三步推理。

---

## 7. 这 3 个 Step 应该怎么理解

### 7.1 Step 1：`ocr_only`

代码语义：

- 直接对整图做 OCR
- 返回 OCR 可视化图

它验证的是：

- OCR helper 是否可用
- OCR text / meta 是否能正确写入 `runtime_result.json`

### 7.2 Step 2：`sam_mask`

代码语义：

- 用 `"sign"` 做 GroundedSAM2 mask
- 返回 mask 高亮图

它验证的是：

- GroundedSAM2 helper 是否可用
- `text / meta / annotations / mask_scores` 是否能正确回传
- helper trace 是否正确记录

### 7.3 Step 3：`mask_crop_then_ocr`

代码语义：

1. 先按 `"sign"` 做 DINO crop
2. 再对 crop 图做 OCR
3. 最后把 crop 图作为本 step 的最终输出图

它验证的是：

- 一个 step 内多次 helper 调用是否成立
- trace 顺序是否正确
- GroundedSAM2 的中间产物能否作为 OCR 输入
- 最终 `runtime_result.text/meta` 是否来自最后一次 helper

所以 Step 3 不是“两步 runtime”，而是：

> 一个 step 里串了两个 helper 的组合操作

这正是大 pipeline 设计里允许的模式。

---

## 8. 这 3 个 Step 是顺着的吗

不是。

虽然输出目录名是：

- `step_001`
- `step_002`
- `step_003`

但这只是 smoke 脚本为了方便阅读给出的顺序编号，不代表它们构成同一条真实 trajectory 的连续三步。

关键点是：

- 每个 case 的输入 `visible_images` 都是同一张 `root_0.png`
- `step_002` 没有读取 `step_001` 的输出图
- `step_003` 也没有读取 `step_002` 的输出图

因此它们更像：

- planner 可能给出的 3 种候选单步路线

而不是：

- 同一条轨迹中已经被 planner 连续选中的 3 个连续 step

---

## 9. 为什么 Step 3 没有 OCR 返图

这是这次最容易误解的点。

Step 3 的 executor 代码逻辑是：

```python
crop = _call_dino_crop(...)
ocr = _call_ocr_assist(image_obj=crop["image"], ...)
result = crop["image"]
```

这里最后一行显式指定：

- 本 step 最终输出图像 = `crop["image"]`

而 runtime 当前的设计是：

1. **最终输出图像** 取 executor 代码最后的 `result`
2. **最终文本和 meta** 优先取最后一次 helper 调用结果

所以 Step 3 的结果是一个“混合型 step”：

- 输出图：crop 图
- 输出 text：OCR 文本 `"18"`
- 输出 meta：OCR 结构化结果

这不是 bug，而是当前 executor 写法和 runtime 语义共同决定的结果。

如果想让 Step 3 返回 OCR 标注图，只需要写成：

```python
result = ocr["image"]
```

如果想同时保存 crop 图和 OCR 图，则需要在更高层定义：

- 当前 step 的“主输出图”是什么
- 中间 helper 图是否也要显式落盘

这个问题在正式接大 pipeline 前应该尽早冻结，否则下游会对 `images` 和 `meta` 的配对关系产生歧义。

---

## 10. 这次例子代表的真实任务场景

这个 smoke 使用的图像里，有一个类似标签/牌子的局部区域，OCR 读出来的数字是 `18`。

它对应的真实任务很像：

- “价格标签上写的数字是多少？”
- “牌子上的年份/数字是什么？”
- “画面里这个 sign 上写了什么？”

这类任务的合理路线通常是：

1. 先判断整图 OCR 是否已经够用
2. 如果不够，再定位目标区域
3. 对局部区域做 crop / 放大 / OCR

所以这次 3 个 case 对应的是 3 种常见单步策略：

### 路线 A：直接整图 OCR

适合：

- 目标文字已经足够大
- 背景不复杂
- 不需要先定位

### 路线 B：先做 mask/定位可视化

适合：

- 需要先确认目标区域是否找对
- 下一轮 planner 需要看到被高亮的目标

### 路线 C：先 crop，再 OCR

适合：

- 整图 OCR 不稳
- 目标很小
- 需要去背景干扰后再读文字

---

## 11. 一个更像真实 Planner 的输出应该是什么样

对于“牌子上的数字是多少”这种问题，更像真实 planner 的输出可以是：

```json
{
  "can_answer_now": false,
  "global_chain_cot": "先尝试直接读取；如果整图 OCR 不稳，再先定位牌子再裁剪再 OCR。",
  "suggestions": [
    {
      "suggestion_id": "s1",
      "suggestion_cot": "直接 OCR。",
      "steps": [
        {
          "step_id": "step_a",
          "step_goal": "直接读取图中的数字",
          "capability_plan": [
            {"order": 1, "capability": "_call_ocr_assist", "instruction": "直接对当前图做 OCR"}
          ],
          "executor_instruction": "写代码直接对当前图做 OCR，返回 OCR 可视化图。"
        }
      ]
    },
    {
      "suggestion_id": "s2",
      "suggestion_cot": "先定位 sign，再 crop，再 OCR。",
      "steps": [
        {
          "step_id": "step_b",
          "step_goal": "裁出 sign 区域并读取其文字",
          "capability_plan": [
            {"order": 1, "capability": "_call_dino_crop", "instruction": "定位 sign 并返回单个最可信 crop"},
            {"order": 2, "capability": "_call_ocr_assist", "instruction": "对 crop 图做 OCR"}
          ],
          "executor_instruction": "先 crop sign，再 OCR crop。返回最有助于下一轮判断的图。"
        }
      ]
    },
    {
      "suggestion_id": "s3",
      "suggestion_cot": "先 mask 确认目标，再看是否继续。",
      "steps": [
        {
          "step_id": "step_c",
          "step_goal": "确认 sign 的位置是否可靠",
          "capability_plan": [
            {"order": 1, "capability": "_call_sam_mask", "instruction": "高亮 sign 区域"}
          ],
          "executor_instruction": "用 mask 高亮 sign，返回高亮图和定位信息。"
        }
      ]
    }
  ]
}
```

这个例子里：

- `s1` 对应 Step 1
- `s3` 对应 Step 2
- `s2` 对应 Step 3

所以把这次 smoke 看成“planner 的三个候选单步解的手工模拟”是对的。

---

## 12. 一个更像真实 Executor 的写法应该是什么样

### 12.1 如果 planner 选中“直接 OCR”

executor 局部 thought 可以是：

```text
The target is a sign-like text region. First try direct OCR on the whole image because the text may already be legible.
```

代码可以是：

```python
ocr = _call_ocr_assist(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    visualize=True,
)
print("ocr_text:", ocr["text"])
result = ocr["image"]
```

### 12.2 如果 planner 选中“先 crop 再 OCR”

executor 局部 thought 可以是：

```text
The text may be too small in the full image. First crop the sign region, then run OCR on the crop.
```

代码可以是：

```python
crop = _call_dino_crop("sign", based_on="box", max_crops=1, padding=4)
print("crop_text:", crop["text"])
ocr = _call_ocr_assist(image_obj=crop["image"], visualize=True)
print("crop_ocr_text:", ocr["text"])
result = ocr["image"]
```

注意这里把最后一行写成 `ocr["image"]`，这样最终返图就是 OCR 标注图，而不是单纯 crop 图。

### 12.3 如果 planner 选中“先看 mask 再决定”

executor 局部 thought 可以是：

```text
First verify that the sign region can be reliably localized. A mask view is useful for the next planning round.
```

代码可以是：

```python
mask = _call_sam_mask("sign", multimask_output=False)
print("mask_text:", mask["text"])
print("mask_meta:", mask["meta"])
result = mask["image"]
```

---

## 13. 这次功能是否满足接入要求

对 runtime 层来说，当前已经基本满足接入要求。

### 13.1 已满足

1. 能读 executor 代码并单步执行
2. 能读取当前可见图片
3. 能通过 helper 调 OCR 和 GroundedSAM2
4. 能采集真实 `images / text / meta`
5. 能记录 `observed_helper_calls`
6. 能保存 `stdout / stderr`
7. 能落标准 `runtime_result.json`
8. runtime 没越权去改 trajectory / messages

### 13.2 已验证的具体能力

1. OCR helper 单步成功
2. GroundedSAM2 mask helper 单步成功
3. 单 step 内串联多个 helper 成功
4. GroundedSAM2 -> OCR 的链式数据传递成功
5. GroundedSAM2 本地路径问题已修复，不再依赖启动 cwd

### 13.3 还未完全验证

1. planner 真输出接到 executor 真代码生成
2. runtime 结果写回 `messages.json`
3. trajectory store / resume
4. frontier / judge
5. exporter
6. 多轮 replanning

因此：

- **可以接入 runtime 层**
- **还不能宣称整个 pipeline 已经端到端 ready**

---

## 14. 建议的下一步

建议后续按下面顺序推进：

### 第一步：冻结当前 runtime 语义

尤其要明确：

1. step 最终输出图到底取什么
2. 最后一个 helper 的 `text/meta` 是否默认成为 step 的 `text/meta`
3. 中间 helper 图是否要额外落盘

### 第二步：补 trajectory / messages 的真实写回

把 runtime_result 接到：

- `trajectory.json`
- `messages.json`
- planner round 历史

### 第三步：接 planner / executor 真循环

做一次最小闭环：

1. planner 给 2 到 3 个 suggestion
2. 选一条
3. executor 只写当前一步
4. runtime 执行
5. planner 重新读取新结果再规划

### 第四步：补 judge / frontier

至少先有：

- cheap filter
- keep / prune
- terminal state

---

## 15. 当前功能的正确定位

如果用一句工程化的话来描述当前状态：

> 现在已经完成了 offline branching SFT pipeline 中最关键、也最先该落地的“单步真实执行引擎”；它已经不是 demo mock，而是未来正式 runtime 组件的雏形。

但如果用项目整体视角来描述：

> 当前主要完成的是“schema + 文档 + runtime/helper 执行层”，而不是“完整 planner-executor-judge-export end-to-end 系统”。

---

## 16. 一句话版本

这次 `runtime_helper_smoke` 的 3 个 step，本质上是对同一张 root 图手工模拟出的 3 个候选单步策略；它已经证明 runtime/helper 层可以接入大 pipeline，但还没有证明整条 branching pipeline 已经跑通。
