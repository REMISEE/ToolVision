# 6 CodeImageTool Helper 新增改动维护说明

日期：2026-03-27  
状态：active 维护文档  
目的：说明 `CodeImageTool` 当前 helper 机制已经做到哪一步、后续新增/改名/删除 helper 时应该改哪些地方、如何做最小验证。

---

## 1. 当前状态

这份文档对应的是“先补 tool，不补 wrapper”之后的状态。

也就是说，下面这些点现在已经成立：

1. helper trace 已经接入
2. `_execute_code()` 已经能返回：
   - `result`
   - `message`
   - `helper_result`
   - `helper_trace`
   - `stdout_text`
   - `stderr_text`
3. `execute()` 已经会把下面这些内容放进 metrics：
   - `observed_helper_call_count`
   - `observed_helper_calls`
   - `stdout_text`
   - `stderr_text`

对应代码位置：

- helper 注入与 trace 包装：
  [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L580)
- helper trace 统一记录入口 `_run_helper(...)`：
  [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L629)
- `_execute_code()` 扩展返回与 stdout/stderr 捕获：
  [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L856)
- `execute()` metrics 组装：
  [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L1021)

注意：

- 旧文档 [5_runtime_wrapper_最小实施说明_20260327.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/5_runtime_wrapper_%E6%9C%80%E5%B0%8F%E5%AE%9E%E6%96%BD%E8%AF%B4%E6%98%8E_20260327.md) 中关于“`observed_helper_calls` 还没补”的描述，已经只代表当时状态，不再代表当前代码。
- 当前成功态 `ToolResponse.text` 仍然会在真实文本后追加固定 follow-up 文案，位置在：
  [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L43)

---

## 2. 现在 helper 名字到底在哪里生效

一个 helper 的“对外名字”，当前其实同时影响三层：

### 2.1 executor code 里怎么调用

这是通过 `_create_safe_globals()` 注入的。

例如：

```python
safe_globals.update({
    "_call_ocr_assist": _call_ocr_assist,
    "_call_ground_box": _call_ground_box,
})
```

所以：

- 只有放进 `safe_globals.update(...)` 的 helper，executor code 才能直接调用
- 这里本质上就是当前 helper 的“注册点”

### 2.2 runtime trace 里显示什么名字

这是通过 `_run_helper("helper_name", ...)` 决定的。

例如：

```python
return _run_helper(
    "_call_ocr_assist",
    lambda: self._call_external_model(...),
)
```

所以：

- `observed_helper_calls[].name` 不是自动从函数名反射出来的
- 而是你手动传给 `_run_helper(...)` 的那个字符串

### 2.3 文档 / prompt / demo / 旧代码兼容

如果 helper 名字已经在这些地方出现了：

- planner / executor prompt
- demo script
- 手工 smoke code
- 旧轨迹里的 executor code

那么只改函数定义不够，外部引用也要一起改。

---

## 3. 后续改 helper 时，哪些情况需要改别处

### 3.1 只改 helper 内部实现，不改名字

例如：

- `_call_dino_crop()` 内部参数收敛
- `_call_blur_bg()` 的默认阈值调整
- helper 返回的 `meta` 更丰富

这种情况通常只要改 helper 内部实现，不需要改外部调用点。

前提是：

- helper 名字不变
- 返回协议仍然保持：

```python
{
    "image": ...,
    "images": ...,
    "text": ...,
    "meta": ...,
}
```

### 3.2 新增 helper，但复用已有 service backend

例如新增 `_call_ground_outline()`，底层还是走 `"grounded_sam2"`。

这种情况至少要改 3 处：

1. 在 `_create_safe_globals()` 里新增 helper 函数
2. 用 `_run_helper("新名字", ...)` 包起来
3. 在 `safe_globals.update(...)` 里暴露这个 helper

通常不需要改 service registry。

### 3.3 新增 helper，而且需要新的 service backend

例如新增 `_call_depth_estimate()`，但项目里还没有 depth service adapter。

这种情况除了 3.2 的 3 处，还要补：

1. 新的 adapter/client
2. `service_adapter_registry`
3. 对应 config

也就是说：

- helper 名字注册在 `safe_globals`
- 后端模型名注册在 `service_adapter_registry`

这是两层不同的“注册”。

### 3.4 helper 改名

例如把 `_call_ocr_assist` 改成 `_call_ocr_read`。

这种情况要一起考虑：

1. `safe_globals` 暴露名变了
2. `_run_helper("...")` 里的 trace 名也要变
3. 旧 executor code 里所有 `_call_ocr_assist()` 都会失效
4. prompt / demo / 文档里的名字也要同步

如果不想一次性打断旧代码，推荐先保留 alias。

### 3.5 删除 helper

如果删除 `_call_focus` 这类旧 alias，技术上很简单，但风险是旧代码会直接报：

```python
NameError: name '_call_focus' is not defined
```

所以删除前至少要确认：

1. 当前 prompt 样例里不再教模型用它
2. demo / smoke 不再用它
3. 历史轨迹是否还要 replay

---

## 4. 推荐改法：新增一个 helper 的最小步骤

下面用“新增 `_call_ground_outline()`，但继续复用 `grounded_sam2`”举例。

### 第一步：写 helper 函数

```python
def _call_ground_outline(
    text_prompt: str,
    image_index: Optional[int] = None,
    image_obj: Optional[Any] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    **kwargs,
):
    call_kwargs = {
        "_operation": "outline",
        "text_prompt": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    }
    call_kwargs.update(kwargs)
    return _run_helper(
        "_call_ground_outline",
        lambda: self._call_external_model(
            "grounded_sam2",
            _select_image(target_index=image_index, image_obj=image_obj),
            call_kwargs,
        ),
    )
```

### 第二步：暴露到 safe globals

```python
safe_globals.update({
    ...
    "_call_ground_outline": _call_ground_outline,
})
```

### 第三步：确认后端真的支持这个 operation

如果 service 端还不认识：

```python
"_operation": "outline"
```

那 helper 注入虽然成功，运行时还是会失败。

所以新增 helper 时要分清楚：

- Python 层 helper 是否已经接上
- service 端 operation 是否已经存在

---

## 5. 推荐改法：helper 改名时先做 alias

如果想把：

```python
_call_ocr_assist
```

迁到：

```python
_call_ocr_read
```

建议先分两步。

### 第一步：并存

```python
def _call_ocr_read(...):
    return _run_helper(
        "_call_ocr_read",
        lambda: self._call_external_model(...),
    )

def _call_ocr_assist(...):
    return _call_ocr_read(...)
```

然后在 `safe_globals.update(...)` 里同时暴露两个名字。

### 第二步：统一 prompt / demo / 文档

等新名字已经稳定后，再决定是否删除旧 alias。

如果一开始就硬改名，模型生成和旧 executor code 都会一起受影响。

---

## 6. 当前实现的几个注意点

### 6.1 helper trace 只记录最小信息

当前 trace 只记录：

```python
{"order": 1, "name": "_call_ocr_assist", "status": "ok"}
```

不会记录：

- 复杂参数
- 大图像 payload
- 大块中间结果

这是有意为之，先满足 runtime wrapper / judge / replay。

### 6.2 `_call_focus` 现在是独立 trace 名

当前 `_call_focus` 不再只是“调用一次 `_call_ground_box()` 然后复用它的 trace”。

它会在 trace 里明确记成：

```python
"_call_focus"
```

这样观测结果才和 executor code 真实写法一致。

### 6.3 stdout/stderr 已经在 worker 内捕获

这一步已经做完，所以后面的 wrapper 不需要再自己想办法从 Ray worker 外面抓 `print()`。

wrapper 只要消费：

- `metrics["stdout_text"]`
- `metrics["stderr_text"]`

即可。

### 6.4 `ToolResponse.text` 目前不是纯 runtime canonical text

当前成功态 `ToolResponse.text` 的拼法是：

```python
(helper_text or message) + SUCCESS_FOLLOWUP_TEXT
```

这意味着：

- 它同时承载了“真实 helper 语义”
- 也承载了“继续思考/继续调工具”的控制文案

所以后面如果要做更干净的 offline export，最好再拆：

1. runtime canonical text
2. prompt-only follow-up text

但这一步当前还没做。

---

## 7. 改完后最小验证什么

推荐至少做 3 级验证。

### 7.1 语法检查

```bash
python -m py_compile CodeVision/verl/tools/code_image_tool.py
```

### 7.2 直接测 `_execute_code()`

最简单的无 helper 测试要确认：

1. `stdout_text` 能拿到 `print()`
2. 没有 helper 时 `helper_trace == []`

最简单的 helper 测试要确认：

1. helper 成功时 trace 写 `ok`
2. helper 失败时 trace 写 `error`

### 7.3 再测 `execute()` 的 metrics 组装

至少确认这些字段真的会出现在 metrics：

```python
metrics["observed_helper_call_count"]
metrics["observed_helper_calls"]
metrics["stdout_text"]
metrics["stderr_text"]
```

---

## 8. 当前这一步到底完成了什么

如果按 [5_runtime_wrapper_最小实施说明_20260327.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/5_runtime_wrapper_%E6%9C%80%E5%B0%8F%E5%AE%9E%E6%96%BD%E8%AF%B4%E6%98%8E_20260327.md) 的分解来看：

当前已经完成的是：

### 第一步：给 `CodeImageTool` 补 helper trace

已经落地的内容：

1. `__helper_trace__`
2. helper 成功 / 失败记录
3. `_execute_code()` 扩展返回
4. `execute()` metrics 补 `observed_helper_calls`
5. worker 内 stdout/stderr 捕获

还没做的是：

### 第二步以后

1. `offline_sft_pipeline/runtime/code_image_runtime_wrapper.py`
2. request / output dataclass
3. `runtime_result.json` 落盘
4. wrapper smoke

---

## 9. 一句话版本

现在 helper 的“注册点”就是 `safe_globals.update(...)`，helper 的“观测名”就是 `_run_helper("...")`；以后新增或改名时，优先检查这两处，再看是否需要补新的 service adapter。
