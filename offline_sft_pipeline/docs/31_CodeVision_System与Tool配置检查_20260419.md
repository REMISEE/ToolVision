# CodeVision System 与 Tool 配置检查

日期：2026-04-19

## 结论

当前导出的 `CodeVision-SFT` 风格数据，**格式是对的**，可以被 CodeVision 现有的 LLaMA-Factory SFT 流程读取。

但当前 `system` 内容是**旧版 CodeVision prompt**，和我们现在导出的数据行为已经不一致。主要问题有：

1. 没覆盖我们新增的外部工具与 helper。
2. `flip` 示例和真实数据常用写法不一致。
3. 没说明新的 `image_index` / 可见图片时间线语义。
4. RL / eval 路径不会直接信任数据里的 `system`，而是强制替换成 `sp.txt`，所以只改数据集不够。

## 1. 当前数据格式是否正确

原始 CodeVision-SFT 样本格式位于：

- `/data/home/suchenghao/ToolVision/CodeVision-SFT/codevision_sft.json`

样本顶层字段是：

- `conversations`
- `images`
- `metadata`
- `system`

我们当前导出的样本与这个结构一致，训练格式本身没有问题。

## 2. SFT 是否会读取数据里的 `system`

会。

CodeVision 的 SFT 配置位于：

- `/data/home/suchenghao/ToolVision/CodeVision/LLaMA-Factory/examples/train_full/qwen3vl.yaml`

其中指定：

- `dataset: codevision_sft`

而数据字段映射位于：

- `/data/home/suchenghao/ToolVision/CodeVision/LLaMA-Factory/data/dataset_info.json`

这里 `codevision_sft` 的配置是：

- `messages -> conversations`
- `images -> images`
- `system -> system`

也就是说，**LLaMA-Factory 的 SFT 会读取样本里的 `system` 字段**。

## 3. RL / eval 路径是否也读取这个 `system`

不完全是。

CodeVision 的 RL recipe 位于：

- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/qwen3_vl.sh`
- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/eval.sh`

这两个脚本都显式打开了：

- `+data.replace_system_prompt=True`
- `+data.new_sp_path=recipe/codevision/config/sp.txt`

对应实现位于：

- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/uvtr.py`

`uvtr.py` 会在读取消息后，把第一条 system message 直接替换成：

- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/config/sp.txt`

所以结论是：

- **SFT 路径会读数据里的 `system`**
- **RL / eval 路径会强制改成 `sp.txt`**

因此如果后面要统一新版 prompt，至少要改两个地方：

1. 数据导出时写入的 `system`
2. `recipe/codevision/config/sp.txt`

## 4. 当前 system 的主要不一致点

当前 system 内容来源还是旧版 CodeVision 提示，里面嵌了旧的 `<tools>` 块。它的问题不是“格式错误”，而是“内容过时”。

### 4.1 新工具 / 外部工具没有写进去

当前 RL tool 配置文件位于：

- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/config/code_image_tool_config.yaml`

里面已经支持很多新的 helper：

- `_call_ocr_assist`
- `_call_manual_box`
- `_call_manual_crop`
- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`
- `_call_manual_depth`
- `_call_ground_depth`
- `_call_count_assist`

同时还配置了外部服务：

- `paddleocr`
- `grounded_sam2`
- `depth`
- `countgd`

但当前 `sp.txt` 仍然只是旧版泛化描述，没有把这些能力说明清楚。

### 4.2 `flip` 写法不一致

当前 `code_image_tool_config.yaml` 中的示例写法仍然是：

- `image.transpose(Image.FLIP_LEFT_RIGHT)`
- `image.transpose(Image.FLIP_TOP_BOTTOM)`

但原始 CodeVision-SFT 数据里，真实常见写法主要是：

- `ImageOps.flip(image)`

检查结果：

- `ImageOps.flip`：253 次
- `FLIP_TOP_BOTTOM`：0 次
- `FLIP_LEFT_RIGHT`：0 次

这说明 **prompt / tool 示例和训练数据中的真实分布不一致**。

这部分应该统一，否则模型会同时学到两套不同表述。

### 4.3 `image_index` / 图片时间线语义需要进 prompt

当前新工具配置已经写明：

- `image_index` 是可见图片时间线上的索引
- root image 先出现
- 每次成功 tool 返回的图会追加到后续可见列表

这是我们导出数据非常关键的行为语义。

但旧 system 没把这个规则说清楚。

如果不补进去，模型对多张图和 tool 返回图的索引关系会学得不稳定。

### 4.4 新导出数据中确实在用这些新能力

以我们导出的复杂题数据为例，已经实际出现：

- `image_index`
- `_call_manual_crop`
- `_call_dino_crop`
- `_call_ocr_assist`
- `_call_count_assist`
- `_call_ground_depth`

所以这不是“未来可能会用”，而是**当前训练数据已经在用**。

## 5. 除了 system，还应该看哪里

### 5.1 `sp.txt`

这是 RL / eval 真正生效的 system prompt 文件：

- `/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/config/sp.txt`

如果后面只改导出数据里的 `system`，RL / eval 仍然会继续使用旧版 prompt。

### 5.2 `code_image_tool_config.yaml`

这个文件本身已经比 `sp.txt` 新，但里面的示例文本仍然需要和最终 system 对齐，尤其是：

- `flip` 示例
- helper 使用说明
- `image_index` 规则

否则会出现：

- system 一套说法
- tool schema 另一套说法
- 数据里第三套写法

### 5.3 数据导出脚本中的 system 来源

当前导出脚本会把旧版 CodeVision-SFT 里的 `system` 直接复制到新样本里。

相关脚本：

- `/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/export_codevision_sft_dataset.py`
- `/data/home/suchenghao/ToolVision/offline_sft_pipeline/scripts/export_easy_codevision_sft_dataset.py`

所以后面如果要换新版 system，不能只改 CodeVision repo 里的 `sp.txt`，还要同步改导出脚本的 system 来源。

## 6. 当前最小必要修改范围

如果目标是“让 SFT / RL / eval 都一致地学到我们的新版工具行为”，最少需要改这三处：

1. `offline_sft_pipeline` 导出脚本使用的 `system` 文本来源
2. `CodeVision/recipe/codevision/config/sp.txt`
3. `CodeVision/recipe/codevision/config/code_image_tool_config.yaml` 中与 system 不一致的示例与说明

## 7. 建议纳入新版 system / tool 文案的点

后面改 prompt 时，至少要把下面这些点写进去：

1. 支持外部 helper，而不只是传统 PIL 基础操作。
2. `image_index` 的时间线规则：
   - root image 先出现
   - tool 返回图会追加到可见列表
3. 优先用最直接、最可解释的操作。
4. 可以组合 helper 与普通 PIL / NumPy / OpenCV 操作。
5. `flip` 示例统一成和真实训练分布一致的写法。
6. OCR / grounding / counting / depth 都属于可直接调用的能力。

## 8. 当前判断

当前状态可以概括为：

- **数据格式正确**
- **字段映射正确**
- **system 会被 SFT 读取**
- **RL / eval 会覆盖 system**
- **真正需要改的是 prompt 内容的一致性，而不是数据结构**

所以接下来的动作应该不是改 schema，而是：

1. 先定新版 system 文案
2. 再同步改：
   - 导出脚本的 system 来源
   - `sp.txt`
   - `code_image_tool_config.yaml` 的文字说明与示例

在这之前，不建议继续生成最终 merged 数据集，否则后面还要重导一次。
