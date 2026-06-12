# Benchmark Comparison

分数统一为 `accuracy x 100`。  
如果某一列是 `-`，表示该论文或本地没有这组数。

---

## 1. 主表：base / final 对照

这张表只放最核心的对照列：

- `Local 3VL-8B Base`：你本地 `Qwen3-VL-8B-Thinking` base 结果
- `Local 3VL-8B SFT`：你本地当前 SFT checkpoint 结果
- `THYME 2.5-7B Base`：THYME 论文里的 `Qwen2.5-VL-7B`
- `THYME Final`：THYME 最终方法
- `CodeVision 3VL-8B Base`：CodeVision 论文里的 `Qwen3-VL-8B-Thinking`
- `CodeVision 3VL-8B Final`：CodeVision 最终方法

### 1.1 可以直接比的 benchmark

| Benchmark | Split | N | Local 3VL-8B Base | Local 3VL-8B SFT | THYME 2.5-7B Base | THYME Final | CodeVision 3VL-8B Base | CodeVision 3VL-8B Final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V* | Attribute | 115 | 67.8 | 80.0 | - | 83.5 | - | - |
| V* | Spatial | 76 | 52.6 | 77.6 | - | 80.3 | - | - |
| V* | Overall | 191 | 61.8 | 79.1 | 76.4 | 82.2 | 77.5 | 82.4 |
| HRBench-4K | Single / FSP | 400 | - | 85.0 | 85.2 | 91.0 | - | - |
| HRBench-4K | Cross / FCP | 400 | - | 64.3 | 52.2 | 63.0 | - | - |
| HRBench-4K | Overall | 800 | - | 74.6 | 68.8 | 77.0 | 72.4 | 77.1 |
| HRBench-8K | Single / FSP | 400 | - | 75.3 | 78.8 | 86.5 | - | - |
| HRBench-8K | Cross / FCP | 400 | - | 60.3 | 51.8 | 57.5 | - | - |
| HRBench-8K | Overall | 800 | - | 67.8 | 65.3 | 72.0 | 68.1 | 73.4 |

### 1.2 只能参考，不能和 CodeVision 论文硬比的 benchmark

原因：

- CodeVision 对 OCRBench 用的是变换版 `OCRBench-T`
- CodeVision 对 ChartQA 用的是 `ChartQAPro-T`

所以这里只保留本地和 THYME。

| Benchmark | Split | N | Local 3VL-8B Base | Local 3VL-8B SFT | THYME 2.5-7B Base | THYME Final | CodeVision 3VL-8B Base | CodeVision 3VL-8B Final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | Human | 1250 | - | 82.5 | 72.5 | 80.0 | - | - |
| ChartQA | Machine | 1250 | - | 83.8 | 94.9 | 92.2 | - | - |
| ChartQA | Overall | 2500 | - | 83.1 | 83.7 | 86.1 | - | - |
| OCRBench | Regular Text | 50 | - | 96.0 | - | - | - | - |
| OCRBench | Irregular Text | 50 | - | 88.0 | - | - | - | - |
| OCRBench | Artistic Text | 50 | - | 88.0 | - | - | - | - |
| OCRBench | Handwriting | 50 | - | 66.0 | - | - | - | - |
| OCRBench | Digit String | 50 | - | 76.0 | - | - | - | - |
| OCRBench | Non-Semantic Text | 50 | - | 84.0 | - | - | - | - |
| OCRBench | Scene Text VQA | 200 | - | 80.0 | - | - | - | - |
| OCRBench | Doc VQA | 200 | - | 79.0 | - | - | - | - |
| OCRBench | KIE | 200 | - | 76.0 | - | - | - | - |
| OCRBench | Handwritten Math Expr | 100 | - | 52.0 | - | - | - | - |
| OCRBench | Overall | 1000 | - | 77.1 | 88.4 | 86.3 | - | - |
| CountBench | Overall | 510 | - | 90.8 | - | - | - | - |
| FSC147 | Val | 1286 | - | 75.1 | - | - | - | - |
| FSC147 | Test | 1190 | - | 76.5 | - | - | - | - |

---

## 2. 论文里的 SFT 阶段结果

这张表只回答一个问题：  
**论文里报出来的 SFT checkpoint 到底是什么水平。**

### 2.1 THYME 论文里的 SFT

THYME 的 SFT 表里，最接近“最终 SFT checkpoint”的是 `Math Data Annealing` 这一行。

| Benchmark | Split | THYME 2.5-7B Base | THYME SFT | THYME Final |
|---|---|---:|---:|---:|
| V* | Overall | 76.40 | 79.58 | 82.2 |
| HRBench-8K | Overall | 65.50 | 65.12 | 72.0 |

说明：

- THYME 论文没有给 V* 的 split 级 SFT 数据
- THYME 论文也没有给 HRBench-4K 的 SFT 数据

### 2.2 CodeVision 论文里的 SFT

CodeVision 论文的 SFT ablation 是 `Qwen2.5-VL-7B-SFT`。  
它报的是 7B，不是 8B。论文里**没有给 8B-SFT**。

| Benchmark | Split | CodeVision 2.5-7B Base | CodeVision 2.5-7B SFT | CodeVision 2.5-7B Final |
|---|---|---:|---:|---:|
| V* | Overall | 74.6 | 71.7 | 83.7 |
| OCRBench-T | Rot180 | 70.2 | 57.0 | 72.3 |
| OCRBench-T | Verti | 17.0 | 35.8 | 67.4 |
| ChartQAPro-T | Rot180 | 23.4 | 23.2 | 30.8 |
| ChartQAPro-T | Hori | 19.5 | 20.9 | 30.1 |
| MVToolBench | Overall | 18.1 | 26.6 | 60.1 |

这个表最值得注意的点：

- CodeVision 的 `7B-SFT` 不是全面上涨
- 它在 `V*` 上甚至比 `7B Base` 低：`74.6 -> 71.7`
- 真正大幅提升是在 RL / final 阶段完成的：`71.7 -> 83.7`

---

## 3. 你本地当前结果，怎么读

如果只看你现在最关心的结论：

| Benchmark | Local 3VL-8B Base | Local 3VL-8B SFT | 结论 |
|---|---:|---:|---|
| V* Overall | 61.8 | 79.1 | 本地 SFT 提升明显 |
| HRBench-4K Overall | - | 74.6 | 低于 THYME / CodeVision 最终值 |
| HRBench-8K Overall | - | 67.8 | 低于 THYME / CodeVision 最终值 |
| ChartQA Overall | - | 83.1 | 接近 THYME base/final 之间 |
| OCRBench Overall | - | 77.1 | 明显低于 THYME |
| CountBench Overall | - | 90.8 | 两篇论文都没报，单独看即可 |

---

## 4. 你本地工具使用诊断

这张表不是分数表，是行为表。

| Benchmark | Local 3VL-8B SFT | Mean Turns | 解释 |
|---|---:|---:|---|
| V* | 79.1 | 2.00 | 基本没用工具 |
| HRBench-4K | 74.6 | 2.03 | 基本没用工具 |
| HRBench-8K | 67.8 | 2.04 | 基本没用工具 |
| CountBench | 90.8 | 2.00 | 基本直接答 |
| ChartQA | 83.1 | 3.87 | 有工具调用 |
| OCRBench | 77.1 | 4.12 | 有工具调用 |
| FSC147-Val | 75.1 | 2.00 | 基本直接答 |
| FSC147-Test | 76.5 | 2.00 | 基本直接答 |

---

## 5. 最短结论

1. 你要的 `Qwen2.5-VL-7B base`、`Qwen3-VL-8B-Thinking base`、以及论文里的 `SFT` 数据，现在都单独整理出来了。  
2. THYME 的 SFT 提升比较平滑；CodeVision 的论文结果说明 **SFT 不一定直接变强，最终大提升主要靠 RL**。  
3. 你本地当前 `3VL-8B SFT` 在 V* 上已经明显高于本地 base，但从 `mean turns=2.0` 看，**这个提升还不是稳定靠工具触发出来的**。  

---

## 6. 来源

- 本地 metrics:
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_vstar/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/vstar_base/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_chartqa/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_ocrbench/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_countbench/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_hrbench4k/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_hrbench8k/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_fsc147_val/metrics.json`
  - `/mnt/users/maodelin-20251119/ToolVision/CodeVision/saves/CodeVision/tools_a100_2gpu_fsc147_test/metrics.json`
- THYME:
  - https://arxiv.org/html/2508.11630v1
- CodeVision:
  - https://arxiv.org/html/2512.03746v1
