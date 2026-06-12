# 17 Step 4：scripted 入口与目录结构说明

日期：2026-03-30  
状态：已新增  
目的：说明当前新增的 `run_single_sample_pipeline.py` 怎么用、它会产出什么目录结构、哪些组件是 fake/scripted、哪些组件仍然是真实主逻辑。

---

## 1. 这个入口脚本是做什么的

新增脚本：

- `offline_sft_pipeline/scripts/run_single_sample_pipeline.py`

当前只支持一种模式：

- `--mode scripted`

它的定位不是：

- 真实模型入口

而是：

> 一个正式的、可直接运行的 scripted demo 入口，用来把当前 v01 pipeline 的目录结构和多轮执行语义跑出来给人看。

也就是说，这个脚本的目标是：

1. 让你不用进 `unittest` 也能直接跑一条 sample
2. 跑完后能看到 store 目录结构
3. 明确哪些部分是主逻辑，哪些部分只是 fake

---

## 2. 当前哪些组件是 fake

这次新增的 fake 组件都放在：

- `offline_sft_pipeline/pipelines/scripted_components.py`

这里面以下内容全部是 fake/scripted：

### 2.1 `ScriptedPlannerClient`

它不会调用模型。

它只是按：

- `(trajectory_id, round_idx)`

返回预先写好的：

- `PlannerOutput`

### 2.2 `ScriptedExecutorClient`

它不会调用模型。

它只是按：

- `(trajectory_id, step_idx)`

返回预先写好的：

- `ExecutorStepOutput`

### 2.3 `ScriptedRuntime`

它不会执行真实 executor code，也不会调用 deployed helper 服务。

它只是：

1. 在 step 目录里写：
   - `stdout.txt`
   - `stderr.txt`
   - `runtime_result.json`
   - `output_0.png`
2. 构造假的：
   - `ExecutorRuntimeResult`

### 2.4 `ScriptedJudgeBackend`

它不会调用 judge model。

它只是返回预先写好的：

- `overall_score`

### 2.5 `build_three_round_demo_scenario(...)`

它也是 fake 的。

它的作用是把一条固定的三轮 demo 场景组装出来，供：

- 回归测试
- scripted 入口脚本

共同使用。

---

## 3. 当前哪些组件不是 fake

下面这些虽然在 scripted 模式里接的是 fake 输入，但模块本身是真实主逻辑：

### 3.1 `OfflineTrajectoryStore`

路径：

- `offline_sft_pipeline/core/store.py`

这是真实 store。

它真的会：

- 建目录
- 写 `messages.json`
- 写 `trajectory.json`
- 写 planner/step/judge 记录

### 3.2 `OrchestratorV01`

路径：

- `offline_sft_pipeline/pipelines/orchestrator_v01.py`

这是真实 orchestrator 主循环。

它真的会：

- 调 planner client
- 做 selection
- fork child
- 调 executor client
- 调 runtime
- 回写 messages
- 调 judge
- 推进多轮 frontier

### 3.3 `JudgeClient`

路径：

- `offline_sft_pipeline/pipelines/judge_client.py`

它虽然现在接的是 fake backend，但它把分数转成：

- `JudgeRecord`

的这层逻辑是真实的。

---

## 4. 新增后的结构怎么理解

当前跟 scripted 入口有关的关键文件是：

### 主逻辑

- `offline_sft_pipeline/core/store.py`
- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/pipelines/executor_client.py`
- `offline_sft_pipeline/pipelines/judge_client.py`

### fake / scripted 组件

- `offline_sft_pipeline/pipelines/scripted_components.py`

### scripted 运行入口

- `offline_sft_pipeline/scripts/run_single_sample_pipeline.py`

### 语义回归测试

- `offline_sft_pipeline/tests/test_orchestrator_v01.py`

---

## 5. 怎么运行

当前推荐直接跑：

```bash
python offline_sft_pipeline/scripts/run_single_sample_pipeline.py
```

也可以自定义输出目录：

```bash
python offline_sft_pipeline/scripts/run_single_sample_pipeline.py \
  --output-dir offline_sft_pipeline/outputs/scripted_sample_pipeline_runs \
  --run-id demo_manual_001
```

脚本会输出一份：

- `scripted_run_summary.json`

并把同样内容打印到 stdout。

---

## 6. 跑完之后会生成什么目录

脚本会先生成一个 run root。

大致结构如下：

```text
<output-dir>/
  <run-id>/
    inputs/
      root.png
    scripted_run_summary.json
    store/
      samples/
        <sample_id>/
          root_sample.json
          artifacts/
            img_root_0.png
          trajectories/
            <trajectory_id>/
              trajectory.json
              messages.json
              planner/
                round_000.json
                ...
              steps/
                step_001/
                  executor_cot.md
                  executor_code.py
                  runtime_result.json
                  stdout.txt
                  stderr.txt
                  output_0.png
              judge/
                step_001_cheap_filter.json
```

其中：

### `inputs/root.png`

这是 scripted demo 用的输入图。

它本身也是 fake demo 图。

### `store/`

这里才是真正的 pipeline 落盘根目录。

也就是说：

- 你后面真正要看 pipeline 结构，主要看这里

### `scripted_run_summary.json`

这是给人读的总览文件。

它会标出来：

- 当前模式
- 哪些组件是 fake
- trajectory ids
- 推荐先看的文件
- 一个 tree preview

---

## 7. 跑完先看哪几个文件

如果你只是想快速理解结构，建议按这个顺序看：

### 7.1 summary

- `scripted_run_summary.json`

先看：

- 这次 run 放在哪
- 哪些组件是 fake
- 结果里有哪些 trajectories

### 7.2 root 轨迹

- `store/samples/<sample_id>/trajectories/<root_trajectory_id>/trajectory.json`

看：

- planner history
- root 是否变成 `expanded`

### 7.3 child 的 messages

看任意一个 child 的：

- `messages.json`

你能直接看到：

- executor step message
- tool result message
- final answer message

### 7.4 planner round

看：

- `planner/round_000.json`
- `planner/round_001.json`

这里能看到：

- `global_chain_cot`
- `suggestion_cot`
- suggestions

### 7.5 step 目录

看：

- `steps/step_00x/`

这里能看到：

- `executor_cot.md`
- `executor_code.py`
- `runtime_result.json`
- 输出图

---

## 8. 为什么还要保留测试

虽然现在已经有：

- `run_single_sample_pipeline.py`

但测试文件仍然应该保留。

原因是：

### 8.1 脚本是给人看结构和手工运行的

它更像：

- demo / inspect 入口

### 8.2 测试是给代码回归用的

它更像：

- semantic contract

以后如果有人改了：

- selection
- budget
- child copy
- visible image 传播

测试会第一时间报错，而脚本不会自动帮你拦住这些回归。

所以当前建议是：

- 脚本保留
- 测试也保留

两者职责不同，不冲突。

---

## 9. 当前这个 scripted 入口最适合做什么

最适合做的是：

1. 向人展示当前目录结构
2. 让人 inspect 一条真实落盘的 trajectory 样例
3. 验证 store/orchestrator/messages 的语义
4. 在没有模型 API、没有 helper 服务的情况下先看清 pipeline 壳子

不适合做的是：

1. 验证模型能力
2. 验证 helper 服务质量
3. 验证真实 OCR / SAM / crop 输出
4. 替代将来的真实样本运行入口

---

## 10. 一句话版本

当前新增的 `run_single_sample_pipeline.py` 是一个 scripted demo 入口：

> 主循环、store、messages、trajectory 落盘都是真实的；planner / executor / runtime / judge 的结果来源是 fake/scripted 的，目的是先把 v01 pipeline 的结构和执行语义完整跑出来给人检查。
