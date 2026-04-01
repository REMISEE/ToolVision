# 4 Trajectory 分叉与存储约定 TODO

日期：2026-03-26  
状态：待继续细化  
目的：把后续关于 trajectory 树结构、目录落盘、frontier 管理、resume 的待办集中列出来，后续逐项冻结。

---

## 1. 当前结论

已经基本对齐的是：

- 逻辑上是一棵树
- 存储上是多条 trajectory 记录
- frontier 只维护当前活跃叶子

但下面这些还没有完全冻结。

---

## 2. TODO 列表

### TODO 1：trajectory_id 命名规则

待定问题：

- root trajectory 如何命名
- child trajectory 是否直接带 `suggestion_id`
- 是否需要人类可读命名

建议候选：

- `traj_<sample_id>__root`
- `traj_<sample_id>__r0_s1`
- `traj_<sample_id>__r1_s2`

### TODO 2：父节点状态怎么表示

待定问题：

- 父节点 fork 出子节点后，是继续 `running`，还是进入内部 parked 状态
- schema 里是否要新增 `parked`

V0.1 暂时建议：

- 只有叶子是 `running`
- 父节点不再执行，但保留

### TODO 3：frontier 更新规则

待定问题：

- planner 一轮 3 个 suggestion 是否都 fork
- judge 后保留 top-k 还是阈值过滤
- 什么时候直接停止整棵样本树

### TODO 4：目录布局

待定问题：

- sample 级目录长什么样
- trajectory 级目录长什么样
- planner / steps / judge / export 各放哪

建议至少包含：

- `trajectory.json`
- `messages.json`
- `planner/`
- `steps/`
- `judge/`
- `exports/`

### TODO 5：messages.json 的真实格式

待定问题：

- 生成态 messages 是否直接采用最终训练态格式
- assistant 的 tool call 是存 XML 字符串，还是存结构化字段
- tool message 如何引用新图 artifact

### TODO 6：resume 规则

待定问题：

- 中断后优先恢复哪个 trajectory
- 通过什么字段判断某步已完成
- `pending_execution` 如何更新最稳

### TODO 7：planner 历史如何落盘

待定问题：

- 每轮 planner 输出单独一个 JSON 还是合并文件
- 是否保留未被选中的 suggestions 全量原文

建议：

- 每轮一个独立 JSON
- trajectory 里只保留索引

### TODO 8：judge 历史如何落盘

待定问题：

- cheap filter / committee / final_select 是否分文件
- 是按 step 存，还是按 trajectory 存

建议：

- 每次 judge 一条独立记录
- trajectory 里只保留引用

### TODO 9：导出阶段如何选轨迹

待定问题：

- V0.1 是否导出全部终止轨迹
- 是否需要同时导出失败 / pruned 样本

当前倾向：

- 先导出全部终止轨迹
- 后处理再筛高质量子集

### TODO 10：图像 artifact ID 规则

待定问题：

- root 图和 step 新图的 ID 是否统一规则
- 一个 step 多张图如何编号

建议候选：

- `img_root_0`
- `img_step_001_0`
- `img_step_001_1`

---

## 3. 推荐后续推进顺序

建议按这个顺序继续冻结：

1. `trajectory_id` 命名规则
2. sample / trajectory / step 目录布局
3. `messages.json` 真实格式
4. resume 规则
5. frontier 更新策略
6. export 选择策略

---

## 4. 一句话版本

这份文档先不下最终结论，只把 trajectory 树和落盘系统后续必须回答的问题列全，后面按顺序逐项冻结即可。
