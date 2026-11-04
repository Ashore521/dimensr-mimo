# 受约束匹配闭包（1.7.0 开发中）

`sharing_strategy='closure'` 保留原随机共享的完整神经、GP、NSGA-II与常数polish路径，然后在剩余训练预算中提出新的紧凑共享图。默认仍为`random`；该选项要求`use_sharing=True`及`final_selection='precision'`并显式提供精度/结构门槛。

```python
config = SearchConfig(
    sharing_strategy='closure',
    final_selection='precision',
    acceptance={
        'nrmse': [1e-4, 1e-4],
        'shared_graph_size': 24,
        'independent_parameters': 6,
        'expanded_total_nodes': 30,
        'maximum_output_depth': 6,
    },
)
```

闭包最多两轮、每轮16个父系统、每轮最多24次新系统评价、合计最多48次；这些评价计入原`max_evaluations`与`wall_seconds`。匹配图仍须完整联合拟合并通过原量纲、结构和数值域检查。候选必须逐输出满足父代训练误差10%+1e-8保护，基础候选不被替换或重新拟合。

全部搜索结束后才使用验证数据。基础池按原precision规则选择参照；最终模型同时满足全部原门槛，以及相对该参照的逐输出训练与验证误差10%+1e-8保护。没有可行新选择时保留基础选择及其原接受/未接受状态。验证无效的新增图被记录并拒绝，不触发续搜；基础候选验证失效会明确报错。

这些训练/验证边界不保证测试误差不退化。`enclosed`也只代表有限数值筛查。是否推荐必须由冻结对照的完整精度/结构、逐输出测试保护和后续新任务验证决定。

`diagnostics_['sharing_closure']`保存基础池与完整基础搜索摘要、实际评价计数、父子记录、保护拒绝及验证后留存数。`post_selection_`保存基础选择和两组逐输出边界；保存重载会重算选择和基础池摘要。本选项尚处于冻结开发对照阶段，不能将构造测试或执行机制通过当作科研收益证据。

第一轮`experiments/protected_closure_20261028`技术核验34/34、逐输出保护34/34，接受数9→10，但指定紧凑图为23，未达到≤20，因此未通过采用门槛。其完整源码/输入/运行结果保留。训练图诊断证实第一轮耗尽48次额度阻止了已可达的递归共享；24+24修订的`experiments/balanced_closure_20261028`已完成34/34技术与输出保护、17/17基础池一致、接受数9→10，并恢复图20及机器精度，独立审查通过该批开发门槛。仍有7/17未完整接受，也尚无新任务验证，因此默认保持不变。

来源与复现入口：`experiments/search_lineage_20261028/REPORT.zh-CN.md`说明初始改动依据，各实验的`run_bounded.py`包含外部总时限监督。已有冻结目录不得原地重跑，应在新目录复制/固定协议与输入后执行。
