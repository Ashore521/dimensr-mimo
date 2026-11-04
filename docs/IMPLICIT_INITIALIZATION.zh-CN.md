# 有理系数联合初始化（1.7.0 开发中）

`SearchConfig(initialization='implicit')` 在原 `compound` 特征和提案基础上，加入通用分母偏置联合初值。默认仍为 `legacy`；它不增加返回种子名额或完整系统评价上限。

对于有合法偏置单位的有界特征 d(X)，考虑分母 d(X)+b，以训练关系 y·d=N−b·y 同时求分子外层系数和 b。此线性子问题固定 d 的外层系数为1；嵌套参数仍由后续正常完整 MIMO 常数拟合处理。初始化按普通表达式的直接输出误差排序，训练分母非有限或绝对值≤1e-12时拒绝，不使用保护除法。

隐式与原提案共同竞争同一前12或调用者指定limit，所有完整系统仍执行神经/GP/NSGA-II、联合常数拟合、量纲/数值域、Rademacher及既定选择。初始化计算消耗原墙钟预算，并可能改变后续搜索轨迹，不能声称保持旧模型或保证测试改进。

`initialization_diagnostics`记录隐式被试分母、线性子问题、有效提案尝试、分母拒绝及返回位置。`history`的initialization阶段记录实际含新初值的系统尝试、新评价和有效数；新评价/有效数排除缓存命中及结构/预算拒绝。有效提案尝试数不等于唯一树数。

训练侧依据见`experiments/rational_seed_lineage_20261028/REPORT.zh-CN.md`；正式消融由`experiments/implicit_rational_20261028/PLAN.md`及`PROTOCOL.md`规定。该批34/34技术核验通过，完整接受10/17→15/17，但逐输出保护仅30/34，采用门槛失败，默认不变。后续保留原搜索的可选精度补充见`PRECISION_REPAIR.zh-CN.md`；两批均属于历史开发证据。
