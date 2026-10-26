# 原生接口与物理坐标交付结果

## Material Passport

类型：软件接入回归与真实拟合交付核验。状态：VERIFIED。范围：已知合成开发任务，无新测试集评分、盲测或实际工程数据结论。源码和阈值在FREEZE.json中预冻结。

六系统全部完成，运行耗时117.94秒；6/6独立核验通过。单工作进程、数值库单线程，单元外部60秒、fit52秒，最高单元耗时44.44秒，无超时。初始捕获器遗漏的三个normalized结果留存为无效参考，详见baseline/INCIDENT.md；corrected参考全部与第七轮已发布结果一致。

|单元|参考来源|搜索与池|最终选择|独立复算|
|---|---|---|---|---|
|products/legacy|baseline|完全一致|原weighted一致|通过|
|dimensional_frequency/legacy|baseline|完全一致|原weighted一致|通过|
|dimensional_frequency/normalized|baseline_corrected|完全一致|归档precision一致|通过|
|partial/normalized|baseline_corrected|完全一致|归档precision一致|通过|
|high_dimensional/normalized|baseline_corrected|完全一致|归档precision一致|通过|
|affine_lengths/physical|预冻结新示例|真实拟合完成|搜索/物理均accepted|通过|

五个参考单元同时核对了完整候选池、变异计数、去计时阶段事件、候选训练/验证误差、最终选择及预测。六单元都有实际神经生成/参数更新、GP新评价、多代NSGA-II新评价和局部Rademacher计算证据。

物理示例在[8,12]×[90,110]长度域内，训练96行、验证48行；仅用训练数组建立坐标。两输出还原为约`2*length0+3*length1+5`和`length0-length1+2`。逐输出训练/验证NRMSE约5.30e-15、4.47e-15。保存/重载和独立JSON节点解释与坐标包装预测逐元素一致。

源图size17、拟合参数5；物理图size31、参数13（5拟合+8固定坐标）、展开节点42、深度8。固定坐标开销如实计入物理限额。显示公式可被SymPy整理，计算及审计仍依据原始DAG。

85项测试通过；sdist和wheel构建通过。隔离Python直接从wheel导入，明确阻止experiments导入后，全部六模型重载及物理点预测通过。PACKAGING.json记录wheel哈希；validation/保留测试与构建日志。

生产默认保持legacy/weighted。当前precision遍历搜索坐标中的候选复杂度，物理交付独立检查所选图，不承诺任意缩放下全池最小物理图。没有精度可行解时保留not_met；非恒等坐标未提供原物理数组时不认定精度通过。enclosed为有限数值筛查。

前序提交9a30c4d、41a7db7、13471be的GitHub Actions已实时核查成功，见CI_PRIOR.json。本批云端状态需按本批提交另行核查。

下一批：冻结含噪声及未用于既有轮次的新公式族研究矩阵，完整保留失败；随后开展单位明确的实际数据交付验证。当前总目标继续推进。
