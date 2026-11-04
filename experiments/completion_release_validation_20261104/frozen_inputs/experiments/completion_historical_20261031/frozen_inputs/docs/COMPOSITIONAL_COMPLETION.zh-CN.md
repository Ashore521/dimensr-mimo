# 完整基础搜索后的复合候选补充

`SearchConfig(compositional_completion=True)` 是默认关闭的开发选项。它要求
`initialization='compound'`、`sharing_strategy='closure'`、
`final_selection='precision'`、单位及逐输出约束，并与 `precision_repair` 互斥。
生产默认仍为 `legacy` 初始化和 `random` sharing；旧 repair 的训练精度触发规则不变。

完整神经、GP、NSGA-II 和 polish 先按原流程完成。闭包先记录原始 base 数量、
图/常数/误差摘要、缓存指纹和评价数，新选项随后始终进行一次有界补充，不因 base
训练精度达标而跳过。补充和后续原两轮各最多 24 次匹配闭包共用原总评价与截止时间。

每个输出调用原 `structured_seeds(compound=True, implicit=True)`，与新 log/sqrt
逆变换线性初值竞争同一个最多 `min(12, population_size//n_outputs)` 的返回名额。
新机制只读训练 X/Y、声明单位和算子，不读验证或测试数据、任务标识或真实公式。

新外层固定为 `f(h)`、`x*f(h)`、`x/f(h)`，按语法顺序使用最多 16 个变量。
log 的训练内层目标是 `exp(r)`；sqrt 是 `r²` 且要求 `r>=0`。
潜在响应依外层分别取 `y`、`y/x`、`x/y`。绝对值不超过 `1e-12` 的除数、
非有限中间量、逆变换溢出及 log 逆变换下溢至零均拒绝，不裁剪或平移标签。

线性基按固定顺序枚举变量、变量平方、两不同变量乘积，最后加入声明逆单位常数
允许的归一化变量/平方；去重并按原语法及严格求值筛选后最多保留 24 个。
每个内层只用一个或两个同单位基；存在合法目标单位常数时分别考虑有/无截距。
每个实际最小二乘问题记账。无量纲非负平方权重可吸入已有合法尺度，或在平方内
加入已声明的合法尺度；负权重只走合法外乘。新图逐一接受原节点/深度和 grammar
回放检查。没有相应常数或算子时不会合成未声明的终结符。

裸根继续参赛；存在无量纲常数和乘法时还生成初值为 1 的可拟合外幅值：
`a*f(h)`、`(a*x)*f(h)`、`(a*x)/f(h)`。变量形式把尺度放在变量分支，以保留
较深一元分支的深度空间。幅值变体共享同一次线性求解，随后只参加原完整 MIMO
联合拟合。逆变换线性初值不保证一般外幅值下精确、返回或收敛。

所有合并候选均用自身严格 `Node.evaluate` 在原始训练响应上重算
`MSE/max(var(y),1e-12)+1e-8*nodes`，按分数、节点数、repr 排序。
同拓扑只保留真正最低分常数，完全同分保留首项；来源跟随该常数，不合并落选来源。
这统一重排原 initializer **返回的根** 与新根，不声称重新严格排序旧 initializer
全部内部提案，也不修改旧 repair 的排名。

平方吸收仅用于构造新图，不改写旧模型，不宣称全浮点定义域等价。
例如 `0*square(1e200)` 有非法中间量，而 `square(0*1e200)` 可以有限。
新图的有效性、分数和最终原生域审计均从自身重新计算。

输出列表按相同索引组成系统；某输出缺该索引时复制固定原父对应根，不循环列表。
原父按平均训练误差、复杂度和图摘要排序固定。系统至少含一个实际 implicit 或
compositional 来源的根；成功/失败缓存均跳过，最多 12 次缓存外系统评价。
每次使用原 Evaluator 的全输出联合拟合、单位/域及 Rademacher 检查，不额外 refine。
只有每输出训练误差 `child<=parent*1.1+1e-8` 才保留。后续验证过滤和原 base
训练/验证双保护选择不变；训练保护不代表未知测试保护。

`diagnostics.sharing_closure.compositional_completion` 含触发、预算、父、有效/保护、
缓存/结构拒绝、短列表计数及事件。`initialization_diagnostics` 每输出保存原 initializer
计数、新基/变换/线性及拒绝计数，以及 `returned_sources`。每个返回来源含
`kind`（structured/implicit/compositional）、返回与提案位置、最终树/常数/分数；
新来源另含 transform、form、variable、basis_indices、intercept、outer_amplitude。
评价事件记录实际 `implicit_outputs`、`compositional_outputs` 和逐输出 `root_sources`。
顶层线性总数为旧 initializer 的全部 linear_subproblems 加新变换线性问题；
implicit_linear_subproblems 是其中子集，不应再次加总。
history 的 closure 阶段另记录 `completion_new_systems` 和 `completion_protected_systems`。

初始化、基枚举、每次线性求解和系统组合合作检查同一 Evaluator deadline。
Timeout 保留已完成的部分工作计数、base 和已留存扩展；单次 NumPy/BLAS 不可抢占。
旧模型先按原保存配置核验选择/来源记录，之后才迁移新增 False 默认，因此可再存再载。

当前文档描述实现和构造验证。正式历史/开发矩阵、新锁定任务、稳定核心独立审查、
打包安装及发布验收仍按各自协议单独完成；构造测试通过不等于科学采用。
