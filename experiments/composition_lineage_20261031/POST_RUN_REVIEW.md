# 复合初始化谱系诊断：独立运行后窄审

2026-10-31。结论：固定4次initializer均返回，原计数、训练摘要、全部保存提案分数容差及前12返回列表相容；完整排名只有2/4通过。范数两输出的近并列交换已被独立复现，因此保持正式 `VERIFICATION.status=incomplete`、`verification_passed=false` 和外层退出1。不能用差异小或前12不受影响替代事前完整排序门槛。本次没有发现冻结字节、输出封存、统计或报告主要观察的实质冲突。

报告中的局部语法形状统计成立，可以作为下一阶段机制假设来源；不构成完整搜索归因、拟合收益、采用或新盲验证。以下独立检查通过，指复核与既有未通过结果一致，不表示把诊断改记为通过。

## 实际检查与完整性

本次只新增本报告、post_review_audit.py及post_review_checks.json。未修改冻结源、trace、raw、分数、排名、正式收据或主代理REPORT；没有运行initializer、lstsq、非线性拟合、fit/search、正式诊断入口、汇总入口或pytest。

独立核55份FREEZE输入当前字节及55份frozen_inputs原字节归档，清单与FROZEN_INPUT_ARCHIVE逐项相同。16份output工件的实际集合与VERIFICATION.outputs完全一致，全部摘要匹配；逐调用结果绑定的raw/trace摘要匹配。复算结束后再核55当前输入、55归档与16输出，仍无漂移。FREEZE→VERIFICATION、VERIFICATION→LINEAGE_SUMMARY及汇总脚本摘要交叉一致。原训练来源模型在new_task ALL中的成员/摘要及原ALL→FREEZE链相符。

四次调用顺序、9131、两个固定无噪声case、limit12/compound=True/implicit=False，与冻结规格及原模型逐输出initializer计数一致。BUDGET为60秒、4调用、数值单线程；STARTED时间按调用顺序单调且均在原入口时段内。EXECUTION记录worker returncode=0、termination_reason=null、wall=`47.05505418777466 s`、initializer_completed_calls=4、verified_calls=2。按冻结入口代码，核验false导致外层SystemExit(1)，与主任务提供的终态一致；不是worker崩溃或超时。无重跑、额外调用或缩分母的工件证据。

## 独立数据、分数和排名

审查脚本没有导入诊断实现、task generator或initializer；按冻结公式/SeedSequence独立重建两个96×2训练输入/标签，未生成validation/test。训练摘要分别为：

- logarithmic_gain_n0：`49fad8eeedb6bd4ed0c0ddba509c3e8dc7648338ec5569563284bd5dc1d7a400`
- algebraic_norm_n0：`c8898afa1161445cdc9c844f22a6ad584338223fd6fd0f5b61a9f1a1aa70faf4`

两者均与FREEZE及原模型training摘要完全相同。另写严格JSON树解释器，检查常数叶数/顺序、逐节点非有限掩码，重算全部1849个保存合法提案的训练MSE、NRMSE、原方差归一化分数和size惩罚。全部636个已准入特征向量也独立重算，其字节摘要全部与raw相同。

1849个重算分数逐值等于保存的recomputed_score；与initializer原分数全部满足事前rtol=1e-10、atol=1e-12组合容差，最大绝对差为 `1.4779288903810084e-12`，没有把atol误当成唯一容差。使用独立构造的冻结Node表示、原size/repr tie-break重排完整列表，得到的全部索引逐项等于保存的独立重排；实际返回的12棵树/常数与原前12、独立前12逐项相同。

| 调用 | 合法提案 | 最大原分数差 | 完整排名 | 前12 |
|---|---:|---:|---|---|
| logarithmic_gain/output0 | 254 | 2.7755575615628914e-17 | 匹配 | 匹配 |
| logarithmic_gain/output1 | 178 | 2.2737367544323206e-13 | 匹配 | 匹配 |
| algebraic_norm/output0 | 785 | 1.4779288903810084e-12 | ranking_unresolved | 匹配 |
| algebraic_norm/output1 | 632 | 8.881784197001252e-16 | ranking_unresolved | 匹配 |

未解决的完整排名差异恰为：

- norm/output0零基153↔154：原分数 `0.4331641347657319`、`0.4331641347657333`；严格树重算后均为 `0.4331641347657333`，随后原size/repr次序交换两项。
- norm/output1零基21↔22：原分数 `0.011752504767209479`、`0.011752504767209494`；重算后均为 `0.011752504767209489`，原tie-break交换两项。

两对均在前12之外。它们与线性向量估计和严格树计算路径中的近并列浮点差异相容；本次未重新求解原线性子问题，不能唯一归因于某一BLAS/归约步骤。事实不改变原rank门槛，也不授权重跑、降门槛或改排原候选。

## 事件和语法形状的独立核验

逐行解析全部trace，共1224个admit_return、148个denominator_call、39580个proposal_boundary。admit成功计数与raw已准入特征数一致；实际分母调用列表与raw逐项相同。所有边界树的size/depth独立重算与记录一致，结构拒绝布尔均符合15节点/6层，源边界行仅为已审唯一锚点198/308。

| 调用 | 边界检查 | 结构拒绝 | 原线性子问题计数 |
|---|---:|---:|---:|
| logarithmic_gain/output0 | 6425 | 6150 | 6342 |
| logarithmic_gain/output1 | 4762 | 4528 | 4726 |
| algebraic_norm/output0 | 10664 | 9853 | 10638 |
| algebraic_norm/output1 | 17729 | 17095 | 17648 |
| 合计 | 39580 | 37626 | 39354 |

39580/37626是可重复边界事件数；39354是四次原计数器记录的线性子问题数，本次未重做这些求解。1849应精确解释为四次调用各自去重proposal字典长度之和，不是全批唯一图数，也不等同39580类未去重尝试事件。按这一口径理解主REPORT的“不能等同唯一候选数”，数值没有冲突。

独立定义与主汇总相同的语法谓词并重算全部阶段集合，LINEAGE_SUMMARY每个字段、形状示例、rank差异和总量全部相同：

1. **对数任务：** 两个输出各有1个根log、内部含square的已准入特征，均size9/depth6。output0直接外尺度包装的记录为size11/depth7，在198行结构条件处被拒绝。包含该语法形状的边界尝试分别154和36次，全部结构拒绝；合法提案和返回列表均0。该观察发生在合法提案排名之前，不能解释为“只因没排进前12”。
2. **变量直接乘log-square：** 对数两调用的已准入feature、合法proposal、returned及已捕获边界均为0。它是本次保存路径的缺口，不扩展为所有语法可能树或完整神经/GP搜索都未出现。
3. **范数形状：** 两个norm调用在feature/proposal/returned/boundary中，`sqrt(add(两分支各含square))`均为0。该谓词仅核特定语法形状，未穷举代数等价范数表达式。合法提案排序中的两次交换不会增删这些阶段集合，因此不影响这项局部零计数事实；仍不能据此称整个诊断完整通过。

读取主REPORT和summarize.py后，其主要观察及有限因果措辞与本次独立重算相容。主汇总是封存后处理，未进入原调用，摘要绑定正式VERIFICATION；本次没有把主汇总自动当作独立审查。对数包装深度拒绝可以确认，但“去掉包装/新增复合构造会拟合成功、保持保护或提高新任务接受”等只能是后续待检验假设。

## 覆盖与未知

独立数值覆盖：2个训练split、全部636个已准入特征向量、全部1849个保存提案、完整排序/实际48个返回树及常数。独立结构/计数覆盖：全部39580个边界、1224个admit返回及148个分母调用，55输入和16输出完整摘要链。保存的原计数与原模型逐字段一致，但本次没有重新求解39354次线性子问题、重跑initializer、重放完整生成控制流或重审每一trace树的量纲推导；来源是此前已审核且本次摘要一致的量纲语法。

未生成或使用验证/测试数组，未运行非线性拟合、MIMO搜索、隐式repair、神经/GP/NSGA-II或额外种子。本次没有运行时访问监控，训练隔离依据冻结代码、输入规范、来源摘要及仅训练重建；没有联网核发布状态或重新核平台session日志。

本批只观察两个无噪声任务首种子的每输出compound初始化一次，不支持完整搜索、另一种子、含噪声或implicit阶段归因，不构成拟合收益、采用或发布证据。正式rank未通过及原新任务7/16失败必须继续保留。主任务可以据这些局部观察设计下一份通用机制假设和新冻结方案，不能静默修正本批输出、重新解释为盲验证或降低原精度/保护/结构要求。

## 审查证据SHA256

| 文件 | SHA256 |
|---|---|
| FREEZE.json | 5c4386a0bef1b28f530145eb274c35f389c5aa3f98bf4934bd756df50d8d3680 |
| VERIFICATION.json | c86a27e947881cb56463699e1b9aa1161f6559e1ebe9af55629f5f9e168e4a71 |
| EXECUTION.json | bb201137e87dc9616d10df73378b0737a2bf81f529c68bd99a5ca3aca5440455 |
| FROZEN_INPUT_ARCHIVE.json | 02594bea23cba61190a31aaca76d8ec349fa00b8f3f537ca22c887d407f523df |
| LINEAGE_SUMMARY.json | 1f433d91ffb0d16789917de38e39abaeccc36376d64b7131343550d8630af3e8 |
| summarize.py | 3d0ed04c257290af25948ad6620b809c2f0a6a199a6c758eba22d35385576b30 |
| post_review_audit.py | 56471f2b46af48083deb72e2e806ee503c6c1d70a77d66c8dbcd4f8602a29011 |
| post_review_checks.json | 13b7c756254ee715a295c4a071071d883106b110d591841b683e8372896f96ca |
