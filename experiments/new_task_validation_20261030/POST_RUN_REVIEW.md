# 固定新公式族验证：独立运行后审查

2026-10-30。结论：独立检查支持32/32系统技术通过、16/16基础身份一致、8/8未触发单元整池/cache/选择一致及32/32配对输出保护；完整接受由5/16增至7/16。候选没有满足事前16/16完整接受要求，**本批整体验收失败，adoption_pass=false**。宏平均误差和图规模改善不能抵销该失败。当前审查范围未发现冻结、计数、原始图误差或汇总的实质证据冲突。

新增完整接受均来自 bilinear_saturation 的两个无噪声单元；其他族没有接受增益。不能把本次结果称为四族普遍发现收益、全部新任务解决、完整研究目标完成或版本发布依据。本矩阵已被观察，后续只能作为开发/回归证据。科学接受和下一阶段决策仍属主任务。

## 实际检查和独立性

本次只新增 POST_RUN_REVIEW.md、post_review_audit.py 和 post_review_checks.json；未修改冻结输入、source、frozen_inputs、runs、已有审查/日志或正式分数。没有运行 fit/search/refit、initializer、正式评分/核验入口、pytest、额外试验或子代理。

独立脚本不导入本批case生成器、scorer或verifier；根据已冻结公式和SeedSequence规范自行重建16对数据，以另写的原始JSON图解释器重算32所选系统train/validation/observed-test/clean-test，并用Fraction逐节点核拓扑、输入/常数/输出量纲、参数身份、共享图和全部结构指标。另实际调用锁定生产实现的Estimator.load、PhysicalModel.load及audit_domain：32个模型和32个物理模型对train/validation/test预测与独立raw逐元素一致，32次原域复核与保存审计一致且enclosed。这部分是独立调用既有加载/域实现，不是重新实现了加载器或形式域证明。

对全部69个repair和461个closure事件，共530个扩展事件，独立解释父/子原图、重算训练误差和每输出父保护，复核实际新评价、有效/保护数、验证留存、完整训练池/验证子集重建及两阶段会计；从保留模型重算所有SUMMARY字段，并逐字段与正式SUMMARY比较。全部比较相容。

实际数值差：所选测试观测/干净误差与正式分数逐值相同；训练/验证误差最大绝对差分别 `5.551115123125783e-17`、`2.7755575615628914e-17`。事件父/子误差最大差分别 `2.7755575615628914e-17`、`2.220446049250313e-16`，530个保护布尔均一致。独立训练SD的axis=0归约与模型逐列归约最大差 `1.1102230246251565e-16`，在浮点归约精度范围内，没有尺度来源变更。16对训练/验证联合字节摘要全部匹配FREEZE/result；形状为96×2、48×2，测试192×2。

## 冻结、封存与资源

- 40份live src的完整路径集合和SHA均匹配CANDIDATE_LOCK；与FREEZE及本轮source副本一致。715份FREEZE输入当前字节及frozen_inputs中的715份原字节副本全部匹配，FROZEN_INPUT_ARCHIVE.files逐项等同FREEZE.files。34份history_sources副本摘要匹配来源清单；TEST_VALIDATION的40份src与11份日志摘要匹配。
- ALL_MODELS_FROZEN所列352份运行工件与实际runs文件集合、摘要完全相同。32份CELL及其全部文件均在ALL中，CELL清单也与各目录实际文件集合相同；逐CELL时间按矩阵顺序单调。数值复核及加载/域核验结束后，再核全部715当前输入、715归档输入和352运行工件，仍无漂移。
- 时间链：FREEZE `1793352965.693415` → TRAINING_STARTED `1793352967.9549253` → ALL `1793353862.9523091` → SCORES `1793353889.6189206` → VERIFICATION `1793353932.479808`。FREEZE→ALL→SCORES→VERIFICATION交叉摘要相符。结合此前已审评分入口，这支持全模型封存后统一评分；本次未作额外运行时文件访问监控。
- FREEZE、score、verification的job顺序、32个唯一系统及16对固定任务/种子完全一致，两臂各16，所有resource/result/score为complete、returncode=0，verification逐行verified=true。不存在缩分母、换种子或补跑的工件证据。每系统有效配置符合冻结规格，配对仅precision_repair不同。
- 原1200评价、52秒合作fit、60秒单元外部上限、单线程串行、2880秒BUDGET及600秒评分核验预留均保留。实际最大评价433，最大保存fit elapsed为 `28.781000000002678 s`，最大单元外wall为 `38.73500000000058 s`。EXECUTION外wall `975.2486333847046 s`、termination_reason=null、returncode=0；BUDGET至完成核验 `966.8210391998291 s`。本轮没有触发截止边缘，不能据此声称已实测任意原子运算抢占；.2秒采样和preflight独立启动窗口的原边界不变。

## 身份和实际机制

16对基础候选count、payload摘要、base_evaluations和base_search_fingerprint全部相同；基础候选payload摘要本身逐模型重核。8个未触发单元的完整candidates、最终有效cache指纹、best原图/常数/训练指标及validation误差与base相同。该身份不要求带额外flag/报告/时间字段的整个模型JSON字节相同。

全部32系统从真实events中均有策略参数变化>0、有效neural>0、GP新评价>0；每系统至少两不同NSGA-II step分别有新评价，本批实际均为4个。保存的Rademacher计数>0，与正式机制收据一致；开关本身未用作执行证据。原图单位、搜索大小边界、共享关系、保存重载和原域核验均通过。

repair触发由基础池全部输出训练误差与原nrmse门槛重建，共8次：logarithmic_gain无噪声2次、algebraic_norm四次、bilinear_saturation无噪声2次。训练父排序、实际事件隐式根位置、父/子保护及验证留存均与保存报告一致。其余8个训练池已经全输出训练达标，跳过与协议一致，未按最终结构或测试失败启动repair。

| 机制 | base | repair |
|---|---:|---:|
| matching新评价/有效 | 222/222 | 239/239 |
| matching训练保护/验证留存 | 113/113 | 131/131 |
| repair新评价/有效 | 0/0 | 69/69 |
| repair训练保护/验证留存 | 0/0 | 10/10 |

所有事件满足原repair≤12、closure≤48及每轮≤24上限；每系统总评价等于base+repair+closure新评价。所有530个child在本轮均有效，保护拒绝仍保留事件与预算；训练保护留存合计254，验证留存也254。按已保存initializer诊断重计52693个隐式线性子问题和89个获胜隐式返回根；69个实际新系统均保存获胜隐式根来源。此处独立核的是保存计数、位置与图来源，没有重新运行底层线性初始化来证明52693次底层调用。

最终有效cache为保存指纹的身份检查，本次没有完整运行时cache字典，不能独立枚举所有历史失败/成功cache命中。所有扩展事件图去重、来源与会计已复核，这不冒充完整cache重新执行。

## 原门槛和失败的独立重建

| 指标 | base | repair |
|---|---:|---:|
| 完整精度与结构接受 | 5/16 | 7/16 |
| 宏平均干净测试NRMSE | 0.018020825660018117 | 0.01602532135442316 |
| 平均shared_graph_size | 22.9375 | 21.875 |
| 干净逐输出保护 | — | 32/32 |

5个base已接受单元全部保留；接受增益2、损失0。原单位/域、无噪声各train/validation/test≤1e-4、有噪声观测≤.03/干净test≤.02、共享图24/参数6/展开30/深度6门槛保持。16/16候选完整接受这一主条件为false，因此总体adoption=false；不是技术核验失败，也不能以exit0代替总体接受。

| 族 | base→repair接受 | repair触发 | 新/有效 | 保护/验证留存 | 接受增益 |
|---|---:|---:|---:|---:|---:|
| logarithmic_gain | 0/4→0/4 | 2 | 21/21 | 2/2 | 0 |
| logistic_gain | 4/4→4/4 | 0 | 0/0 | 0/0 | 0 |
| algebraic_norm | 0/4→0/4 | 4 | 37/37 | 4/4 | 0 |
| bilinear_saturation | 1/4→3/4 | 2 | 11/11 | 4/4 | 2 |

两个新增接受是bilinear_saturation_n0/9131和9132，图分别22→13、21→13；repair参数3、总展开18、深度4，各train/validation/clean误差均≤ `3.5773374188062213e-16`。三个族有实际验证留存，但完整接受增益只覆盖一个族；logistic零触发不能用作repair新增发现证据。近浮点精度只对应声明数据和域筛查，不是解析恒等式或域外外推证明。

以下9个repair单元仍未完整接受，均保留分母和原失败：

| 单元 | 超出原门槛的实际项 |
|---|---|
| logarithmic_gain_n0/9131 | 两输出train/validation/test精度；图27>24，参数7>6 |
| logarithmic_gain_n0/9132 | 两输出train/validation/test精度；图26>24，参数7>6 |
| logarithmic_gain_n1/9131 | 图29>24，参数8>6；观测/干净精度通过 |
| logarithmic_gain_n1/9132 | 图25>24，参数7>6；观测/干净精度通过 |
| algebraic_norm_n0/9131 | 两输出train/validation/test精度；图28>24 |
| algebraic_norm_n0/9132 | 两输出train/validation/test精度；图26>24 |
| algebraic_norm_n1/9131 | output1观测train/validation/test及干净test精度；图26>24 |
| algebraic_norm_n1/9132 | output1观测train/validation/test及干净test精度；图29>24 |
| bilinear_saturation_n1/9132 | clean output0=`0.021897967820013676`>.02；结构及观测精度通过 |

这张表是验证过的门槛差距，不是搜索失败原因的唯一诊断。读取当前REPORT初稿，其主要数值、逐单元最大误差、整体验收失败和有限发现作用口径与本次复算相容。报告中对数含噪声池的7/1个绝对精度+结构可行计数与保存pool记录一致；该计数未包含所有相对保护约束，不能据此认定完整可接受候选被错误丢弃。本次没有以测试标签重选、调参或判断后续修复方案优劣。

## 未覆盖项与后续边界

共5330份保存候选。本次独立数值重算覆盖32所选系统和530扩展事件，没有重算所有5330候选的每个训练/验证指标；该全量核验由已冻结的正式VERIFICATION和此前审过的原始JSON验证路径提供，本次不能冒称又独立完成了一遍。native加载器会重建precision选择和审查其保存元数据，但这也不等同所有候选数值误差的独立重算。未重新计算Rademacher值或initializer，未重复全套测试或跑资源边缘试验。

新任务口径限定于已审34份可访问注册/定义来源的完整输出对比较，不是删除/未知历史或全部解析等价任务的穷尽排查。四族为作者合成任务，输入正域、远离奇点，两个种子同时改变数据与优化；两噪声层X不同，不作配对噪声因果比较。不支持公共标准benchmark、实测工程接受、统计显著性、总体成功概率或普遍优越性。

当前源码默认仍为legacy/random、precision_repair=False。当前本地HEAD与冻结base_commit同为 `652b05884269c7563b101e0a14a9b15f749bbc6f`；工作区pyproject标记1.7.0不等于发布完成。本次没有联网检查远程发布状态，也不以本报告授权默认/发布。总体研究目标继续，任何后续诊断应以训练/验证及已保存事件为依据，算法改变另立冻结阶段；不得降低本批原门槛或返工本批测试选择。

## 关键证据SHA256

| 文件 | SHA256 |
|---|---|
| FREEZE.json | 728fb302de07cc262f43657e479e7afd0c543744c25b0655a20f2fdcaadc89a9 |
| ALL_MODELS_FROZEN.json | cc618d4e4f415ef59bdce6455d41a9a9a1634c1f9e33344ab9950c2a45c96593 |
| SCORES.json | e9cd27d925ae7e62c781260995d4b7085f70b6ae43b621bdb6780ea1320db2e3 |
| VERIFICATION.json | 80a0c4c959d43ff330d4458be39b25165794b7c4e4c54f3f47af75b297365ec7 |
| SUMMARY.json | 1788c2786973e5d9d2c8c5b2573f2d0a5ed70c6a0e78b8e4006d60f24a64bc21 |
| FROZEN_INPUT_ARCHIVE.json | 8b871e7dff8b1bc4c8f7b49b35eccc828f5ba72b907d11cb90c16f327c0695c3 |
| EXECUTION.json | bc6ca0bafa1b3ac79637d8f3c80774b2696ac3779b9060aeacd0a4516d9c4406 |
| post_review_audit.py | 8e383fa482316b9f6137b10d0a9205e41ebe62c9f783941941e8a0ccec5caf67 |
| post_review_checks.json | 71395b7fb4075940f44529bd5e79352d92735b1897e70efd0850ec4ab64afa9e |
