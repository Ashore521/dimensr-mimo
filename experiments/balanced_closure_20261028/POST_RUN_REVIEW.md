# 两轮均衡闭包运行后独立审查

结论：本次限定范围核验支持本轮34系统技术通过、34/34配对干净输出保护、17/17基础保留一致，以及原compact和其余采用门槛全部通过。`adoption_pass=true`在冻结历史开发批次口径下得到支持。未发现阻断这些结论的剩余实质问题；这不等于新任务验证、默认切换、发布接受或完整科学目标完成，最终决定由主任务作出。

仅写本报告。未修改冻结源码、协议、门槛、结果或历史审查；未训练、fit、搜索、pytest、新增实验或派生代理。确定性复算只评价封存图/常数，没有重新拟合、续搜或按测试标签改选模型；未纳入另外的有理初始化观察。

## 证据完整性、分母和资源

重新读取本轮PROTOCOL、FREEZE、BUDGET、TRAINING_STARTED、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、EXECUTION及必要单元文件。FREEZE.files的120项、ALL_MODELS_FROZEN.files的374项、34个CELL_FROZEN内部文件清单均实际逐文件核SHA256通过。引用链一致：

- FREEZE：`abbb7e68bec12fd1382f61f9cf348e4e4ced08d0a4edcb40c29f59f54b017abf`
- ALL_MODELS_FROZEN：`f038bebc48ed89bfeff78c7bc6af9943a7f759216a1781918f6049915f7d32de`
- SCORES：`a677b165c4337eac2ffa40ac1f1de2113760721d7c4705a55a6eea179fcdbfdd`

BUDGET实际内容与FREEZE内budget一致，冻结→训练→全模型封存→评分→核验时间顺序成立。34唯一jobs与score/verification逐项同序，random/closure各17，无缺失/重复/缩分母；34项resource/result/score均complete、进程返回0、verified=true。

EXECUTION.returncode=0、termination_reason=null，外部704.658810377121秒；VERIFICATION内部700.0779986381531秒、budget_met=true，低于原2880秒。逐单元外部最大21.98400000000038秒，模型diagnostics.elapsed_seconds最大15.170999999998457秒，低于60/52秒。闭包history阶段时间合计约9.031999999995605秒；这是闭包阶段记录成本，不是配对总时间差或纯拟合耗时。未用少于上限的实际时间声称所有机器和任务都能满足该上限。

## 每轮调度与实际机制

实际读取34个events.jsonl重算：每项policy参数变化>0、neural有效候选>0、GP新评价>0、NSGA-II至少2代且有新评价、Rademacher调用>0。闭包附加机制按冻结批次口径满足四项非零；存在0新增单元，不能写成17个closure单元都发现新图。

逐事件统计各轮新增评价，所有单元每轮≤24、两轮总≤48、原系统总≤1200。总评价random6142→closure6536，增加394；闭包394新/394有效/111训练父保护通过/111验证后留存，与SUMMARY一致，无未记账评价。damped_n0/701实际两轮 `[24,24]`、48新/48有效/34保护，所选最佳图来自round=1。其他单元可因机会不足或无保护子代只执行部分额度，未转用未消费额度扩张下一轮上限。

17/17对：closure基础candidate前缀与random完整candidates字典精确相同；base_search_fingerprint与random search_fingerprint、base_evaluations与random总评价数，以及operations/changed/novel/successful均一致。由此本批保留基础状态的声明得到运行证据支持，超出本批的有墙钟同轨迹仍不能仅据开关保证。

本次额外逐事件检查：父代在当轮训练平均误差/复杂度/图摘要排序前16中，来自基础或先前轮留存状态，每轮父代数≤16；确定性matched_proposals重建的图/warm与每事件精确对应；同轮已执行提案排名不逆序，新增评价图无重复。完整训练pool摘要可由基础payload+训练通过子状态重建。未重新拟合或重演整个搜索。

## 独立数值复算与原门槛

采用自写JSON逐节点NumPy解释器，按原始算子和中间非有限传播计算，未复用graph.evaluate计算这些误差。实际完成：

- 5357个最终候选、10714组train/validation误差；394个父及394个有效子训练误差，均符合原rtol=1e-9/atol=1e-11容差。训练/验证数组哈希逐项吻合FREEZE，归一化与训练观测标准差符合1e-15相对浮点归约精度。
- 34个模型实际load重算选择后，predict(test)与自写原图解释器逐位相同。34项observed/clean测试误差均符合原rtol=1e-12/atol=1e-14；独立按原train/validation/observed-test/clean-test绝对阈值与结构/precision状态严格重算接受9/17→10/17，仍有7/17 closure未完整接受。
- 17对×2输出的干净保护按原 `closure <= random*1.1+1e-8` 重算为34/34。宏平均干净NRMSE为random `0.002617556289020336`、closure `0.002614428507410396`；共享图均值 `22.941176470588236`→`21.88235294117647`，与SUMMARY一致。全17项均完整，未用幸存样本均值。
- damped_n0/701：random图27/6参数，closure图20/4参数，closure展开28、最大深度6，满足原一般结构门槛及专门图≤20要求。逐输出train/validation/clean-test严格独立复算的最大误差为 `3.2449972428016087e-16`，低于原1e-12。封存train为 `[1.700552365049716e-16,3.2449972428016087e-16]`，validation为 `[1.9018485879161261e-16,3.071626013496416e-16]`；没有用一般图≤24上限替代compact门槛。

因此technical、原34/34输出保护、接受数不降、宏平均不升、批次闭包机制及compact条件均成立，支持本批compact_damped_gain_preserved=true与adoption_pass=true。相较前轮原图23导致的采用失败，本轮图20门槛确已通过，前轮失败仍是原样保留的历史证据。

## 数据角色和剩余边界

FREEZE/全模型封存test_labels_accessed=false，各result.test_scores_computed=false均成立；未漂移worker仅取训练/验证，score在全模型封存guards之后才生成test/clean标签。源码、哈希链与时间记录支持先封存后评分，未做OS级访问审计。本次使用测试标签仅核验已有封存结果，没有调参、重选或续搜。

本轮与先前诊断、历史任务有关，PROTO明确为历史开发回归，不能转述为盲验证或新任务泛化保证。34/34保护仅指本批既定10%+1e-8数值边界；上限相同也不等于实际消费相同，本轮closure比random多394次评价，不能单靠本批判断其在相同实际评价数下优于其他追加提案方式。

base_search_fingerprint只散列有效cache状态，未保留全部失败提案身份序列；基础前缀、有效cache指纹及计数一致不能提升成每个失败提案轨迹完全相同的证明。已核父排序/前轮关系/实际matched图和warm/执行次序，未完整复演缓存/结构/warm筛掉的未评价提案，也未独立重做全部Rademacher数值、连续域interval盒子或物理坐标导出；34项现有VERIFICATION的相关通过记录与封存哈希已核，但本次不重复其全套计算。enclosed仍是有限浮点筛查。

科学接受边界：本轮恢复指定紧凑收益并通过本批冻结采用门槛；生产默认仍random、production_default_changed=false。剩余有理式发现问题、新任务/新种子验证和整体研究目标仍未解决。本审查不接受默认切换或发布，也不以本批通过代替这些后续证据。
