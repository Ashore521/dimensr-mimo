# 训练精度补充：正式开发批独立运行后审查

结论：本次独立检查支持34/34技术与配对身份通过、34/34干净输出保护，以及原预定开发采用门槛通过。完整接受由10/17增至11/17，新增接受来自rational_n0/702；SUMMARY全部字段与本次从原模型/分数/事件重建的汇总一致。未发现本次审查范围内的实质证据冲突。六个repair单元仍未完整接受；本轮不构成新任务/种子证据，不改变默认，也不批准1.7.0发布。科学接受及下一阶段设计由主任务决定。

## 实际检查与保存边界

本次只新增POST_RUN_REVIEW.md及post_review_checks.json，未修改FREEZE.files、source、frozen_inputs、runs、既有审查、协议、日志或评分。没有运行study/worker/initializer/fit/search、pytest、正式评分或核验脚本，没有重新选择模型、补跑或派生代理。

独立使用标准库核摘要、配置、分母、时间、事件、配对身份与全部SUMMARY字段；另用项目环境NumPy 2.5.4，按照冻结源码相同任务、种子、公式和随机流重建本轮原有数组，仅用于复核现有模型。17对训练/验证联合字节摘要全部匹配FREEZE与result，形状96×2、48×2，测试192×2。未新增case、种子、训练或实验。

对全部34个所选模型以自写原始JSON图解释器重算train/validation/observed-test/clean-test误差，对全部所选图独立核拓扑、单位、参数维数与四项结构指标；四类误差与保存值最大绝对差为5.204170427930421e-18。对788个matching及15个repair事件，重算父和有效子系统训练误差与逐输出父保护，最大父/子误差差分别2.0816681711721685e-17、6.661338147750939e-16，803个事件的保护布尔全部一致。

初次尺度检查将矩阵axis=0 std与逐列std做了严格相等比较；两种NumPy归约顺序产生最高1.1102230246251565e-16差异。复核冻结Evaluator实际逐列std后，34系统尺度逐值完全一致，没有数据或归一化变更；检查过程及修正说明保留在post_review_checks.json。

## 冻结、封存与预算

- 230份FREEZE输入当前字节全部匹配；FROZEN_INPUT_ARCHIVE.files与FREEZE.files逐项相同，frozen_inputs中230份按repo相对路径复制的输入也全部匹配。含40份本轮source副本，均与被冻结live src及其输入归档一致。独立数值复核结束后再次核全部输入/归档与运行摘要，仍无变更。
- ALL_MODELS_FROZEN的374份运行工件、34份CELL_FROZEN及各自文件清单逐项匹配。FREEZE→逐CELL→ALL_MODELS→SCORES→VERIFICATION交叉摘要全部匹配，各CELL时间按任务顺序单调且在TRAINING_STARTED与ALL之间；ALL清单含每个CELL及其工件。
- 全局时间为FREEZE 1793349829.6355176、TRAINING_STARTED 1793349830.5175378、ALL_MODELS 1793350528.8474898、SCORES 1793350558.0152256、VERIFICATION 1793350600.1673515，顺序一致。结合已审核评分入口，这支持全模型封存后统一评分；本次没有额外的运行时文件访问监控。
- 34个job与score、verification按顺序完全相同且唯一，两臂各17个，全部resource/result/score complete、returncode=0、verified=true，没有缩分母或补跑证据。每系统model.config=result.effective_config，符合冻结配置、seed与case门槛；配对有效配置唯一差异是precision_repair。
- 实际配置均52秒合作式fit、60秒外进程上限、1200评价、数值单线程/串行；预算2880秒含600秒评分核验预留。全部新评价数≤1200，最大保存fit elapsed为18.0秒，最大单元外wall为24.389999999999418秒。EXECUTION returncode=0、termination_reason=null、外wall=775.148854970932秒；VERIFICATION elapsed=770.5447947978973秒，均低于2880秒。本次没有触发边缘截止来测试抢占能力；52秒仍非单次BLAS任意时刻可中断保证，60秒外监督仍有0.2秒采样粒度。

## 基础身份、触发与实际机制

17对原base_candidate_count、base_candidate_sha256、base_evaluations、base_search_fingerprint全部相同；每份基础前缀的实际候选payload摘要也重核一致。15个未触发单元的整个验证候选池、最终有效cache摘要、best完整系统/常数/训练指标及validation误差与base相同。该相同不包含有额外flag/报告和时间字段的整个model JSON字节。

只在damped_n0/702和rational_n0/702触发；从各自原base前缀训练errors与既定acceptance.nrmse重算结果相同，固定父的平均训练误差、复杂度、图摘要排序一致。15个实际repair事件的原索引与implicit_retained_indices一致，实际隐式输出根共21次；本轮事件没有索引缺根情形，缺根warm路径的证据仍为运行前构造测试与源码审查，不能说本轮实际覆盖了该分支。

| repair触发单元 | 缓存外新评价 | 有效 | 无有效child | 父保护拒绝 | 保护/验证留存 |
|---|---:|---:|---:|---:|---:|
| damped_n0/702 | 8 | 7 | 1 | 7 | 0/0 |
| rational_n0/702 | 7 | 6 | 1 | 4 | 2/2 |
| 合计 | 15 | 13 | 2 | 11 | 2/2 |

这里“无有效child”为事件child=None的新评价，不是完整单元执行失败；两个失败均保留并计入评价预算。两触发单元各有4/5个nonimplicit_skips，cache_skips和structure_rejections均为0，stop_reason均completed。保存diagnostics合计implicit_linear_subproblems=6815、implicit_retained_indices计21，与SUMMARY及verification一致；6815是保存的线性子问题计数，本次未重新执行线性初始化来证明每次底层调用。15个完整系统事件均含实际保存的隐式来源，13个有联合拟合后有效child，2个通过固定父训练保护并实际入池，故实际机制证据超过配置开关。

matching两臂均394次新评价/394有效；base保护及验证留存111，repair为115，合计788/788/226/226。每单元至多两轮、每轮≤24，总≤48；repair各单元≤12。全部34系统满足总评价=base_evaluations+repair_new+matching_new；各事件new_evaluation和valid/protected/rejection数一致。跨两阶段事件图不重复；完整训练池摘要由原base+repair保护留存+matching保护留存重建一致，本轮全部扩展验证留存、无validation_rejections，验证候选子集也逐项相同。5474份已保存候选是两臂合计候选数，不是本次独立全量重拟合/全候选重算数。

34个系统从实际events均有4个不同NSGA-II step分别new_evaluations>0，每系统policy_delta>0、有效neural>0、GP新评价>0及Rademacher计数>0。closure运行事件与保存history分账一致；没有使用旧base+matching会计等式漏掉repair。有效cache摘要是保存指纹的配对身份核验；本次没有完整运行时cache字典，不能独立枚举或重建所有历史成功/失败缓存命中，也未重新计算Rademacher值。

## 指标与原采用门槛的独立汇总

下表从原SCORES逐行重汇总，并用全部34个所选图的独立原始数据计算核对，而非只复述pass字段：

| 指标 | base | repair |
|---|---:|---:|
| 完整接受 | 10/17 | 11/17 |
| 每系统干净NRMSE再宏平均 | 0.002614428507410396 | 0.0022923970183390385 |
| 平均共享图规模 | 21.88235294117647 | 21.176470588235293 |
| 干净逐输出保护 | — | 34/34 |

按repair≤base×1.1+1e-8逐输出重算34/34保护；全部10个base已接受单元仍接受。原绝对训练/验证/观测测试、干净测试、四项结构、总接受不降、宏平均不升及配对身份门槛均保留。用模型实际指标和结构重判完整接受与SCORES一致；SUMMARY的methods、pairs、分母、机制汇总和每个technical/adoption布尔全部与本次重建对象逐字段相同。

- **rational_n0/702：** base图27、6参数，train/validation/clean最大误差分别0.010707490583237413、0.009508337948239428、0.008688776191983036；repair图15、4参数、总展开21、深度5，对应最大误差分别2.826005429389069e-16、3.1822894606352185e-16、3.250286231277542e-16，满足完整接受及各≤1e-4。其单位与冻结定义一致，输入域为time[0.1,1.6]、length[0.5,1.5]，输出length及length²；有限浮点样本近机器精度不等于解析恒等式或域外外推证明。
- **damped_n0/701：** 未触发，base/repair所选图均20、4参数，训练/验证/干净误差均≤3.244997242801608e-16，满足图≤20与各≤1e-12。这里是保留原base紧凑结果，没有产生新的额外压缩收益。
- **前轮implicit退化的四个指定输出：** 本轮均未触发且保留base候选池与选择；独立重算的clean值两臂完全相同：polynomial_n1/702/output1为0.0033008511008706942，damped_variant/711/output1为0.0011550392786305306，rational_variant/711/output0为0.003581147993589743，polynomial_variant/711/output1为0.0003334414905537481。它支持本轮保护事实，不将前轮退化唯一归因于某个算子、拟合或选择阶段。

repair仍有以下六个未完整接受单元，继续保留原分母和失败：

| case/seed | 当前超出的原绝对门槛 |
|---|---|
| damped_n0/702 | output1训练/验证/测试精度及图规模25>24 |
| damped_n1/701 | 图规模30>24、参数7>6 |
| damped_n1/702 | 图规模26>24 |
| rational_n1/702 | 图规模27>24 |
| damped_variant/711 | clean output0误差0.02193441936406219>.02、图27>24、参数7>6 |
| rational_variant/711 | 图规模28>24 |

damped_n0/702补充没有保护留存；所选output1的train/validation/clean仍约0.0042042578/0.0035554388/0.0039823244，不能称已解决。表中是已验证门槛差距，不是对搜索失败原因的唯一诊断。

## 未检查项与后续边界

本次独立重算覆盖全部34个所选系统与803个扩展父子事件，并未重新计算全部5474候选的所有训练/验证误差；原正式VERIFICATION及被冻结脚本提供该全量候选核验的既有证据。未独立重新加载Estimator/PhysicalModel、重跑区间域审计、重算Rademacher或线性初始化；34份已保存域审计均enclosed，与score/verification一致，但enclosed只代表有限浮点筛查，不是形式证明。训练/测试访问隔离依据已审核源码、冻结/封存时间和摘要链，未进行额外的进程文件访问取证。平台主/子模型元数据依据主任务已核验日志，本次未独立重读平台日志。

本轮支持“保留原基础搜索、仅按训练缺口补充”的历史开发结果及预定门槛通过，没有全面解决六个剩余单元，也没有授权以本轮测试结果返工候选、改门槛、补跑或择案例切换算法。主任务可据此决定本开发批是否接受，再另行设计独立新任务/种子验证；生产默认和1.7.0发布仍不自动获准。

## 关键证据SHA256

| 文件 | SHA256 |
|---|---|
| `FREEZE.json` | `23c9a00582b6ef4987a6e56c9948a73403cc7908b783e791af507068cb6f7326` |
| `FROZEN_INPUT_ARCHIVE.json` | `d2d6b6e90b3d8362c8c958810887aae2dbf020754a0cee0a65856e2c6c1eb50c` |
| `ALL_MODELS_FROZEN.json` | `26b81c68d4cee5f2cdbc4c0c27d1d4b48e6f63348bf5804b0dad1d824c9e6031` |
| `SCORES.json` | `8400d7d16708096b0635c73f69b9050becdc2f6ff71e3e0059ffb2a48e029692` |
| `VERIFICATION.json` | `fa58d9cfb331b30d7967e34b2ae65a96b22dd4e3b27fd383134abcbcf8c0c777` |
| `SUMMARY.json` | `0a47773e8cbb5ec5f3669f218514e0bfd1f91e5ed118afe7add1c256ad6e21b6` |
| `BUDGET.json` | `cc53855436f98f9559d23321f70c99dfd5adc8104d8b682de7ab13e1b7815219` |
| `EXECUTION.json` | `6f2e5bc9bd2d957133d23a6d72d98dd547413882c3dfe0235dc451a5951ad5a3` |
| `post_review_checks.json` | `3c02e271a894829cd3e6cb6e4e590195843e8ed362a222a8e7a1b708e384e536` |
