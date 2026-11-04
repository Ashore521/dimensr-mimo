# compositional_completion：历史固定对照独立harness审查

2026-10-31。**当前审查范围无剩余实质harness阻断。独立技术判断支持主任务完成preflight、最终冻结/源码副本/原字节归档核验后，按原预算执行本历史批。不得自动启动development。** 本报告不代表正式运行、科学采用、默认改变或发布通过；最终执行和接受决定属于主任务。

## 范围与发现处置

本工作包连续审查共享PLAN/DESIGN_REVIEW/CORE_REVIEW、两阶段PLAN/PROTOCOL、共享harness.py/verification.py、八个阶段入口、两个新测试模块及旧隔离复用依赖。没有重做已完成独立核心审查；核心报告SHA为`25f10893b445bb0a37c4d9e1d0c5c63a187f5eb4c7730bcafa4ec47cc6c35b51`，其41份源码快照逐字段等于当前候选锁。

1. **前阶段门槛类型缺口，已修复。** 初版harness.py:82以truthiness检查technical_pass/adoption_pass。独立纯内存AST fixture显示，七项hash链完全匹配但这两字段为字符串'false'或整数1时仍被接受；bool False正确拒绝。本检查没有落盘或伪造生产接收收据。主任务将两字段改为`is True`，最终同一内存fixture仅接受True/True，其余拒绝。新增类型测试先复现4项失败，修后通过。
2. **审查日志漏冻，已修复。** 主任务在检查中补充确认freeze_inputs未枚举harness_review_validation，新增构造先复现此缺口。最终harness.py:144加入该目录；本审查实际只读枚举清单，确认本阶段harness_review_validation/static_checks.log在输入路径中。原5失败日志和后续107通过日志均保留。两项修复只涉及共享helper两行及对应开发harness测试，41生产源未变。

当前未发现其他有证据支持的实质阻断。首版worker的final_readonly.log绑定14个文件，仍原样保留；其中helper和开发测试因上述主任务修复与早期绑定不同，不能把旧绑定误称最终版本。最新TEST_VALIDATION逐文件绑定当前代码，差异恰为full-suite之后声明的两文件。

## 矩阵、参数、输入与阶段顺序

本阶段固定17对/34系统/34配对输出，来源任务/spec/seed不变，臂顺序按种子奇偶反转。两臂均保留原完整compound神经/GP/NSGA-II、联合拟合、polish、precision选择和24+24 matching；有效配置仅交换precision_repair/compositional_completion两个互斥flag。共享基础配置读取原precision_repair FREEZE.config并核原ALL→FREEZE；两阶段同配置、同41源码锁。worker只取train/validation，规格、单位、常量目录、边界和接受阈值不注入oracle或真式。preflight还必须逐case比较旧spec与96×2/48×2训练验证摘要；本次未调用它或生成这些数组。

`check_stage_gate`对development核真正历史阶段POST_REVIEW_ACCEPTANCE：status=accepted，sha256键集合恰为FREEZE、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、POST_RUN_REVIEW和共享CANDIDATE_LOCK七项，逐值与当前文件摘要相同；再核技术/科学bool True、VERIFICATION.status、VERIFICATION→FREEZE/ALL/SCORES、ALL→FREEZE、SCORES→ALL、同source锁及同config。缺失、科学失败、未接受、变更config/锁/summary或畸形bool均拒绝。harness没有写接收收据，也不复用旧accepted_as_independent_post_review；历史入口不会启动下一阶段。

## 冻结依赖和原证据

实际只读核当前41份src的完整集合/摘要、CANDIDATE_LOCK及CORE_REVIEW_CHECKS快照一致。本阶段原来源的230份frozen_inputs按旧FREEZE原hash全匹配，归档清单及FREEZE交叉hash相同；旧ALL的374份运行工件全匹配。本次不要求已修改的live src继续匹配旧算法，而是区分旧归档原字节与新执行依赖。

共享freeze_inputs包含旧FREEZE.files实际路径形成的live依赖/测试fixture闭包，同时包含旧frozen_inputs每项原字节、manifest、旧seals/评分核验/summary/独立后审和全部旧运行工件；另含当前src及新增本阶段source副本、全部顶层tests、两阶段入口和协议、当前阶段md/json、共享设计/核心审查/锁/检查、TEST_VALIDATION/PREFLIGHT、core/core_review/harness/harness_review递归非pyc日志与辅助文件、pyproject和文档。development另冻结历史七项接收链及POST_REVIEW_ACCEPTANCE。冻结前核旧归档/运行原hash，冻结后按实际新清单核live与source副本，主任务另建原字节输入归档收据。

本检查点清单里当前HARNESS_REVIEW尚未存在是正常的审查中状态，本报告现已写入；PREFLIGHT必须由主任务在门槛成立后生成，未伪造。development缺历史新seals/后审/接收收据是预期阶段阻断，不作为既有代码缺失。正式FREEZE前仍须核最终实际清单，不能用本次预估路径数代替最终冻结事实。

## 测试隔离、封存和预算

每阶段独立importlib加载cases、旧precision_repair verifier及其compound raw base，显式绑定各自HERE/cases/study，不修改历史模块单例。case测试接口先检查当前ALL存在，真正scorer在任何test标签读取前执行候选锁、阶段gate及完整旧guards：FREEZE/ALL全部文件hash、交叉hash、test未访问标记、FREEZE→TRAINING_STARTED→按序CELL→ALL时间、各CELL及列明工件在ALL中的成员/hash。当前冻结矩阵与source锁也在guards核对；SCORES再绑定ALL及评分时间。仅文件存在不是完整隔离证明。

入口/preflight/freeze/train/每cell/worker及评分入口检查source锁，训练每cell还核冻结输入。已有CELL只核后跳过，有job未封存要求事故复核；失败/超时/未启动均保留，统一ALL后才评分。真实训练接口拒绝test，worker不传truth_graph或公式。未进行额外运行时文件访问监控；本结论来自已审代码和构造测试，不能冒充正式批已证实零泄漏。

原每批2880秒BUDGET含600核验预留，52秒合作fit、60秒cell进程树、1200评价，单worker串行、OMP/OpenBLAS/MKL/torch单线程；外监督.2秒采样，保留独立60秒preflight启动窗口，不称从入口严格2880秒。没有借下一批预算或自动续跑。合作式BLAS不可任意抢占。入口发现执行工件拒绝直接重启；坏VERIFICATION按未通过保存外部执行结果。外部硬kill、解释器崩溃或不可写磁盘可能无法写完Python收据，届时保留现场并报告不完整。

## 验证、身份和新补充来源

原raw路径保留所选图train/validation/test重算、全部候选训练验证误差、单位/原结构/共享图、native/PhysicalModel真实load和预测比对、保存预测摘要、原域审计及precision选择重建。逐系统真实策略变化、有效neural、GP新评价、至少两不同NSGA-II step各有新评价、Rademacher正计数仍为硬要求；enclosed只是有限浮点筛查，不是连续域证明。

新completion回放核always模式、固定训练最优原父、原训练eligible记录、每输出≤12返回根、canonical/单位/grammar/15节点/6层、原Y上的MSE/var+1e-8*nodes分数及score/size/repr顺序。旧implicit原位置与合并后位置分开，新变换核log/sqrt、direct/multiply/divide、外幅值、basis索引0..23和原图外形。逐事件root_sources必须来自同一返回位；缺位只复制固定父对应根，重建完整graph/warm后比对，不循环补齐。None子和new/valid/protected分账；每输出父×1.1+1e-8保护由raw重算。

旧linear_subproblems已包含implicit子集，核总线性=旧全部+新变换全部、implicit另列；截止途中诊断保留但不冒称返回根。新补充最多12，closure每轮24/总48，所有新评价共用1200/52秒；总评价=foundation+对应补充+matching，完整训练池、验证过滤子集、拒绝摘要及runtime/history计数均回放。缓存外事件图跨阶段不重复、不在foundation候选图中；完整运行时失败/成功cache字典本次没有重新枚举，原core的实际cache跳过逻辑及保存指纹仍为证据边界。

每对candidate/control的foundation完整payload/count/hash/evaluation/cache相同；control还逐model/result/delivery/pool/test score比较旧repair臂。仅归一声明时间字段、版本/schema元数据、新False默认、job方法改名和派生source_model_sha256；不删数值、参数、结构、评价、有效cache指纹、候选payload、选择或算法事件。旧原生load先按旧payload验证来源再迁移；control的训练触发/skip用旧repair回放，skip还检查旧同批base整池/cache/选择。新始终补充模式没有假设skip身份。

matching每臂new/valid/protected/validation_retained汇总均>0仍为技术硬门槛。补充实际变换/implicit来源、线性、返回根、缓存外系统、有效/保护/验证留存和接受增益按族另列。没有有效新根或留存不能宣称新增复合发现；本比较同时改变始终补充与初值构造，事件来源不支持单组件因果归因。

## 科学门槛与失败收据

保留旧repair的11/17接受及原剩余六失败；34/34输出保护；全部原已接受保留；宏干净误差与平均图不升。damped_n0/701须完整接受、图≤20/参数≤4且train/validation/clean各输出≤1e-12；rational_n0/702须完整接受、图≤15/参数≤4且对应误差≤1e-4。没有把历史门槛改成17/17都接受。

全部原case绝对精度/结构、单位和域门槛保留。科学通过必须同时满足所有条件，不能以宏均值抵销单输出退化。summary分别保存technical/adoption、完整34/17/34分母、逐对保护、foundation/control身份、特殊目标或candidate_all、失败原因及真实来源。旧接受保留依赖control完整身份，不是重新命名一个不同基线。

缺失、重复、错误case、负/非有限/不完整误差、结构或非法计数阻断通过；没有完整预定指标时方法均值null，不只平均成功子集。有限错误包装在原scorer的cell try内转scoring_failed；整阶段异常写SCORING_FAILURE。verify把budget/guards/score准备异常转为全部预定失败行，来源hash不存在为null，不读模型/标签；既有score/verification/scoring_failure拒绝覆盖。相关构造覆盖这些路径，没有假造生产通过收据。

## 实际测试证据与未知

读取最终TEST_VALIDATION并独立核当前validated_input_files、日志及suite receipt摘要，均匹配。完整394项/89.40s发生在门槛类型与日志目录两行修复之前；修后全部两个harness模块107项/2.79s通过，4项新类型用例使总398个不同用例分批覆盖。不能称最终单次398全套通过。原103项harness、394全套、初次旧guard隔离失败/修复、5个门槛复现失败及后107通过日志均保留，不把失败抹成通过。

本人没有重复pytest、正式case生成、initializer/search/fit、评分或正式PREFLIGHT/FREEZE/BUDGET。实际构造只执行提取的gate AST与纯内存假hash/文档，不创建生产POST_REVIEW_ACCEPTANCE；其余是只读源、AST/路径、hash、清单和日志核对。只新增本阶段HARNESS_REVIEW、HARNESS_REVIEW_CHECKS及harness_review_validation/static_checks.log；不改生产代码、测试、协议、锁、旧结果或核心审查。

未验证正式数据、墙钟边缘、实际新补充/拟合效果、逐对正式身份或全池正式数值；这些只能由冻结执行和独立POST_RUN_REVIEW确定。未核安装态打包或远程发布，也未重读平台运行模型日志。无当前harness阻断不等于全面科学接受；生产默认及发布边界不变。

## 最终审查绑定

| 文件 | SHA256 |
|---|---|
| shared/CANDIDATE_LOCK.json | 51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452 |
| shared/harness.py | ef0e2873cf42b2672282280e6bbceb300b76a5b1a1ebdf17b80ad7bddffc28fc |
| shared/verification.py | 14f505143f8a371aa8db6bef745e006b01f86b8c9c8a87a05632e376825d6f5d |
| 本阶段PROTOCOL.md | c37f79493ae579b27b309f0cf7337285f7827b1b4a7024e0a7027ce3051b28fa |
| 本阶段TEST_VALIDATION.json | cd42870eb9a7926673f76d11ffe331b3e5b8f878b3b62a102ddb2f2efe085283 |
| 本阶段HARNESS_REVIEW_CHECKS.json | 3c5bcf2df3c76909325026ba854d2a26109c98946a70fe15f6480062ca5d1300 |
| harness_review_validation/static_checks.log | f9d2d68316d9632d24d617dcddba1f5caf31c79cd6ddd051532453b459e6b35d |
