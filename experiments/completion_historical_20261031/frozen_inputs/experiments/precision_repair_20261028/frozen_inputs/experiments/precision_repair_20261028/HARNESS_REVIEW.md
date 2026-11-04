# 训练精度补充 harness：冻结前独立审查终态

What this change does: 在原17个已观察任务/种子对上比较 base 与 repair，保存完整搜索、补充和闭包证据。全34系统封存后统一评分，并独立解释原始JSON图、重算指标、回放两阶段事件及全部采用门槛。

**结论：本次发现的冻结依赖/source副本缺项及CELL时间链核验缺口均已修复并完成受影响复核。当前已审核心与harness无剩余执行前阻断。主任务可在核对下表稳定字节、将本报告及最终测试收据纳入冻结后，按原授权顺序执行preflight→freeze→受限正式批次→全模型封存→评分→核验；本报告不授予新预算、补跑、门槛变更、科学采用、默认变更或发布。最终验收由主任务决定。**

## 范围、实际检查与版本

本次仅写本文件，未改PRE_RUN_REVIEW、核心、harness、测试、数据、历史及失败。使用ponytail-review审查实现与调用链；先完整审查稳定harness，再复核主任务完成的两项计划内修复。读取study/worker/evaluate/run_bounded/PROTOCOL和test_precision_repair_study，追踪compound_search、first_round supervisor、second_round.previous、native_api raw/guards、typed delivery/audit、任务定义及实际导入的旧selector。相关核心范围见PRE_RUN_REVIEW:35起追加段，该文件字节未变。

本次执行的命令只做文本/AST/摘要/路径存在性检查，没有运行pytest、preflight、freeze、initializer、fit、训练、搜索、正式数据生成、评分或核验脚本，也未派生代理。测试结果来自主任务已有日志，不声称本审查独立重新执行了数值计算。检查时尚无本轮PREFLIGHT/FREEZE/BUDGET/runs/ALL_MODELS/SCORES/VERIFICATION，因此以下运行链是实现检查，不是本轮已发生运行的认证。

- TEST_VALIDATION中15份日志字节摘要与40份当前src摘要全部匹配；AST展开新增harness测试为44项。核心6个新增/挂接文件与此前核心终态摘要一致。
- 已查看source_suite_01.log的163 passed in 76.08s；它排除当时仍在编辑的本轮新harness模块。最终harness_review_fixes_02.log为52 passed in 2.09s，包含44个当前新harness用例和8个旧兼容用例。207表示163+44个不同用例的分批覆盖，不是单次207项调用，也不把8项重复兼容测试另加一次。
- 原core_targeted_01的27通过/1失败、harness首轮13失败/16通过、次轮1失败/28通过及随后所有通过日志均保留。已查看首轮Fraction JSON fixture失败与次轮预期报错顺序不匹配的具体记录；没有删除失败、重命名成正式运行成功或据测试分数改科学阈值。

## 发现并已关闭

1. **冻结清单漏掉已导入旧依赖与source副本。** 初版study.freeze未列native_api.verify导入所需precision_delivery/selector及其guarded selector，也未列部分完整测试依赖、两个旧fixture；source副本虽复制却不在核摘要集合。当前study.py:76-94已纳入相关17个旧helper目录顶层.py、两项补充fixture和每个source副本，统一受原verify_files核验。指定旧模型、两份旧源码fixture及新增两fixture当前均存在；17目录均存在。新冻结构造测试检查依赖、fixture键、复制内容及副本冻结键。该问题已关闭；本次没有实际生成source副本或FREEZE。

2. **评分前未验证逐CELL封存时间链。** 初版复用guards只检查FREEZE≤TRAINING_STARTED≤ALL，verify_cell只核CELL所列文件摘要；CELL早于训练、晚于ALL或顺序倒置可漏过。当前evaluate.py:29-45包装原guards，逐任务要求TRAINING_STARTED≤CELL时间、各CELL单调且≤ALL，ALL明确包含CELL当前摘要和其每个工件摘要，并核实际CELL文件集合；base.guards被显式替换，复用的evaluate实际调用新guards，verify入口也使用它。新增7状态构造测试覆盖有效、过早、过晚、倒序、缺CELL、缺工件及CELL改字节。该问题已关闭；旧helper未改。

## 合同逐项核对

**矩阵、预算及失败。** study.matrix维持原6个family/noise任务×2种子加3变体、2回归，共17对34系统，奇偶种子反转臂顺序。study.configuration沿first_round冻结配置保留24 population、4 epochs、2 GP代、4 selection代、1200评价、3个polish及原完整神经/GP/Rademacher；两臂均compound/closure/precision，worker唯一有效配置差异是precision_repair。worker及closure_evidence独立重建并核有效配置、seed、case门槛。

52秒为同一Evaluator合作式fit截止，60秒为原进程树单元监督；原supervisor每0.2秒采样/检查、kill子进程与主进程，核验对complete单元保留原0.5秒采样容差。原数值线程环境均为1，torch线程为1，串行单worker。2880秒从BUDGET.start_unix起算，训练在剩余600秒前停止启动新单元；评分和核验共享该截止，外层继续监督全部流程。入口preflight仍有原单独60秒启动窗口，BUDGET在freeze生成，因此不能把该实现描述成从入口到退出的严格2880秒上界；此口径沿用既有harness，没有本轮增加LIMIT。合作检查不能抢占已进入的单次BLAS/NumPy，外监督有采样粒度。

train保留not_started_budget、失败、timeout的resources/traceback/日志，并封存该单元；已有CELL只核验后跳过，不补跑，未封存但有job的单元要求事故复核。外层遇既有执行证据拒绝重启。全局进程被终止可能留下未封存检查点或未完成汇总，应报告执行失败并保留现场，不能据现有部分结果缩分母或自动补跑。

**冻结及评分隔离。** freeze纳入当前src及对应source副本、全部顶层测试、harness源码、全部当前Markdown（含PLAN/PRE/本HARNESS_REVIEW/PROTOCOL）、训练审计、PREFLIGHT、TEST_VALIDATION与日志、pyproject、旧helper与兼容模型/fixture。记录依赖版本、base commit、预算、各任务spec及训练验证数组摘要。输入来源live src与source副本在同一冻结清单；训练每单元前、评分及核验入口均核FREEZE.files。

训练worker只请求train/validation；cases.training_data拒绝test，训练与验证分别96/48，测试192只在封存后由评分器请求。cases生成器内部使用latent定义构造各分区数据，不把干净测试指标传入触发或候选选择。所有模型全封存前不调用评分。guards核freeze摘要、全模型工件、test_labels_accessed及FREEZE→TRAINING_STARTED→逐CELL→ALL时间链；score保存ALL摘要和评分时间，verify要求ALL≤score并保存FREEZE/ALL/SCORES摘要及核验时间。所有当前文件字节及最终全模型时间链仍须正式运行后核对。

**repair及闭包回放。** evaluate.repair_evidence:110起仅以原base前缀训练errors和既定nrmse重算触发，不含结构、验证或test。重建固定父的平均训练误差/复杂度/图摘要排序，核每输出seed上限、implicit_denominators≤64、implicit_retained_indices及原索引；缺索引根用root_payload比较完整根结构、参数别名和warm数值。部分deadline diagnostics允许保留，但不能被当作完整系统新评价。

extension_evidence逐事件核父属于原base/已保留池、repair父固定、完整图结构/单位及warm维数/有限性，用独立raw解释器重算父子各输出训练误差并重算×1.1+1e-8保护。child=None的失败不计valid/protected；缓存外评价按new_evaluation总和计数，故新评价失败也消耗预算。跨两阶段新图摘要不得重复；核心及构造测试另保证基础成功/失败缓存图不refine。此处不是完整失败缓存字典的独立重放。

evaluate.closure_evidence:157起核matching每轮≤24、至多两轮/48，新图须缩小；repair≤12；总评价严格等于base_evaluations+repair新评价+matching新评价。完整训练池严格等于原base+repair保护留存+matching保护留存，并核完整训练池摘要。对两类扩展分别raw重算验证有限性/输出边界，重建验证留存子集、类型计数和拒绝摘要；全部拒绝继续留在事件证据中。核模型history与runtime事件的两阶段分账，没有调用旧closure verifier错误的base+matching总量等式。precision基础参照仍仅原前缀，核base_choice摘要和原训练/验证保护。

**配对身份、实际机制与科学门槛。** paired_identity核17对原base count/hash/evaluations/有效cache摘要；未触发repair时进一步要求整个验证候选池、最终有效cache摘要、best完整状态及validation误差均相同。墙钟扰动引起不一致会使technical/adoption失败，不能以同预算替代相同轨迹证据。

study.mechanism在原policy变化、有效neural、GP新评价要求外，额外要求至少2个不同NSGA step各自有新评价；统一评分及base.verify_cell继续要求实际Rademacher和机制证据。summarize要求两臂matching分别在批内new/valid/protected/validation_retained均>0，repair批内trigger/implicit线性子问题/获胜implicit索引/implicit完整系统新评价/valid/训练保护留存均>0，单独报告repair验证留存。允许单元0，但不能以flag为执行证据。

继承的compound评分/verify_cell对原始JSON图、重载、物理模型、所有保存候选训练/验证误差、所选测试误差、单位/结构及所选域审计进行核对。summarize拒绝缺失/重复score或verification，34系统与34个配对输出的分母固定；不完整单元不计通过、均值在缺单元时为None。原无噪声1e-4、噪声观测.03/干净.02、图24/参数6/展开30/深度6、全部base已接受保留、总接受不降、宏平均干净误差不升、每输出≤base×1.1+1e-8，以及damped_n0/701图≤20和各精度≤1e-12、rational_n0/702完整接受且各精度≤1e-4全部保留；均值或结构收益不能抵销任一保护失败。

## 检查边界与主任务下一步

Not checked: 未执行真实34系统，未重新生成数组、实际加载本轮模型或独立跑raw/单位/域/指标核算，也未实际触发52/60/2880秒监督；未从运行时完整缓存重建全部失败图或按保存种子树重新运行implicit生成。原索引implicit来源核验依赖保存的diagnostics与已审核心生成顺序，不能当作独立重新生成全部种子的证明。逐候选训练/验证重算由正式verifier计划执行，域重新审计主要针对所选系统；不是对全部候选各域的独立重审。enclosed仍仅有限浮点数值筛查。平台实际模型依据主任务核验的外部日志/收据，本次没有独立打开平台会话日志。

主任务应先确保最终本报告/收据及下表版本进入FREEZE，并确认全部所需输入摘要与source副本一致，再按既有授权执行。正式输出若有失败或配对身份不一致，保留原门槛、失败、完整分母及检查点；不得据测试结果返工候选或补跑。即使开发批adoption_pass=true，仍是已观察任务的开发回归，不是新任务/种子验证；默认、发布和最终科学接受继续由主任务按原合同决定。

Verdict: 当前审核版本无剩余执行前阻断；可以进入原授权冻结与受限执行，尚无本轮科学结果可接受。

## 最终检查点SHA256

| 文件 | SHA256 |
|---|---|
| `experiments/precision_repair_20261028/study.py` | `756b786e49d399e5bfbdec2a47b8631029b3a5efb3b1c4b3f4f021aab4ae6e12` |
| `experiments/precision_repair_20261028/worker.py` | `f11bf25d66ffc1c014aed1d0bcacc323e3fccc09bd59607dd5d780bb4eceef9e` |
| `experiments/precision_repair_20261028/evaluate.py` | `e9ac3ffacc4cf193069fb7e47201a88e5b80d8946b81e51f485a6828c442cdd1` |
| `experiments/precision_repair_20261028/run_bounded.py` | `1d8d7d35a8292ed63d6387c93c9eef96985ddcdcf9ac1d6a411d864b335a65ed` |
| `experiments/precision_repair_20261028/PROTOCOL.md` | `c2e972bc031ba450f6386a19cc74400ed40e7963936f947a4d9296772d2a3707` |
| `experiments/precision_repair_20261028/PLAN.md` | `5e246f1112644519c171d34328f759f68681e0471c58bf4f37c1d9af658a7b05` |
| `experiments/precision_repair_20261028/PRE_RUN_REVIEW.md` | `bbafcd6fe2975e3fbcb3b80681049bc180fa45d7ede11b52fe9f539c35507b63` |
| `experiments/precision_repair_20261028/TRAINING_GATE_AUDIT.json` | `5191d2567e228129479642d76b290a41151639773fdb4ad9f2c0353739e694ec` |
| `experiments/precision_repair_20261028/TEST_VALIDATION.json` | `71999c8ea6bcdb3c39edd99cbbd14b8e278cd562b10be90880c392bc74dc6036` |
| `tests/test_precision_repair_study.py` | `3e702f1ddfd044214d5d4ba3636d8f7a46c981e21001f72c8541ae13793d7f6a` |
| `experiments/precision_repair_20261028/validation/source_suite_01.log` | `1a3e8872302ffa3068fb210dd27a30b5235f680fc2724285d6a2f98832b0fa28` |
| `experiments/precision_repair_20261028/validation/harness_review_fixes_02.log` | `28fe21057dd39783393044ea3522eec446695451cfcb1fdb45f81d7af67befa5` |
| `experiments/compound_search_20261027/evaluate.py` | `035caf7a5e35f6ca75bd1e010270f8fec6a47b1b92da9392b4a4ba9e83f3b5e5` |
| `experiments/compound_search_20261027/study.py` | `ff37c6cd90548f9d0eb8157cde3a7de9866ad8124e9d298475808fbae1c22d26` |
| `experiments/compound_search_20261027/cases.py` | `d241301c0739022311318bc4fe6c1116e328916e6fdb44b97548db2629330873` |
| `experiments/first_round/study.py` | `2c5240c45e6dd5b19f52f9597cf68d07cd6797c89843ea04c72e53b08eed55ac` |
| `experiments/native_api_20261026/verify.py` | `2d206e2e29c4ad595edf470051916c99cdaf94c95224ce01461e423fa901eef5` |
| `experiments/precision_delivery_20261026/selector.py` | `98afdfe7e9414c71ee066c6f65dd989009a305cd0faf83531175a2fe3200bcd8` |
| `experiments/guarded_selection_20261026/selector.py` | `ed0cc8d7c938de870ff7d4e8bedb682c45b9e4002f88d0adc0d4e9a401505c6e` |
