# 新任务验证：稳定实施与 harness 独立窄复核

2026-10-30。结论：本次审查发现的整阶段失败收据缺口已在新批次局部适配层修复，并完成限定复核；当前审查范围未发现剩余实质阻断。稳定 harness 与已审核设计的矩阵、源码锁、隔离、原预算、机制及科学门槛相容。此结论支持主任务继续核验其执行前条件，不代表新任务已运行、算法科学验收通过、生产默认可改变或版本可发布。

## 范围和方法

只新增本报告；未改 PRE_RUN_REVIEW、DESIGN_AUDIT、cases、生产源码、测试、历史工件或其他作者文件。未调用正式数据生成、initializer、fit、search、评分或pytest，未派生代理。通过只读源码追踪、标准库 AST/路径/摘要检查及读取已完成日志核证。设计层的四组真式单位、域、语法可表达性和来源比较沿用本审查者已保存的设计检查点，没有重新选择任务或重复全部设计探索。

完整阅读本轮 study.py、worker.py、evaluate.py、run_bounded.py、PROTOCOL.md、cases及两个新测试文件；追踪 precision_repair_20261028 的独立加载评分器、compound_search_20261027 的 raw scorer/verify_cell、first_round 的串行监督和机制统计、native_api 的原始JSON解释器/require、typed_coverage 的交付/候选池、regressor.load 的单位及precision选择重建。修复后只复核 evaluate/study入口/协议及相关新测试变化，没有重做已核稳定部分。

## 已发现并解决的实质问题

初始本轮 evaluate.py（SHA256 `63336e62020f0ba15853bcac1bbf82aaa0beab33e1627baef1cce39669f31071`）的 verify 在逐cell异常处理外读取 SCORES；继承 scorer 的评分预算检查、resources读取也在cell异常处理外。若整个SCORES缺失/损坏或评分阶段提前异常，原 `evaluate(); verify()` 会在生成固定分母核验收据之前中断。原测试只覆盖文件存在而 rows=[]，不能证明整个文件缺失时有32行失败收据。该问题不把失败误判为成功，但违背本轮明确要求的失败可审计交付。

最终修复限于本轮适配层：

- `evaluate.py:200` 的 run 只尝试一次评分；整阶段异常写 SCORING_FAILURE 后进入 verify，既有 SCORES/SCORING_FAILURE/VERIFICATION 拒绝覆盖。
- `evaluate.py:148` 的 verify 先固定 study.matrix 的32个job，将budget、guards、score读取和索引准备放入同一异常边界。前置异常使32job逐一 verified=false；score_rows保持空，宏均值为null，不读取模型/标签、不把缺失摘要伪造为存在。缺失、坏JSON、坏行、评分budget/resource异常均有新增构造覆盖。
- `evaluate.py:47` 拒绝非有限及负误差、负记录均值、不完整二维指标和非法结构值；非法输出不能通过平均值抵销。
- `study.py` 正式run路由改为调用上述 run，无重试评分或补跑搜索。

源码与 `review_fixes_02.log` 支持上述修复。不可写磁盘或外部硬kill仍可能使Python来不及写收据，最终协议已明确此边界；届时必须依据外监督失败及已有现场报告未完成，不推定通过。本次没有实际触发外部硬kill或磁盘失败。

## 矩阵、配置、源码与冻结输入

`study.py:53` 矩阵固定8个case×9131/9132×base/repair，32个唯一系统、16对、32配对输出，种子奇偶反转臂顺序。worker只接受两臂。`study.py:36` 在入口/preflight/freeze/train/每cell和独立worker路径检查40份src的完整路径集合和SHA；本次实际重核40项和集合全部匹配候选锁。`study.py:58` 读取前轮冻结config，先核旧ALL的FREEZE交叉摘要；锁中的前轮SUMMARY摘要也实际匹配。

worker仅补入同一seed、compound、closure、precision和case acceptance，两臂唯一配置差异是 precision_repair。原24种群、1200评价、15节点/6层、4神经轮/每轮2GP、4NSGA-II、常数拟合/polish和24+24 matching保持。真图、真系数、公式字符串及跨输出配对规则未传入 estimator；worker只请求train/validation，PhysicalModel/交付使用单位、名称、边界与接受规格。case用目录分离，输入输出identity；两噪声层X不同已在协议明确，不解释成配对噪声因果结果。

`study.py:84` freeze_inputs 纳入旧FREEZE完整依赖/fixture路径闭包、旧FREEZE/SUMMARY/POST/HARNESS/评分核验/归档收据及旧ALL列明全部运行工件、新src/顶层测试/本轮脚本、全部当前Markdown审查、日志和TEST_VALIDATION、四类锁/审计、34份history_sources、PREFLIGHT和pyproject。`study.py:99` 为当前40源码另存source副本并将副本本身纳入FREEZE；预算版本/训练验证摘要随冻结保存。HARNESS_REVIEW在冻结前已存在时会由Markdown通配纳入；主任务应核最终实际清单与原字节归档收据，不能用计划的路径集合代替最终冻结事实。

本次重核前轮FREEZE的230份输入及旧ALL的374份运行工件全部匹配，旧配置链/SUMMARY锁及34份来源副本均匹配；未修改历史。已核本轮TEST_VALIDATION的40份src摘要与候选锁逐字段相同，11份日志摘要全部匹配。初期只读清单中PREFLIGHT/TEST_VALIDATION尚不存在符合当时阶段，不作为开发缺失；最终审查时TEST_VALIDATION已写入并核验，PREFLIGHT/正式FREEZE仍未产生。

## 封存、标签与模块隔离

`worker.py:18` 只用training_data(train/validation)，测试接口拒绝test；case.generate(test)的ALL存在检查只是一层标签闸。真正评分入口在任何标签读取前调用本轮guards，先核候选锁，再复用完整旧guards：FREEZE/ALL全部文件摘要、FREEZE交叉摘要、未读test标记、FREEZE→TRAINING_STARTED→逐CELL→ALL单调时间、每CELL及其列明工件在ALL中的成员和摘要。`evaluate.py:148` 再核ALL→SCORES时间、score交叉摘要及冻结矩阵。

importlib分别新建repair verifier及其compound base实例，本轮只重绑这些实例的 HERE/cases/study 和独立base.error/guards；历史模块单例保持原绑定。被引用的函数通过各自独立globals访问本轮路径和case。构造测试显式核模块身份、函数globals、旧绑定不变，以及guards拒绝时尚未请求标签。新 finite-error 包装保留原数学误差公式，只在原cell try内将非有限结果转成scoring_failed；不会改变旧 scorer 单例或让异常单元变成成功。

训练逐cell核FREEZE文件；封存CELL只核验保留，已有job未封存要求事故复核，不重跑。全模型终态封存后统一评分；失败、超时、未启动保留矩阵。此次未做运行时文件访问取证，隔离结论限于源码、路径/摘要设计及已完成构造测试，不能冒充实际批次已证实零标签泄漏。

## 预算、技术与科学门槛

`run_bounded.py:15` 在60秒preflight窗口后沿BUDGET监督全进程树，.2秒轮询；数值库与torch单线程，串行一个worker。系统60秒外部上限、52秒合作fit、1200评价；study在启动前保留600秒评分核验，总2880秒由freeze中的BUDGET起计。既有执行证据拒绝直接重启；不因失败补跑。原合作式原子数值调用边界、采样精度和入口与BUDGET起点差异均在协议保留。本次没有实测这些截止边缘，也不把相同上限说成相同实际工作量。

复用逐系统验证包括：模型/PhysicalModel真实load，保存预测摘要，独立原始JSON解释器对所选train/validation/test预测逐元素核对；交付单位/共享图及四项结构重算；全部候选train/validation误差复算、pool诊断、precision选择重建；原域重审；实际策略更新、有效神经候选、GP新评价、至少两不同NSGA-II step各有新评价和Rademacher。域enclosed仍只是有限浮点筛查，不是形式证明。

repair与closure各自回放父/子原图、训练误差和固定父保护，检查隐式来源/原索引缺根回退、触发只看基础训练池、实际缓存外评价、原12/48及每轮24边界、有效/保护/验证留存、完整池重建、事件/history一致，总评价为base+repair+closure。16对基础count/hash/evaluations/cache身份、所有skip对的完整候选池/最终cache与所选图参数/验证误差一致要求均保留。每臂matching new/valid/protected/validation_retained汇总全部>0仍为技术硬门槛。

`evaluate.py:62` 的summary将技术与科学分开：完整唯一32score/32verification集合、每臂16完整有限指标、所有verified、机制及身份是技术条件；科学条件还同时要求repair16/16完整接受、32/32逐输出干净误差≤base×1.1+1e-8、base接受全部保留、宏干净误差与平均图均不升。完整接受仍由训练/验证/观测test及干净test绝对误差、结构、单位/域组成，未混入旧damped/rational特定任务门槛。缺失/重复/非法值阻断通过；宏均值只在整臂指标完整时产生，不对成功子集平均。正常摘要另给固定每族/噪声层和16对逐单元结果。

repair跨族实际作用按四族分别记录trigger/隐式子问题与根/new/valid/protected/validation留存及接受增益。零触发或无留存可技术通过，但discovery字段及claim_scope不会把它称作repair跨族新增发现收益。没有全体新结果前不能预判这一分支或整体门槛；实际事件不替代性能门槛，性能均值不抵销单输出退化。

## 测试证据和未检查项

读取最终 `TEST_VALIDATION.json`（SHA256 `4bc681a73e9f5cbad2950d85912419f6cb02fbadafb91f40686986344b7ff520`）及日志，而未重跑测试：完整source_suite_01为251 passed，发生在最后失败收据修复前；随后review_fixes_02为95 passed，包括最终45个新harness、6个case和44个旧repair harness用例。因此是258个不同用例的分批覆盖，不能写成“最终单次258全套通过”。前期case测试使用默认缺算子配置的3失败/3通过日志保留，改为既有冻结研究配置后6 passed；目标公式与生产算法未因该失败更改。此前78/80/81/82通过日志及两次review修复记录均保留，未用反复运行扩大性能样本分母。

构造用例覆盖唯一flag/禁止test与真图注入、源码新增/缺失/变化拒绝、每cell锁、冻结传递依赖/源码副本、隔离模块绑定、前置guards、32/16/32分母、负/非有限/缺失指标、全部科学与身份门槛、每臂matching事件及无触发/无留存口径；旧44用例补充repair/closure原图来源、会计、保护、验证拒绝、缺根、CELL成员/时间链等。测试收据中的主/子模型身份依据主任务已核验日志，本次未重新读取平台session元数据。

未做正式数据生成/数组摘要复核、真实训练或任何新任务分数核验；未重算Rademacher或initializer、未做额外性能/资源试跑、未对所有测试重新执行独立回放。当前并无本轮PREFLIGHT、BUDGET、FREEZE、TRAINING_STARTED、ALL、SCORES、VERIFICATION、SUMMARY或EXECUTION；不存在可接受的新任务结果。

执行前剩余条件：主任务接受本次审查后，按原授权完成preflight和最终冻结清单/源码副本/原字节输入归档核对，确认审查后源及协议未漂移、全部必要文件和预算收据在链中，再决定是否执行。正式批次完成后仍需独立POST_RUN_REVIEW；默认、总体研究目标与1.7.0发布均不自动获准。

## 最终已审工件摘要

| 文件 | SHA256 |
|---|---|
| study.py | e6ef0d0353abc091d7f3d80998daa939f57a05ae21d7077265c4cefa0519bc2a |
| worker.py | ec47f817e62817cdeab1eac8014d50cfe2fc58831b689406b525cb92a159e9db |
| evaluate.py | 52c74c08c3b0f5b965078284dde39ea6eded7cd709581364fc0821014840f4bc |
| run_bounded.py | 8be1b17575575537952335b1da287f7755bb122cefcd65d80e52bc9f948d3a15 |
| PROTOCOL.md | cfabfccb8af78dcec8f9fb113e8ccc35376ce441c3b4583d5362bb7b402e990a |
| tests/test_new_task_study.py | d64e0ee9ea064075bb63ceb15f531c9b9f72d2c14b122961cbedfc5e4e3784a9 |
| validation/review_fixes_02.log | fa4fd763c175f2d060055536c3420e5e8e5a8b6434427a282baaa55419d9a048 |
| validation/source_suite_01.log | 0e3d65e13d0b18bc3297a897b5ecf089fc2be3b4ef55df1e7570df95f5de630c |

