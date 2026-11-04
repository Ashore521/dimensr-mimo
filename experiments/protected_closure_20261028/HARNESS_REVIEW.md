# Protected closure 运行脚本窄范围独立审查

结论：本次运行前窄范围审查完成，初查发现的预算文件未绑定与闭包实际留存门槛未对齐均已修正并复核；当前限定范围未发现剩余实质阻断。此结论仅针对冻结前脚本、协议与确定性构造，不代表34系统运行、科学成效或发布接受。最终冻结、执行和接受决定由主任务负责。

唯一写入本报告；PRE_RUN_REVIEW.md原文未改，末次实读SHA256为 `d9bcd94365aaa0456427dc036330f05c6665a9b25079d93d46a55394c65fd497`。未执行fit、训练、基准或pytest，没有启动新34系统实验，未派生代理。

## 来源和版本

已审读本目录 PROTOCOL.md、study.py、worker.py、evaluate.py、run_bounded.py，tests/test_closure_study.py，以及实际复用的 compound_search_20261027 的 cases/study/worker/evaluate、first_round 监督器与机制统计、second_round previous依赖装载、native_api读写/JSON解释器、typed_coverage交付/门槛和noise_families数据定义。核心变更相对上次审查为PACKAGE_VERSION=1.7.0及每事件new_evaluation布尔；后者通过实际calls增量记账，未见改动提案、拟合、保护或选择。

末次受审SHA256：

- PROTOCOL.md：`27e9e5ff94711dcd396db2a1be62c59e4453b2c8691534c5e5a96334b1948aab`
- study.py：`1bb1faebff6ce66b9dc10ca89598f8625ce6751343af89287e09cfa6f9b79a8f`
- worker.py：`6a10cf1f37d6ac027f7577d9cfceb65a6a492a086b5c9ef2a4b07ecf1727b3a5`
- evaluate.py：`5e71003116c99527c29163b0b32715685193b2a6c6feefa53f047bf248c5ca4c`
- run_bounded.py：`95fa0a3eac482ff4e14de13cd4d4614b273cd76b7069ab9d5737b1a77463cb6d`
- tests/test_closure_study.py：`1b241cf549774b5b686ac694a67e0a54725674a1252977490610665f1c90ae71`
- src/dimensr_mimo/sharing.py：`d6f62512722cc43cc47da86ffee5b3f75b786cadb0a884eeb166810f6349e19e`
- src/dimensr_mimo/regressor.py：`c432e7a2f7b704c0f0c0b95b4d75855360314539a9f27850182d091f663259b3`

## 初查问题及修正

1. **预算与审查清单。** 初始freeze未将BUDGET.json与本报告纳入文件清单；budget()及外部监督直接读取该文件，缺少冻结字节约束。当前study.py:68纳入HARNESS_REVIEW，:77-78写BUDGET后纳入其路径，再生成FREEZE.files哈希；旧random对照使用的physical_pool历史genetics字节亦明确纳入。train每单元及评分guards都会核该清单。该问题在源码范围已解决；尚未实际生成FREEZE来核验最终清单实例。
2. **有效闭包门槛。** 初始summarize仅要求整批训练protected>0，全部扩展验证拒绝仍可能满足该项，且协议未明确逐单元/批次。当前PROTOCOL.md:21明确原神经/GP/NSGA/Rademacher机制逐系统要求通过；闭包附加机制按本批合计验收，new_evaluations、valid、protected、validation_retained均须>0，允许单元无共享机会时如实记0。evaluate.py:118-119将四项合计非零纳入technical，从而也约束adoption；新构造测试验证validation_retained全0时技术失败。修正与主任务明确采用的批次口径相符，不应在结果中宣称17个closure单元都产生了有效共享。

## 固定矩阵、预算与失败处理

- study.matrix实际为34个唯一单元、17个任务种子对、random/closure各17；偶数种子反序。原三族×两噪声×两种子为12对，三变体711及products/boundary601再5对。17对各2输出，配对输出保护分母为34，而非68系统输出混作配对分母。
- 实读配置为population24、1200评价、52秒fit、每输出15节点/6层、4轮神经/每轮2GP、4NSGA-II、常数25步、polish50步/3候选、compound/expanded/phased/precision；共享策略由新worker明确限制为random/closure，原默认random未改。
- 复用first_round.run_job实际启动当前本目录worker.py；其进程树每0.2秒监督，达到60秒单元限则终止。run_bounded独立监督整个study进程树，使用冻结预算起点+2880秒，并覆盖评分核验；study.train预留600秒，不够启动的项写not_started_budget并照常封存。轮询和进程退出存在亚秒开销，资源核验允许60秒后最多0.5秒，不能将其说成连续时间的零超调硬证明。
- BUDGET计时从冻结阶段开始；外部监督在BUDGET出现前另给60秒启动/预检窗口，因此2880口径是协议所写“冻结至评分核验”，不包括此前全部构造测试与预检时间。
- 已有CELL_FROZEN仅核hash后跳过；有job却未封存立即要求incident检查，无自动补跑。超时/失败/预算不足均保留原文件、完整单元分母；score/verification已有文件禁止覆盖。外部总限中断可能留下未封存单元或未完成核验，不能据此补跑或报告通过。

## 数据隔离、门槛与核验

- worker只调用training_data(train/validation)，该接口拒绝test；无按task名称识别真值的新增算法分支。数据生成与原compound矩阵复用相同case/seed/split；96/48/192及原量纲/噪声定义沿用。评分guards核源码/依赖、每单元/全模型封存哈希与先后时间后，base.evaluate才生成test及clean标签。该隔离是源码与保存证据层面，未作OS级文件访问审计。
- 原绝对观测/干净门槛及结构阈值均复用cases.design和delivery.accepted：无噪声1e-4、1%观测.03/干净.02，图24/参数6/展开30/深6，误差以训练观测标准差归一化。development接受同时核train/validation与precision状态，test接受独立核观测、干净及结构；domain与机制另在技术核验中要求通过。
- 所有34项score分母与冻结jobs长度核对，逐项job顺序/状态核对，失败行进入verified=False。任何缺失/失败阻止technical/adoption；方法宏平均仅全17项complete时给出，不用幸存样本均值代替。
- closure_evidence核有效config三方一致、外部返回码/资源、总评价≤1200、事件new_evaluation之和、base_evaluations+new_evaluations总账、基础前缀摘要、父状态来自基础/已留存集合、图更小/结构/量纲、独立JSON解释器重算父子训练误差、训练保护与有效/拒绝/留存计数。重建完整训练pool摘要，再复算新增候选验证合法性与有效子集/拒绝图ID；重算基础选择摘要并核最终train/validation边界。base.verify_cell另重载模型、逐候选复算train/validation、核原图/物理图/单位/域/所选test预测与接受。
- summarize保存配对基础池/基础cache摘要是否与random相同，墙钟轨迹差异如实报告，不强行解释为路径保证。该比较不能单靠摘要还原全部基础cache；本核验也未重新生成完整提案排序或独立复演整个搜索。
- 推荐要求technical、34/34配对干净输出保护、closure接受数不降、宏平均干净误差不升及compact_damped_gain_preserved。后者明确固定damped_n0/701 closure，共享图≤20，逐输出训练/验证/干净test误差全部≤1e-12。图21即使接受也不能通过该收益门槛。通过本批亦只支持历史开发回归，默认不自动改、有理发现缺口及新任务验证仍未完成。

## 实际无训练检查与限制

- 直接调用tests/test_closure_study.py两项函数（worker被mock，传入未写入路径），在门槛修订前后均通过；没有运行pytest或fit。独立检查矩阵34唯一/17对及所有上述配置值。
- 用合成零误差两输出Candidate、内存read替身和小型构造训练/验证数组执行真实closure_evidence：一致父子记录通过；新评价计数、完整训练pool哈希、验证留存计数三种不一致分别被实际拒绝。未生成test标签，也未写模型/实验文件。
- `git -c core.whitespace=cr-at-eol diff --check`实际通过。仓库.gitattributes对核心与冻结实验设-text字节保护；前报告普通diff --check的CRLF提示由此解释，不应改历史冻结字节来消除提示。原报告保留，本报告作此补充。
- 主任务通报全套136测试及最新2项流程复测通过、首轮缺worker.write导入的失败日志保留；本审查已读到write导入修正，但未独立重跑全套测试、读全部旧日志或验证平台会话元数据。
- 尚未生成FREEZE、训练34项、实际观察52/60/2880截止、评分干净误差或完成运行后独立数值核验。实际成本、闭包覆盖、图20收益和推荐门槛是否通过均未知。

审查接受边界：可进入主任务决定的冻结/已约定开发对照步骤；当前无剩余实质阻断，不新增实验授权，不接受默认切换或科学完成声明。
