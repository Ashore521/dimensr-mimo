# 两轮均衡闭包：代码与流程运行前独立审查

结论：本次窄范围独立复核完成，当前未发现阻断按固定方案冻结/执行的剩余实质问题。每轮24次保留了有保护子代且原全局预算未耗尽时的第二轮机会；总限仍为两轮/48新增评价及原1200评价、52秒fit、60秒外部、2880秒全局。此结论不证明新联合拟合、域检查、验证选择或图20收益能通过；最终冻结、执行与科学接受由主任务负责。

仅写本文件；旧轮、代码和其他文件未修改。未执行fit、训练、搜索、pytest、新实验或派生代理。沿用此前核心/流程审查，不重复重算上一轮全部5352候选或测试分数。

## 受审来源和最小差异

已读PLAN、PROTOCOL、TRAINING_DIAGNOSIS、4个运行脚本、两份变动测试和TEST_VALIDATION；对旧轮冻结输入归档逐字节比较当前src，唯一生产变动为src/dimensr_mimo/sharing.py：

- 评价循环前保存round_start=report['new_evaluations']。
- 每轮本地新增评价数达到24时break。

旧48总限、2轮、16父排序、匹配器、warm排名/去重、正常联合拟合/数值域/Rademacher、父训练保护、基础池前缀及最终precision训练/验证保护均未变。限额按实际calls增量累计，缓存或到期前未启动评价不占新增评价；不存在额外refine或新增搜索RNG消费。

与旧4脚本逐行比较并规范化module目录路径后，worker.py/run_bounded.py相同；study.py只把旧HARNESS审查项改成本轮TRAINING_DIAGNOSIS，保留本PRE_RUN_REVIEW及其他冻结材料；evaluate.py仅增加两轮各new_evaluation之和≤24检查。两个harness的矩阵、配置、失败处理与科学门槛未变。

受审SHA256：

- src/dimensr_mimo/sharing.py：`72e8c3a400c10fd2ee2a827de51c4962637a07bba9024715732dc81fe01bc5f6`
- study.py：`400ed163c4f0148db83f3c26abdf307ee7e486243aa1c3d56acfbdd91a4f5b6d`
- evaluate.py：`c46579a8d9a69a28f1cad9d3613ae0752fa33946a3476ecddc9d4315f43d58a0`
- PLAN.md：`f638081cc51cdcf917a8119a9019512c3fe2ee5c5b5f93d2ef8935ab292a440e`
- PROTOCOL.md：`1723e34bf4a7e94c58b48914b3b3afd2a3112cbf32f9c8d0e4a73769cb03b364`
- TRAINING_DIAGNOSIS.md：`1406a256b788967619c881e423ac3e39ff4d91866b7540395d4f70c41102b056`
- TEST_VALIDATION.json：`f3d5f61f688cf6aec03773915c01bcad80e4ceec77dcebdb21bae3d687f5b7d1`

## 调度和训练侧依据的实际核对

直接调用调度构造函数，Evaluator与提案源为stub，不调用拟合：原剩余资源充分时实际按 `[24,24]` 完成48次；原总资源只剩1次时按 `[1,0]` 停止，calls增量正确。额外构造第一轮仅5提案，实际 `[5,24]` 共29次，未将第一轮未用19次转给第二轮；中途deadline构造为 `[3,0]`。新harness对单轮25个new_evaluation=true的记录实际拒绝，报 `Closure round cap exceeded.`。

“两轮均有机会”不是无条件执行两轮：原全局deadline/max_evaluations先耗尽、无通过父保护的子代或无新可评图时，仍按原退出条件停止。这与PLAN的资源优先和至多两轮表述一致。第一轮24次后只有已通过子代可进入第二轮父池，不能据warm预测直接留存未拟合图。

为核对训练侧动机，仅读旧damped_n0/701已存Candidate/事件的训练字段，并用inputs(train)取得96行X，没有调用latent、generate、validation/test评分或新的拟合。实际确认：base147等于event6父状态，event6子等于candidate163；前24事件22个保护子代，candidate163在原第二轮top16中排第15。对其确定性matched_proposals枚举得到图20/4参数，图摘要 `af5ad975139e23cc80806e20e7d4ce9780eacccd984398b1fff2310d42f71fc1`，继承warm相对父训练预测标准化变化精确 `[0,0]`。

上述证明已保存训练状态存在递归提案机会，支持检验24+24分配，不证明正常拟合后结果、interval合法性、最终验证选择或clean-test紧凑收益。未独立重建诊断文的全部cache排除与40图排名；该更完整诊断仍依其已列历史trace/hash证据，不把本窄检查写成完整反事实训练。

文档初读时TRAINING_DIAGNOSIS的“current源码等于旧sha”可能被误读为新调度代码；主任务已在首部明确该报告形成于调度变动前、current指旧诊断时点。本项已解决，不修改旧诊断证据或旧审查结论。

## 冻结、资源、门槛与数据角色

- 34唯一单元、17对random/closure、偶数种子反序与原case/seed一致。新旧harness配置验证均通过，1200/52/60/2880及600秒评分预留保持；数值单线程、0.2秒进程树监督、完整模型先封存后测试评分、失败/超时不补跑和全分母保留沿用已审查路径。
- freeze清单包含src/tests/本轮脚本/所依赖旧生成与监督/JSON解释器/历史genetics字节、PLAN、PROTOCOL、TRAINING_DIAGNOSIS、本报告、TEST_VALIDATION及validation日志；BUDGET写后纳入SHA。实际FREEZE实例尚未生成，本次只核其源码路径和当前材料。
- 基础random路径仍先完整结束、closure后才验证评分；训练侧调度不读取验证/test标签，也不按任务名选图。新目录使用已观察开发任务，不能称新任务/盲验证。正式新数据/新种子验证和剩余有理式问题仍未完成。
- 新evaluate逐轮≤24检查叠加原≤48/≤1200、事件实际新评价总账、基础前缀与选择/父子训练保护/完整pool与验证子集核验。旧保存重载路径和数值合法性未改，上一轮详细审查边界继续适用。
- 采用门槛保留原逐输出绝对精度/结构、34/34干净配对10%+1e-8保护、接受数不降/宏平均干净误差不升、批次闭包new/valid/protected/validation_retained均>0，以及指定damped_n0/701图≤20且每输出train/validation/clean-test≤1e-12。一般图≤24的接受上限不能替代专门图≤20门槛，失败不抵销，默认不自动改变。

## 历史保护与验证状态

直接验证FROZEN_INPUT_ARCHIVE.files与原FREEZE.files完全相同、指向原FREEZE哈希；127份frozen_inputs归档字节全部吻合，包含即将变动的旧测试。旧ALL_MODELS_FROZEN的374份运行文件也全部吻合。旧源快照/结果未覆盖。当前根目录源码/测试已按新方案变化，因此旧FREEZE对当前根目录的直接guard会预期失败；复现旧轮须使用保存的旧字节/路径，不能把新代码冒充旧轮代码。

直接调用两项harness构造函数，在旧/新模块各2项通过；worker被mock，没有fit或临时文件写入。查看3份已保存测试日志并核SHA：首次9 passed/3个pytest临时目录错误；workspace basetemp重跑12 passed；全套138 passed in58.50s。TEST_VALIDATION记录这些分母及旧失败保留。本审查没有独立执行pytest或全套测试，平台模型元数据记录也未自行重新查平台日志。

CRLF-aware `git -c core.whitespace=cr-at-eol diff --check`实际无输出，通过；不以更改冻结文件换行来满足普通diff检查。

剩余未知：尚未冻结/训练新34系统，实际两轮留存、图20是否通过拟合/域/最终选择、每输出测试保护与接受数/均值均未知。没有独立重跑前轮或新增实验来代替这些证据。当前无剩余实质阻断，可由主任务决定按已定预算冻结并执行；本审查不新增实验授权或预先接受科学成效。
