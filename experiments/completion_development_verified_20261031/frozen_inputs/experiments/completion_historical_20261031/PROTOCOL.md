# completion历史固定对照执行协议

矩阵为precision_repair_20261028原17对/34系统及原case/spec/seed，属于已观察历史回归。入口为 `.venv/Scripts/python.exe experiments/completion_historical_20261031/run_bounded.py`；只运行本批，绝不自动启动development。

完整科学门槛同时要求34/34技术、17/17 foundation身份和17/17旧control身份、34/34干净输出≤control×1.1+1e-8、control全部接受保留、宏干净误差及平均共享图不升。接受数从完整score行计数；不能把原11/17说成17/17或隐藏原六单元。damped_n0/701完整接受、图≤20/参数≤4、每train/validation/clean≤1e-12；rational_n0/702完整接受、图≤15/参数≤4、每train/validation/clean≤1e-4。原各case精度、图24/参数6/展开30/深度6及单位/域门槛保持。六单元和新增失败在SUMMARY逐项列出，原数据、来源及失败不改。

独立后审通过并由主代理明确接收后，主代理写本目录POST_REVIEW_ACCEPTANCE.json，精确schema见下；harness不自行创建接收收据。

## 实现与封存链

两臂完整配置只交换 precision_repair/compositional_completion：base=True/False，completion=False/True。其余配置读取原precision_repair冻结config；共享候选锁CANDIDATE_LOCK.json检查当前src的完整路径集合与全部摘要，expected_source_count若存在必须匹配，不硬写40或41。入口、preflight、freeze、每个训练cell/worker及评分入口检查source锁。preflight要求当前spec及train/validation摘要逐项与对应旧FREEZE一致。测试标签接口在本批ALL_MODELS_FROZEN存在前拒绝；worker仅接train/validation，真图和oracle不传入算法。

共享harness.py/verification.py仅供这两批使用。每批importlib隔离加载旧cases和precision_repair/evaluate及其compound raw verifier，显式绑定当前目录/cases/study，不改旧模块单例。旧raw JSON解释器、全候选train/validation复算、native/PhysicalModel实际保存重载、单位/结构/域、Rademacher和完整机制检查均保留。每系统至少两个不同NSGA-II step各有新评价；matching各臂new/valid/protected/validation_retained汇总均>0。

训练按冻结job顺序串行；逐CELL封存模型或失败文件，再ALL封存。复用FREEZE→TRAINING_STARTED→按序CELL→ALL的时间、工件成员与hash核验；评分必须在ALL后，SCORES/VERIFICATION保存交叉摘要。enclosed仍为既有有限浮点数值筛查。已封存CELL只核验后跳过，未封存但有job则停止并要求事故检查；入口遇任何既有执行收据拒绝重启，不补跑。

## 新模式回放与控制身份

新模式要求mode=compositional_completion_v1、triggered=True；不套用旧训练精度触发条件。固定父来自原foundation按平均训练误差、复杂度、图摘要排序。completion最多12缓存外完整系统评价；之后matching每轮24/共48，全部计入同一1200评价和52秒合作截止。总评价必须等于foundation+相应补充+matching，训练池必须等于foundation+保护留存补充+matching留存，验证过滤的子集和拒绝摘要也须重建一致。

逐输出returned_sources须顺序、种类、原proposal索引、canonical树、常数、实际原始训练Y分数及score/size/repr排名一致。分数按MSE/max(var(Y),1e-12)+1e-8*nodes重算；不是变换域残差或Evaluator NRMSE。新变换来源须匹配log/sqrt、direct/multiply/divide和外幅值结构，basis索引固定在0..23，声明单位、原grammar.trace和15节点/6层不放宽。旧implicit原返回索引与合并后索引分开验证。

每个完整系统事件的root_sources必须严格来自对应输出返回位；缺位从固定父复制并核missing_roots，不循环。用保存树/常数与父重新构图，核完整graph/warm身份。原raw解释器重算父/子训练误差与每输出父×1.1+1e-8保护；缓存外评价、None子、valid/protected和所有计数分别记账。部分deadline/全局预算停仍保存部分工作，不能说完成了未执行的初值或评价。

控制臂与旧封存repair臂逐model/result/delivery/pool及测试score比较。只忽略明确时间字段（constant_fit/evaluation/elapsed/cpu/phase/worker wall/worker cpu seconds及monotonic）、版本元数据、新增False默认配置和包含这些元数据的source_model_sha256；不忽略评价数、参数/预测/误差、结构、完整候选payload、cache、选择、诊断事件或其他算法指标。原生重载仍先按原payload核来源，不把新默认先写进旧文件。控制的旧precision_repair触发/skip规则用原verifier回放，未触发时还核与旧同批base的完整池/cache/选择身份。candidate与当前control的foundation完整payload/count/hash/evaluation/cache须全对一致；新candidate没有“未触发跳过身份”门槛。

补充真实来源/线性工作/返回根/缓存外系统/有效/保护/验证留存和接受增益按族报告。没有有效新复合根或验证留存，不能声称新增复合发现；该比较同时改变始终补充模式和新初值，事件来源不证明单组件因果效应。完整原事件与stop_reason在逐cell模型及VERIFICATION保留。

## 冻结输入与失败边界

freeze清单含当前src及其本批source副本、全部顶层tests、新两批入口/共享helper、阶段所有md/json、TEST_VALIDATION、PREFLIGHT、当前CORE/HARNESS审查及core/harness日志。共享core_validation/core_review_validation递归纳入非pyc原字节，包括IMPLEMENTATION.md、IMPLEMENTATION_CHECKS、package_versions、verify_implementation.py、.diff、主测试日志及收据。读取旧FREEZE.files键形成实际依赖/测试fixture闭包，同时将旧frozen_inputs逐文件原字节和原manifest/seals/来源model一起冻结，区分旧版本证据与本次live源码。freeze前核旧输入archive及旧ALL所列运行文件原hash；不要求已变化的live src继续假装旧版本。主代理另生成本轮输入归档/收据。

每批2880秒BUDGET含600秒核验预留，52秒合作fit/60秒cell进程树上限、单worker串行，OMP/OpenBLAS/MKL/torch各一线程。外监督.2秒采样，BUDGET从freeze生成时起算，另沿原60秒preflight启动窗口。单次NumPy/BLAS不可抢占；不称入口到退出严格2880秒。禁止借另一批预算或自动续跑。

非法、缺失、重复、错误case或NaN/Inf/negative误差/结构指标均按完整预定分母失败，方法均值为null。原scorer的非有限误差在cell try内拒绝，保留scoring_failed行；整体scoring异常写SCORING_FAILURE.json。坏/缺SCORES、guards失败或过期会写全部预定VERIFICATION失败行和SUMMARY，交叉hash缺失时为null；不覆盖既有SCORES/VERIFICATION/SCORING_FAILURE证据。外进程强杀、磁盘不可写或解释器崩溃可能阻止收据写入，必须保留现场并由主代理报告执行不完整，不能把没有收据解释为成功。外监督将坏VERIFICATION视为未通过并保留原字节。

本协议不代表正式运行通过，不改生产默认，也不授权提交/发布或替代独立运行后审查。

## 前阶段接收收据的精确schema

文件位于 `experiments/completion_historical_20261031/POST_REVIEW_ACCEPTANCE.json`，由主代理在独立后审被接收后写入：

```json
{
  "status": "accepted",
  "sha256": {
    "FREEZE.json": "<historical FREEZE sha256>",
    "ALL_MODELS_FROZEN.json": "<historical ALL sha256>",
    "SCORES.json": "<historical SCORES sha256>",
    "VERIFICATION.json": "<historical VERIFICATION sha256>",
    "SUMMARY.json": "<historical SUMMARY sha256>",
    "POST_RUN_REVIEW.md": "<historical independent review sha256>",
    "CANDIDATE_LOCK.json": "<shared compositional_completion_20261031/CANDIDATE_LOCK.json sha256>"
  }
}
```

sha256键集合必须精确匹配上述七项；其余审查说明可放顶层。还要求historical SUMMARY.technical_pass/adoption_pass均true、VERIFICATION.status=passed，核VERIFICATION→FREEZE/ALL/SCORES及ALL→FREEZE、SCORES→ALL交叉摘要；FREEZE.candidate_lock_sha256须等于共享锁当前摘要、FREEZE.config与本批固定config相同。这是新阶段收据，不能复用或改写旧阶段 accepted_as_independent_post_review 收据。缺失收据是正式启动阻断，构造测试不会伪造生产接收。
