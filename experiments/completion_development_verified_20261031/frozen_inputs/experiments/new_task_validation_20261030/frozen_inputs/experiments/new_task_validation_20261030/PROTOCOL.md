# 固定候选的新公式族验证协议

本协议实现 PLAN.md 的固定比较，科学结果在正式全模型封存和核验前未知。候选生产源码由 CANDIDATE_LOCK.json 的40份文件及前轮 SUMMARY 摘要固定。生产默认保持不变；不得根据本批表现改算法、删任务、调阈值、补跑或追加预算。

## 矩阵、接口与配置

4公式族 logarithmic_gain、logistic_gain、algebraic_norm、bilinear_saturation，各含0/1%噪声、9131/9132种子，共16对、32系统、32配对输出。固定样本数train/validation/test=96/48/192。所有变量在声明单位的[.2,1.4]范围内；输入输出identity变换。具体公式、变量顺序、单位目录和门槛以被冻结 cases.design 为准。

每个split的输入流使用 SeedSequence([20261030,seed,split_code,CASES.index(name)])，噪声另用411/422/433代码，输入代码为111/222/333。噪声层包含在case索引，因此同族同种子的n0/n1使用不同X；本批不作配对噪声因果比较。每一base/repair对的数据完全相同，并核同一训练/验证摘要。噪声SD为干净训练输出SD×.01；评分归一化仅用训练观测SD，不使用测试统计。

configuration逐字读取已审核 precision_repair_20261028/FREEZE.json.config。worker仅补入相同的seed、compound、closure、precision selection和case acceptance；两臂唯一差异为 precision_repair。原神经策略、GP、NSGA-II、联合拟合、单位/域检查、Rademacher、常数拟合/polish和24+24 matching预算全部保留。worker只调用training_data(train/validation)，不调用真图或test。cases.generate(test)在ALL_MODELS_FROZEN存在前拒绝；评分器另核完整封存链后才请求标签。静态oracle仅作为单位/语法/结构/有限浮点域的设计证据，不注入拟合。

## 执行、冻结与不可补跑

入口为 `.venv/Scripts/python.exe experiments/new_task_validation_20261030/run_bounded.py`。串行、单worker，OMP/OpenBLAS/MKL/torch各一线程。每系统52秒合作fit、60秒进程树上限、1200次完整系统评价。外监督每.2秒采样，合作检查不能任意抢占单次NumPy/BLAS。最多32×60=1920秒单元名额，仍使用原2880秒BUDGET，包含600秒评分核验预留；BUDGET在freeze生成，入口另沿用60秒preflight启动窗口，因此不称从入口到退出的严格2880秒上界。

入口、preflight、freeze、train和每cell/worker都检查候选40份source集合及摘要，漂移即停。训练每cell还核FREEZE.files。已经封存的CELL只核验并保留；未封存而有job的现场要求事故复核。timeout、failed、not_started_budget均封存且保留完整分母，不补跑。外入口发现BUDGET/supervisor.log/EXECUTION已有执行证据即拒绝重启。

freeze输入含新全套源码、全部顶层测试、所有当前Markdown审查文件、所有测试日志及TEST_VALIDATION、PLAN、CANDIDATE_LOCK、NOVELTY/ORACLE/DESIGN审计、history_sources来源副本、PREFLIGHT、依赖版本、训练验证摘要、pyproject及源码副本。旧已审核FREEZE.files键集合提供完整历史导入/测试fixture依赖闭包，旧FREEZE及SUMMARY/POST_RUN_REVIEW、旧运行seal和其所列全部工件也进入本轮清单。新source副本本身被列为冻结输入；源与副本分别核摘要。主任务另创建被冻结输入归档及其收据。冻结不改历史文件。

每cell封存后才创建ALL_MODELS_FROZEN。隔离importlib加载旧precision_repair/evaluate及其compound base，显式重绑各自HERE/cases/study，保留原raw解释器、完整候选训练验证复算、模型/PhysicalModel保存重载、extension/repair/closure回放、资源及会计核验。历史模块单例不被改写。复用原guards核FREEZE→TRAINING_STARTED→逐CELL单调时间→ALL时间、CELL及各工件在ALL中的成员/摘要；评分与核验再核ALL→SCORES的时间和交叉摘要。域enclosed仍只代表原有限浮点筛查。

## 固定验收与报告

逐系统技术要求实际策略更新、有效神经候选、GP新评价、至少两不同NSGA-II step各有新评价、Rademacher、保存重载、单位和原域筛查。matching每臂汇总new/valid/protected/validation_retained均须>0。总评价严格为原base+repair新评价+matching新评价。16/16原base count/hash/evaluations/cache身份，以及所有跳过repair对的整池/cache/最终选择身份必须一致。

主科学门槛全部同时成立：候选16/16完整接受；32/32干净输出误差≤base×1.1+1e-8；所有base已接受保留；完整分母上的宏平均干净误差及平均共享图均不升。完整接受保留无噪声train/validation/test≤1e-4，1%噪声观测≤.03、干净test≤.02，图≤24、参数≤6、总展开≤30、输出深度≤6，及单位/域要求。没有旧17/34矩阵或特定damped/rational目标门槛。

失败、缺失、重复、非有限及不完整指标均阻断总体通过；有缺失/非法值的方法宏均值为null，expected固定为每臂16、总32，不只平均成功子集。隔离base.error在归一化误差非有限时于cell内抛出异常，由原评分器捕获为scoring_failed行，避免NaN使严格JSON写入整体失败。评分整阶段异常另写SCORING_FAILURE，随后不重试评分而写固定分母失败核验；整份SCORES缺失、损坏、行结构不合法或封存核验失败也产生32条失败记录，缺少的来源摘要为null。外部硬终止或磁盘不可写可能阻止Python写完这些收据，届时以外监督失败及冻结矩阵/已保存现场明确报告未完成，不能推定通过。SUMMARY按family/noise和逐case/seed配对报告，不允许总体均值抵销单项门槛。

repair_discovery_evidence独立列trigger、线性子问题、实际隐式获胜根、缓存外完整系统新评价、有效/训练保护/验证留存、接受增益和损失。repair无触发或无留存不构成技术失败；零触发只能支持跳过行为和本矩阵的系统接受，不能声称repair新增发现的跨族作用。实际新评价但无留存、或无完整接受增益时同样直报。接受门槛未过则adoption_pass=false，即使technical_pass=true；此小矩阵只作描述性作者合成任务验证，不声称显著性、总体成功率、公开标准benchmark或工程达标。

正式批次后另需独立POST_RUN_REVIEW；本协议与构造测试不构成运行通过证据。版本打包、安装态验证、最终独立发布审查和发布授权仍属后续阶段。
