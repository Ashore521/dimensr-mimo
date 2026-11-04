# 锁定候选新输出组合验证协议

本协议落实本目录PLAN.md。任务和真式由主代理在41源候选锁定后编写，设计独审记录于DESIGN_REVIEW.md/DESIGN_CHECKS.json。本目录只实现stage接线、前置gate、新训练验证preflight及无历史同任务control的核验汇总；算法保持锁定。

## 固定任务、配置与预算

四族log_affine_ratio、reciprocal_quadratic_norm、oscillatory_affine_saturation、exponential_affine_saturation，各含0/1%噪声与10241/10242种子，共16对/32系统/32配对输出。train/validation/test=96/48/192，两个输入范围[.2,1.4]；单位、公式、常量单位目录按cases.design。噪声层X独立，SeedSequence首项20261104。噪声幅度只按干净训练SD，误差按训练观测SD。

base启用precision_repair，completion启用compositional_completion，互斥；其余配置逐字承接共享候选锁和原冻结config。保留完整compound神经/GP/NSGA-II/联合拟合/Rademacher、原父及base训练验证保护、12次有界补充和24+24 matching。两臂按种子奇偶交错。默认legacy/random保持。

入口为 `.venv/Scripts/python.exe experiments/completion_release_validation_20261104/run_bounded.py`，工作目录D:/符号回归项目/dimensr-mimo。复用原worker/train/bounded，单worker串行，数值线程均1。单批2880秒含600核验预留，1200完整系统评价、52秒合作fit、60秒cell外限、原60秒preflight启动窗口不变。gate与Python导入占用外限，不保证fit获得完整52秒；单次NumPy/BLAS不可抢占。失败保留，不补跑、不追加种子或借预算。

## 前置开发接收与待定状态

completion_development_verified_20261031必须有终态和独立后审接收。缺任一必要文件时，gate返回/抛出明确的`pending_missing_prior`及缺件列表，正式bounded入口在启动子进程前拒绝。这个状态不表示通过，也不跳过前置。

前置POST_REVIEW_ACCEPTANCE.json精确schema：

```json
{
  "status": "accepted_for_release_validation",
  "sha256": {
    "FREEZE.json": "<prior hash>",
    "ALL_MODELS_FROZEN.json": "<prior hash>",
    "SCORES.json": "<prior hash>",
    "VERIFICATION.json": "<prior hash>",
    "SUMMARY.json": "<prior hash>",
    "EXECUTION.json": "<prior hash>",
    "FROZEN_INPUT_ARCHIVE.json": "<prior hash>",
    "POST_RUN_REVIEW.md": "<prior hash>"
  },
  "candidate_lock_sha256": "<shared candidate lock hash>"
}
```

sha256键集合必须恰好八项。前置technical严格True、VERIFICATION passed且严格整数32/32、rows逐项同序/唯一/verified严格True、budget_met严格True、EXECUTION整数0且无终止/verification_passed严格True。前置adoption_pass必须是实际bool，可False；gate不强造科学成功。

核前置FREEZE/ALL/SCORES/VERIFICATION交叉摘要、源码/config、冻结输入当前字节及原档、全部运行文件、CELL成员/hash/单调时间、实际job/resource状态和外限；运行目录集合恰为预定32个job，额外attempt拒绝。保持原600/2880及52/60、单线程参数。终态在原BUDGET截止内，EXECUTION计入口窗口，wall不超过原窗口加批上限。该核对结合冻结的不补跑代码与唯一工件，未声称从未保存的外部进程历史证明没有任何操作。

还复核原历史failed与独立隔离核验passed/接收双链，保留原false、原失败和原档，不创建原历史准入收据。前置科学失败按用户允许的opt-in发布口径如实报告，不降低本批16/16等科学门槛。

## 新数据隔离与核验语义

本地CANDIDATE_LOCK.source_files须等于共享41源锁的完整路径/摘要，parent_candidate_lock_sha256须匹配共享锁。每入口/worker/cell继续检查锁，任何漂移拒绝。

新preflight只调用train/validation，核96/48形状、有限值和训练观测非零SD，记录16个数据摘要；逐一验证有效配置，不读取test。收据绑定local/parent锁、cases/PLAN/DESIGN_CHECKS摘要、完整jobs/config/specs。freeze前和评分guard重核绑定。静态oracle和真式只进入审计/冻结规范；原worker传给拟合器的是训练/验证数组、单位目录、名字、bounds和公开配置，不传truth_graph、真系数或公式。

所有CELL模型或失败先封存，再ALL封存后统一test。隔离加载原raw verifier，重绑cases/study/HERE，保留原始图解释、全池train/validation复算、补充/closure来源和会计、native/physical保存重载、单位/结构/原域及全机制。enclosed仍是原有限浮点筛查。

NewTaskVerifier只覆盖verify/summarize中涉及旧control和本批汇总的逻辑，其他方法继承原Verifier。新任务没有历史同任务模型，verify不调用control_identity、不写archived_control_identity字段；显式调用该方法会拒绝。SUMMARY以`historical_control_comparison=not_applicable_no_same_task_archive`说明边界。同批两臂foundation完整payload/count/hash/evaluation/有效cache指纹逐对一致；指纹不证明未保存失败cache或RNG全状态身份。

## 全分母与固定门槛

技术要求32/32完整核验、真实神经更新/有效候选/GP/至少两NSGA step各有新评价、联合拟合/Rademacher、单位域、重载及补充事件核验。matching各臂new/valid/protected/validation-retained均>0。verified只接受严格True，计数拒绝bool/负数，数值拒绝bool/NaN/Inf/negative，结构使用非负整数。

科学采用仍需candidate16/16完整接受、32/32输出≤control×1.1+1e-8、control全接受保留、宏clean误差和平均图均不升。无噪声train/validation/observed/clean≤1e-4；1%噪声观测≤.03、clean≤.02；图≤24、参数≤6、展开≤30、深度≤6及原单位/域条件。任何单项失败不能由均值改善抵销。

缺失、重复、错误case/job、失败、超时和非法指标保留32系统/16对/32输出，缺失方法均值null。复用SCORING_FAILURE和全32失败核验收据，不覆盖既有证据。新补充来源、工作与接受增益按族报告，不从整体比较作单组件因果归因。外hardkill或磁盘不可写可能阻止最终收据，须保留现场报告未完成。

## 冻结来源与当前边界

保留前置依赖闭包全部旧冻结文件，含原失败与修正链；添加前置开发完整终态/输入归档/运行源，以及本目录代码、计划、锁、cases、测试、日志、设计/harness审查、oracle和新颖性审计。54份history_sources原字节与DESIGN_CHECKS指定13份源FREEZE注册表都入清单并核摘要。120条历史注册记录含86条公式记录和34条省略公式记录，不能称120个独立公式。该有界排查不证明外部盲基准新颖性或任意历史表达式不等价。

54份来源的权威版本是NOVELTY审计保存的history_sources快照。主代理后续为当时未冻结的前置开发PLAN补充I/O/归档说明，不要求该live草案永久等于早先快照；SOURCE_VERSION_NOTE.json单独记录两个版本并冻结，plan_drift_01.log保留。前置正式FREEZE另行绑定其最终PLAN与执行源码。41生产源码锁、54归档原字节和13冻结注册表的强制摘要核对不放宽，原NOVELTY/DESIGN文件不改。

新目录测试临时fixture无需递归作为执行依赖；测试源码、完整日志及审查脚本/收据仍冻结。旧闭包已冻结依赖一项不删。构造fixture不代表正式拟合/科学验收；正式运行前还需独立harness审查、主代理最终测试收据、前置接收与冻结验收。本工作包没有正式preflight/freeze/fit/search/test评分或发布。

Windows长归档通过前置stage的局部archive_io处理：文件存在性检查使用disk_path，sha/verify_files及保留原字节copy_original支持extended path且幂等。study仅显式适配原freeze构建段的I/O，manifest始终使用正常ROOT相对逻辑key；worker/train/bounded不变。不会全局patch共享sha/Path或改平台注册表，也不删除超长依赖来通过检查。
