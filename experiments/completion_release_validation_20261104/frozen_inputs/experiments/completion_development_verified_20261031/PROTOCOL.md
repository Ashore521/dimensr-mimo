# 接续开发批执行协议

本协议落实本目录PLAN.md，承接原completion_development_20261031尚未使用的一次16对/32系统预算。原准备目录保持未运行，原历史failed判定保持原字节；被独立接收的隔离核验作为新增证据。该矩阵已观察，本批属于开发回归。

## 固定比较与入口

四族×0/1%噪声×9131/9132，以及96/48/192划分、公式/单位/范围/噪声流、训练尺度和奇偶交错臂顺序，均继承原开发PLAN和已冻结case实现。base为precision_repair=True/compositional_completion=False，completion相反；共享41源锁、compound、完整神经/GP/NSGA-II/联合拟合、closure24+24和precision选择保持原值。preflight仍须核原case/spec和训练验证摘要。

唯一入口是 `.venv/Scripts/python.exe experiments/completion_development_verified_20261031/run_bounded.py`，工作目录为D:/符号回归项目/dimensr-mimo。四个薄入口复用原共享harness的监督、训练、封存、评分及失败处理，只将study/cases/HERE绑定本目录。原模块单例不变。evaluate使用已冻结的SealedVerifier(study)，不修改该类或原Verifier。

2880秒BUDGET含600秒核验预留；1200完整评价、52秒合作fit、60秒cell外限、单worker串行及数值单线程保持。原preflight独立启动窗口60秒。gate和解释器启动占用cell外限；52秒是fit合作上限，不保证每cell实际可用52秒。单次NumPy/BLAS仍不可抢占，外监督有.2秒采样粒度。没有第二份开发预算、失败补跑或预算借用。

## 原失败与修正通过的双证据链

gate先核共享CANDIDATE_LOCK完整src路径集合和每份摘要，以及原准备目录不存在正式启动证据。随后要求修正目录POST_REVIEW_ACCEPTANCE.status精确为`accepted_for_sealed_historical_correction`，sha256键集合精确为：

```text
FREEZE.json
VERIFICATION.json
SUMMARY.json
EXECUTION.json
FROZEN_INPUT_ARCHIVE.json
POST_RUN_REVIEW.md
CANDIDATE_LOCK.json
```

前六项来自completion_verification_repair_20261031，锁来自共享compositional_completion_20261031。顶层可含审查说明；sha256映射不能缺键、多键或漂移。接收收据只由主代理写，harness不会制造接收。

原历史十项ARTIFACTS摘要必须与修正FREEZE.original_sha256完全相同。原SUMMARY.technical_pass/adoption_pass严格False、VERIFICATION.status仍failed、EXECUTION.verification_passed严格False；原POST_REVIEW_RECEIVED明确接收原失败，不能变成历史准入。核原FREEZE→ALL→SCORES→VERIFICATION交叉链与原输入/归档、运行文件的全部原hash。

修正VERIFICATION必须passed，expected/verified严格整数34，rows严格34、逐行verified严格True、job与原34任务逐项同序且唯一；budget_met/inputs_unchanged严格True。修正SUMMARY的34系统/17对/34输出/34保护、完整分母、technical/adoption及verification_of_sealed_results全部要求正确的类型和值，original_verdict为failed。修正FREEZE.no_fit_search_or_rescoring、EXECUTION.no_training_performed/verification_passed严格True，returncode严格整数0且无终止。

所有时间均有限且拒绝bool。原封存/评分/核验在修正开始之前，修正verified_unix位于修正起点和deadline之间；deadline不超过原历史BUDGET起点+2880或修正起点+600，修正wall不超窗口。检查的是已完成核验的记录时序；新开发批使用尚未使用的独立原预算，不要求当前日期仍处在历史截止内。

最后核原和修正FREEZE输入当前字节及frozen_inputs归档、archive清单/计数/交叉摘要，原ALL所列全部运行文件；计数由实际manifest导出。共享锁摘要必须与原历史FREEZE相同；当前基础config与原历史及锁内shared_configuration逐字结构一致。两份原stage PLAN摘要也须匹配锁，确保任务未替换。

## 验证规则与冻结闭包

SealedVerifier仅在control和旧repair都具有完整epoch、全神经提案、GP条目、有限有序时间、正截止裕量和未命中global deadline时，分类phase_budget中三个既定调度时间字段；其余原Verifier身份/轨迹/候选/cache/参数/预测/选择/score比较保持。该规则不证明完整失败cache或RNG状态相同，也不支持一般“计时不影响算法”的结论。

freeze_inputs完整保留旧冻结依赖闭包，替换原历史目录永不生成的POST_REVIEW_ACCEPTANCE.json路径。补入原失败十项证据、原输入当前字节/原档/运行文件/失败后审来源，以及修正的代码、协议、测试、日志、审查、FREEZE/结果/接收和frozen_inputs。新目录gate、四入口、test_stage、协议、实现/检查收据与日志由原闭包纳入。独立审查后的归档修正仅排除本stage两个validation目录的嵌套临时fixture子树，保留顶层代码、日志、收据与全部旧依赖。临时现场在本地完整保留。

训练worker只接train/validation，test接口在本批ALL_MODEL封存前拒绝。逐CELL和ALL先封存所有模型或失败标记，再按原时间/成员/hash链评分。原始JSON解释、全候选训练验证复算、native/physical重载、单位/域、Rademacher和各系统实际神经/GP/至少两NSGA新评价step均保留。matching各臂new/valid/protected/validation_retained均须大于0。

固定科学门槛仍为candidate16/16完整接受、32/32干净输出≤base×1.1+1e-8、原七接受全保留、宏干净误差和平均图不升。无噪声三split≤1e-4；1%噪声观测≤.03、干净test≤.02；图≤24、参数≤6、展开≤30、深度≤6及原单位/域门槛保持。技术通过不能抵销科学失败。补充来源和接受增益按原规则报告，整体比较不作单组件因果归因。

非法、缺失、重复、NaN/Inf/negative指标继续按32完整分母失败，缺失均值null；复用SCORING_FAILURE及坏/缺SCORES的全分母失败收据，不覆盖旧失败、不自动重跑。外hardkill、解释器崩溃或磁盘不可写可能阻止最终收据，应保留现场并报告未完成。

## 当前交付边界

Windows归档I/O使用本目录archive_io.py的局部extended-path适配。逻辑ROOT相对清单、原hash和路径越界检查保留，sha/复制/归档verify_files在磁盘访问时转换路径，已带extended前缀的路径保持幂等。旧已冻结的深层依赖全部复制并复核，已有冲突归档拒绝覆盖。共享harness、算法和系统长路径设置保持原状。

本工作包只完成构造测试和只读链检查，没有正式数据生成、preflight/freeze、fit/search、评分或新矩阵结果。首次Win32长路径fixture错误及修复后日志保留。正式执行还须本接续harness独立窄审、主代理最终测试收据和冻结验收；发布与后续研究接受由主任务另行处理。
