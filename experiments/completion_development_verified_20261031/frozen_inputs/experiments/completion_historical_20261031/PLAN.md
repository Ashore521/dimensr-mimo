# 通用复合补充：历史固定对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-31
- Verification Status: UNVERIFIED
- Version Label: completion_historical_v1

## 目标与来源

在完整静态 MIMO 神经/GP/NSGA-II 搜索中比较已审查 precision_repair 与新的 compositional_completion 整体机制，检验是否保留历史精度、紧凑与逐输出保护。机制设计以 ../compositional_completion_20261031/PLAN.md 及 DESIGN_REVIEW.md 为准。原始配置、17个任务/种子对、数据与接受规格来自 ../precision_repair_20261028/FREEZE.json 及其原字节归档；控制臂对应该阶段 repair 臂。本批是已观察的历史开发回归。

## 固定设计与预算

- 原17对、base/completion共34系统，按原种子奇偶交换臂顺序；不增删任务、种子或失败。
- base: precision_repair=True, compositional_completion=False；completion: False, True。其余有效配置与原冻结配置相同，两臂均 compound 初始化、closure 24+24、原 precision 选择。
- 候选源码使用共享 CANDIDATE_LOCK.json 的完整动态 src 清单；本批与后续 development 使用同一源码锁和配置。不得读本批结果后改变下一批算法、配置或门槛。
- 每批独立2880秒BUDGET，含600秒评分核验预留；每系统52秒合作fit、60秒外监督、1200次完整评价；数值单线程、单worker串行。preflight启动窗口维持原60秒。失败/超时不补跑、不借后续预算。
- 原完整搜索与polish先完成并冻结foundation；候选只在剩余共同预算内作最多12次缓存外MIMO补充，再执行原24+24 closure。无逐输出非线性拟合。

## 运行前条件与执行

核心与harness稳定，独立CORE_REVIEW/HARNESS_REVIEW无实质阻断；相关回归及完整测试通过并形成TEST_VALIDATION；共同候选锁覆盖全部src；preflight核训练/验证摘要；冻结源、依赖版本、协议、测试/审查/日志和输入原字节。所有旧结果、失败和归档保持原样。

工作目录 D:/符号回归项目/dimensr-mimo，入口 `.venv/Scripts/python.exe experiments/completion_historical_20261031/run_bounded.py`。监控本目录supervisor.log、BUDGET.json、逐单元CELL/FAILURE及最终EXECUTION.json。不得提前生成或读取本批测试标签；所有模型及失败标记先封存为ALL_MODELS_FROZEN，之后统一评分和核验。非法、缺失、重复或非有限指标按固定全分母失败；缺失均值为null，不删样本。

## 固定验收

1. 34/34系统技术通过：各系统真实神经更新与新候选、GP新评价、至少两个NSGA-II step各有新评价、Rademacher、单位/数值域、native/physical重载、全候选训练/验证及所选测试独立核算。每臂matching汇总新评价/有效/训练保护/验证留存均大于0。
2. 17/17对原foundation的完整候选payload/hash、评价数与有效cache一致。控制臂须与旧封存repair臂的完整候选、缓存、选择、数值和结构一致；仅可忽略新增默认配置字段及计时等非算法元数据，不可忽略指标差异。
3. 34/34配对输出干净测试误差不超过控制×1.1+1e-8；当前11个完整接受全部保留，平均干净误差与平均共享图均不升。原各case绝对精度/结构阈值照原规格执行；继续逐项报告原剩余六个未完整接受单元。
4. 指定damped_n0/701保持原精度且图≤20、参数≤4；rational_n0/702保持原精度且图≤15、参数≤4。
5. 新补充逐项报告来源、原implicit/新变换线性子问题、返回根、缓存外评价、有效/训练保护/验证留存及拒绝原因。始终调用的新模式不适用旧repair未触发的完整身份门槛；不能把flag或提案数当实际评价。整体比较不支持分组件因果归因。

## 交付与后续门槛

保存完整冻结与输入归档、逐单元工件/失败、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、EXECUTION、REPORT和独立POST_RUN_REVIEW。主代理接收后审时另写POST_REVIEW_ACCEPTANCE：status为accepted，sha256逐项绑定FREEZE、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、POST_RUN_REVIEW及共享CANDIDATE_LOCK。仅本批技术、科学门槛及后审全部通过才进入development；失败保留，不改变本批或下一批门槛。

本计划不代表运行成功。即使两批通过，仍需稳定候选锁定后的新任务验证、安装态打包核验、最终独立审查与新发布授权；生产默认不变，1.7.0未发布。
