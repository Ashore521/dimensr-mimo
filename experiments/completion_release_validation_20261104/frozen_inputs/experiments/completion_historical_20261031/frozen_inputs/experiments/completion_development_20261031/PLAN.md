# 通用复合补充：已观察新公式族的固定开发对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-31
- Verification Status: UNVERIFIED
- Version Label: completion_development_v1

## 目标与来源

沿用 ../new_task_validation_20261030 的四族、两噪声层、两个种子9131/9132，共16对/32系统，检验新的整体补充机制能否同时达到原完整精度/结构接受与逐输出保护。该矩阵已被观察和用于训练侧诊断，本轮明确为开发回归，不能称新盲验证。数据、划分、接受规格以旧PLAN/FREEZE及原字节归档为准；控制臂对应旧repair臂。

## 固定设计、顺序与预算

base为precision_repair=True/compositional_completion=False，completion为False/True；均compound、closure24+24及原precision选择。使用与completion_historical相同的共享CANDIDATE_LOCK和配置，动态覆盖全部src；禁止看历史批结果后调整本批候选、数据、预算或阈值。训练/验证/测试仍96/48/192，1%噪声标度只来自训练数据。

历史批须技术和采用门槛通过、独立后审被接受，且POST_REVIEW_ACCEPTANCE的status及全部sha绑定可核，才启动本批。历史失败不得自动进入本批。本批独立2880秒BUDGET含600秒评分核验预留，原60秒preflight窗口、52秒合作fit/60秒外监督、1200次完整评价、数值单线程及worker串行不变；不补跑、不借预算、不增加测试后重选。

## 运行前条件与执行

机制设计与核心审查见 ../compositional_completion_20261031/PLAN.md、DESIGN_REVIEW、CORE_REVIEW；本批harness须独立审查并通过构造测试，最终完整测试及收据先完成。固定本计划及协议、源锁、训练/验证摘要、依赖版本、全部输入和审查日志后，使用 `.venv/Scripts/python.exe experiments/completion_development_20261031/run_bounded.py`，工作目录 D:/符号回归项目/dimensr-mimo。

只监控本批supervisor.log、BUDGET、逐单元收据与EXECUTION。先封存32个模型/失败标记为ALL_MODELS_FROZEN，再统一测试评分。非法、缺失、重复及非有限指标按固定分母失败，缺失均值null；失败保留。旧源、全部历史及失败不改。

## 固定验收

1. 32/32系统技术通过：真实神经更新及有效新候选、GP新评价、至少两个NSGA-II step各有新评价、Rademacher、单位和原数值域、native/physical重载、全候选训练/验证及所选测试核算。matching每臂新评价/有效/训练保护/验证留存汇总均大于0。
2. 16/16对foundation完整候选payload/hash、评价数、有效cache一致；控制臂与旧封存repair臂的完整候选、cache、选择、数值及结构相同。可忽略新增默认字段及非算法时间等元数据，不能忽略误差或结构差异。
3. 候选16/16完整接受，32/32配对干净输出误差≤base×1.1+1e-8，旧repair七个完整接受均保留，宏平均干净误差和共享图不升。
4. 每系统两个输出同时通过原绝对精度与结构：无噪声train/validation/test NRMSE≤1e-4；1%噪声观测NRMSE≤.03且干净测试≤.02；共享图≤24、独立参数≤6、总展开节点≤30、最大深度≤6。原训练观测标准差作误差标度，单位和域条件不变。
5. 逐族/噪声/种子报告误差、结构、接受与失败原因；补充实际来源、线性工作、缓存外评价及有效/保护/验证留存独立记账。新始终补充模式不适用旧未触发身份门槛，不以开关当执行证据，不把整体对照归因为单个构件。

## 交付与证据边界

保存FREEZE、BUDGET、完整输入归档、逐单元模型与失败、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、EXECUTION、REPORT及独立POST_RUN_REVIEW。技术通过不替代16/16采用门槛；任何失败按原口径保留。不得按已见测试结果调参、注入真式、改触发或重选。

本轮为小矩阵描述性开发证据，不宣称统计显著性或任意新任务成功概率。即使两批通过，仍缺新的锁定候选新任务验证、打包/安装态核验、最终独立发布审查与新发布授权；生产默认不变，1.7.0未发布。
