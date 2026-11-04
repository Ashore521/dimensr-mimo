# 1.7.0 锁定候选发布验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-04
- Verification Status: PLAN_ONLY
- Version Label: completion_release_validation_v1

## 发布目标与候选

用户要求继续至发布 1.7.0，然后暂停优化。候选承接 compositional_completion_20261031/CANDIDATE_LOCK.json 的全部 41 份源码，在编写本批任务之前另存本目录 CANDIDATE_LOCK.json。先完成已定 completion_development_verified_20261031 开发批与独立后审，随后执行本批，最后完成安装包和独立发布审查。

本批用于描述锁定候选在新输出组合上的表现。采用门槛与软件发布分别记录。1.7.0 提供显式启用的 closure、implicit、precision_repair 和 compositional_completion 接口，默认继续使用 legacy/random。科学门槛失败会进入发布说明和后续优化清单。此发布阶段只进行验证和必要工程修复，算法优化在发布后暂停。

## 固定矩阵

四个双输出合成族，各用无噪声与 1% 噪声、种子 10241/10242。16 对、32 系统，对照 precision_repair=True，候选 compositional_completion=True，其他参数完全沿用已冻结共享配置。顺序按种子奇偶交错，训练/验证/测试为 96/48/192。两个输入均取 [0.2,1.4]。每个族的两个噪声层有独立 X。随机种子序列首项固定 20261104。

事前指定的公式和单位如下。T 为时间单位，L 为长度单位，常数的物理单位以使各算子合法为准。

| 族 | 输入单位 | 输出公式 | 输出单位 |
|---|---|---|---|
| log_affine_ratio | 1, 1 | log((1+0.4*x0)/(1+0.7*x1)); x0*log((1+0.4*x0)/(1+0.7*x1)) | 1, 1 |
| reciprocal_quadratic_norm | 1, L | 1/sqrt(1+x0**2); x1/sqrt(1+x0**2) | 1, L |
| oscillatory_affine_saturation | T, 1 | sin(0.6*x0)/(1+0.5*x1); cos(0.6*x0)/(1+0.5*x1) | 1, 1 |
| exponential_affine_saturation | T, L | exp(-0.4*x0)/(1+0.6*x0); x1*exp(-0.4*x0)/(1+0.6*x0) | 1, L |

这些任务由作者在候选锁定后编写，已有实验包含其组成算子和部分相关结构。新颖性审计须限定到已检索的历史注册、公式和图，明确报告重叠。此批属于新输出组合验证，外部盲基准和实测数据仍属于后续研究范围。

真式图仅用于冻结前的静态语法、单位、定义域和结构预算审计，使用预设五个固定点核对 NumPy 表达式。搜索入口只接收训练/验证数组和公开配置。测试标签必须等全部模型与失败封存后才能生成。实际数据和模型不得进入任务设计或算法修改反馈循环。

## 预算、指标和验收

单独一次 2880 秒批预算，包括 600 秒核验预留。每个系统 52 秒合作 fit、60 秒外监督、1200 次唯一系统评价，单线程串行。启动 preflight 窗口 60 秒。失败保留完整分母，禁止补跑、追加种子或扩大预算。此预算仅用于此前目标要求的锁定候选新任务验证。

沿用原绝对精度和结构阈值。无噪声 train/validation/observed-test/clean-test NRMSE 均 <=1e-4，1% 噪声观测误差 <=0.03、干净测试 <=0.02。共享图 <=24、独立参数 <=6、展开节点总数 <=30、输出深度 <=6。所有误差按训练观测标准差缩放。

技术验收要求 32/32 完整运行、原神经/GP/至少两 NSGA 新评价、联合拟合、Rademacher、单位/域、全池重算、native/physical 重载和机制事件核验。两个臂的完整基础搜索池、计数和指纹逐对一致，matching 各臂 new/valid/protected/validation-retained 均 >0。本批具有同时拟合的新对照，单独报告该基础身份检查。

科学采用要求候选 16/16 完整接受、32/32 输出误差 <=对照*1.1+1e-8、对照全部接受保留、宏干净误差和平均图规模均不升。来源事件只说明实际候选路径。全批比较包括始终补充策略和新增逆变换初值，组件因果效应须未来独立消融。

## 实现与冻结

复用冻结 shared harness 的 worker、配置、矩阵、训练监督及原 Verifier 的数值和补充事件复算。新目录仅实现任务、stage 绑定、前置接收链、独立新数据 preflight、冻结输入与本批新对照的汇总。由于本批没有同任务的历史拟合，对照身份字段必须明确定义为同批 foundation 身份，汇总不生成“archived_control_identity passed”的声明。

新增代码经构造测试和独立设计/harness 审查后冻结。输入闭包包含源码、执行依赖、配置、计划、锁、历史新颖性审计、静态 oracle、构造测试与日志、前置开发终态和后审。所有源字节单独归档。原历史失败及隔离核验通过两条链均保留。

交付逐单元结果、FREEZE/BUDGET/ALL_MODELS_FROZEN/SCORES/VERIFICATION/SUMMARY/EXECUTION、输入归档、报告与独立后审。软件发布另行完成源码与隔离安装态测试、源码包内容检查、独立发布审查、GitHub main 提交及 CI 核验。
