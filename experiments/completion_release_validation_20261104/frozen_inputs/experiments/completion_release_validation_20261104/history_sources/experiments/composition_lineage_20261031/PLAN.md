# 复合一元结构初始化谱系诊断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-31
- Verification Status: PLAN_ONLY
- Version Label: composition_lineage_v1

## 问题与范围

已封存new_task_validation_20261030技术通过，但完整接受7/16、整体验收失败；new_task_pool_audit_20261030确认无噪声对数族及代数范数族均缺训练精度和结构同时达标的候选。固定问题是：原compound初始化在哪些环节限制复合一元结构——特征生成/准入、提案结构拒绝、初始训练误差排名、前12截断？

仅观察两个已用任务logarithmic_gain_n0/9131、algebraic_norm_n0/9131，每任务两个输出，固定4次原structured_seeds调用，compound=True、implicit=False、limit=12。选择来自两族已核训练缺口，两个种子均有缺口后固定首种子；不取测试表现最差者，不搜索其他种子或公式。该范围不能代表另一种子、含噪声任务、repair隐式阶段或神经/GP完整搜索。

不改生产源码，不加特征或提案，不提供真式/oracle，不拟合完整MIMO系统，不调用神经、GP、NSGA-II、非线性拟合、保存模型的选择器或测试评分。原initializer内部线性最小二乘和残差投影是被观测对象，按实际次数报告。结果属于已观察开发任务的机制诊断，不是盲验证、科学采用或发布依据。

## 来源与输入

生产实现和cases.py均从new_task_validation_20261030/frozen_inputs加载，40份源及案例生成器须与原FREEZE中的摘要一致。两个原base/model.json只用于配置、单位、训练摘要和initializer计数；模型含验证字段，但本脚本不读其验证指标。仅调用training_data(name,'train',9131)，不读取额外clean、validation、test数组，也不调用truth_graph。训练数组摘要必须与原模型diagnostics.data_sha256.training一致。

冻结诊断脚本、方案、独立PRE_RUN_REVIEW、原40源/案例生成器/两个模型/原FREEZE与ALL_MODELS_FROZEN，以及本轮BUDGET与依赖版本。全部输入原字节复制至本目录frozen_inputs，执行后再核。不得修改或覆盖任何历史工件。

## 观测与核验

使用Python事件追踪原函数，不修改返回值、局部变量、随机状态或源码。逐事件flush记录admit尝试图、常数、是否准入与实际返回行；fit_terms分母调用；原单特征与多项式/有理提案结构边界判断处的图、初始常数、size/depth、有限误差与是否越界。边界行通过冻结源码的唯一文本锚点确定，不猜行号。

每次原函数返回后，先保存完整raw特征、分母、提案分数/常数、返回列表和计数，再开始断言验证。独立JSON树解释器重算全部提案初始训练误差、排序和实际前12；误差分数须在rtol=1e-10、atol=1e-12内相容，完整排序及前12必须逐项一致。近并列导致排序不同也记ranking_unresolved，不能称通过；不重跑或改变排序规则。原initializer计数须与两个原base模型对应输出逐字段相同。

诊断JSON将非有限数值按原位置编码为nonfinite标记（nan/+inf/-inf），不因记录这些值而向原函数抛数值序列化异常；真实I/O错误仍明确失败。独立解释器校验常数数量和消费顺序，并逐节点传播非法中间值。严格树分数与原线性估计不相容时保存raw、失败收据并停止余下未启动调用，不修改原分数或静默删提案。

报告区分：从未被该次initializer提出、提出后被结构条件拒绝、进入合法提案但排名落后，以及已经返回。没有目标拓扑的符号等价穷尽匹配，不凭字符串缺失声称全搜索从未生成目标。后续归因如要使用具体形状，只能读取本轮完整已保存谱系；不得为预想修复构造更有利候选后注入原排名。

## 资源、执行与停止

入口 `.venv/Scripts/python.exe experiments/composition_lineage_20261031/diagnose.py`，现有项目环境。单进程数值单线程，4调用串行；沿用原谱系诊断60秒外部总墙钟上限，含worker导入、冻结、归档、追踪和核验，不增加上限。外监督检查活进程并在硬截止终止该worker；不与任何正式实验并行。worker仅导入/本地计算，不生成子进程。

已有BUDGET/FREEZE/EXECUTION/worker.log任一执行证据即拒绝重启。失败或超时保留已落盘trace/raw、日志、EXECUTION和固定4调用分母，不补跑、不加时；跟踪开销超限仍如实失败。独立预审无实质执行阻断、必要构造自检及源摘要核对后才执行。方案属于用户已授权的连续受限训练侧诊断，不引入新的优化批次、外部调用或公开发布。

每调用先写STARTED，正常错误尽力保存FAILURE；完整逐调用核验收据绑定raw/trace摘要。外监督无论VERIFICATION缺失或损坏都按未通过处理并保存固定4行执行状态，区分passed、failed、ranking_unresolved、invalid_receipt、interrupted_or_failed和not_started。部分raw不能证明完成。硬终止可阻止worker写完Python收据，磁盘不可写也可能阻止外监督收据；此时保留实际现场，不声称完整通过。

## 成功与后续

成功须4/4原调用完成、原计数及训练摘要一致、所有独立分数/完整排序/返回列表一致、源和归档字节一致、无额外拟合/验证/测试访问。交付trace/raw/逐调用核验、VERIFICATION、EXECUTION、结果报告及独立后审。任何未完成保留明确分母与错误。

这一诊断只确定下一步通用机制假设。算法改动、候选增强试验、资源或触发策略变化均另立新冻结阶段并独立审查；不得用本次诊断降低原精度、结构或逐输出保护要求。
