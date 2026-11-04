# compositional_completion_v1：稳定核心独立审查

2026-10-31。What this change does: 新的默认关闭模式在原完整基础搜索/polish之后、验证之前，合并旧implicit返回初值与新的log/sqrt逆变换初值。补充最多12次缓存外完整MIMO评价，再执行原匹配闭包与原base训练/验证双保护。

**结论：本次审查未发现有证据支持的实质阻断，实现满足修订设计，可进入正式harness审查和冻结准备。正式实验启动仍须主代理完成最终全套集成检查、两批harness独立审查及输入/依赖冻结；本报告不是这些步骤的替代，也不是科学采用、默认切换或发布验收。** 最终接受决定由主代理作出。

审查绑定 PLAN SHA256 `3d561857fee731eba08cdf3c0fac757fb16a494bd6032d8420779f0ddfe270a5`、DESIGN_REVIEW SHA256 `c9e2852417db630d79a196af293d67d6e973f5a8bdd094655006de5a3d5222d5`，核心completion.py SHA256 `5210be43a043ce38527da8a675b35efc28d7f7fd08cb8488beeb1be708782318`。全部当前源码摘要见CORE_REVIEW_CHECKS.json及core_review_validation/audit_checks.json。

## 实际验证与版本

- 独立读取新模块、四个连接文件的完整增量与调用链，并连通核对旧repair/structure、Grammar、Node/Builder/System、Evaluator、联合拟合、selection/acceptance、保存重载及physical路径。实现者的IMPLEMENTATION、收据、文档和保留失败日志亦已检查，没有改写它们。
- 独立核验BASE_SOURCE_LOCK所列40份旧源归档原字节，核当前差异仅config、sharing、regressor、acceptance四文件；另新增completion.py。其余36份旧源码未变。40个当前hash、三个新增源码/测试/文档hash、六份实现测试日志hash均与IMPLEMENTATION_CHECKS相同；源码主体没有夹带本工作包之外的改动。当前41份源码完整快照已记录。
- 实际运行最终相关测试集合：`tests/test_compositional_completion.py tests/test_precision_repair.py tests/test_protected_closure.py tests/test_physical_delivery.py tests/test_physical_selection.py tests/test_implicit.py`，**75 passed in 31.44s**。这包含最终新33项；不是重复引用实现者此前74项收据。命令和日志位于core_review_validation/pytest_related.log，pytest临时工件也只在该独占目录。使用项目.venv、数值单线程，tests/conftest.py另将Torch设为单线程。
- 新增只读核查脚本core_review_validation/audit.py，独立构造log/sqrt × direct/multiply/divide × bare/amplitude共12个小图；逐个以原grammar回放、核最终预测幅值及逆变换内层值，含不同物理单位。另验证只有已有逆单位尺度、没有无量纲常数时的合法平方权重吸收。全部通过。脚本没有调用initializer、拟合器或search；相关pytest本身包含获授权的小型initializer/联合拟合/完整fit构造用例，不能将整个审查称作“零拟合”。

测试针对当前源树，不是wheel安装态核验。实现收据环境为Python3.12.10、NumPy2.5.4、SciPy1.18.1、Torch2.14.1+cpu、pytest9.1.1；本次检查另记录实际运行版本及导入路径。当前源码本地版本1.7.0和editable安装分发元数据不能视作已发布或安装包通过。

## 算法、单位与数值语义

1. **可选模式与原基础链保留。** `config.py:25–67`新增严格bool默认False，要求compound/closure/precision，借既有约束继续要求sharing、units及逐输出constraints，与precision_repair互斥。`regressor.py:282–293`仅在原polish完成后传入grammar并分别记录completion/repair历史；`sharing.py:67–81`先冻结原base，再用互斥分支调用补充。四份实际diff没有改变神经、GP、NSGA-II或polish主体。旧repair源码和默认legacy/random均未变；测试中关闭神经只是构造用例，正式矩阵仍须证明真实全机制执行。

2. **固定有界通用枚举。** `completion.py:65–109`按变量、平方、不同变量交叉积、合法归一化的固定语法顺序去重/严格数值筛选后截断24基；外层变量取前16。基来自全部声明变量，外层变量限制与基限制并未混同，18变量构造测试核到该顺序。截断在按目标内层单位筛选之前，这与实现文档明确口径一致；没有按任务名、真式或测试结果调整顺序。每个内层仅使用同单位1/2项和可选合法截距，不伪造单位常数或算子。

3. **单位推导和逆变换正确。** `completion.py:112–163`以U为输出、V为外变量单位，direct潜在响应为y（单位U），multiply为y/x、单位U−V，divide为x/y、单位V−U；log仅允许潜在响应无量纲、内层无量纲，sqrt内层为潜在单位两倍。实际代码使用Fraction单位运算，最终Node和grammar.trace再核。inverse拒绝近零训练除数（绝对1e-12）、非有限潜在值、sqrt负响应、平方溢出、exp溢出或下溢到零，不裁剪或平移标签。`completion.py:211–215`另查最终一元分母有限且大于原阈值，标签侧合法不会自动放行最终除式。

4. **平方吸收及幅值条件落实。** `completion.py:45–62`只对非负无量纲线性权重吸收到合法平方尺度，已有尺度乘sqrt(weight)，负权重走合法外乘；缺合法尺度时拒绝。当前基构造每个归一化基至多一个独立尺度，常数遍历顺序与canonical一起重排，没有把参数别名或截距折叠。新图仅继承数值初值，`completion.py:200–215`重新检查自身边界、严格预测、原Y分数及最终分母。新33项实际覆盖0*square(1e200)与square(0*1e200)的原生区间域差异，未将二者当全浮点等域。`completion.py:193–197`保留bare，并在合法无量纲常数及mul可用时增加a=1的外幅值；变量乘/除的尺度确在变量分支。新测试与本次手工12图均核到位置及幅值语义。参数仍接受原联合拟合器的constant_bound及域修复，不把线性warm分数当最终拟合精度。

5. **原始Y排名和实际来源一致。** `completion.py:37–42,226–277`将原initializer返回根及新根统一canonical，以严格Node.evaluate重算 `MSE/max(var(y),1e-12)+1e-8*nodes`，按(score,size,repr)取同一limit；非法图和非有限预测拒绝。同拓扑只有更低分才替换常数和来源，同分保留首来源，不把落选来源合并成“真实执行”。旧原initializer内部提案排名未被本模块重写，文档准确区分“返回根再排名”；不冒称已修复原谱系近并列排序失败。原Y与逆域残差偏好相反的实际测试、来源常数获胜测试均通过。

## MIMO、截止、会计与保护

- `completion.py:280–368`固定原base训练最优父，所有输出候选按相同索引组合，短列表复制固定父对应根，无循环补齐；至少有一个实际implicit/compositional根才评价。Builder只构造完整多根图，不做逐输出非线性拟合。缓存键包含原图结构，成功/失败缓存均跳过，不因新warm或来源而refine缓存图。总calls以原Evaluator调用增量记账，独立上限12和cfg.max_evaluations并存。
- `_check_deadline`覆盖基枚举、每形式/线性问题前后、新根处理、合并排名；旧structured调用也收到同一evaluator.deadline。系统调度另查同deadline和全局calls。Timeout只捕获可预期截止，保留部分线性诊断及base/已有留存；一般编程错误不被吞掉。测试覆盖构造/旧调用/线性问题中截止，以及已有子被留存后截止。单次BLAS仍不可抢占，生成形式/choices等少量Python原子构造不构成硬实时保证；正式外60秒监督由harness负责。
- 旧structured的linear_subproblems已经包含implicit子集；`completion.py:307–314`的总线性数=旧全部+新全部，implicit另列不重复相加。实际求值、线性、拒绝、返回根及获胜来源都有分账；events保存固定父、graph/warm、child/None、new_evaluation、两个来源输出集合和逐根来源。返回根计数不等于被调度完整系统数，更不等于有效/保护/验证留存，正式harness须继续分别验证，不能拿flag或returned_roots替代实际新评价。
- `completion.py:355–368`每输出训练子≤固定父×1.1+1e-8才留存，失败child仍占缓存外预算；`sharing.py:88–140`后续匹配仍最多两轮各24/总48，沿用原局部父保护和同一Evaluator。原base候选字节摘要断言仍在。新补充消耗剩余总预算，故未承诺每单元都会执行满48次匹配。
- `acceptance.py:26–67`最终保护基线始终来自base_count前缀，先核hash，递归base同时关闭两flag，再构造原训练/验证各×1.1+1e-8边界，未把补充加入保护基准。新增flag仅在原配置存在该键时复位，避免老payload来源hash污染。已有逐输出训练或验证一项退化即保留base的测试75项中实际通过。原始base完整身份、扩展验证无效剔除、缓存失败、短列表、独立12上限及剩余全局预算均有测试。
- `regressor.py:400–454`没有为新flag改变加载核验顺序：先按保存config/来源重建选择并比较，再以新默认迁移。旧无两flag模型、旧repair模型、新completion模型的保存/重载及二次保存测试通过，base hash篡改拒绝；相关旧测试另覆盖老选择来源篡改拒绝。physical相关测试亦通过。诊断存在于模型和history，重载不会重新运行搜索。

## 无阻断不等于全面接受

此次未发现须修代码的问题，不增加猜测性缺陷。下列是已声明或尚未验证的边界，均非本次发现的新bug：

- 非单位外幅值只是可表达图路径；a=1处逆变换、原Y排名和同一前12竞争不保证该变体被返回或收敛。sqrt潜在响应非负限定及24基/16外变量截断仍限制覆盖。不能声称任意复合表达式都能发现。
- 本次75项相关测试不是全套，也不是两批冻结矩阵。原base身份在构造数据中通过，不证明52秒墙钟边缘的每个正式单元必定有相同base；正式逐对身份核验失败必须保留。没有以新结果放宽原门槛。
- 未审查正在并行编写的harness/verification，没有启动历史17对或已观察16对矩阵，没有生成正式case数组、重新评分旧任务、检查新盲任务、重新计算历史全候选/Rademacher、核验安装包或网络发布。
- 原谱系ranking_unresolved、已观察新族7/16失败以及历史六失败均未修改；新模块测试通过不能覆盖这些历史状态。后续两批顺序、每批独立2880秒/600预留、52/60秒/1200评价、不补跑、全模型封存后评分及所有原精度/结构/保护门槛仍须正式协议和收据证明。

本审查只新增CORE_REVIEW.md、CORE_REVIEW_CHECKS.json和core_review_validation内检查/日志/pytest临时工件，未改生产源码、生产测试、PLAN或旧证据。源码hash将在交付前再核。

Verdict: 核心审查无实质阻断。完成最终集成测试、harness独立审查和冻结后，由主代理决定正式执行；科学目标尚未验收。
