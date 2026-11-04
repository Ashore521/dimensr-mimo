# DimenSR 对历史 MIMO 真实数据的固定复测协议

日期：2026-10-25（Asia/Shanghai）。本轮已获用户授权完成本地实现、运行与交付；不发布、提交或推送结果。实验目录独立于两个 Git 仓库。

## 研究问题和比较边界

对八项既有真实多输出任务，按输出独立训练原 DimenSRRegressor，然后合并预测。名称固定为“逐输出独立 DimenSR”，不称原生联合 MIMO。数据是 Concrete Slump、Parkinsons、Air Quality、Gas Turbine、SML2010、Tetouan power、Water treatment、Video transcoding，不使用 DimenSR 先前的 Airfoil/Concrete Strength/Yacht 数据。

所有测试集已经在历史研究中检查过，本轮为固定历史划分复测，不是新盲测。模型仅用拟合行训练，候选选择仅用验证标签；所有 76 个系统得到终态记录并整体封存后，才由单独评分入口对外层/旧测试标签评分。禁止按测试表现改配置、换种子、删任务或追加搜索。

## 固定网格和历史参照

| 组 | 任务 | 划分与种子 | DimenSR 系统 CPU 上限 | 历史主参照 |
| --- | --- | --- | ---: | --- |
| legacy | Slump、Parkinsons、Air Quality | v1 原留出；0、1 | 20 秒 | real_blind_v2 selected joint；旧算法固定 3 generations，**非同预算**；无同留出 independent |
| cv | 上述三项 | real_blind_v3 原 3 层外折 × 种子 0–4 | 20 秒 | v3 joint、independent；只作为补充开发 CV，不与旧 test 混合 |
| new | Gas、SML2010 | v3 原留出；0–4 | 20 秒 | v3 joint、independent |
| external | Tetouan、Water、Video | external_transfer_v1 原留出；0–4 | 25 秒 | released joint25、independent25 |

总计 76 系统（6 + 45 + 10 + 15），标称搜索上限总和 1595 CPU 秒。复用可匹配的历史 mean、Ridge、Extra Trees，仅作预测背景；其预算未与符号搜索匹配。历史 external 未存训练指标的字段保持缺失。首次三个任务增加 45 个 CV 系统是为了得到同划分的联合/独立参照，不把 CV 冒充旧留出结果。

历史版本由各冻结记录及当前快照联合定位：v2 freeze git_head dcd38e565ea4220e099a324e7ec1e8f758f6dee2；v3 训练源 5e0df6a（文档记录）；external 算法 01eefe18cfb46310aff39f834298a311f8eae653、实验源 ddfe8726a345333f785d1d848b09b12b66ef6068。本次当前 DimenSR commit 及两仓库状态、包版本、逐文件哈希在 FREEZE.json 中实际记录。两个 src 包复制到本目录 source/，适配层不得修改包源码。

## 数据和预处理

读取已有缓存，逐文件 SHA-256 与各 manifest 核对。沿用原始特征顺序、目标顺序、缺失值处理、拟合上限 1200、组/时间隔离与间隔。Water 实际仅 250 拟合行。legacy 拟合子采样种子为 20261023；v3 为 20261024 + fold；external 已缓存原预处理结果。完整行 ID、组 ID、行哈希、源 manifest、特征名、目标名、均值/标准差都写入 splits/。

X 使用对应历史拟合行的均值/标准差标准化，零标准差置 1。external 缓存已经标准化，绝不重复变换。y 保持原数值单位；误差归一化使用历史拟合目标总体标准差（ddof=0），零尺度沿用历史 fallback。DimenSR 的符号量纲声明全部为 dimensionless，与历史无量纲设置一致；不推断未知物理单位，不声称验证了量纲约束优势。

## DimenSR 配置与预算适配

SearchConfig 保留当前默认值，仅 max_nodes=13（与历史结构上限一致）以及 random_state 按历史种子设置；max_depth=6、population_size=64、epochs=6、gp_generations=2、selection_generations=3、max_evaluations=5000、默认七算子、RNN/GP/单位/尺寸多样性/Rademacher 开关均保留。准确完整配置以 CONFIG.json 和各模型 JSON 为准。预算为上限，搜索自然结束时不补烧预算、不重启。

每系统总额平均分到 m 个输出，每个输出得到 B/m；顺序训练，不转移剩余时间，不给每输出完整 B。DimenSR 原 API 仅有评估次数预算；独立适配层在候选调用边界用 process_time 检查 CPU。75% 总 CPU 后通过原阶段计数停止第一阶段，剩余阶段仍走原候选筛选、NSGA-II 和最终验证选择；100% 后不再拟合新候选或 refine，只返回已完成缓存。保留原三代 NSGA-II 和原选择规则，不实现新的候选选择器。首个常数候选允许完成，以免初始化占满极小预算时无可用模型；所有实际超额记录。

计时从 fit 前开始，包括 Grammar/Evaluator/Policy 初始化、种子/候选生成、拟合、数值合法性检查和策略更新。原子数值调用、结构初始化或策略更新可能越过检查点。最终 Pareto/验证选择、表达式导出、保存、重载、审计开销另列；Python/torch 导入和数据加载另列。记录实际 CPU、墙钟、初始化、搜索、选择/导出及超额，**只称相同标称系统 CPU 上限，不宣称严格相同 CPU、FLOPs 或优化工作量**。搜索机制和选择器本就不同，不作单组件因果归因。

## 资源、失败和检查点

复用工作区隔离环境 .venv，不改已有依赖。环境版本随 freeze 保存。最多两个单线程子进程，BLAS/OpenMP/torch 都设 1；每个工作进程 RSS 上限 2 GiB。可用物理内存不足 4 GiB 时退为单进程，低于 2 GiB 不启动。系统墙钟超时沿历史设置：legacy 240 秒、cv/new 120 秒、external 300 秒。运行前记录当前进程和内存；不终止不属于本轮的进程。

按任务分批。每输出落盘模型、训练/验证指标、预测和失败记录；每系统写终态。已完成记录不可覆写或换种子。超时、OOM、非有限预测、异常均计失败；整体冻结允许失败终态，不删失败以换取完整率。中断后跳过已验证的终态；无终态目录需人工审查已有进程与输出后恢复，禁止覆盖。

## 指标、表达式与合法性

每任务/种子/输出/分区报告 RMSE、NRMSE=RMSE/拟合目标 SD、R²。任务均值对输出和种子等权；CV 另对外折等权。八项旧留出与三项补充 CV 分开汇总，不把种子当独立数据集做显著性推断，不计算任务总体显著性。

保存可移植模型 JSON 和标准化变量表达式；z_j=(原输入_j−训练均值_j)/训练标准差_j，输出保持原单位。模型中的 raw tree 是数值域审计依据，SymPy 展示式不能代替原始计算图。复杂度并列给出原始树节点数、DimenSR 原混合复杂度，以及将每个非常数输出表达式视为一个权重为 1 的基函数、截距为 0 时的精确值感知共享 DAG 节点数 + 输出连接数 K；纯常数叶按非零截距计数。后者与 MIMO 同一计数 convention，但表示形式不同，不能把较小 K 自动解释为搜索效率或泛化优势。跨输出精确公共子树可复用，单独存储成本同时列出。

选定模型在训练输入盒上做 audit_domain；全模型封存后，对训练 + 验证 + 观察到的评分输入所覆盖的扩展盒审计，并报告评分点超训练盒比例。enclosed 是现有数值区间筛查通过；unresolved 不是已证明非法，有限点成功也不是全域证明。扩展审计不得筛掉差模型。

## 验证与交付

冻结前通过合成数据适配层测试，包括限时、阶段/缓存保留、原选择、保存/重载和复杂度；检查全部数据划分与原预处理一致。冻结后先完成全部模型；验证重载的训练/验证预测逐元素一致，整体封存，再统一评分。最终独立重算所有 DimenSR 训练/验证/评分指标、核对冻结哈希、引用/对照聚合和预测，保存 VERIFICATION.json。交付中文报告、任务对比表、逐输出 CSV、全部表达式和模型/配置/源代码/计时/失败凭据。原始数组与私人文档不复制进公开仓库，本目录仅作本地交付。
