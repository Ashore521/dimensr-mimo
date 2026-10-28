# 1.6.0 matched sharing 独立科研审查

状态：本次限定范围独立审查完成。基线 commit：96563546ce543c9f34f14bcdd8145cd3129d23d0；审查对象为其上未提交的 1.6.0 改动及本目录冻结证据。

## 已核查事实与初步结论

- 已读项目 AGENTS.md、ResearchReviewer.toml，以及本轮 PROTOCOL.md、study.py、evaluate.py、worker.py、REPORT.zh-CN.md、SUMMARY.json、PACKAGING.json、FREEZE.json、ALL_MODELS_FROZEN.json、VERIFICATION.json 和相关源码/测试；大文件由程序化逐字段及哈希检查补齐。
- sharing.py 使用算子顺序、变量编号、单位及局部参数别名模式匹配，忽略参数数值。提案严格缩小 System.size；Builder.finish 不合并参数数值。结构匹配并非等价化简。
- genetics.py 只在 matched 模式原 share 事件的一半切换匹配提案；random 模式的短路条件不额外消耗随机数。Evaluator 仍执行联合常数拟合、结构/数值域检查；正常选择/保留流程未被新代码绕过。
- 固定矩阵为 17 对、34 系统、34 个比较输出。SUMMARY 与报告一致：完整接受 random 9/17、matched 12/17；保护 31/34；宏均干净测试误差 0.002617556289020336 → 0.00268245542528866，推荐失败。默认 random 保留；该证据不能支持默认切换或优越性推广。
- 训练 worker 仅请求 train/validation，评价脚本先 guards 检查全部模型封存再读取测试值。本轮使用已检查历史任务及开发变体，协议和报告明确不是盲评。哈希/时序链独立复核通过；它支持声明的执行顺序，不能证明历史任务从未被看过。
- 本范围未发现必须阻断保持默认不变的 opt-in 发布的实现或科研报告问题。推荐切换默认的门槛明确失败；最终发布/接受决定由主任务作出。此结论不构成全面接受。

## 实际完成的独立检查

1. 104 个 FREEZE.files、374 个 ALL_MODELS_FROZEN.files 哈希全部通过；39 个冻结源码与当前源码逐字节相同。34 个 CELL_FROZEN 的子文件哈希通过，且单元封存不晚于全模型封存。FREEZE → TRAINING_STARTED → ALL_MODELS_FROZEN → SCORES → VERIFICATION 的时间与哈希链通过。
2. 冻结任务、评分及核验记录均为相同顺序的 34 项且全部 complete/verified。34 个 effective_config 的方法、种子、1200 评价上限、52 秒 fit、compound 初始化与 precision 选择均通过；实际评价数不超上限，外部墙钟均小于 60.3 秒。记录总核验时间 685.1785061359406 秒，小于 2880 秒。两个方法实际最大评价数分别为 415/423；公平含义是相同资源上限，不是相同已消费评价数。
3. 使用本轮 summarize 从 SCORES.rows / VERIFICATION.rows 重算 SUMMARY，逐字段完全相同。匹配操作 attempts/changed/novel/successful = 451/272/212/208。三个保护失败值与报告一致：damped_n1/702 输出1 0.001899074480700409 → 0.004924144694852561；rational_n0/702 输出2 0.008688776191983036 → 0.012689473376547095；rational_variant/711 输出2 0.0008602285845567621 → 0.006934745381372824。
4. 独立加载全部 34 个 model.json / physical.json，重载字典一致；逐项核对单位和原图结构。以独立 JSON 解释器复算全部所选图的训练/验证/测试预测，与 estimator 和 PhysicalModel 精确一致；测试 observed/clean NRMSE 按训练观测标准差重算通过。全部 5276 个已保存候选的训练/验证 NRMSE 重算通过。训练/验证数组哈希与冻结和 result 记录一致。根据 saved history/diagnostics 重查每项实际策略更新、有效神经候选、GP 新评价、至少两代 NSGA-II/新评价与局部 Rademacher 计数，均通过。
5. 短测试首次运行 tests/test_matched_sharing.py 与 tests/test_matched_study.py：7 passed，1 setup error。错误为沙箱 pytest 默认临时目录无法创建，不是被测断言失败。随后只复跑 mock-worker 那一项，以内存注册的 tmp_path 夹具返回现有路径，测试中唯一 worker 调用已被 monkeypatch、不会写盘：1 passed。该复跑验证 worker 参数差异与非法方法处理，不验证 pytest 的真实临时目录功能。6 个结构匹配测试验证局部别名、变量/单位、非等价提案、正常联合拟合、非法域/结构拒绝、默认兼容及40种子随机数消耗；第7项验证完整矩阵和失败分母。
6. 已读 check_wheel.py 与 PACKAGING.json，独立核对现有 wheel 的 SHA256 和检查脚本 SHA256 均与记录一致。PACKAGING 记录的隔离实验导入、34重载及旧模型兼容检查未由本审查再次在 wheel 中重跑。已读归档 validation 日志尾部：126 passed、sdist 26 passed；这些属于已有日志证据，不冒充本次独立执行全套。

## 关键来源及解释边界

- src/dimensr_mimo/sharing.py:8、:23、:51、:55：签名、路径/规模限制、前64匹配、最多16个严格缩小图的提案。src/dimensr_mimo/graph.py 的 Builder.replace/finish 保留单位、拓扑和明确参数身份；不会按数值相近合并。
- src/dimensr_mimo/genetics.py:36 与 config.py:17：matched 调度与默认 random。src/dimensr_mimo/evaluation.py:64、:86：原结构拒绝、联合拟合和数值域流程保留。
- src/dimensr_mimo/regressor.py:117、:249、:272、:306：匹配候选经过原 evolve/选择流程；验证值只排列已冻结池。已保存候选池不等于每个搜索阶段所有被评价图的完整谱系；本审查不能据此定位某次父子提案的因果贡献，也不能排除未保留图曾可行。
- experiments/matched_sharing_20261028/study.py:33、:59、:85 和其调用的 first_round/study.py:45：固定矩阵、冻结、顺序运行与资源监督。沿用 compound_search_20261027/worker.py 的训练侧入口，不访问测试 split；compound_search_20261027/evaluate.py:29、:38、:64 负责封存后评分/核验。本轮 evaluate.py:49、:52 保留全技术门槛与逐输出保护。
- REPORT.zh-CN.md、SUMMARY.json：9/17 → 12/17 的绝对接受增加与31/34保护失败、宏均误差上升同时成立；不能用增加的接受数抵销保护失败。damped_n0/701 的紧凑且机器精度图可作为本开发矩阵内结果；不能推广为普遍优势或新盲评。

## 未检查、未知与待决

- 未重跑搜索、训练或基准；未独立重建 wheel/sdist、执行全套安装测试、检查远端 CI 或发布。主任务负责发布门槛。
- 数值域源码及保存 audits 已审读；未重新遍历 interval 审计的所有盒子。enclosed 只代表有限浮点数值筛查，不是连续域证明或工程保证。
- 本轮保护失败已足以否决推荐；原因尚不能从现有父子记录确定。后续诊断或行为变更需要新冻结方案，不能按已读测试结果改选本轮模型。
- 主任务通报其已从平台日志独立核验审查会话 01a18190-e97e-7ed3-a531-5d11fcde2364：parent 01a1818f-08fc-7bf2-9312-1004f778dd59、agent_path /root/release_160_review、ResearchReviewer、gpt-6.1-sol/high。本审查未自行读取该平台日志，不将启动参数或代理自述作为元数据证明。

Verdict：本次限定范围无 opt-in 发布阻断；保持默认 random，维持推荐失败。最终接受决定由主任务作出。

唯一写入物为本报告；未修改历史证据、算法或门槛。未开展搜索、训练或新实验。
