# 通用有理系数初始化：运行前独立审查

结论：设计、核心实际实现、构造测试源码和新实验流程的独立窄审完成。发现的NSGA-II机制门槛缺口已在新harness中修复，当前未发现阻断按固定方案冻结/执行的剩余实质问题。主任务可据既定授权决定冻结并执行34单元；本审查不增加预算或实验授权，不预先接受性能/结构成效，最终科学接受由主任务作出。

## 范围与实际检查

仅写本报告，不修改代码、测试、历史材料或冻结证据。未运行 initializer、fit、search、pytest、数据生成或新实验，未读取新实验 clean/test 指标或标签，未派生代理。读取主任务已有 pytest 日志和测试源码。实际模型和推理等级没有平台元数据核验。

已读本轮 PLAN.md、上轮 rational_seed_lineage_20261028/REPORT.zh-CN.md 和 POST_RUN_REVIEW.md；核对 structure.py、config.py、regressor.py，以及相连的 seeds.py、evaluation.py、fitting.py、_single/expression.py、_single/grammar.py。另读 balanced_closure_20261028 的方案、既有预审和 study.py，确认固定预算与此前审查边界。代码审查采用 ponytail-review/ponytail 的连通调用和最小必要改动原则；既有报告不替代本次实际检查。

初审时三份受改生产文件逐一与 balanced_closure_20261028/source 对应快照做 SHA256 比较，全部一致：

- structure.py：`2a8d01e7ae569012a520ba57308415b10b48d0381030735136779251767f8c6e`
- config.py：`51912d5033a7fbd5b440b270d5e53f0773e1dfe17b05d7b0a8beeae437223509`
- regressor.py：`c432e7a2f7b704c0f0c0b95b4d75855360314539a9f27850182d091f663259b3`

本轮固定 PLAN.md 当时 SHA256 为 `ad427db132ede096145441ab7689cf02d37a609b1584239f750818c34618a271`。旧训练观测报告记载的第22/13位及其计数作为已有、已独立审查的阶段证据使用，本次没有重新调用旧初始化或再次复算全部旧观测。

## 设计核对

1. 对 D=d+b，`y*d=N-b*y` 的线性子问题符号正确。若 d 单位为 U_d、输出为 U_y，偏置 b 必须为 U_d，分子和线性目标均为 U_y+U_d；每个分子乘子必须为该分子单位减对应特征单位。通过已声明常量单位选择、普通 Node 和 grammar.trace 可维持语法/量纲合法性。固定 d 外层系数只规范当前子问题，不保证嵌套参数全局可辨识。
2. 交叉乘积最小二乘与原输出最小二乘具有不同权重，因此只能作为初值生成。按 N/(d+b) 的训练误差与既有大小项共同排序，随后仍经完整 MIMO 的正常拟合、原始算术、探针和区间检查，是检验该假设的适当路径。不能从训练点 D 非零推断完整域无极点，也不能把区间 enclosed 写成解析证明。
3. 每个候选必须在求得自己的 b 后检查 D 是否有限且所有训练点绝对值大于1e-12；不能用原 d 的检查代替。分子预测必须移除 `-y*b` 列贡献，再除 D；否则会把交叉乘积拟合量误作 N。零列、秩亏和巨大系数不代表成功，必须由有限性、直接分数和后续完整域接受约束。
4. 初始化只接收划分后的训练 X/Y；现有 fit 先分割/接收验证数据，再调用初始化。通用分母特征、单位和训练列构造不需要任务名、真值系数或验证/测试标签。已经观察过的任务/种子及指定 rational/damped 门槛属于历史开发回归，不能报告成新任务盲验证。
5. 固定前 limit、现有初始化份额及相同 max_evaluations/wall_seconds 保留总预算比较。新增线性计算会占用初始化时间并改变入选种子与后续轨迹，故不能称保持同一基础搜索，也不能要求同提案数。52秒 fit 是合作式限制，严格60秒仍依外部监督；单个线性代数调用可能越过合作式检查点。
6. 记录必须区分隐式子问题次数、有效提案、实际返回位置、完整系统新评价和有效系统；同树新旧提案去重后应归因实际获胜的初值。完整系统缓存命中或预算/结构拒绝不能当作隐式 warm 已重新拟合的证据。计数大于零也不等于科学精度/结构通过。

## 核心实现稳定版窄审

逐份用 `git diff --no-index` 对照 balanced_closure_20261028/source 的三份源码，并追踪新增参数的调用者、Node.canonical 的参数重排、Builder.tree/finish、Evaluator.__call__ 与完整 fit_constants 路径。PLAN 摘要未变。受审版本 SHA256：

- src/dimensr_mimo/structure.py：`2fe3ac4ee5762c307d6cee9625677425271581c15f245052c1f3ca6f39c6ae5d`
- src/dimensr_mimo/config.py：`be9d7871a6c3a1788fd04ad1b8b730cd1c39ef6219805011c244bd78da8d30cb`
- src/dimensr_mimo/regressor.py：`cc635471045349f95789cf3083cf22baddc21bd666624f10ffa85959d6d790e6`
- tests/test_implicit.py：`bab88611490e16790dcd3451f99322b0114724d14dfb677dbac4952c3d3abc39`
- outputs/pytest-implicit-targeted.log：`ab0176f58ba7ce3cac827ce0d60b87ab3511c18670cdad05e48d0d0a379c2f8d`

实际确认：

- structure.py:207-313：`fit_offset` 仅增加必选 `-y` 基底列，偏置取基底末列系数；分子用所选特征系数和可选截距重新计算，没有把 `-y*b` 当作分子。D 对每个拟合后的 b 单独做非有限/绝对值<=1e-12拒绝，再组装普通 add/div 节点，canonical 同步重排常量，grammar.trace 后按 tree.evaluate 的直接训练误差重算最终排名分数。
- structure.py:316-334：旧 additive/rational 路径先完成，再在原特征上选择大小<=5、具有合法偏置常量单位、根非add/sub的前64分母。旧提案先进入同一个按树去重的字典；只有分数严格更优的新初值覆盖实际获胜值并进入 implicit_proposals。之后没有旧支路再次覆盖该标记，所以 implicit_retained_indices 对应实际返回的隐式获胜初值，而非仅仅同形状曾被尝试。
- config.py:24,52-53 与 regressor.py:146-208：默认仍为 legacy；implicit 路由同时开启原 compound 特征，旧调用默认 implicit=False。每输出 structured 名额、完整系统组合顺序、原 initialization_fraction 和全局预算上限未增加。额外计时位于原 fit start/deadline 之后，未形成独立免费预算。
- regressor.py:159,175-182：structured 返回位置加上 typed 前缀长度，并按实际 `k % len(seed_lists[j])` 判定系统是否含隐式初值。new_evaluations 取完整 evaluator.calls 增量；valid 同时要求 candidate 非空且 calls 增加。Evaluator 在 cache/预算/结构拒绝前不增 calls，增 calls 后沿原完整 MIMO fit_constants 及数值域/Rademacher 路径执行，因此这两个新增字段没有把缓存命中或未启动评价当成新有效隐式系统。jobs 是尝试数，允许包含缓存或拒绝；valid 不表示各科学接受门槛通过。
- implicit_valid_proposals 在通过直接误差/语法检查后计数，发生在树去重之前，因此是合格提案尝试数，不是唯一拓扑数。implicit_denominators 是被调用分母数，即便目标溢出或无合法分子而提前返回也会计入；解释结果时须结合 implicit_linear_subproblems、返回位置及完整系统计数。

已读 tests/test_implicit.py：共4个测试函数，分母拒绝参数化为2项，因此隐式测试为5项。它们覆盖有量纲双输出构造的合法树及直接预测误差、与旧冻结compound的树/常量/计数精确对照、注入零和非有限分母偏置后无隐式返回、正常完整系统有效新评价与save/load预测一致。最后一项的验证数组取自同一构造训练数组子集，只用于代码路径/序列化测试，不能作为独立泛化证据。

已有日志为合计9 passed in 8.53s，包含上述5项及主任务所报compound的4项；日志本身没有逐项名称，本审查没有独立执行或从日志重建命令。没有据此声明全套测试或正式任务性能通过。目标源文件的 CRLF-aware `git diff --check` 实际无输出；未改历史换行。没有新增依赖。

## 待完成检查和决策

- 核心源码、上述构造测试及下文新harness窄审已完成；全套与随后针对门槛修复的定向日志已检查。正式冻结时仍需生成PREFLIGHT/FREEZE实例、记录最终源码/测试/报告/预算摘要和依赖版本。定向通过不能代替正式隐式种子运行证明。
- PLAN 的34/34技术及逐输出10%+1e-8保护、所有原compound已接受单元保持、接受数/均值不退化、damped_n0/701图<=20和各误差<=1e-12、rational_n0/702完整接受且各误差<=1e-4均应分别检查，不能用均值补偿指定失败。生产默认保持 legacy，新任务/种子验证和最终审查仍在本轮范围之外。

截至本阶段，已有构造测试通过日志支持完整系统执行路径，但尚无正式34单元运行证据，也没有历史任务性能提升证据。

## 新实验流程初审及待修项

已读取新 study.py/worker.py/evaluate.py/run_bounded.py/PROTOCOL.md、tests/test_implicit_study.py，并追踪 compound_search 的 evaluate/study、first_round 的 run_job/mechanism、second_round 的 previous 导入替换、compound cases.training_data 及 typed delivery。未调用这些脚本或构造函数。

实际确认 worker 用 job.method 显式覆盖 initialization，且两臂均设置 closure/precision，未复用旧 worker 的强制 compound 路由。配置复核按冻结共享配置+case接受条件+seed+该方法构造完整 expected SearchConfig，并对模型 config/result.effective_config 做摘要相等检查。矩阵由原17对组成34唯一单元，奇偶种子反序；全局与单元预算、600秒保留、单线程和0.2秒外部进程树监督均沿原路径；已封存单元只核验后跳过，未封存但存在job的中断单元报错，失败无补跑。

训练路径只请求 train/validation，training_data 拒绝其他 split；所有模型封存和摘要guard后才由复用evaluate生成test。原 JSON raw 解释器、真实保存重载预测、全部候选train/validation重算、单位/结构与原域筛查路径保留。新verify要求34/34分数行，并按冻结job序列逐一核对；失败保留verified=false而不缩分母。采用门槛分别保留逐输出保护、compound每个已接受单元不丢、接受总数/均值、指定damped和rational，不用总数改善抵销另一case失败。

**发现的实质问题（已修复，见终态复核）：NSGA-II代数门槛起初未按协议实现。** 流程初读时 study.py 直接把 `prior.mechanism` 赋给 mechanism；该函数来自 compound_search_20261027/study.py:104-108，只检查 `nsga_generations>=2` 和全部代 `nsga_new_evaluations` 总和>0。PROTOCOL要求至少2代各有新评价。例如新评价序列 `[1,0,0,0]` 会被继承检查接受，但只有1代工作。regressor.py:264-273 可以在没有新子代、尚未到全局期限时继续记录后续代，因此不是不可达的虚构输入。最小修复是在本轮新harness中按events统计 `new_evaluations>0` 的代数，要求>=2，并补该反例构造测试；不要修改旧冻结helpers。

初读的三项harness测试源码覆盖方法路由/训练split限制、均值不能抵销已接受case丢失/机制缺失/指定rational失败/超时分母，以及initializer模型history与events漂移。已读日志保留首次worker括号SyntaxError、第二次测试预期AssertionError但实际ValueError、第三次3 passed in1.37s。没有据此声明修复后的NSGA门槛或全套测试已经通过；最终版本及日志待窄复核。

## 流程终态复核

修复后的 study.py:28-34 仅在本轮包装 prior.mechanism，按events中phase=nsga2且new_evaluations>0的唯一step数要求>=2，两臂共享同一检查。tests/test_implicit_study.py 新增反例 `[1,0,0,0]` 拒绝、`[1,1,0,0]` 通过的构造测试，原helper未修改。该项已解决。TRAINING_DIAGNOSIS 新副本首部也明确“当前没有实现或试验”指旧诊断时点，不改写历史原报告。

实际用标准库核对 TEST_VALIDATION.json 的6份validation日志，SHA256全部匹配；对4脚本及2份隐式测试仅做AST解析，6份语法均通过，没有导入或执行initializer/实验/pytest。日志终态为：全套146 passed in58.76s；之后仅改本轮NSGA正评价代数门槛及其测试，最终harness4 passed in1.47s。没有把它写成“最终全部147项重跑通过”。两次开发失败保留，全部日志及TEST_VALIDATION均在freeze源码清单中。主任务记录了代理平台session元数据，TEST_VALIDATION声明platform_log_verified=true；本审查未自行读取该平台日志，因此不以本代理自述独立证明实际模型/推理等级。

重新核对3份核心源码和tests/test_implicit.py摘要，仍等于上文受审稳定版。最终流程SHA256为：

- study.py：`e785cf29bce2cd474795b4576897bcc0836f2f5525874e6c97973687caae9015`
- worker.py：`0507bbed1440b22246f9da16be29f878193a6ca9878a5afcaa34a235a773c528`
- evaluate.py：`6b478ccf2371d065789e6eb5fa414968348e5911fe3fabce2380c8ba9fa0507b`
- run_bounded.py：`95fa0a3eac482ff4e14de13cd4d4614b273cd76b7069ab9d5737b1a77463cb6d`
- PROTOCOL.md：`24c9f7cac273202d5953e79e3cc60bab0250a698c114ec0cd351d5d61d3e61e5`
- tests/test_implicit_study.py：`503a63dcc39f22b3004c390cbae9aa09e7606373a82647af81db7a1f0f437d45`
- TEST_VALIDATION.json：`069575ae2ca6c39b5113aae326dc541603093ebf1157c45e2fe4371df4816660`

冻结清单覆盖src、全部测试、本轮四脚本/协议/方案/诊断/本报告、validation日志及TEST_VALIDATION、继承的first/second/typed/native/noise/compound/matched/protected/balanced脚本、JSON raw解释器、历史genetics、旧compound行为对照源码及配置来源first_round/FREEZE。BUDGET写后纳入摘要，依赖版本由freeze记录。当前尚无PREFLIGHT/FREEZE/BUDGET实例，符合尚未启动状态；本审查核查的是清单源码与已备材料，未预先声称冻结校验已经通过。

正式verify中，闭包的new/valid/protected/validation_retained按每方法批内合计均须>0，隐式的jobs/new/valid/linear_subproblems/retained_roots批内均须>0。initializer history与events逐字段相等、0<=valid<=new<=jobs且new<=阶段new_evaluations和返回位置界都被检查；这些检查叠加已审过的计数生产路径，仍须正式运行后核查真实结果。

剩余未知：34单元是否在原预算内完整执行、实际隐式有效新系统与全部MIMO机制、数值域合法性、逐输出保护、指定rational/damped精度和结构、平均值及全部接受保持均未产生本轮正式证据。没有未解决的代码/流程实质阻断；没有未见任务验证或生产默认切换结论。
