# 复合初始化谱系诊断：独立执行前审查

2026-10-31。结论：固定四次训练侧观察的设计范围合理，来源及原调用参数一致，但当前观测器存在下述实质执行阻断，**尚不支持进入正式冻结执行**。建议仅修复诊断层和构造测试，再对变化部分窄复核；不需改生产算法、任务、原60秒预算或门槛。

## 需先修复的发现

### 1. 非有限边界值会让trace主动终止原initializer

位置：`diagnose.py:117-139`。原 `structure.py:308` 会对非有限error局部continue。观测器虽然把非有限error写为null，却直接序列化constants，`allow_nan=False`遇到Inf/NaN即在trace回调内抛异常，替代原函数继续拒绝该提案的控制流。

已用提取的trace/emit AST与假frame构造核证：fit_terms结构边界的error=Inf、c=[Inf]、size=8/depth=4，原条件应拒绝该提案；观测器却抛 `ValueError: Out of range float values are not JSON compliant: inf`，trace没有写出该行。没有调用原initializer或正式数据。这证明观测器对该边界并非只观测；尚不能说四次正式调用一定会遇到它。

最小修订：为trace/raw中的非有限数值使用明确、可回放的标记编码，保留原finite/invalid状态和常数位置，不使记录回调因数值本身异常而改变原控制流。磁盘写入失败仍应明确终止并保留失败，不能吞掉真实I/O错误。增加无initializer调用的假frame构造测试，覆盖非有限error与常数、正常常数/树配对和各reject路径。

### 2. 独立raw解释器改变了冻结图的严格数值语义，并忽略多余常数

位置：`diagnose.py:90-103`；对照冻结 `_single/expression.py:82-108`。raw只返回最终值，没有传播任一中间节点的非有限状态，也不校验常数数量。冻结Node.evaluate会把出现过非法中间值的样本标NaN，并要求常数个数与树叶一致。

构造检查：对正输入x0=.4/.8，树 `1 / (1 / (x0-x0))`，当前raw返回[0,0]；从冻结源码独立提取的纯evaluate方法返回全NaN。向同一树多给一个常数99，raw仍返回[0,0]，冻结evaluate则拒绝Incorrect number of constants。该反例只是解释器语义/配对边界测试，不是向正式排名注入候选，也不是证明当前四次调用已包含这种树。

最小修订：独立解释器逐节点累计非有限掩码，验证常数叶数、消费顺序及完全消费，最终结果不得掩盖非法中间值。原initializer的向量/线性误差与严格树误差如不相容，应保留原分数与raw并明确score/semantic_unresolved，不静默改原分数、删提案或继续声称核验通过。增加相应纯构造测试。

### 3. 截止时部分写入的VERIFICATION会使外监督漏写EXECUTION

位置：`diagnose.py:220-222`。worker使用直接exclusive JSON写入，可能被60秒截止终止于VERIFICATION写入中途；main读取存在但不完整的JSON没有异常边界，因此在写EXECUTION前再次异常。

只重放收据读取与写入两条AST语句，令存在的VERIFICATION内容为`{`：得到JSONDecodeError，EXECUTION写入次数为0。没有运行main/worker、创建进程或触发正式预算。worker内部异常或分数断言失败也会直接结束循环，没有逐调用终态核验；固定四调用数虽然保存在BUDGET/FREEZE，执行收据仍不足以直接区分已完成、失败与未启动。

最小修订：外监督对缺失/坏JSON核验收据按verification_passed=false处理，仍exclusive保存EXECUTION，并记录expected_calls=4及从已有独立完成收据确认的completed/failed/not_completed状态；不能凭部分raw推定已核验。正常Python异常应尽力保存当前失败和余下未执行的固定分母，不补跑；硬kill/不可写磁盘保留明确不能写完Python收据的边界。增加部分VERIFICATION、worker异常、超时、数值核验失败及原工件不覆盖构造测试。

## 已确认的设计、来源与隔离

- 观察对象明确为logarithmic_gain_n0/9131与algebraic_norm_n0/9131，各两个输出，共4次compound=True、implicit默认False、limit=12。原regressor对24种群/2输出正是每输出12，使用同一plain grammar和训练X/Y。没有调用完整MIMO拟合、非线性拟合、神经/GP/NSGA-II或最终选择器。
- 本次只读核40份冻结src及冻结cases摘要，全部与new_task_validation FREEZE相符；旧ALL→FREEZE交叉摘要一致，两个9131/base模型的成员/摘要一致。worker先从frozen_inputs/src导入，检查structure/cases实际路径；训练数据在核来源后才生成，随后与原diagnostics.training摘要核对。审查没有调用training_data、structured_seeds、fit/search或诊断入口。
- cases仅请求train，原模型验证字段未参与该脚本运算；没有读取验证/测试指标、额外数组或truth_graph。initializer内部线性最小二乘、相关投影是实际被观察计算，不能把这次工作称作完全零拟合或纯统计。
- 两个边界文本锚点在被冻结structure.py中各唯一一次，实际为198与308行。单特征边界的未canonical树对应`[weight,c...]`，多项式/有理边界对应已组装tree/c；结构判断前记录位置正确。admit返回行可区分实际准入与各早退，分母call、合法proposal总表和前12列表均有保存路径。trace只读局部变量，恢复原trace并关闭流的finally路径存在。
- 这些观测能区分已到admit/已到两个提案边界的结构、合法提案排名和截断。没有记录所有语法可生成树、_compound_products全部前置组合筛选或完整神经/GP谱系；“未观察到/未准入”不能扩张为“整个搜索从未生成”。PLAN已经限制四次调用和非符号等价穷尽的结论边界，该限制应保留。

## 排名、保存和预算核验

正常返回后先exclusive保存raw，再重算全部已保存提案的训练MSE/方差分数、完整排序与前12，并检查返回常数逐值相同；原计数逐输出与旧模型一致，独立排序使用原size/repr tie-break。排序近并列不一致会标ranking_unresolved而不通过，不应放宽这一预定规则。分数不相容或其他断言异常当前会中断，需按发现3补明确失败终态。

来源、诊断脚本、方案、PRE_RUN_REVIEW、BUDGET、测试收据/日志、两个模型和源FREEZE/ALL进入本轮FREEZE并原字节归档；结束后再次核源与副本。修复后新增的测试日志/审计摘要也应进入最终清单或由被冻结报告中的精确摘要约束，避免只冻结过期6项测试记录。

主入口在启动worker之前创建60秒BUDGET，重导入NumPy/冻结/复制/追踪/复算均计入；数值环境单线程，四调用串行，没有worker子进程。外监督等待剩余窗口后kill该worker；不重启、不加时。没有正式执行证据时guard才允许启动，既有BUDGET/FREEZE/EXECUTION/worker.log拒绝重启，write为exclusive。预算是否足够及追踪是否改变实际耗时属于运行后证据，当前未测试或保证。

## 实际检查、测试和剩余门槛

已读AGENTS、前轮pool audit的PLAN/REPORT/DIAGNOSIS/FREEZE、本轮五个稳定材料及其依赖源码。源链和TEST_VALIDATION内script/test/log三项摘要全部匹配；validation.log是6 passed/0.08s。六项只证明防重启、缺review拒绝及exclusive write，没有覆盖trace、原始图语义、常数配对、完整排名或失败收据；不能据此接受诊断核心。没有重复这些测试。

本次只新增本报告及 `PRE_RUN_CHECKS.json`。构造核证仅运行从源码提取的纯raw/evaluate/trace/收据语句片段，使用手工数据、假frame和内存流，没有启动main/worker、调用案例生成器/initializer或创建正式文件。初次AST选择错选较早的模型循环而停止，未写工件；更正为包含trace的循环后继续，过程保存在摘要。当前BUDGET/FREEZE/worker.log/VERIFICATION/EXECUTION均不存在。

修复上述三项并提供相应构造日志/收据后，可复用本审查者只复核变化；主任务再作最终冻结及执行决定。没有对四次正式输出、训练摘要、实际计数、排名匹配或60秒完成作任何通过声明。本诊断即便成功，也不能证明全搜索缺口、另一种子/噪声层归因、新构造拟合成功、科学采用或发布。

## 本检查点摘要

| 文件 | SHA256 |
|---|---|
| PLAN.md | 085e46321ad6a8c2c9702b2cef354b593ea16c0b70477f251d4f7bef2245aa4f |
| diagnose.py | b5c8c736094fdd0b772e06dacca919f21f539b1b0075a6e8648fea01708c3fb2 |
| test_observer.py | a3f38d793d4843c0d7e5a5c39f890c65b1a28906d2477f7f6162752df4af3710 |
| TEST_VALIDATION.json | cdd3ee2113b6aa4d7dddb89ceb36e9f93a5347c9209e3e4370d96d4efcb6f75d |
| validation.log | de1de1be84f54e69eec5a1aabbb19f4c997393b64553bd6e594d605b59d91d27 |
| PRE_RUN_CHECKS.json | 54d2980db517aeefc73f410a4358a8b1a93be3f3889fb8e6dfadd05d4b388080 |

## 修订后窄复核：当前执行前结论（2026-10-31）

本节为最新结论；上文保留初版问题和构造证据，不表示这些问题仍未解决。对脚本 `05ea98f12a593b3bcd1ede48e938938dfb0c58452370e90a63cb552dc470ea7e`、同步PLAN、测试及收据完成窄复核后，**三项原阻断均已消除，当前范围未发现新的实质执行阻断。从独立技术审查角度，允许主任务按原四调用、60秒方案进入最终冻结及执行阶段。** 这不替代主任务的最终执行决定，不增加资源，不预判四调用完成、排名核验通过或任何算法收益。

### 原发现的处置

1. **非有限观测值。** `diagnostic_value:31-38` 递归保持列表位置，以`{nonfinite:nan/+inf/-inf}`编码非有限浮点；`emit`和exclusive `write`均使用该编码。原真实I/O异常继续向外报告。trace在单特征直接边界尚无error时将finite_error记为null，不再把未计算误记成有限。测试从真实emit/trace AST构造正常大小、超节点、超深度并含Inf/NaN/-Inf常数的边界，检查标记、finite_error及拒绝计数，不调用initializer。原发现1关闭。
2. **严格raw与常数配对。** 模块级`raw:41-67`先计常数叶数、按树顺序消费并确认无剩余，逐节点累积非有限掩码，最终对受影响样本保留NaN。代码与冻结严格数值语义相容；构造覆盖曾失败的非法中间值、常数缺失/多余以及非对称常数顺序。原分数/常数/树仍先保存raw，strict-score不相容会明确失败并停止余下未启动调用，不重写原排名或丢提案。原发现2关闭。
3. **失败收据与固定分母。** 每调用先写STARTED，正常Python异常写FAILURE后停止；只有完整逐调用结果才绑定raw/trace摘要。`call_receipts:70-94`固定四行，区分passed、failed、ranking_unresolved、invalid_receipt、interrupted_or_failed和not_started；failure的initializer_returned须为bool、error须为字符串，partial raw不当完成。`save_execution:97-111`对缺失/坏JSON/错误类型的VERIFICATION保留错误并按未通过处理，仍exclusive写EXECUTION及4行状态。worker最终核源/归档，记录input_integrity_error、完成及verified数量。构造覆盖坏/部分收据、数值失败、partial raw、假Popen worker失败与timeout；不补跑、不覆盖原工件。原发现3关闭。

正常rank不一致仍产生ranking_unresolved，而不是改排序或放宽近并列规则；原计数比较、分数容差、返回常数逐值核验与完整前12要求未变。异常恢复仅用于留下诚实终态，未将错误转成成功。硬kill或不可写磁盘可能无法完成Python收据的边界已明确保留。

### 冻结与测试证据

最终输入清单已追加全部`validation*.log`和原PRE_RUN_CHECKS；旧6项记录和初版构造反例原字节保留，作为历史修复依据。窄审再次只读核40份冻结源码和cases，摘要无漂移，原ALL→FREEZE链匹配；固定case、seed、limit、compound/implicit、训练隔离和60秒外监督均未改变。未重做前轮完整设计、来源或数值审查。

TEST_VALIDATION的当前脚本、测试及三份日志摘要全部匹配。最终validation_03为20 passed/0.52s；validation_02的20 passed/0.58s及旧6 passed/0.08s均保留，它们是重叠测试记录，不能相加声称46个不同用例。本人未重复执行pytest或上述已充分覆盖的构造测试；本次依据稳定源码追踪、已完成测试源码/日志和独立摘要核验。主/审查代理实际模型元数据沿用主任务已核验记录，本次未重读平台session日志。

本窄审仅追加本节，PRE_RUN_CHECKS仍保留旧script哈希与旧反例，未伪装成当前代码的失败。当前没有BUDGET/FREEZE/worker.log/VERIFICATION/EXECUTION，正式initializer调用仍无执行工件。本次未调用main/worker、正式cases/training_data、structured_seeds、fit/search。实际追踪完整性、原计数/训练摘要、全部排序和源/归档末次检查仍须由这一固定执行和独立后审验证；不保证60秒一定够用，不将未观察到的图解释为整个搜索不可能生成。

| 修订后工件 | SHA256 |
|---|---|
| PLAN.md | 24067271368efc778929b81bf3dfe46753ea77297ade87ad556c3d77c6a05739 |
| diagnose.py | 05ea98f12a593b3bcd1ede48e938938dfb0c58452370e90a63cb552dc470ea7e |
| test_observer.py | fece1eb940c44ff8fbd2c8a115d68fb7e5d4b62a05c3c3eb33f829a8a65f37fc |
| TEST_VALIDATION.json | d7de52fd6aabe22da6f4d10281b0c267c96149bba4760e814c2b30244872ab15 |
| validation_03.log | c531df2b89c9d764aa3861fa01ec7c9accdb7cce3793010e2914287685b74b76 |
