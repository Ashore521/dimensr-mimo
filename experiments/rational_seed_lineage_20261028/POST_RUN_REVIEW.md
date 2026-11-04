# 初始化谱系诊断：运行后独立审查

结论：已落盘证据支持“所查目标形状提案已生成，但被初始化前12排序截断”。该二次分母没有在准入、分母 eligibility 或前64分母截断处丢失。审查未发现本次观测结果与冻结脚本口径不一致的实质问题。此结论限定于本历史案例、两个输出和冻结初始化阶段，不解释完整搜索失败，也不证明保留该提案后非线性拟合会成功。

## 实际完成的检查

仅读取已有 JSON/JSONL 与执行记录，并做摘要、字段、树结构、排名和算术一致性检查；未运行 initializer、fit、search、数据生成或测试。未读取 SCORES 或 clean/test 标签。只写本报告；冻结文件及 PRE_RUN_REVIEW 保持不变。代理实际模型和推理等级未验证。

- `FREEZE.json.files` 的45个文件当前 SHA256 全部匹配；`VERIFICATION.json.freeze_sha256` 与 FREEZE 实际摘要相符，`VERIFICATION.json.outputs` 的6个观测文件摘要全部匹配。脚本摘要仍为 `348e72a5887055123148901cb39dbfc1769d8ad8762814191d45ede45375529a`，原预审报告摘要也保持匹配。
- `EXECUTION.json` 记录 returncode=0、termination_reason=null、wall_seconds=8.478016138076782、verification_passed=true；`VERIFICATION.json` 记录 completed_calls=2、status=passed。`BUDGET.json.start_unix=1793193705.1717007`；正式 `balanced_closure_20261028` 的预算起点1793192957.27914加 EXECUTION 墙钟704.658810377121，为1793193661.9379504；本次启动晚约43.234秒。该已落盘时间记录支持没有与该正式批重叠。
- 两输出 raw 和验证输出的 counts 均逐字段等于历史 `search_lineage_20261028/runs/rational_n0_s702_random/model.json.diagnostics.initialization_diagnostics[j]`。提案数149/63，保留特征156/160，线性子问题4837/836，其余计数字段也全部匹配。这里的线性最小二乘是被观测初始化的一部分。
- `output_0_trace.jsonl` 的298行全部可解析：272条 admit_return、26条 denominator_call；输出1的306行全部可解析：279条 admit_return、27条 denominator_call。成功 admit 的树/常量按顺序与 raw.features 的156/160项完全相等；分母事件序列与 raw.denominator_calls、raw.eligible_denominators 完全相等。
- 两输出 raw.proposals 与 output_j.json.proposals 按行核对树、常量、原 score、rank 和 retained 字段，149/63项全部一致。raw.returned_seeds 精确等于各自原排名前12的树及常量。
- 另用标准库 JSON 树重建节点尺寸及原 Node 表示顺序，分别按已保存原 score 和已保存 recomputed_score 的 `(score,size,repr)` 重排。两输出完整序列均为原 rank 顺序，前12与实际返回 seeds 一致；也等于 `independent_order_as_original_indices`。输出0存在1组原数值分数并列，完整三元排序仍匹配。
- 保存的 score 与 recomputed_score 最大绝对差分别为 `6.821210263296962e-13`、`1.3877787807814457e-17`；逐项符合冻结脚本的相对/绝对容差。用已保存 training_nrmse² + 1e-8×树尺寸检查 recomputed_score，最大绝对差分别为 `2.2737367544323206e-13`、`9.094947017729282e-13`。本次未重新生成训练数组，因此这些是对已保存两套分数的审计，没有再次从 X/Y 独立算预测误差。

## 阶段归因

以下索引/rank 均从0开始，均由 `output_j_raw.json` 与 `output_j.json` 对应字段直接定位。

| 项目 | 输出0 | 输出1 |
|---|---:|---:|
| 分母 `add(c0,square(mul(c1,x0)))` 的 features 索引 | 24 | 25 |
| 同分母的 eligible_denominators 索引 | 19 | 20 |
| 同分母的 denominator_calls 索引 | 19 | 20 |
| 全部 eligible / 实际分母调用数 | 26 / 26 | 27 / 27 |
| 目标单项分子提案 rank | 21（第22位） | 12（第13位） |
| 原 score | 0.19722357611449767 | 0.09331046403427641 |
| 初始训练 NRMSE | 0.44409849821238717 | 0.30546743530903 |
| retained | false | false |

两输出所查分母常量均为 `[1.0,1.0]`，可在成功 admit 事件和实际分母调用中交叉定位；两个 eligible 列表均不足64且全被调用，故前64规则未排除这些分母。

输出0目标形状为 `div(mul(c0,x1),add(c0,square(mul(c1,x0))))`，提案常量为 `[1.2471973266826875,1.0,1.0]`；输出1为 `div(mul(c0,square(x1)),add(c0,square(mul(c1,x0))))`，常量为 `[1.2521074518220408,1.0,1.0]`。同样树形中重复出现的常量 token 仍按叶节点分别消费参数，不能把它们解释成强制共享的单个参数。

输出0还有4个其他分子、同一二次分母的提案，rank 为47、54、57、84，亦未进入前12；输出1该精确分母树形只有上述 rank12 提案。归因因此是这些已生成、已评分提案在原初值训练误差排序中落选，而不是直接二次加法分母从未生成。

## 证据边界与待决事项

- FREEZE 记录训练形状 X/Y 均96×2、float64，训练摘要为 `304660ac2e7556e144e4e15a73f11efd914a5fb9ea0c0110d7d4e355c8234864`；本次未重新生成数组复验该摘要。数据角色及调用路径沿用已冻结预审；执行结果声明 nonlinear_fits=0、validation_labels=false、test_labels=false，与冻结代码一致，但不是额外运行时全函数监控证据。
- 初始分母参数为1而非目标训练族的真实参数；观察到初值误差较大，不足以断言任何非线性优化、更多预算、替代排序或联合MIMO构造必然达到原1e-4门槛。本次没有试验该反事实。
- 可以据此形成“如何保留有潜力但初值误差较大的有理式结构”的后续通用算法假设；不据此修改本批门槛/预算，不回填历史结果。任何算法修改与新实验仍需另冻目录及独立审查，最终接受由主任务决定。
