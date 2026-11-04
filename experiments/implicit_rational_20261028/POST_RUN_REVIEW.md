# 隐式有理系数初值：运行后独立窄审

结论：现有冻结证据支持34/34技术与实际机制通过、完整接受从10/17增加至15/17，以及指定rational/damped门槛通过；逐输出保护仅30/34，因此本轮采用失败。平均误差下降、图缩小和接受数增加均不能抵销四个预定保护失败。默认仍为legacy，不能将本轮报告成新任务验证或整体目标完成。最终科学接受由主任务决定。

## 实际检查与版本链

仅写本文件，未修改任何冻结输入、代码、历史结果或既有报告。用标准库读取JSON/JSONL，核摘要、配置、时间、计数及已保存数值并重新汇总；未运行initializer、fit、search、pytest、数据生成、原评分/核验脚本或派生代理。没有按测试误差挑图、改门槛或重跑。

- FREEZE.files的134个当前文件SHA256全部匹配，ALL_MODELS_FROZEN.files的374个运行文件全部匹配；34份CELL_FROZEN的逐文件摘要也全部匹配。当前运行代码/方案/预审/测试/日志和冻结时一致。
- FREEZE→全模型封存→SCORES→VERIFICATION的交叉摘要全部匹配。冻结、TRAINING_STARTED、全模型封存、评分、核验时间单调一致；全模型封存时间1793196470.3001466，评分记录时间1793196497.5820649。结合已审评分源码，这支持评分发生在完整封存后；不是额外运行时文件访问监控。
- 34个冻结job唯一且与34个score及34个verification按顺序逐一相同；两臂各17个，全部status=complete、verified=true，没有遗漏或缩小分母。每单元model.config=result.effective_config，且逐字段符合冻结配置、该case接受条件、seed和method；每一配对两个有效配置的唯一差异为initialization。
- 全部实际配置为closure/precision、1200评价上限、52秒fit；单元resources均returncode=0、complete且<=60.5秒，最大30.735000000000582秒。EXECUTION为returncode=0、termination_reason=null、外部780.0645077228546秒；VERIFICATION.elapsed_seconds为775.5707652568817秒，均小于原2880秒。未提高预算或补跑。
- 训练/验证行数分别96/48；全部result.data_sha256等于FREEZE该case/seed摘要，result.test_scores_computed=false。没有重新生成数组复验数据摘要。

关键证据SHA256：

| 文件 | SHA256 |
|---|---|
| FREEZE.json | `4531ce7b0447af340badb31ca9053108e3e3d2715c3ed013e755caea4c4a8691` |
| ALL_MODELS_FROZEN.json | `7dc662c799940afaa8d9b96e126e96508993dda76ee6a1421ada0e67294e3bf2` |
| SCORES.json | `cf69e4aeb459b4986de3ae1dd4eac8d108c30e125a69b3d7bf8dcdd283f80213` |
| VERIFICATION.json | `f52aa367bf44569ef4f196108e3c0c8265d17bb43974b226ccfa9cc993aed1f0` |
| SUMMARY.json | `e8eb32e3db10ad04333eee94df5cc386c8bfc86af885a3b95e14d5a64708f561` |
| EXECUTION.json | `902ca2ff1e2b72bef8ae2b9cea9921d397ca073450ddcbc41e02fee303af0a7f` |

## 实际机制及数值检查

直接从各模型initialization history与诊断累加并对照运行events和verification：隐式jobs=145、完整系统新评价=145、有效=136、线性子问题=61974、返回隐式根=194，与SUMMARY完全一致。逐单元history各字段与实际initialization事件相同，且0<=valid<=new<=jobs、new不超过该阶段新评价。结合预审确认的完整Evaluator调用路径，这证明新初值确实进入完整MIMO评价，配置开关或候选生成计数不是唯一证据。136表示有效数值系统，不表示136个模型达到最终科学接受条件；194不是唯一拓扑数。

全部34个运行events都记录4个不同NSGA-II step各有new_evaluations>0，超过本轮修复后的至少2代要求。逐单元策略参数变化>0、有效神经候选>0、GP新评价>0、diagnostics.rademacher_systems>0；总评价不超过1200。

闭包compound/implicit分别394/431新评价，valid亦为394/431，父保护与验证留存分别111/284。逐事件总账、每轮<=24、合计<=48及base_evaluations+new_evaluations=总评价均匹配；另用825个事件已保存父/子训练误差重新计算10%+1e-8父保护布尔，全部一致。34个所选模型的训练/验证误差均满足各自保存的基础保护bounds。该基础保护不承诺相对另一初始化方法的测试保护。

所选模型的train/validation/observed-test/clean-test误差均为两个有限非负值，score与verification的clean误差及接受状态相同；delivery训练误差与model.best.errors相符。34个保存的域审计总状态及两个输出均为enclosed。这里核对的是已保存检查结果；本次没有从原始X/Y重算所有预测、全部候选误差、独立重载或重跑单位/区间审计。原VERIFICATION及其冻结脚本完成这些检查的证据仍适用，enclosed仅为有限浮点数值筛查。

## 关键门槛与四个保护失败

从SCORES逐行重新计算的汇总与SUMMARY一致：

| 指标 | compound | implicit |
|---|---:|---:|
| 完整接受 | 10/17 | 15/17 |
| 每系统干净NRMSE再宏平均 | 0.0026144285074103956 | 0.001798470961673235 |
| 平均共享图规模 | 21.88235294117647 | 18.235294117647058 |

逐个配对重算确认所有compound已接受单元仍被implicit接受。damped_n0/701隐式模型共享图18、4参数，train/validation/clean全部<=3.2449972428016087e-16，满足图<=20及各误差<=1e-12。rational_n0/702共享图12、2参数，保存的两输出train/validation/clean均为0，满足完整接受及1e-4门槛；这表示冻结浮点样本上的零误差，不是解析恒等式证明。其原始交付参数单位含无量纲及时间倒数，适用输入bounds为[[0.1,1.6],[0.5,1.5]]，不外推到未审域。

逐输出按 `implicit <= compound*1.1+1e-8` 重算仅30/34通过，以下索引从0开始，全部保留：

| case/seed/output | compound clean NRMSE | 保护上界 | implicit clean NRMSE |
|---|---:|---:|---:|
| polynomial_n1/702/1 | 0.0033008511008706942 | 0.0036309462109577638 | 0.004168366734989283 |
| damped_variant/711/1 | 0.0011550392786305306 | 0.0012705532064935837 | 0.007051445900084936 |
| rational_variant/711/0 | 0.003581147993589743 | 0.003939272792948718 | 0.006294935533310211 |
| polynomial_variant/711/1 | 0.0003334414905537481 | 0.00036679563960912294 | 0.0015790065698637666 |

因此technical_pass=true、adoption_pass=false、production_default_changed=false与原固定门槛一致。隐式仍有damped_n0/702和damped_variant/711两个未完整接受单元，不能用本轮平均改善宣称全部任务已解决。

## 结论边界和待决事项

本窄审未发现上述证据链、全分母、配置、实际机制、关键门槛与汇总的实质不一致；这不是独立重跑全部原始计算后的全面验收。实际模型/推理等级仍依据主任务记录的外部平台元数据，本次未再查平台日志。

四个退化只能先作为已证实的输出级事实。当前没有独立证据将其唯一归因于隐式拟合、初始化排序、后续随机轨迹、闭包或最终选择；任何此类解释都只是待检验假设。下一设计应只使用授权的训练侧证据，不据这里已经看到的测试误差挑选本轮模型。继续保留所有失败和原门槛，生产默认不变；未见任务/种子验证仍未完成。
