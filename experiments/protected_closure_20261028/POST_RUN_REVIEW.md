# Protected closure 正式开发对照运行后独立审查

结论：本次限定范围独立核验支持正式34系统的技术通过、34/34配对干净输出保护、17/17基础路径保存证据一致及接受9→10的记录；未发现阻断这些已报告结论的剩余实质问题。原紧凑收益门槛未达：damped_n0/701共享图27→23，虽各分割误差均在机器精度量级，却不满足原图≤20。因此 `compact_damped_gain_preserved=false`、`adoption_pass=false` 正确，不能推荐默认切换或宣告完整目标完成。最终接受判断由主任务作出。

仅写本报告；未修改冻结文件、历史报告、代码、阈值或预算。未执行训练、搜索、pytest、新实验或派生代理。数值复算仅评价已封存图及常数，不优化参数、不重选模型。

## 来源和完整性

沿用已完成PRE_RUN_REVIEW/HARNESS_REVIEW，并重新读冻结PROTOCOL、FREEZE、BUDGET、TRAINING_STARTED、ALL_MODELS_FROZEN、SCORES、VERIFICATION、SUMMARY、EXECUTION、TEST_VALIDATION及必要运行单元。源码/data生成/评分路径在运行前已审查；本次又以FREEZE哈希核对当前字节，没有把新REPORT当作冻结证据。

直接逐文件检查：FREEZE.files的127项、ALL_MODELS_FROZEN.files的374项全部SHA256吻合，各34个CELL_FROZEN内部清单亦全部吻合。FREEZE→ALL_MODELS_FROZEN→SCORES→VERIFICATION引用一致；BUDGET实际内容与FREEZE内budget一致；冻结→训练开始→全模型封存→评分→核验时间顺序成立。34个唯一jobs与scores/verification逐项同序，无缺失/重复/缩分母，资源/result/score均complete，进程返回0，34/34 verified。

EXECUTION.returncode=0、termination_reason=null，外部743.7035026550293秒；VERIFICATION.elapsed_seconds=737.2388153076172、budget_met=true，均低于原2880秒。逐单元资源最大28.65599999999904秒，模型diagnostics.elapsed_seconds最大20.06300000000192秒，低于60/52秒。闭包history的phase_seconds合计10.642000000007101秒；这是所记录闭包阶段总墙钟成本，不能等同于每个配对总运行时间之差或精确纯拟合成本。

## 机制、基础池与父子证据

- 实际读每个events.jsonl重算，34/34均policy参数变化>0、neural有效候选>0、GP新评价>0、NSGA-II至少2代且有新评价、Rademacher调用>0。闭包四项附加机制按冻结的批次口径全部非零，不将其写成每个closure单元都产生新图。
- 17/17对closure基础候选前缀与random完整candidates字典精确相同；base_search_fingerprint与random search_fingerprint相同；base_evaluations及operations/changed/novel/successful计数也逐对相同。总系统评价random=6142、closure=6567，额外425，符合原共享上限但实际消费并不相等。
- 逐事件直接检查new_evaluation计账与总账：425新评价、424有效、106训练保护通过、106验证后留存；1次新评价未产生有效Candidate，仍计入425分母。全部单元≤48新评价、≤2轮、总评价≤1200，完整训练pool摘要可由基础payload+通过训练保护子状态重建。
- 独立补查原运行核验未逐项强制的控制流：425事件的父代均在当轮按训练平均误差/复杂度/图摘要排序的前16中，且来自基础或先前轮留存状态；每轮父代数≤16。按已审查matched_proposals确定性重建，每个事件图及warm均有精确对应；同轮实际评价事件的训练warm距离/图大小/父误差/摘要排序不逆序，新增评价图无重复。该检查未重新拟合，也未重演整个搜索。
- damped_n0/701所选closure状态直接对应round0事件，父图27→子图23。其final candidates与已保存closure有效child中，训练各输出≤1e-12者的最小共享图也是23；这只排除在这些已保存状态中靠改最终排序得到≤20，不能推广为语法无表达能力、全部基础cache没有更小图，或定位到某个单一生成/拟合原因。

## 独立确定性数值复算与门槛

自写JSON逐节点NumPy解释器，按原始算子与中间非有限传播评价，未调用graph.evaluate实现来计算这些误差。实际复算：

- 5352个最终候选、10704组train/validation误差；425个父状态和424个有效子状态的训练误差，均通过原rtol=1e-9/atol=1e-11容差。
- 34个所选模型的observed/clean测试误差，均通过原rtol=1e-12/atol=1e-14容差；34个真实load重算选择后，predict(test)与自写原图解释器逐位相同。
- 训练/验证数组哈希逐项与FREEZE.data_sha256相同，归一化等于训练观测标准差至浮点归约精度。首次要求二维axis-std与逐列target_scale逐位相等时遇到5.55e-17差别；改用1e-15相对容差确认等价，并用封存的逐列scales复算模型指标。未修改数据或源码。
- 另用已冻结JSON解释器严格复算所选图所有train/validation/observed-test/clean-test原门槛，接受数确为random9/17、closure10/17，仍有7/17 closure未完整接受。唯一新增接受来自damped_n0/701，不能写成全部系统恢复。
- 17对、每对2输出的干净保护按原 `closure <= random*1.1+1e-8` 独立重算为34/34。宏平均干净NRMSE独立复算random `0.002617556289020336`、closure约 `0.002614428507410400`（SUMMARY末位舍入为03995），与汇总一致；未用幸存子集均值掩盖失败。

紧凑收益门槛逐项检查：damped_n0/701 random图27、closure图23；closure严格复算train/validation/observed-test/clean-test的最大误差为 `3.845980099106278e-16`，满足1e-12精度条件，但23>20。原协议要求两者同时满足，不能以图24的一般接受上限替代图20的专门收益门槛。技术/保护/均值/接受数条件通过，compact条件单独失败，因此adoption=false得到支持。

## 数据角色、核验盲点与解释边界

- FREEZE和全模型封存的test_labels_accessed=false、各result.test_scores_computed=false均成立；哈希未漂移的worker只取train/validation，score在全模型封存guards之后才生成test/clean标签。数据角色隔离得到源码、封存链和时间记录支持，未进行OS级访问审计。本次读取测试标签仅用于封存结果复核，没有依测试结果调参、延长搜索或改选模型。
- base_search_fingerprint散列的是全部有效cache状态，不含失败图的完整身份序列；基础prefix逐项相同、指纹/评价/算子计数一致支持本批的基础保留结论，但不能单凭这些字段证明每个失败提案序列完全相同。未另有全cache失败谱系，不能补造该证明。
- 原闭包核验会核父成员及误差、计数和pool；本次进一步核前16父、前轮关系、matched图/warm和已执行提案排序。仍未复演每个被缓存或结构/warm筛掉的未评价提案，也未独立重做所有局部Rademacher数值或连续域interval盒子。现有enclosed只代表原有限浮点筛查，不是形式证明。
- 比较固定了同一1200评价/52秒上限，但closure实际多用425次评价及上述阶段时间；本批不能单独证明其按相同实际评价数优于其他追加提案方式。精度均值改善小且这是已检查任务的开发回归，不能提升为新任务/盲验证或普遍测试保护保证。
- 若继续诊断，当前仅支持训练侧待检验假设：本轮保留池/闭包已存状态没有训练精度达1e-12且图≤20的候选，单改最终排序不能补足该保存池缺口；尚不能区分结构匹配覆盖、父排序/两轮/48边界、联合拟合或域拒绝贡献。不应根据本次干净测试分数调整这些规则；后续方案须另行冻结并接受独立审查，不能补跑本批或放宽原≤20门槛。

最终边界：本次支持技术机制与已列开发结果，原adoption失败应原样保留，production_default_changed=false。未接受默认切换、1.7.0发布或完整研究目标完成；这些决定不属于本次运行后核验。
