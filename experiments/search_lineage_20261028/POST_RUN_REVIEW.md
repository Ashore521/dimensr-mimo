# 谱系诊断运行后独立审查

状态：本次限定范围独立审查完成。未发现阻断本轮训练侧诊断结论的剩余实质问题；不构成后续算法方案或默认切换的接受。

范围：只读本轮 REPORT.zh-CN.md、DIAGNOSIS.json、VERIFICATION.json、冻结/执行记录与必要 trace，核对10项完整分母、训练侧发现/保留/选择结论和下一步方案推理。只写本报告，不重跑训练、搜索、自检或新实验。

## 已独立核查

- 89项冻结文件、110项封存文件哈希全部吻合；FREEZE→ALL_MODELS_FROZEN→VERIFICATION→DIAGNOSIS的哈希引用一致，DIAGNOSIS脚本哈希吻合。10项任务/核验/诊断顺序一致，verified=expected=10，budget_met=true；内部246.68037128448486秒、外部254.2753393650055秒，正常退出且无截止终止记录。
- 逐项直接读取本轮与1.6.0旧model.json，10个search_fingerprint和完整best字典精确相同。
- 直接解析所有trace，状态总数3297；按原训练门槛和原始图结构重新筛选，紧凑达标状态数分别为 damped0 0/3、damped1 5/26、rational0 0/0、rational_variant 19/30、polynomial_variant 14/0（random/matched）。全部以相同图和参数进入final pool。
- rational_n0/702 全部有效状态的最小最大逐输出训练误差分别0.010707459487106602与0.012465996275254186，高于1e-4；五个matched运行的描述性训练父代保护计数独立重算为11/29、10/24、7/16、4/16、2/3。
- selected origin不只图相同，子状态常数也与最终best精确相同：damped0 matched proposal1524，NSGA-II step2，matched_share 24→20；damped1 matched proposal648，GP step3，bind 24→23；rational_variant matched proposal782，GP step4，share 25→24。误差与报告一致。
- 本审查另以独立JSON解释器实际复算3297个有效状态的训练误差、全部最终候选池验证误差，均通过原核验容差；图/状态摘要ID、训练/验证数组与冻结/result三方哈希均通过。该确定性复算未拟合任何参数、未请求test split。

## 解释边界与待核对措辞

rational0证据排除仅靠最终候选排序修复本轨迹的绝对精度缺口；尚不能单独区分结构提案、常数拟合、数值域拒绝或预算覆盖的原因。初稿“后续需要通用复合有理结构发现”曾过于确定；主任务已通报将该段修订为“通用复合有理提案是待检验方向，尚不能区分提案/拟合/域/预算贡献”。这一修订与已核查证据边界一致，本项不再构成未决意见；本审查最后阶段按主任务要求未再重复读取全文。

保留random路径与基础候选、在剩余原预算内研究匹配闭包，是与本轮证据相容的下一方案假设；本轮未证明其可保测试保护或普遍改进，报告也明确尚未实现、验证或推荐。后续仍须冻结提案上限/递归、独立资源口径及验证参照，不能使用既有干净测试分数改选模型。

## 测试隔离、未检范围与最终结论

- 本轮 worker/verify/diagnose 已在运行前和本次审读中核对：训练/验证数据经拒绝test的training_data接口取得；全部有效状态仅复算训练误差，验证只核对冻结最终池；diagnose只读取训练侧谱系、验证描述与旧model的指纹/best，没有读取旧SCORES或干净测试分数，也未选择新模型。保存记录均声明test_data_accessed=false / model_reselection=false，哈希链未漂移。本次独立复算也未读取测试标签。此为源码路径与保存证据支持的隔离结论，不是操作系统级访问审计。
- 父子训练代价不等于干净测试代价；damped1最终bind事件只支持该有效子状态的直接来源及训练误差变化，不能证明bind独立造成与random的干净测试差异。相同fingerprint/best支持这十条谱系与旧对应结果相接，不能推广到其他任务、时间预算或观察器开销。
- 本次未重跑训练、搜索、基准、自检或新实验；未重复核验已发布版本的全套5276候选、构建、远端CI或安装；未独立重新执行连续域/全部interval盒子审计。未统计失败/非法提案是否包含目标有理结构，不能把没有有效达标状态提升为语法表达能力不足的证明。
- 精确证据入口：VERIFICATION.json 的 expected/verified/rows/elapsed_seconds/budget_met；DIAGNOSIS.json 的 rows[*].search_fingerprint_matches_160、selected_matches_160、compact_train_accurate_states、compact_train_accurate_states_in_pool、minimum_training_max_output_error、selected_origins；runs/damped_n0_s701_matched/trace.jsonl 的 variation serial1524、runs/damped_n1_s702_matched/trace.jsonl 的 serial648、runs/rational_variant_s711_matched/trace.jsonl 的 serial782；各状态内容及紧随的phase事件；FREEZE.json、ALL_MODELS_FROZEN.json和EXECUTION.json的来源/预算链。

Verdict：本次限定范围内，10系统训练侧诊断、候选去向与具体父子来源得到支持；没有证据将最终池丢弃已有紧凑达标状态作为首要修复点。保留random路径的匹配闭包及通用有理提案均仅为下一轮待冻结、待对照验证的假设。本审查不新增实验授权，最终方案及接受决定由主任务作出。
