# Protected closure 实现前独立审查

结论：本次限定范围的独立审查完成。初查发现的验证非法候选/冻结池摘要缺口及有限预测引起的误差溢出均已修正并作无拟合窄范围复核；当前未发现剩余算法/数据角色阻断。本结论不等于新34系统实验通过、默认切换接受或完整科学目标完成；最终实验与接受决定由主任务作出。

## 版本、范围与依据

基线 HEAD 实读为 `652b05884269c7563b101e0a14a9b15f749bbc6f`。已读项目 `AGENTS.md`、`.codex/agents/ResearchReviewer.toml`、本目录 `PLAN.md`、`../search_lineage_20261028/REPORT.zh-CN.md` 与 `POST_RUN_REVIEW.md`。审读当前 sharing/config/regressor/acceptance 差异及相关 selection/Evaluator/fitting/genetics/graph/audit/complexity、`tests/test_protected_closure.py`。未重跑10系统训练侧诊断；其结果仅作为方案来源。未核验远端发布CI或实际模型会话元数据。

审查末次读取的 SHA256：

- `src/dimensr_mimo/sharing.py`: `4a7fa3cfe391a66f7da0ff3506d82a044ec351dff9a18acf68d0a35e76b13acc`
- `src/dimensr_mimo/config.py`: `51912d5033a7fbd5b440b270d5e53f0773e1dfe17b05d7b0a8beeae437223509`
- `src/dimensr_mimo/regressor.py`: `8c13af3b113f71ee1e16d31b060314d3166d61152c09bdcdbca47b09ab34d37a`
- `src/dimensr_mimo/acceptance.py`: `60b87dbdf5bb92cb4db91e9a3716bd095cee519fe9eb22cfb07f6a235ab40113`
- `tests/test_protected_closure.py`: `f9c126e9fc361fb35c63aec731bd8d1ba9f58582222613e69afcf300f7085a45`

## 曾发现、现已解决的实质问题

初始 `regressor.py` 对完整训练 pool 冻结摘要，但仅将验证预测 admissible 的候选写入最终 candidates；`sharing.py` 基础数量/摘要也来自验证前完整池。若新增候选验证失效，selector 会报 `Candidate pool differs from its training freeze.`，并可能破坏基础前缀映射。这条 freeze/filter 路径原已存在，闭包新增候选会扩大触发面。另发现仅检查有限预测不足以保证 NRMSE 有限：有限值1e200平方即可溢出。

当前 `regressor.py:284-306` 在 closure 分支同时核对 raw 预测、逐输出误差及 violation/score 的有限性；基础候选异常明确失败，新增候选拒绝并保存图摘要与原因。保留 `full_training_pool_sha256`、基础前缀摘要及验证后新增保留数，为现有 selector 单独保存有效子集摘要。验证拒绝发生在所有搜索结束之后，没有据此补评价、续搜或重拟合。基础候选异常失败保留了原 precision 路径的可用性边界；本修复不保证任意验证输入下基础池都可用。

实际抽取当前源码的“冻结→验证评分”代码块，以合成 Candidate 执行：log 在负验证输入上被拒绝，基础前缀及摘要保持一致；square 在验证输入1e100产生有限预测1e200但误差溢出，也被拒绝，仍只保留基础候选。完整训练摘要与最终有效子集摘要不同并分别保留。该核对不调用 fit、优化器或训练。

## 核心机制核对

- **基础路径与兼容范围。** config 只允许 closure 配合 precision/use_sharing；默认仍 random。genetics 仅 matched 进入额外随机分支，closure 在原神经、GP、NSGA-II、polish及基础池构造结束后调用。闭包不消费基础搜索 RNG、不 refine 缓存图。`sharing.py:70-71` 在进入闭包时记录全部有效 cache 的 `base_search_fingerprint` 与 `base_evaluations`，而非仅基础候选子集。静态分支一致不能单独保证有墙钟运行每次完全同轨迹，需以正式回归运行记录核对。
- **训练/验证角色。** 父排序、warm 距离、拟合和父代保护只使用训练 Evaluator 的 X/Y/scales；不接收验证或测试标签。`regressor.py:278` 之后才统一评分验证；基础precision参照也在此后由同一冻结基础池产生。未见测试标签用于参照、选择或延长搜索。
- **资源与留存。** 至多2轮、每轮16父、沿用原每父最多16提案；按图去重并跳过所有缓存（包括失败缓存）；至多48次新评价，同时受原 calls/max_evaluations/deadline 约束。基础池完整保留为前缀，训练父代保护拒绝的已评价子状态保留在 events。`protected` 是训练通过数，`validation_retained_extensions` 才是验证后新增留存数，不能混称。
- **维度、别名与数值。** 沿用原 matched_proposals 的有序语法/单位及局部参数别名模式；Builder 拓扑重编号，不按常数值近似相等合并，不删除近零项。每个新图仍经正常联合拟合、raw训练/probe/interval合法性及局部Rademacher评价。warm非有限预测变化在拟合前拒绝；这可能减少可修复初始值的覆盖，但与当前冻结的训练侧预筛设计相容，没有据此声称发现完备性。
- **保护与fallback。** `acceptance.py:26-46` 对基础前缀复现原precision选择；原绝对精度、结构、initial-reference训练/验证约束与guard验证边界保留，另加相对基础选择逐输出训练/验证 `1.1*x+1e-8`。无全门槛候选时返回基础选择且继续 not_met，不伪报accepted；未宣称测试保护。
- **重载。** load 重建原weighted anchor并重算precision，比较 best/post_selection/diagnostics；基础前缀摘要、基础选择摘要及逐输出边界重新生成，不只相信存储accepted标记。完整父子训练误差是否真实仍需封存后的独立数值复算；JSON完整性校验本身不是训练误差真实性证明。

## 实际执行的确定性检查

- 直接调用选择构造：训练输出保护、验证输出保护、fallback/非法配置，3项通过。没有调用测试文件中的 fit/Evaluator拟合测试。
- 合成 Candidate + 无拟合 FakeEvaluator，走真实 matched_proposals/protected_closure：两轮、4次新评价、4次保护通过、3次缓存跳过，图大小 `[12,9,9,9,6]`；基础状态精确不变，第二轮实际产生递归共享。
- mock提案源与Evaluator，仅验证调度控制流：48次闭包上限、原总预算仅剩12次、中途deadline在3次停止，实际新增评价分别 `[48,12,3]`。不把mock时间控制当作真实拟合墙钟上限证明。
- 构造完整原生JSON模型，以内存read_text替身调用真实 load：重算选择后序列化往返一致；基础训练边界、验证边界、基础选择hash、基础index、status、基础池hash、closure protected计数共7类单点篡改均拒绝。未向模型文件写入任何构造数据。初始构造用整数单位元数据导致往返规范化不同，改为真实encode输出的字符串单位后通过；这是构造修正，不是源码缺陷。
- 读到新增测试覆盖实际fit的random/closure基础池及完整cache指纹比较、log非法/square溢出、48及剩余总预算边界；本审查未执行这些fit测试。主任务通报相关测试通过，该消息不替代本审查独立运行证据。

## 剩余限制与待决

未执行搜索、训练、基准或pytest；未启动新34系统实验，未变更门槛/预算。未独立执行旧random/matched模型全套重载、真实有墙钟基础轨迹对比、全部域盒子审计或新34系统输出保护/接受率核验。新可选方法的科学成效、历史damped图20收益是否保留、实际48次闭包额外成本仍未知；方案已将新批标为开发回归，不能提升为盲验证或解决有理式发现缺口。

收尾 `git diff --check` 尚报 acceptance.py 新增行的CRLF/trailing-whitespace（34新增行），属于非科学阻断的格式检查失败，交主任务整理；本审查未改源码。除该格式项及尚待主任务正式核验的运行证据外，本次限定范围未发现剩余实质阻断。最终是否冻结并启动已约定对照由主任务决定。
