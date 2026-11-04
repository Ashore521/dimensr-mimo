# 训练精度补充：独立执行前审查

结论：通用训练触发与保留原compound搜索的设计有证据依据，可进入实现与定向检查；本阶段尚不能批准冻结执行。两个必须落实的实现条件是原基础precision参照不能被补充候选改写，以及新增隐式生成须响应剩余墙钟预算。正式科学接受与默认变更仍由主任务决定。

## 审查范围与实际检查

仅写本文件。读取PLAN、TRAINING_GATE_AUDIT、前轮POST_RUN_REVIEW、现有regressor/sharing/structure/evaluation/config/acceptance/selection代码。未调用initializer、fit、search、pytest、数据生成或评分脚本，没有新增实验或查看本轮新评分。代码处于主任务正在修改的工作区；以下为实现前设计审查，实际diff完成后须另做窄复核。

- 用标准库只读重核前轮17个compound模型：每个源model字节SHA256、原closure基础候选数、按selection.FIELDS提取的基础前缀SHA256、配置nrmse、全输出训练达标候选数和minimum_max_training_nrmse，全部与TRAINING_GATE_AUDIT一致。没有读取候选validation_errors进行触发判断，也未读取测试数据或SCORES。
- 17池中确有15池至少一个候选各输出训练误差达标；仅damped_n0_s702_compound和rational_n0_s702_compound为零。它支持通用触发的覆盖事实，不支持按案例名硬编码，也不保证新候选成功。
- 现有regressor.py:259-287在NSGA-II与polish后才运行closure，288行起冻结池、随后评分验证；新阶段插入polish与closure之间可保持训练/验证边界。现有Evaluator只接收训练X/Y，完整调用包含联合常数拟合、域检查和Rademacher；新增系统必须复用此路径，不能以单输出线性拟合替代完整评价。
- 现有acceptance.py:29-48从closure.base_candidate_count/hash重建原基础precision选择，并对扩展候选施加原基础训练/验证保护。因此基础前缀完整性是选择语义的一部分，不只是报告字段。

## 必须落实后复核

1. **原基础池身份必须早于repair冻结。** 当前sharing.py:81-86把传入pool整体记作base。如果直接把base+repair传入现有protected_closure，repair候选会进入基础precision参照，违背PLAN“最终基础参照仍只来自原基础池”。需要显式传递原base的count/hash/cache指纹，并保证base前缀与候选对象/常数/误差不变；完整训练池另记为base+repair保留+closure保留。acceptance递归生成基础模型时还需把新flag复位为false，避免compound/closure/precision配置约束使内部random基础选择失效。

2. **隐式候选生成当前不可按墙钟中断。** structure.py:55的structured_seeds尚无deadline，内部特征、beam及最多64隐式分母循环进行大量线性子问题。只在调用前后检查Evaluator.expired不能保证新增阶段在剩余52秒预算到期即停。新调用应有可选deadline/过期检查并保留旧调用默认行为，不能挤出预算后仍跑完生成；外部60秒监督也必须保留。单次底层线性代数不可抢占的边界应如实记录，不能称为严格任意时刻中断。

3. **计数和缓存以实际评价为准。** 每输出候选上限与完整系统上限须区分：每输出最多min(12,population_size//q)，系统最多12次缓存外新评价。跳过所有已缓存图，包括缓存失败None；不允许refine。至少一个根须来自structured_seeds实际获胜implicit_retained_indices，不得把非隐式候选或未胜出的拓扑算作隐式执行。新评价增加量必须从同一Evaluator.calls取，满足base_evaluations+repair_new+closure_new=最终总量，且共同受原max_evaluations/deadline约束。单位、参数顺序、缺根复制需通过Builder/clone_subgraph保留。

4. **跳过路径与旧模型兼容需要具体证据。** 默认false和未触发路径应不消费搜索随机数、不写缓存、不更改原closure/选择；配置新增默认值后旧JSON须先按原config验证来源hash，再迁移新默认值，不能先补flag导致旧post_selection.source_model_sha256不匹配。需覆盖旧precision/closure模型加载与重存、新flag约束和bool严格类型、基础前缀篡改拒绝、触发/未触发以及失败/到期/预算耗尽路径。实际测试由主任务执行，本审查没有执行测试。

## 证据边界与待决事项

PLAN的52秒/1200评价、原closure每轮24合计48、17对全分母、先全模型封存后统一评分、失败不补跑及原保护门槛均未因本审查放宽。父训练10%+1e-8保护仅保护补充相对固定训练父，不承诺未来测试保护；最终仍需预定跨臂逐输出保护。训练触发是已观察任务上的开发诊断，不是新盲验证。

尚未检查实现diff、harness冻结/封存/评分链、新阶段运行日志、重载回放或本轮结果；未核验平台实际模型/推理元数据。待主任务提交稳定实现及测试证据后复核上述四项，再由主任务决定是否冻结执行。

本次检查的PLAN SHA256为5e246f1112644519c171d34328f759f68681e0471c58bf4f37c1d9af658a7b05，TRAINING_GATE_AUDIT SHA256为5191d2567e228129479642d76b290a41151639773fdb4ad9f2c0353739e694ec。


---

# 2026-10-30 当前核心实现独立审查（追加，保留上文设计审查）

What this change does: 新增默认关闭的训练精度补充。在原完整 compound 搜索及 polish 后、匹配闭包和验证评分前，仅当原基础池没有全输出训练精度达标候选时，提出少量隐式初值并执行完整 MIMO 联合评价。

**核心结论：已复核本次发现的不等长候选索引问题的修复，当前所审核心无剩余阻断。此结论只覆盖下列文件与检查，不批准正式冻结执行、科学采用、默认变更或发布。harness 独立审查及当前完整测试仍待主任务完成。**

## 实际检查与来源版本

本审查使用 ponytail-review，仅写本报告追加段，没有修改核心、测试、方案、原始数据、历史失败或冻结结果。读取 AGENTS、PLAN、TRAINING_GATE_AUDIT、上文旧设计审查及前轮 POST_RUN_REVIEW；沿 repair/structure/sharing/config/regressor/acceptance/evaluation/fitting/generation/genetics/graph/selection 和三份指定测试追踪调用。执行的程序仅为标准库文本、diff 与 SHA256 检查；没有执行 pytest、initializer、fit、搜索、训练、生成实验数据或评分，也未派生代理。测试结果来自主任务保存的现有日志。

- 以 `implicit_rational_20261028/source/src/dimensr_mimo` 为前轮核心来源，并核其 11 份相关源码字节摘要均与该轮 `FROZEN_INPUT_ARCHIVE.json.files` 相符。当前 evaluation/fitting/generation/genetics/graph/selection 六文件与前轮字节完全相同。
- 与该 archive 相比，regressor 的当前增量只有 closure 调用时按需传 grammar 与补充计数字段；此前已有的 1.7 implicit、closure、验证拒绝与旧序列化迁移逻辑没有误归为本次新实现。当前 config 增 flag/严格类型与路由约束，sharing 增 repair 挂接，acceptance 增内部基础模型 flag 复位，structure 增可选合作式 deadline。
- PLAN 与 TRAINING_GATE_AUDIT 当前摘要分别仍为 `5e246f1112644519c171d34328f759f68681e0471c58bf4f37c1d9af658a7b05`、`5191d2567e228129479642d76b290a41151639773fdb4ad9f2c0353739e694ec`，与旧设计审查所记一致。没有重新核算前轮训练门槛审计的17份原模型；该项旧审查证据与本次未变的审计文件分开保留。

## 发现并已修复

1. **不等长非空初值列表曾循环复用较短列表。** 初审时 repair 的 `k % len(seeds)` 在长度 `[2,1]`、`k=1` 时重复输出1的第0个初值，而 PLAN 要求该索引缺根时复制固定基础父根。主任务已将 `repair.py:52,58-62` 改为仅 `k<len(seeds)` 才取 `seeds[k]`，隐式来源索引采用同一边界；否则 clone 固定父根。复读修复后代码与 `test_precision_repair.py:105` 的新构造测试，确认第2个系统根为 `[x2,x0]`、implicit_outputs 仅 `[0]`。文档第5行同步修正。该问题已关闭，不需要改变 PLAN 或预算。

## 核心约束核对

- `config.py:25,31,55-64`：flag 默认 false，严格 `type(...) is bool`；开启必须 compound/closure/precision，precision 原配置继续要求 units、constraints 与完整 acceptance。legacy/random 生产默认没有改变。
- `regressor.py:273-292`：原 neural/GP/NSGA-II/polish 在新阶段前执行，补充完成后才首次计算验证 reference/候选误差。`repair.py:13-20` 的触发仅比较各输出训练 errors 与 acceptance.nrmse，不读图规模、验证误差、测试或任务名。存在达标状态时在接触 grammar、initializer、评价缓存和随机机制前返回；共享闭包仍按原代码继续。该路径新增报告和摘要计算会消耗少量时间，不能据语义无随机变化承诺受墙钟约束的所有正式配对必然字节一致。
- `repair.py:21,35-62`：固定父按平均训练误差、复杂度、图摘要排序；每输出最多 min(12,population_size//q)，按原索引组完整 MIMO、缺索引 clone 固定父根。`clone_subgraph` 保留根内单位/参数别名与父常数，Builder.finish 重排完整图及参数；没有推断跨根相同参数。`structure.py:330-348` 先完成非隐式提案，最后才执行隐式 fit_offset，只有实际刷新该树获胜分数的隐式提案才加入 implicit_proposals；repair 至少一个实际获胜隐式索引才评价。
- `repair.py:24-33,64-82`：成功缓存和 None 失败缓存都按图成员关系跳过；没有 refine 参数。每次新评价增加量取同一 Evaluator.calls 差值，至多12次，失败的缓存外完整评价同样计数；还同时受原 max_evaluations 与 evaluator.deadline 限制。有效子状态每输出须 <= 固定父*1.1+1e-8 才保留，并保存父/图/warm/child/来源/保护结果。此训练父保护不承诺未来测试保护。
- `evaluation.py:58-106`、`fitting.py:12-94`、`graph.py`：新增图继续完整联合常数拟合，保留训练尺度、原始有限性/输出边界、域 probes/区间审计及 Rademacher；没有改成单输出线性结果直接入池。隐式分母初值仍拒绝非有限或样本绝对值 <=1e-12 的分母，其后完整域检查仍适用；不是未审域的解析有效性证明。
- `sharing.py:68-77`：先保存原 base count、候选摘要、有效 cache 摘要及 base_evaluations，再扩展 repair，随后原两轮、每轮24、总48的闭包。pool 顺序为 base + repair protected + closure protected；repair 与闭包计数分别报告。结构上满足 `final_calls = base_evaluations + repair.new_evaluations + closure.new_evaluations` 及训练池长度对应式；真实共同评价构造测试也断言两式。没有将 validation_retained_extensions 等同训练保留数。
- `acceptance.py:26-48,63-65`：precision 基础参照只取原 count 前缀，校验原摘要，内部转 random 后将 precision_repair 复位 false，继续原绝对/结构/训练验证保护。`regressor.py:309-318` 对原基础验证无效抛错、扩展无效另记拒绝，冻结完整训练池摘要；验证不回流搜索。
- `structure.py:71-79,193,219,253,274,344` 和乘积循环：合作式 deadline 覆盖特征 admission、产品、beam、各线性子问题及末端；超时带回部分 diagnostics，repair 保留 base。默认 deadline=None 不改变排序/数值步骤、不调用时钟或随机数。一次正在执行的 NumPy/BLAS 或底层域/Rademacher操作不能抢占，故52秒是合作式 fit 截止，不能描述为任意时刻严格硬中断。本核心变更没有提高预算；60秒外部监督及全局2880秒执行仍需 harness 审查确认。
- `regressor.py:402-450`：加载虽先构造带新默认值的 cfg，验证时仍使用未补字段的原始 d config/version 重建 source，比较原 best/post_selection/diagnostics（包含原来源 hash）；成功后才迁移 asdict(cfg)/当前版本并重算 provenance。基础候选摘要仍拒绝篡改。该迁移逻辑已存在于前轮 archive，本次新增 acceptance flag 复位与其相容。

## 已查看测试证据与未检查项

- `core_targeted_01.log` 原样保留：27 passed、1 failed；失败定位为重载后的 tuple/list JSON 表示差异断言。当前断言改为 JSON round-trip 比较，没有改模型筛选或拟合。
- `core_targeted_02.log`：28 passed in 19.22s，覆盖当时 repair/implicit/protected_closure 三份定向测试；它早于不等长索引修复。
- `core_targeted_03.log`：16 passed in 15.14s，对修复后的 precision_repair 测试集重新运行，含新增不等长非空列表测试。它不是修复后全套测试通过的证据。
- 阅读的构造测试覆盖 flag 约束、恰等训练门槛 skip、不访问 initializer/cache 的 skip、12/全局/到期停止、失败缓存跳过、固定父保护拒绝、空/不等长缺根、真实联合拟合与Rademacher、基础前缀/cache不变、会计等式、未触发完整池/选择/重载一致、旧precision模型迁移与篡改拒绝。deadline 测试实际覆盖 admission/product/linear；beam 检查在本次为静态核对，没有单独注入 beam 超时运行。

Not checked: 本次未独立执行任何数值测试、未检查训练harness、封存/评分链或本轮正式34系统结果，未验证真实52秒边缘/60秒外监督行为，未独立重载或重算误差、单位与域；未重新核实平台模型日志。当前无新结果支持科学采用。主任务仍需完成 harness 独立审查、修复后所需完整测试、冻结输入与来源版本，然后决定是否进入已授权正式执行。

Verdict: 当前所审核心无剩余阻断；正式冻结执行条件尚未齐备，最终验收由主任务决定。

## 本次核心检查点 SHA256

下表标识本次最终复读的字节；任何后续相关源码变化须按影响范围复核。

| 文件 | SHA256 |
|---|---|
| `src/dimensr_mimo/repair.py` | `b0678331860bb4b392514714a0ff6fe6c32da21369638d152bcb79c46fc8d08c` |
| `src/dimensr_mimo/structure.py` | `1d1457390888bc568baee979dbe87004a4977da8461ebb4c312ed481bb4223fa` |
| `src/dimensr_mimo/sharing.py` | `d350a9ff293544b5d110aeb0a831b497a4061e2ee51b5af0d17573ffc7888701` |
| `src/dimensr_mimo/config.py` | `ecdd8ae9c8f02543a77e5d77b961fb6552311ca76250e2832b1c765e02600006` |
| `src/dimensr_mimo/regressor.py` | `85c73085ff85d7a34b0a7b1b0f921bd71c7d1bf1b4c4df6a7609403fcb23f071` |
| `src/dimensr_mimo/acceptance.py` | `d4187c011f31c21d1e96aa97cec74295ff56e4e189ac94f7224000670728cb39` |
| `src/dimensr_mimo/evaluation.py` | `db0804c6dc46b3ac225108f97083cabc15e651f57dd97d06def8a7214a9bba19` |
| `src/dimensr_mimo/fitting.py` | `72cbb8384cd351950ea1201873bb3a92796662d68bd02927155d12020fc48e6d` |
| `src/dimensr_mimo/generation.py` | `f191bb372bffadaa618dcce437c59df8b5e8ab2f874bdf40a6785a5dc0958146` |
| `src/dimensr_mimo/genetics.py` | `b496f27b68c5aac9e182111f732519698dcba7decd579d21307641dbed037a5b` |
| `src/dimensr_mimo/graph.py` | `a3c0ccc5fe7a9f971fdba9c32ecd9b55ae720eb7f276022c76c5effe80bb3f19` |
| `src/dimensr_mimo/selection.py` | `ae704bc97cfe82c56b1c767346abac97c24f5e5e58998ba6572518033c746247` |
| `tests/test_precision_repair.py` | `76eb447c0f1a808d9a29875c745266e10e8d7f0a9609fcb16c6b064f6d82adf7` |
| `tests/test_implicit.py` | `bab88611490e16790dcd3451f99322b0114724d14dfb677dbac4952c3d3abc39` |
| `tests/test_protected_closure.py` | `cca5295d2d1c557c7904e42802b6cd2dcf9754587088edf8e49d7183c327cc2a` |
| `docs/PRECISION_REPAIR.zh-CN.md` | `7004176a7e9368d962ea8969b12a2fc17ce97ae48b0fc25ef0eaa98c107481cd` |
| `experiments/precision_repair_20261028/validation/core_targeted_01.log` | `43bf3d6b3d6961f5b6cda68c26ba98168e08453b8738a12a9ba253aa4e18578d` |
| `experiments/precision_repair_20261028/validation/core_targeted_02.log` | `85e6df05bcabdd897db806aef984d5d15968240d77083524889c24cfea8c56a4` |
| `experiments/precision_repair_20261028/validation/core_targeted_03.log` | `e216204e5e3bd4268261102f728e151d9e326ce9f9959565ef7d9824fc1cb680` |
