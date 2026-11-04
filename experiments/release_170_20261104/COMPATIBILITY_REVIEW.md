# 1.7.0 发布兼容性独立预审

2026-11-04。What this change does: 本候选在原基础搜索之后增加默认关闭的闭包、隐式初值、精度修复及复合补充入口，沿用原完整系统评价和选择保护。审查对象是尚未提交的候选源码、公开接口及发布适配，不是已经发布的 1.7.0。

**结论：在本次实际检查范围内，没有发现有证据支持的发布兼容性阻断。可以继续完成既定实验接收、文档和安装包验证；本预审不是最终发布通过，也不认可尚未完成的新任务性能或默认切换。最终接受由主代理决定。**

## 来源及审查边界

- 比较基线为 `652b05884269c7563b101e0a14a9b15f749bbc6f`（1.6.0）。当前 41 份生产 Python 源码逐字节对应 `experiments/compositional_completion_20261031/CANDIDATE_LOCK.json`，锁 SHA256 `51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452`；同时与该目录 `CORE_REVIEW_CHECKS.json.current_source_files` 完全一致。检查开始和结束均未发现生产源码漂移。
- 阅读 `git diff` 的 config/regressor/acceptance/sharing/structure 增量及新增 repair/completion，连通配置、CLI、公共导出、Evaluator、保存重载和 physical 路径。复用原 `DESIGN_REVIEW.md`、`CORE_REVIEW.md`、precision_repair 的追加核心审查、implicit 与 balanced_closure 原审查；没有重复原核心 75 项测试或正式矩阵。
- 本审查实际子代理日志：`C:/Users/608/.codex/sessions/2026/11/04/rollout-2026-11-04T21-48-48-01a1a6d1-c925-7992-8ad8-da5462267a2a.jsonl`；`turn_context` 时间 `2026-11-04T13:48:51.900Z`，turn `01a1a6d1-c9ee-77f2-bc32-852b5a01cce0`，root turn `01a1a67b-1c60-71c0-b19c-4bde6d4032f1`，实际 `gpt-6.1-sol / high`，角色为 ResearchReviewer。本审查没有派生代理。

## 已完成的检查

1. **公共配置和旧默认。** 与基线配置类逐字段比较，所有旧字段默认值完全相同；仅新增 `precision_repair=False` 和 `compositional_completion=False`，仍为 `legacy/random`。独立穷举 1152 种初始化、共享、选择、两flag、单位、约束和sharing组合，150种合法，全部与公开限制一致；10个非严格 bool 构造全部拒绝。closure 必须 precision/use_sharing，precision 继续要求单位、逐输出约束和完整 accuracy/structure 门槛，两种补充互斥且必须 compound/closure/precision。CLI 仍将配置 JSON 交给 SearchConfig，无另一路绕过配置；原公共 `__all__` 未改。

2. **基础机制及数值边界。** `evaluation.py`、`fitting.py`、`graph.py`、`grammar.py`、`pareto.py`、`selection.py`、`physical.py`、`physical_selection.py`、`audit.py` 九份文件与 1.6.0 Git blob 字节相同；全部41源与已独审核心相同。新增候选仍通过同一 Evaluator 的完整输出联合拟合、原节点/深度、probe/interval 域检查及 Rademacher。补充只接训练 X/Y；原polish之后完成训练池再使用验证数据，没有新增测试标签入口。新 log/sqrt 单位推导、有限值、近零1e-12及严格原图求值边界引用原核心独审和未变源码，本次没有重跑 initializer 或独立重算域/单位全集合。训练/验证保护不能写为未知测试保护或连续域证明。

3. **逐输出保护及预算连接。** `sharing.py:65`先冻结原base前缀，再互斥调用补充；补充使用同一缓存、Evaluator.calls、截止及全局上限，最多12次缓存外系统评价，后续匹配最多24+24。`acceptance.py:26`核base_count/hash，仅由原base递归重建precision参照，关闭两flag后重建，最终同时施加该参照训练和验证的各输出 `1.1倍+1e-8` 边界。新增验证无效图被剔除而不续搜，base无效明确报错。上述连接与原独审一致；没有改阈值或把补充并入保护参照。合作式 deadline 不能抢占单次 BLAS 的原限制仍须保留。

4. **实际旧模型迁移和新模型再存再载。** `review_validation/audit_compatibility.py`在阻止全部 `experiments` 导入的环境中实际加载四份封存模型：1.1.0 native precision、缺两flag的旧compound、缺completion字段的repair，以及当前completion模型。四份原best完全保持；各7个域内固定点预测与原保存图精确相同，再存再载的完整 `to_dict()` 相同。三类旧字段缺失均成功补默认并重建来源。四份模型的选择来源摘要篡改均拒绝，新completion的base摘要篡改亦拒绝。另实际加载/重存一份封存physical模型，完整字典和native/physical固定点预测精确相同。这里是源码导入态检查，不是已安装wheel的证据；physical加载构造与native加载分别检查，未误称physical.load必须间接调用native.load。

5. **发布依赖与CI隔离。** 生产src搜索及上述阻断导入检查未见 `experiments` 或本机绝对路径运行依赖。MANIFEST新增的是按目录顶层指定的研究脚本、说明和明确旧模型/冻结语法夹具，不是运行包对研究目录的依赖；wheel/sdist实际成员仍待构建检查。CI保留Ubuntu/Windows×Python3.10/3.12矩阵，checkout局部环境设置 `core.longpaths`，`--import-mode=importlib`解决两个同名test_stage，`pythonpath=tests`保留原裸跨测试模块导入。没有删除旧测试或放宽断言。

6. **封存守卫仍实际运行。** 新root/experiments conftest按三个指定测试名将HERE指向空隔离目录，原 `pytest.raises(...seal...)` 仍执行。独立按CI导入方式仅复跑这三个入口（历史参数化为两项）：`review_validation/guard_tests.log` 为 **4 passed in 1.63s**，没有fit或候选生成。root conftest、原tests/conftest、原历史测试三个文件均与开发批FREEZE字节摘要一致；原tests/conftest仍为Torch单线程设置。主代理已有 `validation/ci_source_tests_06.log` 为 **507 passed in 90.83s**；本审查读取并记录该现有日志，没有将其声称为独立复跑，也没有扩大测试数量重复全套。

7. **功能和科学措辞。** 读取四份功能说明与 `docs/RELEASE_1.7.0.zh-CN.md` 草稿，当前接口、互斥、默认、共享预算和保护边界与代码相符。历史13/17、对照11/17及均值与隔离核验SUMMARY一致；原历史失败及新合同核验被分开描述，未改旧判定。implicit原30/34保护失败、旧新任务7/16接受不足等仍保留，不能写成一般优势或全面通过。当前草稿明确把开发批、新输出组合和安装态结果留待终态补充；旧阶段说明保留签署时的开发状态，不当作最终发布声明。开发批32/32及16/16接收由另一独立后审负责，本审查未重复其完整复算；新输出组合性能未判定。

## 最终包与提交候选仍需复核

- 完成开发批独立后审接收，以及锁定新任务一次32系统的终态证据、科学判定与独立后审；失败也须保留并准确报告，不能用历史改善替代该批门槛。
- README及最终1.7.0说明同步真实终态；清楚区分opt-in工程发布、每批科学采用和默认不变，核对最终链接与公开文件范围。
- 实际wheel/sdist的41源锁、版本、内容白名单、无私人Word/PDF/原始数组/outputs/凭据/临时目录；隔离wheel安装、阻止experiments导入的native/physical兼容检查，以及sdist真实导入位置和约定核心测试。MANIFEST静态合理不等于成包通过，也不承诺sdist含全部历史研究归档。
- 暂存Git实际blob与锁一致、冻结历史未被换字节、最终提交和远端main精确SHA、该SHA的四个CI矩阵结果。上述均未由本预审完成或预先批准。

本审查仅写本报告、`compatibility_review_checks.json`及 `review_validation/`。实际环境为Python3.12.10、NumPy2.5.4、SciPy1.18.1、Torch2.14.1+cpu，源码导入路径和检查输入摘要保存在收据中。没有fit/search、训练、benchmark、正式数据生成或评分、预算/阈值改动、Git写入、网络发布或历史工件改写。记忆仅帮助定位默认关闭、保留失败和旧保护约定，判断以本次源码与收据为准。

Verdict: 兼容性预审范围内无实质阻断；最终包、新任务结果和发布提交仍待续审，尚不能称1.7.0发布已通过。
