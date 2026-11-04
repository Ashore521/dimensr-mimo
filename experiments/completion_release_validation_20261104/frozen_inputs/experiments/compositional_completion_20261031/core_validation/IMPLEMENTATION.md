# compositional_completion_v1 实现交付

状态：实现及构造/相关回归验证完成，稳定核心独立审查待做。未运行正式历史或开发矩阵，
未进行科学采用、打包发布、提交或推送。2026-10-31。

## 实际改动

- 新增 `src/dimensr_mimo/completion.py`，公开
  `compositional_completion(base,evaluator,grammar)->(retained,report)`；
  初值入口 `completion_seeds(grammar,X,y,limit,deadline=None)->(seeds,stats)`。
- 仅修改已锁源码中的 config、sharing、regressor、acceptance 四个文件。
  `core_validation/{config,sharing,regressor,acceptance}.diff` 是对 BASE_SOURCE_LOCK
  指定原字节归档的实际增量，避免把此前未提交的开发改动算入本工作包。
- 新增 `tests/test_compositional_completion.py`（33项）和
  `docs/COMPOSITIONAL_COMPLETION.zh-CN.md`。
- 未修改 repair、structure、原完整搜索主体、旧冻结源/数据/模型/指标、版本或发布文件。

## 接口和诊断

`sharing_closure.compositional_completion` 保持 repair 的标准 triggered、seed_limit、
new_evaluations、valid、protected、protection_rejections、cache_skips、
structure_rejections、initialization_diagnostics、events、stop_reason、parent 字段。
额外字段为 mode、missing_roots、nonimplicit_skips、implicit_returned_roots、
compositional_returned_roots，以及旧 implicit、新 compositional 和全部线性问题计数。
总 linear_subproblems = 原 structured 的全部线性问题 + 新 compositional 线性问题；
implicit_linear_subproblems 属于原 structured 的子集，不能再重复相加。

每输出 stats 的 `returned_sources` 与 seeds 严格一一对应，含真正获胜的 kind、
returned_index、proposal_index、tree、constants、score。新来源还有 transform、form、
variable、basis_indices、intercept、outer_amplitude；旧来源含 original_index。
`structured_implicit_retained_indices` 保存旧返回列表索引；
`implicit_retained_indices` 与 `compositional_retained_indices` 是合并后索引。
合并拓扑重复只保留最低严格原Y分数的常数及其来源；同分保留首项，不做来源并集。
根求值、基求值、线性工作和拒绝都有独立计数。

事件延续旧 seed_index、parent、graph、warm、child、protected、new_evaluation，
并含实际 implicit_outputs、compositional_outputs、root_sources。缺位来源为
`kind='parent'`，复制固定原父对应根，不循环索引。缓存成功和失败都跳过，补充最多
12次原 Evaluator 缓存外完整系统评价；没有新 Evaluator 或额外 refinement。

配置默认 False、互斥并严格 bool。旧记录仍先用原保存配置核验，然后迁移新默认；
acceptance 的递归 base 仅当原配置含新键时才复位，避免污染老模型来源 hash。
regressor 的 closure history 另外保存 completion_new_systems/protected_systems。

## 已执行验证

运行器为项目 `.venv/Scripts/python.exe`；依赖实际版本保存在 package_versions.json。
所有 pytest basetemp 独立放在工作区 outputs/completion-core-*。

1. `pytest_first.log`：新构造测试初版 29通过/1失败。失败由测试按 Node 对象身份定位
   外幅值参数造成：声明常数 Node 可复用身份，但每次树遍历有独立参数位置。
   已改为按树路径定位；实现参数语义未改变。
2. `pytest_related.log`：新构造 + 旧 precision_repair + protected_closure，54通过。
3. `pytest_final.log`：加入真实非单位幅值/有量纲联合拟合、原生区间域和 physical/implicit
   回归，73通过/1失败。失败是测试错误地要求全部零分子除式均拒绝；近似内层仍可能
   产生合法非零分母。已改为逐个实际返回除式检查分母阈值，并另测精确 0/0 拒绝。
4. `pytest_verified.log`：上述修复后的相同相关集合，74通过，30.45秒。
5. `pytest_source_accounting.log`：加调度结构拒绝+评价后 deadline 留存检查，33通过。
6. `pytest_receipt.log`：最后补充实际求值及返回来源总计数后的新核心33项复核。

相关集合命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_compositional_completion.py tests/test_precision_repair.py tests/test_protected_closure.py tests/test_physical_delivery.py tests/test_physical_selection.py tests/test_implicit.py -q --basetemp=outputs/completion-core-verified
```

最后新核心命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_compositional_completion.py -q --basetemp=outputs/completion-core-receipt
.venv/Scripts/python.exe experiments/compositional_completion_20261031/core_validation/verify_implementation.py
```

构造覆盖合法/非法单位和分数单位、缺常数/算子、固定16变量/24基、节点/深度、
正零负平方权重与已有尺度、0*overflow不等域及原区间审计、逆变换溢出/下溢/负号/
零和近零除数、严格原Y排名与常数/来源配对、裸根/外幅值、短列表、成功/失败缓存、
结构拒绝、12上限/全局预算/deadline、部分计数、真实联合拟合/Rademacher/原生域、
完整base身份、history、老两类模型迁移和新模型重载/篡改拒绝。
测试只用独立构造数据；旧冻结模型仅作读取/迁移验证，不修改原字节。

## 可核对证据和剩余边界

`verify_implementation.py` 逐一验证全部40个锁定旧源的归档字节；当前仅上述四个
获授权旧文件变化，其余36个不变。IMPLEMENTATION_CHECKS.json 保存源码、文档、
测试、日志、依赖版本及方案/设计报告哈希，科学接受明确为 False。

单次 NumPy/BLAS 仍不可抢占。新初值需要与旧 structured 返回根竞争前12，
外幅值只提供可表达的联合拟合路径，不承诺一般系数下返回、收敛或科学精度。
本工作包没有运行全套测试（由主代理集成统一运行），没有正式矩阵预算/科学门槛收据，
没有独立稳定核心审查、harness审查、新锁定任务、打包安装或发布验收。
