# DimenSR-MIMO 1.7.0

1.7.0 增加匹配闭包、隐式有理式初始化、训练精度修复和复合候选补充四个可选入口。完整神经、GP、NSGA-II 和联合常数拟合仍构成基础搜索，新增候选继续接受原量纲、数值域、结构和逐输出精度检查。

## 使用入口

| 入口 | 行为 |
|---|---|
| `sharing_strategy='closure'` | 完成基础搜索后执行两轮匹配共享，每轮至多24次新系统评价，最终以基础选择为参照保护训练和验证误差。 |
| `initialization='implicit'` | 用训练侧线性关系联合初始化有理分子的外系数与分母偏置，随后参加正常完整系统拟合。 |
| `precision_repair=True` | 在原基础池没有全输出训练精度达标状态时，提供最多12次隐式补充评价，再执行匹配闭包。 |
| `compositional_completion=True` | 完成基础搜索后提供一次有界补充，原隐式初值与通用log/sqrt逆变换初值竞争同一返回名额，再执行匹配闭包。 |

默认初始化为 `legacy`，共享策略为 `random`，两个补充开关均为 `False`。`precision_repair` 与 `compositional_completion` 互斥，启用时要求 `compound` 初始化、`closure` 共享和 `precision` 选择。补充与基础搜索共用配置中的总评价与时间预算。

下面的配置适用于两个输出。误差阈值和结构预算应由具体任务确定，`fit` 的输入单位、常数单位及适用域沿用原 API。

```python
from dimensr_mimo import SearchConfig

config = SearchConfig(
    initialization='compound',
    sharing_strategy='closure',
    final_selection='precision',
    compositional_completion=True,
    acceptance={
        'nrmse': [1e-4, 1e-4],
        'shared_graph_size': 24,
        'independent_parameters': 6,
        'expanded_total_nodes': 30,
        'maximum_output_depth': 6,
    },
)
config.validate()
```

原始模型保存、加载与物理坐标交付接口继续支持旧模型。新版本先验证旧文件中的原始配置与来源，再补齐新默认字段。模型 JSON 保存补充提案、缓存外评价、实际来源、逐输出保护和最终选择证据。

## 研究结果

历史17对任务中，复合补充的完整精度与结构接受由对照11/17增至13/17，34/34输出通过预设保护，平均干净测试NRMSE由0.002292397降至0.001754020，平均共享图由21.1765降至19.2353。四个阻尼任务仍未完整接受。

原历史核验因控制模型的三个阶段调度时间字段而判定失败，原收据保留。后续隔离核验在原截止内按独立审查后的合同复核同一批封存模型，34/34通过并取得独立后审接收。两个合同和各自判定均可在实验目录追溯。

已观察开发批完成32/32技术核验，完整接受从7/16增至16/16，32/32输出受保护。平均干净测试NRMSE由0.0160253214降至0.0009509047，平均图规模由21.875降至12.625，全部事前开发门槛通过。独立后审重算32个所选模型、850个补充/匹配事件和384个返回根，确认这些结果。

随后锁定候选的新输出组合批也完成32/32技术核验与32/32输出保护。两臂完整接受均为5/16，平均干净测试NRMSE均为0.0219258070，平均图规模均为22.625，科学采用门槛未达。逐族完整接受为对数仿射比0/4、倒数二次范数4/4、振荡仿射饱和0/4、指数仿射饱和1/4。11个未完整接受任务的原始指标、图和逐项原因全部保留。

新任务独立后审重构32个最终模型、983个补充/匹配事件和384个返回根，确认全部16对最终图、常数及完整最优候选载荷相同。两批后审的复算范围与未覆盖项分别记录在对应POST_RUN_REVIEW中。

本版提供上述显式可选入口。当前证据支持已观察开发矩阵的收益，新输出组合中最终接受与汇总性能持平。两个批次使用不同任务组合，结果应分别阅读。

## 证据与复现

本次源码与研究脚本合计507项测试通过。隔离安装的wheel核对全部41份生产源码，完成135组native/physical模型的加载、保存重载及12个固定种子独立坐标点的精确预测比较，另完成一份旧版native模型迁移。检查时阻止实验辅助模块导入，实际导入位置限定在隔离安装目录。源码包检查与独立发布审查收据保存在发布检查目录。

完整基础搜索后的候选补充规则保存在冻结设计中，旧阶段文档保留各自签署时的状态。发布结果以本说明及对应终态收据为准。

- [匹配闭包](PROTECTED_CLOSURE.zh-CN.md)
- [隐式初值](IMPLICIT_INITIALIZATION.zh-CN.md)
- [训练侧修复](PRECISION_REPAIR.zh-CN.md)
- [复合补充](COMPOSITIONAL_COMPLETION.zh-CN.md)
- [历史原失败](../experiments/completion_historical_20261031)与[隔离核验](../experiments/completion_verification_repair_20261031)
- [开发回归](../experiments/completion_development_verified_20261031)与[新输出组合](../experiments/completion_release_validation_20261104)
- [发布检查](../experiments/release_170_20261104)

复现需要独立目录和新的运行标识，既有预算、模型、失败与冻结收据保留。Windows的长归档路径由研究脚本的局部I/O适配处理，CI检出也启用相应Git选项。科学采用仍按每批事前固定的完整接受、逐输出保护和宏指标门槛判定。

Windows首次检出完整证据仓库时使用以下命令。已有仓库在其根目录执行最后一行，设置仅作用于当前仓库。

```powershell
git -c core.longpaths=true clone https://github.com/Ashore521/dimensr-mimo.git
cd dimensr-mimo
git config --local core.longpaths true
```
