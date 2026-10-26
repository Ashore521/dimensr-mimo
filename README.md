# DimenSR-MIMO

静态多输入多输出符号回归：以整个多根公式系统为个体，联合运行输出条件 GRU、量纲 GP 和两目标 NSGA-II。每个输出保留完整公式，可发现共享子图与显式绑定参数，也可解除共享。

新版从 DimenSR 0.4.0 的完整神经/GP/NSGA-II 流程重建。算法定义、来源映射及验收标准见 [中文架构](docs/ARCHITECTURE.zh-CN.md)。[首轮结果解读](docs/FIRST_ROUND.zh-CN.md)、[完整实验报告](experiments/first_round/REPORT.zh-CN.md)、[独立核验](experiments/first_round/VERIFICATION.json)和[安装复现说明](docs/REPRODUCING.zh-CN.md)均已提供。

首轮七种合成任务、两个种子、五方法共 70 个系统全部完成，预检至评分 29.43 分钟。完整联合方法与完整逐输出独立 DimenSR 均实际执行神经更新、GP 和多代 NSGA-II。联合方法平均 NRMSE 为 7.17e-5，独立方法为 4.21e-6；本轮未显示精度优势。9/14 个联合最终模型采用了共享结构或参数，但共享非线性参数任务尚未自动选出真实共享参数结构。工程机制通过不能代替性能结论。

[第二轮诊断与重复验证](docs/SECOND_ROUND.zh-CN.md)完成 60/60 个系统，耗时 35.80 分钟。仅取消最终选择的复杂度惩罚，使平均测试 NRMSE 从 1.85747e-4 降至 4.88166e-5（降低 73.7%）；但 2/40 个输出超过预定退化阈值，因此维持原默认配置。全部机制与独立重算通过，20 对搜索指纹和候选池一致。两轮均为已知合成开发任务，不能据此声称普遍性能优势。

[第三轮逐输出保护选择器](docs/GUARDED_SELECTION.zh-CN.md)用 40 次实际搜索生成 80 个最终系统，全部核验通过，耗时 18.96 分钟。相对原选择器，平均测试误差降低 70.4%，复杂度增加 4.1%，40/40 个输出满足预定测试保护阈值；相对直接取消复杂度惩罚，测试精度持平、复杂度降低 20.5%。该规则通过本轮任务范围内的推荐门槛，生产默认尚未更改，未见任务及真实数据验证仍待开展。

[第四轮共享结构发现](docs/SHARED_DISCOVERY.zh-CN.md)完成40/40系统，耗时14.82分钟。在两组共用guarded选择器的条件下，仅补齐通用参数化sin/cos种子，就在5/5个共享频率任务中恢复了同一参数身份，控制组为0/5。20配对为5改善、15持平、0退化，40/40输出满足测试保护，平均复杂度降低22.4%。收益集中于已知共享频率任务；公开任务迁移验证随后开展。

[第五轮固定公开任务迁移](docs/PUBLIC_TRANSFER.zh-CN.md)完成18/18系统，全部技术核验通过，耗时13.14分钟，但性能推荐未通过：候选平均NRMSE较原联合上升1.04%，1/12输出超过测试保护阈值，生产默认保持原样。较高输入维数下，追加种子被原初始化180名额截断；6/6配对的实际搜索指纹及候选池相同，本轮差异只归于最终选择。逐输出独立DimenSR的平均误差更低。

[第六轮固定预算与量纲开发层](docs/TYPED_COVERAGE.zh-CN.md)完成36/36系统核验，耗时12.83分钟。同词汇平衡分配使精度和结构同时达标的系统由4/12增至6/12，平均测试NRMSE降低86.7%，24/24输出受保护；逐输出独立DimenSR也为6/12达标。11输入任务的三角种子覆盖已修复，不同单位三角、部分共享与边界组合仍失败，且其候选池内无达标解。每个完成模型包含原始物理变量图、参数单位、适用域及可复算交付；总目标仍在推进，生产默认未变。

[第七轮类别对齐与量纲组合](docs/UNIT_COMPOSITION.zh-CN.md)完成48/48系统核验，耗时18.62分钟。2×2开发回归中，对照、仅类别对齐、仅量纲组合、两者组合分别6/12、8/12、12/12、12/12达标；量纲组合的24/24输出保护通过，而仅类别对齐有2个配对退化。后续研究候选优先采用改动较少的量纲组合；这些结果仍属于已知无噪声任务，尚未验证实际应用，生产默认不变。

## 安装

Python 3.10+，建议独立环境。Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe examples/shared_system.py
.venv\Scripts\python.exe -m pytest
```

Linux/macOS 将 `.venv\Scripts\python.exe` 改为 `.venv/bin/python`。仓库不依赖本地原版目录；源模块及许可已随包提供。

## Python API

```python
import numpy as np
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig

rng = np.random.default_rng(21)
X = rng.uniform(-1, 1, (160, 2))
Y = np.column_stack((X[:, 0] ** 2 + X[:, 1], X[:, 0] ** 2 - X[:, 1]))
model = DimenSRMIMORegressor(SearchConfig(random_state=0))
model.fit(X[:120], Y[:120], X_val=X[120:], Y_val=Y[120:],
          input_bounds=[[-1, 1], [-1, 1]])
print(model.formulas_)
print(model.diagnostics_)
model.save("model.json")
loaded = DimenSRMIMORegressor.load("model.json")
predictions = loaded.predict(X)  # n_samples × n_outputs
print(loaded.audit_domain([[-1, 1], [-1, 1]]))
```

`feature_units` 为 p×7、`target_units` 为 q×7，SI 顺序 L,M,T,I,Theta,N,J；`constant_units` 为允许生成的常数量纲。未提供时无量纲。X/Y 必须是有限二维数组。不同输出按训练标准差归一化，模型不会自动改变输入单位。

`formulas_` 是方便阅读的 SymPy 显示；JSON 中原始 DAG、节点引用和参数编号才是计算及审计依据。预测使用原始除法、log、sqrt，非法输入报错，绝不采用保护算子。

## 命令行

```powershell
dimensr-mimo fit --data train.csv --features x0,x1 --targets y0,y1 --validation validation.csv --config config.json --metadata units.json --output model.json
dimensr-mimo predict --data new.csv --features x0,x1 --model model.json --output predictions.csv
```

CSV 第一行为列名；配置 JSON 对应 `SearchConfig` 字段。metadata 可含 `feature_units`、`target_units`、`constant_units`、`input_bounds`。不提供独立验证文件时固定种子划分验证集；分组/时序数据应自行提供适当划分。

## 运行边界

- `use_rnn/use_gp/use_sharing/use_units/use_rademacher/use_size_diversity/use_constraints/interval_checks` 用于明确消融。默认全部启用。
- `wall_seconds` 为单次 fit 的协作式时间限制；数值库的一次调用和最终选择可能少量超时。严格实验上限由外部进程监督。
- `max_evaluations` 限制唯一系统数，`population_size/epochs/gp_generations/selection_generations` 限制各阶段。日志中有实际新评价和参数更新证据；设置开关不代表阶段已执行。
- 参照是训练初始化的逐输出最佳公式；默认相对容差 10%、绝对 NRMSE 容差 0.02。验证不可行时明确记录最小违反量回退，不能宣称满足全部输出约束。
- 区间检查的 `enclosed` 只是有限浮点区间筛查；`invalid/unresolved` 均被拒绝，可能保守排除合法式子。关闭区间为显式消融。
- 局部 Rademacher 是归一化多输出联合参数 Jacobian 的线性化函数族估计，不是结构搜索泛化保证。
- 保存文件用于推理与审计，不是可恢复训练优化器的检查点。

MIT；[来源与许可](THIRD_PARTY_NOTICES.md)。不发布私人文献或原始数据数组。
