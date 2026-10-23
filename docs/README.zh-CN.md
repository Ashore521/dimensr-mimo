# DimenSR-MIMO

面向多输入、多输出的量纲约束符号回归。算法搜索一组共享的非线性基函数，各输出使用独立的线性系数，并以所有输出的联合残差指导基函数选择。

[English README](../README.md) · [原 DimenSR](https://github.com/Ashore521/DimenSR)

例如：

```text
y1 =  2 a b + 0.5 c
y2 = -1 a b + 3.0 c
```

两个输出可以共用 `a*b` 和 `c`。预测时，每个共享表达式节点在同一批输入上只计算一次。输出之间通过共享字典、联合残差评分和内部常数优化相联系。

## 安装与运行

需要 Python 3.10 或更新版本。在仓库目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/shared_system.py
python -m pytest
```

Linux/macOS 使用 `source .venv/bin/activate` 激活环境。运行依赖为 NumPy、SciPy、SymPy 和 scikit-learn，无需 GPU。目前提供仓库源码安装，尚未发布到 PyPI。

## Python 用法

```python
import numpy as np
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig

rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(160, 3))
Y = np.column_stack((2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2],
                     -X[:, 0] * X[:, 1] + 3 * X[:, 2]))

model = MIMOSymbolicRegressor(
    config=SearchConfig(random_state=42),
    feature_names=["a", "b", "c"],
    target_names=["y1", "y2"],
)
model.fit(X[:120], Y[:120], X_val=X[120:140], Y_val=Y[120:140],
          input_bounds=[[-2, 2]] * 3)
print(model.expressions_)             # 每个输出对应一个 SymPy 表达式
prediction = model.predict(X[140:])   # 形状始终为 (样本数, 输出数)
print(model.audit_domain())
model.save("model.json")
restored = MIMOSymbolicRegressor.load("model.json")
np.testing.assert_allclose(restored.predict(X[140:]), prediction)
```

`X` 的形状为 `(样本数, 输入数)`，`Y` 为 `(样本数, 输出数)`，数值必须有限。名称应与各列一一对应且无重复。省略验证数组时，按 `validation_fraction` 和随机种子进行内部划分。验证集用于选择最终模型；独立测试集必须留在 `fit` 之外。

### 量纲与系数

量纲向量依次为 **L、M、T、I、Theta、N、J**，对应长度、质量、时间、电流、热力学温度、物质的量、发光强度。每个输入和输出各提供一个七元素向量，省略时视为无量纲。

```python
length = [1, 0, 0, 0, 0, 0, 0]
time = [0, 0, 1, 0, 0, 0, 0]
speed = [1, 0, -1, 0, 0, 0, 0]
area = [2, 0, 0, 0, 0, 0, 0]
model = MIMOSymbolicRegressor(
    feature_names=["distance", "duration"],
    target_names=["speed", "area"],
    feature_units=[length, time],
    target_units=[speed, area],
)
# 为 duration 声明严格大于零的边界后，distance/duration 才能通过对应区间检查。
```

系数默认无量纲，因此仅允许量纲相同的基函数连接到对应输出。可以通过 `coefficient_units` 显式列出允许的系数量纲；仅当“输出量纲减去基函数量纲”属于该列表时，连接才合法。截距要求输出量纲本身属于该列表。若仍允许无量纲系数，应在列表中保留全零向量。非线性特征内部的常数始终为无量纲。数值需要事先统一单位；量纲检查不会自动进行厘米到米等换算。

### 算法流程

1. 从输入和常数构建带量纲的初始字典，过滤量纲不兼容、定义域无法通过检查的候选。
2. 按各输出归一化残差的联合投影评分候选，逐步挑选少量共享基函数，为各输出求解线性最小二乘系数。
3. 在保留的字典上组合表达式、扰动内部常数，按预算迭代。
4. 对内部非线性常数进行联合优化，每次更新时重新求解输出线性系数，即变量投影。
5. 从训练误差与复杂度的 Pareto 候选中，按“验证误差 + `complexity_weight × 复杂度`”选择最终模型。

复杂度统计共享表达式图中的唯一节点、非零输出连接以及非零截距。`max_terms` 限制所有输出共同使用的基函数数量，不限制输入数或输出数。该实现是基于 DimenSR 的实用扩展，尚无算法新颖性或优于其他多输出方法的比较结论。

| 配置项 | 默认值 | 作用 |
| --- | ---: | --- |
| `max_features` | 256 | 保留字典大小，至少容纳所有输入和两个常数 |
| `generations` | 4 | 初始搜索之后的组合与变异轮数 |
| `proposals_per_generation` | 160 | 每轮提出的特征数量 |
| `max_terms` | 5 | 共享基函数数量上限 |
| `max_nodes` / `max_depth` | 13 / 6 | 单个特征树的节点数与深度上限 |
| `operators` | `add, sub, mul, div, square, sin, cos` | 启用的运算符 |
| `complexity_weight` | 0.0001 | 验证选择中的复杂度权重 |
| `validation_fraction` | 0.2 | 内部验证集比例 |
| `polish_max_nfev` | 30 | 非线性优化评估预算，设为 0 关闭 |
| `random_state` | 0 | 随机种子 |
| `interval_max_boxes` | 32 | 每次区间检查的划分预算 |

还支持 `neg`、`sqrt`、`exp` 和 `log`。超越函数要求无量纲输入，加减法要求相同量纲。搜索受随机性和预算限制，复杂正确表达式可能无法被发现。

### 数值边界

`input_bounds` 为每个输入的 `[下界, 上界]`，需覆盖实际使用范围和端点。原始表达式树的数值区间检查必须返回 `enclosed`，否则候选被拒绝；`invalid` 和 `unresolved` 都不能进入最终模型。代数约分不会掩盖原始运算中的除零等问题。

这是保守的数值区间检查，并非形式化的全域数学证明。区间依赖或划分预算不足可能使合法公式被拒绝。省略边界时，会从内部划分后的训练输入构造包围盒；它不代表盒外的合法性。位于盒外的验证点只接受逐点数值检查。扩大工作范围前，可调用 `audit_domain(new_bounds)` 重新检查。低预测误差也不等于边界合法、物理因果成立或外推可靠。

## CSV 命令行

CSV 必须包含列名，数据行全部为有限数值。默认将 `--targets` 以外的列作为输入，也可在元数据中指定输入列及顺序。验证和预测按保存的列名对齐，允许列顺序变化。重复列名、缺失列、非数值、NaN 和无穷值会被拒绝；选取具名输入时忽略其他数值列。预测文件保留输出名称。

```powershell
dimensr-mimo fit --data train.csv --targets y1 y2 --validation validation.csv --output model.json
dimensr-mimo predict --model model.json --data new_inputs.csv --output predictions.csv
python -m dimensr_mimo --help
```

可使用 `--metadata metadata.json`：

```json
{
  "feature_names": ["a", "b", "c"],
  "target_names": ["y1", "y2"],
  "input_bounds": [[-2, 2], [-2, 2], [-2, 2]]
}
```

还可写入 `feature_units`、`target_units` 和 `coefficient_units`，其值均为七元素向量的列表。提供 `target_names` 时，必须与 `--targets` 的名称和顺序一致。`--config config.json` 可覆盖配置表中的字段：

```json
{"generations": 6, "max_features": 384, "max_terms": 4, "random_state": 42}
```

JSON 模型保存原始表达式树、拟合系数以及量纲和边界元数据。请使用模型文件交换和恢复预测器；打印的 SymPy 表达式用于阅读与检查。

## 来源、验证与贡献

`src/dimensr_mimo/_vendor/` 保留了原 DimenSR 的 MIT 授权量纲、表达式树和区间检查代码，署名见 `licenses/`。多输出搜索、估计器、命令行及测试在本仓库独立维护。原仓库的实验数据和私人原始文档未包含在此项目中。

提交修改时，请运行 `python -m pytest` 并附小型可复现例子。合成例子与已经检查过的问题属于回归验证，不能作为新的盲测证据。项目按 [MIT 许可证](../LICENSE) 开放。
