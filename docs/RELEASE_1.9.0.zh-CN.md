# DimenSR-MIMO 1.9.0

重复拟合常数时，计算图现在复用前序节点已经完成的有限值检查。遇到非有限输入或中间结果时，按样本检查操作数，保留其向下游传播的行为。安装新版即可使用这项提速。

在两个历史大型计算图上，各用相同的训练数据和已拟合系数作为起点，对原版和新版各执行一次完整候选评价。两对评价得到相同的候选摘要及非计时工作计数。

| 计算图节点数 | 原版 | 新版 | 加速比 |
|---:|---:|---:|---:|
| 1,249 | 47.98 秒 | 20.86 秒 | 2.30 倍 |
| 1,263 | 48.94 秒 | 21.18 秒 | 2.31 倍 |

这些数字衡量单个候选评价。完整搜索的用时还包括候选生成、并行调度和检查点写入。

## 训练集稀疏压缩

```python
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig

config = SearchConfig.accuracy_pareto(
    max_evaluations=3000,
    compression_r2_drop=0.0,
)
model = DimenSRMIMORegressor(config)
model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, input_bounds=bounds)
print(model.diagnostics_["training_compression"])
```

有量纲的数据应在构造回归器时传入 `feature_units`、`target_units` 和 `constant_units`。`compression_r2_drop` 为 R² 的绝对降幅，可以是用于全部输出的单个数，也可以为每个输出分别指定数值。默认值 `None` 关闭压缩。

搜索结束后，算法从训练候选中选出一个完整模型，提取支持的基项，逐步加入能解释剩余训练误差的项，并重估保留项的系数。每个系数所需的量纲必须在常数量纲表中。候选须同时满足各输出训练精度下限、量纲、容量和数值区间检查，并降低整图混合复杂度。已达到的联合训练目标也受到保护。沿用工程默认目标时，所有输出的训练 R² 门槛均为 0.90。

压缩目前支持同量纲、训练方差非零的输出，以及可在限定规模内展开的乘积、整数幂、分式和部分三角函数项。遇到不支持的输出或无法形成合格候选时，诊断记录原因。通过训练检查的新增候选与原候选一起冻结，再参与原有的验证集 Pareto 选择。`training_compression["accepted"]` 表示加入了训练候选池，最终推荐模型见 `model.best_`。
