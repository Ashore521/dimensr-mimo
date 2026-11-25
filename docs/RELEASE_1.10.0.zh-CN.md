# DimenSR-MIMO 1.10.0

## 用自己的数据运行

在 Python 3.10 或更高版本的虚拟环境中安装指定版本。

```console
python -m pip install "dimensr-mimo @ git+https://github.com/Ashore521/dimensr-mimo.git@v1.10.0"
```

将 CSV 的第一行设为列名。以下示例以 `x1,x2` 为输入、`Ux,Uy,Uz` 为输出，列名和个数可按实际数据调整。工程命令使用所给的全部行训练和选模，可将另行保留的测试数据用于拟合结束后的评估。

```console
dimensr-mimo-engineering --data train.csv --features x1,x2 --targets Ux,Uy,Uz --output-dir fit-run
```

`--output-dir` 指向新目录或空目录。`--metadata units.json` 可提供 `feature_units`、`target_units`、`constant_units`、原始单位的 `input_bounds` 及可选的仿射 `coordinates`。每个量纲按 L、M、T、I、Θ、N、J 的顺序给出七个指数。省略元数据时使用无量纲变量和原始数值，输入域取自训练数据。提供严格递增的输入上下界可为恒定输入列指定应用范围。

`--config options.json` 可覆盖工程预设中的字段。例如 `{"parallel_mode":"serial","n_cpus":1}` 启用单进程。默认最多使用四个工作进程。每个新数据集默认从自身初始化阶段取得逐输出训练误差保护参照，可用 `fixed_training_reference` 给出已知的逐输出参照值。[工程预设](../src/dimensr_mimo/engineering_config.json)

## 候选如何进入优选

- 所有输出的训练 R² 均达到 0.90 时，最终从达标的完整模型中选混合复杂度最低者。
- 尚无完整模型达标时，按每组模型中最低的训练 R² 从高到低排序。若有 N 组有效模型，默认入围数为 `min(N, max(50, ceil(0.30 × N)))`。最弱分量同分时比较平均训练误差，再按模型标识决定边界名次。
- 入围模型进入 NSGA-II，随后产生的子代和精修模型也可参与最终筛选。表达式搜索的默认评价阈值为 5,000 次，在完整轮次边界检查，后续优选和精修仍可产生新评价。
- 最终筛选只绘制一次误差与混合复杂度的 Pareto 图。图中标出最终交付模型。若交付时接受了压缩，还会标出压缩前选中的模型。默认压缩要求每个输出的训练 R² 均保持或提高。

## 运行中和结束后查看什么

终端和 `fit-run/progress.log` 按时间顺序追加两行一组的记录。第一行给出阶段、轮次、累计评价数和耗时，第二行并列显示训练 R²、最弱输出、达标数和混合复杂度。复杂度为同一完整模型的分数，显示到小数点后两位。

需要暂停时，在运行目录内创建名为 `STOP_REQUESTED` 的空文件。看到“拟合已暂停”后，移除该文件，以相同参数追加 `--resume` 即可接续本次拟合。已有日志保存在原位置。

| 输出 | 内容 |
|---|---|
| `results.json` | 交付公式、各输出训练 R²、混合复杂度、入围统计与压缩结果 |
| `model.json` | 可在原始输入坐标中预测的模型 |
| `pareto/pareto.png` | 最终筛选图，同图另存为 PDF 和 SVG |
| `pareto/candidates.csv` | 图中各点的候选标识、训练误差、复杂度及入选标记 |

读取交付模型后，按原始输入列顺序传入新样本。

```python
from dimensr_mimo import PhysicalModel

model = PhysicalModel.load("fit-run/model.json")
print(model.formulas)
# X_new 的各列顺序与 --features 一致，取值位于声明的输入域内。
predictions = model.predict(X_new)
```

Linux 环境可安装 Noto Sans CJK SC 字体以显示中文图注。常规 `dimensr-mimo fit` 继续使用其已有的数据划分和选模配置，其进度日志位于输出模型旁的 `*.progress.log`。[常规拟合命令](../README.md#命令行)
