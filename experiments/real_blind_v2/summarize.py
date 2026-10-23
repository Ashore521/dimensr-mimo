"""Render the already-frozen results; this script does not read test arrays."""
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
results = json.loads((STUDY / "holdout/results.json").read_text(encoding="utf-8"))
freeze = json.loads((STUDY / "FREEZE.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "experiments/real_blind_v1/data/manifest.json").read_text(encoding="utf-8"))
assert results["complete"] and all(r["status"] == "ok" for r in results["runs"])
names = {"concrete_slump": "混凝土坍落度", "parkinsons": "帕金森远程监测", "air_quality": "空气质量"}
roles = ["baseline", "selected", "ridge", "extra_trees", "mean"]
summary = {}
for dataset in names:
    summary[dataset] = {role: st.mean(r["test"]["mean_nrmse"] for r in results["runs"]
                                    if r["dataset"] == dataset and r["role"] == role) for role in roles}
overall = {role: st.mean(s[role] for s in summary.values()) for role in roles}
improvement = 100*(1-overall["selected"]/overall["baseline"])
tables = ["| 数据集 | 原版 | 正则改进 | Ridge | Extra Trees | 均值预测 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
for dataset, s in summary.items():
    tables.append("| " + names[dataset] + " | " + " | ".join(f"{s[k]:.5f}" for k in roles) + " |")
tables.append("| 数据集等权平均 | " + " | ".join(f"{overall[k]:.5f}" for k in roles) + " |")
intervals = ["| 数据集 | 相对误差下降 | 改进版减原版 NRMSE 的 95% 区间 | 重采样组数 |", "| --- | ---: | --- | ---: |"]
details = ["| 数据集 / 输出 | 原版 RMSE | 改进 RMSE | 改进 NRMSE | 改进 R² |", "| --- | ---: | ---: | ---: | ---: |"]
cost = ["| 数据集 | 原版训练秒数 | 改进训练秒数 | 原版复杂度 | 改进复杂度 |", "| --- | ---: | ---: | ---: | ---: |"]
for dataset, s in summary.items():
    ci = results["paired_intervals"][dataset]
    lo, hi = ci["selected_minus_baseline_nrmse_95ci"]
    intervals.append(f"| {names[dataset]} | {100*(1-s['selected']/s['baseline']):.2f}% | [{lo:.5f}, {hi:.5f}] | {ci['groups']} |")
    before = [r for r in results["runs"] if r["dataset"] == dataset and r["role"] == "baseline"]
    after = [r for r in results["runs"] if r["dataset"] == dataset and r["role"] == "selected"]
    for j, target in enumerate(manifest["datasets"][dataset]["target_names"]):
        values = [st.mean(r["test"]["rmse"][j] for r in before)] + [st.mean(r["test"][k][j] for r in after) for k in ("rmse", "nrmse", "r2")]
        details.append(f"| {names[dataset]} / {target} | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    cost.append(f"| {names[dataset]} | {st.mean(r['fit_seconds'] for r in before):.3f} | {st.mean(r['fit_seconds'] for r in after):.3f} | {st.mean(r['complexity'] for r in before):.1f} | {st.mean(r['complexity'] for r in after):.1f} |")

text = f"""# 真实数据封存测试与算法优化

在三个 UCI 真实观测数据集上，轻度正则输出头的测试集等权平均 NRMSE 从 **{overall['baseline']:.5f} 降至 {overall['selected']:.5f}**，相对降低 **{improvement:.2f}%**。三个数据集的均值均有小幅下降，但三个配对 95% 区间都跨零，不能据此宣称稳定优势。帕金森跨受试者预测仍弱于均值基线，空气质量仍落后于 Extra Trees。

这是公开数据上的**内部封存留出测试**，不是独立第三方盲测或外部预注册试验。没有将这些数据称为首次公开或预训练未知数据。数据准备者只按协议整理与封存，算法开发只使用训练/验证集；测试打开后不再改算法或参数。

## 预先固定的方法

详见[协议](../experiments/real_blind_v1/PROTOCOL.md)、[来源与清洗](../experiments/real_blind_v1/DATASETS.md)及[数据清单](../experiments/real_blind_v1/data/manifest.json)。混凝土按样本随机划分；帕金森按受试者互斥划分；空气质量按时间先后划分。训练/验证/测试分别为 61/21/21、3514/1019/1342、4164/1388/1389。大训练集预先固定抽取 1200 行，验证和测试不抽样。全部方法使用同一批训练行，输入标准化和输出归一尺度只从训练行估计。

每个符号版本使用种子 0、1：max_features=192、generations=3、proposals_per_generation=128、max_terms=8、polish_max_nfev=10，单进程 BLAS 一线程，最多两个进程并发，单次超时 240 秒。训练计时包括拟合、表达式形成和保存，受并发及本机环境影响。相同搜索设置不等于相同 FLOP 预算。Ridge(alpha=1) 与 Extra Trees(200 棵树、叶最少 2 样本)作为预测参照；后者也先按训练输出标准差归一目标，预测再还原。它们不是同预算符号回归对手。

本次只有一个优化候选：`head_ridge=0.001`，在列尺度归一的线性头上施加 Ridge，截距不惩罚。基线为 v0.1.0；所有源代码、配置、预算、软件版本、失败记录和模型均保存在独立目录，原 DimenSR 与合成实验未改动。默认 `head_ridge=0` 保留原来的 OLS 行为；这次增益不足以支持对所有任务更改默认值。

六次原版验证平均 NRMSE 为 {freeze['validation_scores']['baseline']:.8f}，六次改进版为 {freeze['validation_scores']['selected']:.8f}，下降 1.84%，满足事前固定的“六次成功且平均改善至少 1%”门槛，随后停止调参。没有利用最终测试来选择种子或每数据集单独的版本。

## 封存与最终结果

模型冻结时间：`{freeze['frozen_at_utc']}`；首次打开测试：`{results['opened_at_utc']}`。训练源代码提交为 `{freeze['git_head']}`，冻结清单单独保存在 Git 提交 `bfe3b06`。[FREEZE.json](../experiments/real_blind_v2/FREEZE.json)记录 24 个模型和 66 个输入文件的哈希，测试文件哈希来自准备阶段。[完整结果](../experiments/real_blind_v2/holdout/results.json)保留每个种子、每个输出的误差、R²、计时、复杂度和扩展域审计。

NRMSE = 测试 RMSE / **训练**输出标准差；先对输出取平均，再对两个种子取平均，最终对数据集等权平均。值越低越好。R²使用测试自身总离差，与该 NRMSE 不能简单互换。

{chr(10).join(tables)}

24 次拟合和 24 个冻结模型的测试评估均完成，无超时或失败。全部 12 个符号模型在覆盖训练与测试输入的扩展包围盒上得到 `enclosed`；这只是数值区间筛查，不是形式化证明。测试中位于训练包围盒外的行数：混凝土 6、帕金森 69、空气质量 496；域检查通过不代表预测准确。

### 配对不确定性

1000 次配对组 bootstrap，种子 20261023：混凝土按行，帕金森按患者，空气质量按日重采样；两种算法对同一批抽样计算误差，先跨种子/输出平均。负差值支持改进。区间只反映该固定切分下的组抽样不确定性，不包含重新划分数据的变异；空气质量跨日相关仍可能存在。

{chr(10).join(intervals)}

### 每输出表现与运行开销

以下为两个种子的均值，RMSE 保留原始输出单位。归一坐标不改变预测输出单位。

{chr(10).join(details)}

{chr(10).join(cost)}

## 使用与复现

启用这个可选改进：`SearchConfig(head_ridge=0.001)`。自己提供按组或按时间隔离的 `X_val/Y_val`，通过验证集选择强度。默认零适合继续复现 v0.1.0 的输出头行为；其余配置也要保持相同。模型 JSON 会保存此参数，旧 JSON 仍可加载。

本研究的 `model.json` 期望**已标准化输入**，列名沿用原数据标签，但公式中的变量代表 z=(x-mean)/scale，不能直接解释成原始单位物理定律。均值/尺度保存在同目录 `result.json`；在原始坐标中使用公式时，把每个显示变量替换为该 z 表达式。推理方式：

```python
import json
import numpy as np
from dimensr_mimo import MIMOSymbolicRegressor

directory = "experiments/real_blind_v2/development/concrete_slump_symbolic_s0"
with open(directory + "/result.json", encoding="utf-8") as f:
    receipt = json.load(f)
model = MIMOSymbolicRegressor.load(directory + "/model.json")
# X_raw must follow receipt/model feature order.
Y_pred = model.predict((X_raw - np.asarray(receipt["input_mean"])) / np.asarray(receipt["input_scale"]))
```

先安装仓库开发依赖，再运行 `python experiments/real_blind_v1/prepare_data.py` 获取带哈希核验的数据。旧版通过 `git archive v0.1.0 src/dimensr_mimo`提取到独立目录；`run_batch.py --source ... --output 新目录 --comparators`运行开发集，改进版另传 `--config experiments/real_blind_v2/config.json`。复现实验应使用新的结果目录和新的冻结清单，不能覆盖已发布记录。冻结记录来自当时 Windows/Python 3.12 环境，精确版本位于各 result.json；源文件换行符、运行时间和 pickle 序列化也可能使新运行字节哈希不同。

`freeze_models.py --baseline 原版开发目录 --selected 改进开发目录 --output 新FREEZE.json`核验选择门槛并冻结；随后`evaluate_holdout.py --freeze 新FREEZE.json --output 新测试目录`评估。首次冻结文件中的拟合树二进制只留本地，因此历史冻结清单不能直接在缺少这些文件的新克隆中执行；先重新训练并新建清单。不要把本轮已公开的测试成绩继续当作盲测调参依据。

## 证据边界

混凝土测试只有 21 个样本且未按生产批次隔离；帕金森只有 9 个测试患者，UPDRS 是插值评分；空气质量仅单站点、存在传感器漂移、缺失筛选和时间相依，且官方新旧许可说明冲突。原始与处理后的数值数据、拟合树二进制只留本地，仓库公开脚本、来源、哈希、符号模型和汇总结果。没有进行真实物理量纲恢复、因果识别、临床验证、上游方法同预算对比或跨站点外部验证。

完整回归测试 **70 项通过**，共享系统示例通过；新增用例覆盖共线噪声下的系数稳定性、截距不惩罚、非法参数、保存加载和验证标签隔离。测试揭盲后只整理报告与使用说明，没有更改训练代码、配置或模型。
"""
(ROOT / "docs/REAL_DATA_BENCHMARK.md").write_text(text, encoding="utf-8")
(STUDY / "summary.json").write_text(json.dumps(dict(datasets=summary, overall=overall,
    relative_improvement_percent=improvement, paired_intervals=results["paired_intervals"]), indent=2), encoding="utf-8")
print(json.dumps(dict(overall=overall, relative_improvement_percent=improvement)))
