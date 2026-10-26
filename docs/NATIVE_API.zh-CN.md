# 原生搜索配置与物理坐标交付

1.1.0 将已验证的研究候选接入正式 Python 包。`initialization='dimensional'` 使用平衡的量纲种子和合法系数归一化的结构组合；`final_selection='precision'` 在训练结束、候选池冻结和验证评分后执行绝对精度与结构验收。默认仍为 `legacy` 初始化和 `weighted` 选择。

## 显式配置

```python
from dimensr_mimo import SearchConfig, DimenSRMIMORegressor

limits = dict(
    nrmse=[1e-4, 1e-4],
    shared_graph_size=24,
    independent_parameters=6,
    expanded_total_nodes=30,
    maximum_output_depth=6,
)
config = SearchConfig(
    initialization="dimensional",
    final_selection="precision",
    acceptance=limits,
    random_state=601,
)
model = DimenSRMIMORegressor(config, feature_units, target_units, constant_units)
model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, input_bounds=bounds)
print(model.post_selection_["status"])
```

每个输出分别约束训练和验证 NRMSE，分母为训练标准差。候选还须满足训练/验证的逐输出退化约束以及相对原 weighted 选择的验证保护界。只有同时通过全部门槛的候选才能按共享图规模、独立参数数、展开节点数、深度的顺序排序。没有可行解时保留 guarded 回退，状态明确为 `not_met`；这仍可预测，但没有通过声明的验收。

`final_selection='guarded'` 单独启用验证保护选择；只有 `precision` 接受 `acceptance`。所有精度和结构门槛必须在拟合前完整给定；输出数量不匹配、禁用量纲或逐输出约束时立即拒绝 precision 搜索。保存、重载和再保存保留验收记录；原生 guarded/precision 模型重载时会从冻结池重放选择，并核对结果。

量纲向量声明允许运算的维度，不能替代常数量纲目录。例如对长度输入构造 `sin(k*x)`，目录中必须有逆长度参数；不会把带量纲输入直接送入三角函数。`dimensional` 改变通用初始化覆盖，不增加初始化名额、搜索评价上限或常数拟合预算。

## 缩放和原物理变量

完整可运行示例见 [physical_coordinates.py](../examples/physical_coordinates.py)。示例用训练数组建立坐标，显式提供验证数组，拟合后还原到原始物理量：

```python
from dimensr_mimo import AffineCoordinates, PhysicalModel

coordinates = AffineCoordinates.from_training(X_train, Y_train)
model.fit(
    coordinates.inputs(X_train), coordinates.outputs(Y_train),
    X_val=coordinates.inputs(X_val), Y_val=coordinates.outputs(Y_val),
    input_bounds=coordinates.input_bounds(physical_bounds),
)
physical = PhysicalModel.from_estimator(
    model, coordinates, physical_bounds, physical_limits,
    training_data=(X_train, Y_train), validation_data=(X_val, Y_val),
)
physical.save("physical.json")
loaded = PhysicalModel.load("physical.json")
prediction = loaded.predict(X_physical)
```

这里的 scale 是无量纲的数值伸缩，变换后的坐标保留用户声明的量纲；offset 携带对应输入或输出单位。此约定与把标准化坐标定义为无量纲变量不同。数值单位转换须先统一数据和量纲声明，不由这个接口猜测。坐标统计只使用传入的训练数据；调用者应先完成训练/验证拆分。

还原直接在原始 DAG 中代入 `(x-offset)/scale` 和 `scale*y+offset`，不重拟合、不做代数化简，原始除法、log、sqrt 及非法中间值传播保留。预测拒绝声明域以外的输入。适用域须通过有限数值筛查；`enclosed` 不是连续域形式证明。公式字符串仅供阅读，复算应使用 JSON 中原始节点、参数编号和根引用。

交付包括：

- 原物理变量公式、完整 DAG、显式参数身份和 SI 量纲；
- 拟合参数与固定坐标参数的角色、各参数使用的输出；
- 搜索发现的共享与坐标代换产生的共享，分别记录；
- 源图和完整物理图两套复杂度、物理适用域及数值筛查；
- 搜索坐标验收和物理坐标验收，分别记录。

固定坐标参数也计入物理图复杂度。`physical_limits` 应在拟合前另行冻结；搜索坐标达标不能替代物理图达标。非恒等坐标若未提供原物理训练/验证数组，精度状态为 `not_assessed`，不会仅凭理论缩放不变性认定通过；已知结构不达标仍记为 `not_met`。提供数组时直接用还原图复算，记录样本数和数据哈希。模型保存的误差记录可供核对，外部复验仍需原数组。

当前 precision 选择按搜索坐标图复杂度遍历候选；物理导出检查所选图，尚未按任意缩放后的物理图复杂度重新遍历整个池。因此本接口没有承诺得到全池最简单的物理公式。验收范围为提供的训练/验证样本，不等于未知数据或工程应用保证。

## 本批验证

`experiments/native_api_20261026` 保存改动前源码、修正后的参考捕获、正式运行源码、配置及数据哈希、过程事件和独立复算。最初三个 normalized 基线因捕获脚本遗漏 balanced 调度被保留为无效参考；仅使用 `baseline_corrected` 中与第七轮完整池吻合的三个参考。两个 legacy 基线有效。

正式矩阵为五个行为等价重跑和一个物理坐标示例；单工作进程、数值库单线程、每单元外部 60 秒、fit 内部 52 秒、整个矩阵 400 秒上限。六单元全部完成，运行总耗时117.94秒，6/6独立核验通过。五个重跑的搜索指纹、候选池、去计时阶段事件及最终选择均与对应参考一致。

物理示例的训练/验证逐输出NRMSE分别约5.30e-15和4.47e-15，搜索与物理验收均通过。源图size17、独立拟合参数5；物理图size31、完整参数13，其中8个为固定坐标参数；展开节点42、深度8。该例展示还原与计量正确，不能用源图的更小复杂度代替物理交付开销。

全套85测试通过，sdist/wheel构建通过。在隔离Python进程中直接从wheel导入、禁止导入`experiments`，六模型加载及物理预测通过。所有案例均为已知开发回归，不新增盲测或性能优势结论。详细结果见[本批独立核验](../experiments/native_api_20261026/VERIFICATION.json)。
