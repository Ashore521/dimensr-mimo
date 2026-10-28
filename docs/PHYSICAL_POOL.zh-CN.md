# 1.5.0 全候选池物理坐标选择

原`PhysicalModel.from_estimator`继续导出estimator已经选中的图。新入口`select_physical`遍历冻结候选池，用原物理变量恢复每张图，重新核算精度和复杂度，再选择一个完整达标系统；不改变原estimator、不重新拟合参数。

```python
import json
from dimensr_mimo import select_physical

physical, report = select_physical(
    model,
    coordinates=coordinates,             # 拟合时使用的AffineCoordinates；省略为恒等变换
    input_bounds=physical_bounds,
    physical_limits=physical_limits,      # 预先固定的逐输出NRMSE及四个结构门槛
    training_data=(X_train, Y_train),     # 原物理坐标中的原训练观测
    validation_data=(X_val, Y_val),        # 原验证观测
)
with open('physical-selection.json', 'w', encoding='utf-8') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
if report['status'] == 'accepted':
    physical.save('physical-selected.json')
```

新拟合模型记录训练/验证数组的形状和规范float64字节哈希，重选时先变换回搜索坐标核验。旧模型没有这些哈希时，检查行数、训练归一化及全部候选的存储误差（rtol1e-7、atol1e-10）；报告明确这种一致性不等于原数组字节身份。错误数据、漂移候选池或不匹配坐标会拒绝处理。

完整可行集必须同时通过量纲、有限数值域、绝对训练/验证精度、物理结构、原训练/验证参照约束和原weighted锚点的逐输出验证保护界。排序依次为共享图规模、独立参数数、展开节点、最大输出深度、验证平均误差及原池索引。所有固定坐标参数均计入物理结构。

无完整可行候选时状态为`not_met`；合法回退优先减少相对约束违反量与验证平均误差，可能比原图更大，也不保证测试误差改善。全部物理图无效时模型为`None`，报告保留所有失败原因。`accepted`也只表示预定训练/验证门槛通过，不是测试精度或工程应用保证。

[24系统历史回归](../experiments/physical_pool_20261027/REPORT.zh-CN.md)核验3,636个候选，24/24技术通过，完整接受仍为10/24，输出保护53/54，性能推荐失败。搜索和交付默认均未变；本接口补齐原物理坐标中的全池选择能力，不宣称新增发现或普遍性能优势。
