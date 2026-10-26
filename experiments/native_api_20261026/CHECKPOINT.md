# 当前检查点

- 正式goal仍active，无token预算；本批完成不能替代总目标达成。
- 原生API及物理坐标交付已完成：版本1.1.0，显式dimensional/guarded/precision配置，默认legacy/weighted不变；原物理图、固定/拟合参数角色、单位、共享、域、完整复杂度、独立误差复算与保存重载。
- 85测试通过，六运行单元117.94秒、6/6核验通过，五个搜索与池等价；sdist/wheel通过，隔离wheel禁止experiments导入后六模型重载通过。
- 初始normalized捕获器漏Observation调度，三结果保留为无效参考。baseline_corrected已核对与第七轮601对应normalized完整指纹/池/变异计数完全一致。
- affine_lengths真实fit：搜索与物理验收accepted；原物理NRMSE约5.30e-15/4.47e-15；源图size17、参数5，物理图size31、参数13(含8固定)、展开42、深度8。不是新盲测。
- 冻结后没有更改src；源码副本在source/src；脚本和数据哈希见FREEZE.json；模型冻结和独立核验均通过。当前没有仍在运行的实验进程。
- 当前precision只按搜索坐标图遍历选择，物理导出检查所选图；尚未实现任意缩放下全池物理图复杂度最小化。真实工程容差也尚未确定。
- 下一步按顺序自主推进：含噪声和新公式族分层矩阵（新目录、单进程单线程、有限总预算、先冻结后评分）；随后单位明确的实际数据验证。保留全部历史失败，不重复清空、不更改原DimenSR仓库。
- 原DimenSR已实时确认干净，HEAD=7fbcefadd105fe6266c8149cc05420de8ca74a76。
- 已找到GitHub connector fetch actions/runs，可直接查询push运行；9a30c4d、41a7db7、13471be均success。查询head_sha必须用完整SHA，短SHA会返回空；本批提交后需再核查自身CI。
- 不派子代理，不更新memory；已应用ponytail和ARS实验工作流。实验必须新目录，私人Word/PDF不可发布。
