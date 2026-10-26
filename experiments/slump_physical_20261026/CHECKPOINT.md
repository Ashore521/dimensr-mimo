# 当前检查点

- 正式goal active、无token预算，目标未完成；用户要求按顺序自主推进，不重复审批。不得再次清空，不派子代理，不更新memory。
- 已发布API1.1.0=a28bbd7（85测试、6/6等价/物理验证、wheel、四平台CI成功）；含噪声/新族=7575874（36/36技术核验，推荐失败，23/24保护）。当前主包仍1.1.0，默认legacy/weighted。
- 本轮Slump 9/9返回，3/9完整技术核验（仅独立），9/9交付补充核验。六个联合模型NSGA-II新评价0；所有9系统均未通过精度及完整物理结构门槛，dimensional仅5/9保护。完整证据和失败均保留，不重新标为成功。
- 根因：初始化约31–33秒，神经第二轮消耗全部52秒，按评价数的selection_budget_fraction未给墙钟预留；NSGA-II只有空事件。独立每输出20秒均完成，机制有效。
- 下一步计划已在experiments/phase_budget_20261027/PLAN.md登记，写入先于Slump测试评分，仅根据阶段运行事件。需要实现可选phased时间调度、保持global默认，阶段1占1-selection_budget_fraction；剩余epoch分配剩余时间，神经50%/GP85%/更新尾段，NSGA-II保留总墙钟尾段。保持52/60秒和1200评价。
- 下一轮6个新联合fit：legacy/dimensional×801/802/803；本轮6个global联合和3独立作为历史对照，不必重跑。所有数据、划分、单位、变换、常数目录、精度/结构门槛完全不变。先测试、冻结、新目录运行，再统一评分。阶段恢复不能替代精度达标。
- 当前没有运行中的实验。Slump数据位于outputs/slump_physical_20261026的development.npz和sealed.npz，加载函数在data.py；测试只有该批ALL_MODELS_FROZEN.json存在才放行。后续拟合仍仅调用train/val。
- 数据官方哈希、历史行号顺序和单位已核对；当前真实日期已到2026-10-27，目录仍保留启动日期。第三方源为UCI CC BY4.0；不发布原始数组或其他私人材料。
- 后续仍需解决阻尼/有理复合项发现，以及任意缩放情况下物理图的全池结构优选；当前precision只按搜索坐标图选择，PhysicalModel独立检查所选图。
