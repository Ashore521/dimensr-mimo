# 冻结前配置预检记录

2026-10-28，预检调用 SearchConfig.validate 时发现沿用的基础配置 wall_seconds=None，phased 分配要求显式时限。历史 worker 会在 fit 前设置52秒，本轮协议同样固定52秒；已将该值提前写入冻结配置，并增加配置可验证测试。未改正式实验预算或算法。

预检失败后一次冻结命令因 PREFLIGHT.json 不存在退出；该调用未进入训练。此时没有 BUDGET.json、FREEZE.json、TRAINING_STARTED.json 或 runs。记录后重新执行预检，成功后才允许首次冻结和运行。此问题属于准备阶段，不是失败实验补跑。
