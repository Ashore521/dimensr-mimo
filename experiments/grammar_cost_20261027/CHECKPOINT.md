# 当前检查点

- 正式goal active、无token预算。按用户授权自主推进，不再次清空、不派子代理、不更新memory。
- 1.2.0=4b9d14d734dacf82353d4636305c9c63da5c2dc4，已推送，Actions40363300727成功。其六phased单元完整机制2/6，精度0/6，保护16/18。
- 当前1.2.1精确整数量纲计算和实例内特征缓存：90测试、构建、sdist五项针对性回归、隔离wheel通过。无墙钟搜索完全一致；ABBA工作负载提速23.7%。第一LRU候选只提速约2.3%，保留但未采用。
- 当前批已封存评分：6/6完整机制和物理复算，0/6完整精度/结构，14/18输出保护，平均测试NRMSE0.64876→0.69622。所有负结果保留；默认global/legacy/weighted不变。
- 原始Slump数组仍在outputs/slump_physical_20261026；所有历史模型和对照保留。原dimensr HEAD仍7fbcefadd105fe6266c8149cc05420de8ca74a76，干净。当前无实验运行。
- 下一项已在experiments/reference_budget_20261027/PLAN.md登记：可选展开引用预算，让生成mask/cost/trace和最终in_bounds一致；必须新冻结批次，不能改当前冻结源码或复用结果。
- 后续必须处理阻尼/有理复合项发现，以及全池物理坐标结构优选，不能把本轮机制恢复当总目标完成。
- 发布推送曾被自动审批以private标记误判拒绝。已只读核验：128新增文件没有模型/日志private标记，唯一命中是“不发布私人资料”的文本；全部数据为UCI CC BY4.0，官方ZIP哈希一致，origin正确且GitHub连接器确认权限。相同推送重审后成功，未绕过审批。证据在outputs/publication_review_4b9d14d.json。
