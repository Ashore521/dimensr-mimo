# 当前检查点

- 正式goal active、无token预算；用户要求按顺序自主推进。不要再次清空，不派子代理，不更新memory。
- 已发布1.2.0=4b9d14d，CI40363300727成功；1.2.1=6ca205094ffeb3db3cfadaa9d8bab94e6379b6a4，CI40367279747成功。
- 当前1.3.0新增reference_budget=expanded可选配置，token默认保持。引用实际展开size/depth统一用于cost/mask/递归预算/trace，缓存键包含结构。硬数值域等条件不变。
- 本批六系统完成封存/评分/独立核验。技术6/6、物理6/6、110/110神经提案结构合法、18/18输出在10%+1e-8保护界内；0/6完整精度/结构达标。平均测试NRMSE0.69622→0.67975。dimensional/801第一输出仍变差约9.6%，明确保留；不能称零退化或工程通过。
- 98测试通过，构建/隔离wheel通过，sdist13项相关回归通过。冻结至主核验340.26秒。当前无实验运行。
- 下一项experiments/compound_discovery_20261027/PLAN.md已登记：只用训练侧诊断阻尼/有理复合项词汇及筛选缺口，不能注入目标公式或真实系数；诊断后冻结明确矩阵/预算与消融，再运行。用户现有授权持续有效，不重复问审批。
- 后续仍需原物理坐标全候选池复杂度选优。总目标未完成，不能仅因本轮机制全部通过而complete。
- 所有历史失败、源码、21个历史对照保留。原dimensr HEAD仍7fbcefadd105fe6266c8149cc05420de8ca74a76；原始Slump数组位于outputs/slump_physical_20261026。没有公开原始数组或私人Word/PDF。
- Git日志/PROFILE.txt有工具原生末尾空行，保留原始证据；diff --check使用core.whitespace含cr-at-eol，排除原生日志，不输出百万行CRLF警告。提交前核对FREEZE和ALL_MODELS_FROZEN中的Git blob哈希。
