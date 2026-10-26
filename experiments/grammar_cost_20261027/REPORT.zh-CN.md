# 精确语法提速历史回归

## Material Passport

模式run/validate。版本1.2.1；六新增系统，十五历史系统保留。主核验6/6、交付补充核验6/6；完整精度/结构验收0/6、相对上一轮输出保护14/18。完整报告见[结果解释](../../docs/GRAMMAR_COST.zh-CN.md)。

profile_run.py只读训练数据；PROFILE.txt包含instrumented耗时，不能当提速基准。第一候选及其EQUIVALENCE.json保留，其LRU改动未采用。第二候选EQUIVALENCE_V2.json为ABBA无profile计时，种子和提案逐字节一致。正式批次按study.py freeze、study.py train、evaluate.py、verify_artifacts.py顺序执行，拒绝覆盖既有模型；FREEZE.json、CELL_FROZEN.json、ALL_MODELS_FROZEN.json、SCORES.json与VERIFICATION.json形成哈希链。

复现须使用本批source快照及依赖版本，原始UCI缓存不公开提交。工作负载比对可从candidate_v1/grammar.py恢复第一候选；其历史脚本中的LRU清理接口仅属于第一候选，不应对当前包运行。所有历史失败和源码保留。

统计边界：三随机种子不是三个独立数据集；这些配对仅用于固定历史任务的描述比较，不作显著性或总体优势推断。保留完整分母、逐输出退化和not_met。无失败补跑、事后门槛变更或验证集进入后续搜索；有墙钟截断使语义等价的提速产生不同搜索路径。检查选择偏差、重复观测、多重比较、绝对与相对变化、因果外推、随机不确定性、遗漏失败、数据泄漏、过拟合、数值合法性和应用边界；没有新盲评证据。
