# 已观察开发矩阵：接续独立核验修正后的历史证据

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-31
- Verification Status: UNVERIFIED
- Version Label: completion_development_verified_v1

## 为什么使用新目录

原completion_historical_20261031正式核验因三个未列入白名单的history调度时间字段而失败，原false及其全部冻结输入永久保留。其独立后审证实仅这三字段差异、保存截止裕量为正及其他保存轨迹一致。completion_verification_repair_20261031按独立预审协议，在原历史截止内仅对相同34模型做条件更严格的只读核验：34/34技术和历史采用门槛通过，原模型/score/选择未改；该新核验仍需独立后审被主代理接收。

原completion_development_20261031的脚本/计划已被历史FREEZE绑定，且从未启动，不能原地改gate。这个新目录承接其尚未使用的一次32系统预算；不增加任务、种子、拟合名额或另开一个额外开发重复。原准备目录保持未运行。

## 完全保持的比较

科学设计逐项继承 ../completion_development_20261031/PLAN.md：旧new_task_validation四族×两噪声层×种子9131/9132，共16对/32系统、32配对输出；训练/验证/测试96/48/192，误差与噪声仅按训练尺度。该矩阵已经观察，只称开发回归。

同一41源CANDIDATE_LOCK、同一配置、同一冻结case/spec/data摘要和奇偶交错臂顺序。base为precision_repair=True/compositional_completion=False，candidate相反；均完整compound神经/GP/NSGA-II/联合拟合、原closure24+24与precision选择，原foundation/父保护/base训练验证双保护。无算法修改、真式注入或按已见测试调参。

唯一尚未使用的原development单批2880秒BUDGET（含600核验预留）、52秒合作fit/60秒系统外监督、1200评价、单线程串行、原60秒preflight启动窗口；不借预算、不重跑失败、不补单元。全模型/失败封存后统一测试评分，非法/缺失/重复/非有限保留固定完整分母，缺失均值null。

完整技术要求、16/16 foundation和旧repair control身份、matching每臂new/valid/protected/validation_retained>0、候选16/16完整接受、32/32输出≤base×1.1+1e-8、原七接受保留、宏干净误差与平均图不升，全部保留。无噪声三split≤1e-4；1%噪声观测≤.03且干净≤.02；图≤24、参数≤6、展开≤30、深度≤6；原单位/数值域、Rademacher、神经/GP/至少两NSGA新评价、native/physical重载和全池核验不变。

## 新接续gate与身份规则

新gate不改原历史false，而同时绑定两条证据链：

1. 原历史FREEZE/ALL/SCORES/VERIFICATION/SUMMARY/EXECUTION及独立后审/接收原失败的摘要必须与隔离核验FREEZE.original_sha256逐值相同；原technical/adoption仍false、VERIFICATION仍failed。原966输入与374工件/归档按原hash可核，候选源码锁与配置保持原值。
2. 隔离核验FREEZE、VERIFICATION、SUMMARY、EXECUTION、FROZEN_INPUT_ARCHIVE及独立POST_RUN_REVIEW必须被一个新接收收据精确绑定，且新verification 34/34、status=passed、budget_met/inputs_unchanged严格True，新summary technical/adoption严格True、original_verdict=failed及verification_of_sealed_results严格True；新execution正常退出、无终止、verification_passed/no_training_performed严格True。核新verified_unix不晚于新deadline，新deadline不超过原历史BUDGET起点+2880或新起点+600，并核原失败链与新冻结输入未漂移。
3. 共享CANDIDATE_LOCK完整src集合/摘要、原历史FREEZE.candidate_lock_sha256及新接收收据锁摘要一致；基础config与原历史/原计划相同。保留原准备开发PLAN摘要与锁内该计划摘要一致，证明科学任务没有替换。

接收收据只由主代理在隔离核验独立后审被接收后写入 ../completion_verification_repair_20261031/POST_REVIEW_ACCEPTANCE.json：status必须为accepted_for_sealed_historical_correction；sha256键集合精确为FREEZE.json、VERIFICATION.json、SUMMARY.json、EXECUTION.json、FROZEN_INPUT_ARCHIVE.json、POST_RUN_REVIEW.md和CANDIDATE_LOCK.json。前六值来自修正目录，最后来自共享源锁。其他说明可在顶层，但不能缺键/多键/放宽类型。原历史目录绝不创建准入POST_REVIEW_ACCEPTANCE。

本批旧control比较使用已冻结修正的SealedVerifier，其余验证逻辑继承原Verifier。仅在control和对应旧repair均满足完整epoch/神经全提案/GP条目、有限合法时间、正截止裕量和无global deadline时，分类model.history phase_budget的三个时间字段。其他配置/轨迹/候选/cache/参数/数值/选择/score及全旧表分母仍严格相同；任何截止命中或其他漂移按失败。不能一般声称计时不影响算法或完整失败cache/RNG状态相同。

## 实现、冻结与交付

本目录新增最小stage gate、四个薄入口、协议及独立构造测试，复用原harness的监督/训练/评分/失败处理。freeze_inputs保留原依赖闭包，只以新修正接收链替代原历史目录中永不生成的POST_REVIEW_ACCEPTANCE；不能删其他原失败证据。补入修正脚本/协议/测试/日志/审查/冻结及归档、原历史输入/运行文件和新gate/test/审查证据。所有旧文件只读。

先构造测试、独立harness窄审、接收修正后审、锁与原字节归档预检，再执行一次。交付完整freeze/源副本/输入归档/逐cell/ALL/SCORES/VERIFICATION/SUMMARY/EXECUTION/REPORT及独立后审。入口预定 `.venv/Scripts/python.exe experiments/completion_development_verified_20261031/run_bounded.py`。

当前仅方案，未实现/未测试/未正式启动。本批即使通过仍不是盲验证；之后仍需新的锁定候选新任务验证、打包/安装态核验、最终审查与新发布授权。生产默认不变，1.7.0未发布。
