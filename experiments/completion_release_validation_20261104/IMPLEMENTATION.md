# 新输出组合stage实现交付

本目录gate/study/worker/evaluate/run_bounded/test_stage及本说明由harness工作包写入。PLAN/cases/test_cases/审计/history_sources和设计独审属于其他作者，均只读；所有src、根tests、旧stage、共享harness/verification及旧冻结证据未修改。

## 接线与最小适配

study手动绑定隔离的新cases和原first_round supervisor，复用原configuration/matrix/mechanism/worker/train/bounded。新preflight仅为新训练验证规范记录摘要和任务/源码锁绑定，不比较不存在的同任务旧数据。gate核本地/共享src集合、parent锁和config，以及前置开发八键接收和完整技术/预算/封存链；实际科学adoption bool独立保留，可False。缺前置是pending_missing_prior，不能启动子进程。

NewTaskVerifier继承已审Verifier的原始JSON、模型/物理重载、全池和补充/closure回放，只明确覆盖verify/summarize：删除历史control调用/成功字段，以本批foundation完整payload/count/hash/evaluation/cache指纹为身份依据；严格bool/数值/计数和32全分母保留。没有通过control_identity恒True来伪造历史对照，显式调用反而拒绝。

新增任务无特殊damped/rational目标门槛，保持PLAN指定16/16接受、32保护、control保留、macro误差/平均图不升。用户允许的opt-in软件发布与科学adoption分开；本工作包不作发布决定。

只读初审发现额外的54来源live校验会阻断前置开发PLAN后续工程说明，其history_sources原快照仍完全匹配。主代理明确采用审计归档版本口径，因此移除这项额外live约束，继续强制54归档原字节、13冻结注册表及41生产src；新增构造证明合法live草案变化可用、归档/冻结注册表变更拒绝。保留plan_drift_01.log和首次final_audit_01.log，不改原NOVELTY/DESIGN；SOURCE_VERSION_NOTE由主代理提供并进入冻结闭包。这没有更改任务、算法、阈值或预算。

主代理随后提供前置stage的archive_io.py，适配旧冻结依赖中真实超长路径。本stage直接复用其disk_path/sha/verify_files/copy_original，并只在本地显式重现freeze的构建段以替换存在性/摘要/复制I/O。资源字段、顺序、逻辑key、config和数据摘要与原合同相同；共享harness、sha和注册表不patch，旧依赖不删。

## 实际构造测试

项目解释器`.venv/Scripts/python.exe`，仅运行本目录test_stage.py；未重跑main已完成的6项实际train/validation case构造检查。精确命令和依赖版本见IMPLEMENTATION_CHECKS.json。

```powershell
.venv/Scripts/python.exe -m pytest experiments/completion_release_validation_20261104/test_stage.py --basetemp=experiments/completion_release_validation_20261104/harness_validation/p1 -q
.venv/Scripts/python.exe -m pytest experiments/completion_release_validation_20261104/test_stage.py --basetemp=experiments/completion_release_validation_20261104/harness_validation/p2 -q
```

首轮39 passed in 8.77s；补入seal/绑定与pending外入口构造后46 passed in 9.39s。新增长路径构造的第三轮47通过/1失败：测试预期旧helper的RuntimeError，但主代理稳定helper保留原ValueError；已仅修正本目录测试预期，第三轮失败日志保留。最终结果见IMPLEMENTATION_CHECKS.json。

局部long-path适配及fake-freeze逻辑key检查的最终第四轮为48 passed in 9.22s；第三轮失败未删除。测试均为同一新harness集合的逐步补充，不把39/46/47/48相加计为独立覆盖数。

按主代理明确的历史快照口径补充三项来源版本构造后，第五轮51 passed in 9.15s，作为当前最终覆盖。仅本目录gate/构造检查按该口径修正，原案例/审计/设计报告未改；首轮交付audit的live PLAN漂移错误及差异日志继续保留。

覆盖前置科学False/True都按实际通过技术接收，缺件/多键/摘要/严格类型/数量/截止/额外attempt/CELL时间/归档/source/parent锁/config漂移拒绝；source/cases/原verifier隔离绑定；无历史control声明；32完整分母和全部科学门槛；verify不会调用历史control；fake96/48训练验证preflight及绑定；原依赖/13注册表/历史源冻结；pending在进程启动前阻断；当前CELL时间/成员和preflight绑定在test访问之前核验。

JSON receipt、假模型、固定fake数组和绕过native重算的路由stub仅用于构造检查；没有调用实际新任务采样、initializer、fit/search或正式test评分。stub通过不等于对新模型数值/物理结果已验收。既有原Verifier数值实现复用，未来正式批仍须全量原生核验与独立后审。

## 只读交付审计

harness_validation/audit_current.py读取当前锁/审计/前置状态、检查源与原档、清点冻结闭包和正式工件、提取本轮实际模型记录，输出IMPLEMENTATION_CHECKS.json和日志。若前置仍缺失，保存真实pending及具体缺件，不生成接收、preflight或运行工件。输入hash和日志hash在收据中；持续写入的audit stdout不递归自hash。

前置运行期间，其最终frozen_inputs源副本/BUDGET/PREFLIGHT等派生归档可能尚未完成。交付审计仅在pending状态下将来自前置FREEZE.files的这些预期路径列为待定，路径仍完整保留；前置正式gate仍要求终态archive及全部hash通过。此次诊断首次拒绝日志final_audit_02.log保留，没有据此放宽正式核验。

54历史来源与13源FREEZE注册表按实际审计记录冻结；120注册记录不被解释为120独立公式。新fixture树保留在本目录harness_validation中，但不递归列为执行输入；所有源码/完整日志/审查收据仍冻结，旧依赖闭包不删。

## 剩余门槛

前置开发终态及独立接收尚需按实际状态完成；新harness还需独立审查、主代理最终收据和冻结。未运行正式新任务数据、preflight/freeze、拟合/搜索、评分或发布，32系统的性能/时长/科学门槛未知。全量前置gate与解释器导入占用60秒外限，未保证每cell可用完整52秒；单次BLAS、hardkill和磁盘失败边界保持。新批科学失败必须报告，不能为发布改阈值或优化算法。
