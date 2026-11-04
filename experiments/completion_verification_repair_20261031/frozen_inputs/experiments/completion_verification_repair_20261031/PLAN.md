# 封存结果的核验修正方案（尚未执行）

## 目的与前提

原completion_historical_20261031的34系统正常完成，17个control因history三类运行时配额/截止字段差异而身份核验失败，原technical/adoption均false。只读逐字段审计找到17/17对各12个差异，其余model/result/delivery/pool/score全部相同。源码显示三字段实际参与阶段截止，不能一般宣称它们永不影响算法；接受差异的必要条件是本次完整保存的非计时轨迹、候选、cache、参数、选择及所有数值仍一致。

原批独立POST_RUN_REVIEW已完成并确认17/17对的保存轨迹一致；当前/旧神经阶段最小截止裕量2.49/2.184秒，GP为4.9648/4.6432秒，各神经阶段24提案全部完成。报告明确原批仍failed，支持另立隔离核验协议。原FREEZE、VERIFICATION、SUMMARY、EXECUTION、全部模型与失败日志永不覆盖。本目录的方案不是执行或采用已通过收据。

## 唯一允许的变化

在原冻结normalizer基础上，仅为model.history中phase=phase_budget的条目区分epoch_allowance_seconds、neural_until_seconds、gp_until_seconds。核三值为有限合法数值，原52秒配置及资源上限不变；保留它们的全部原值与差异清单。不得以后缀通配忽略其他seconds字段，不扩大到config、候选误差、评价计数、结构、参数、cache、选择或其他history字段。非phase_budget的同名字段仍逐值比较。

使用原34个封存模型及其原数据规范/种子，只做预测和验证的重计算，不调用initializer、优化器、搜索、重新拟合，不改变原模型选择或生成额外任务/样本。原加载器/验证器重建选择以核对既有结果照常，不把任何不同选择写回模型。额外要求每个control及其旧repair的四个phase_budget条目均完整，神经24提案完成、实际事件结束早于各自神经/GP截止；缺项、截止命中或deadline_reached即失败。全部原技术/科学条件、17对/34系统/34输出分母、原11接受和两指定紧凑目标、均值不升、真实机制及逐输出保护均照原标准核，缺失或非计时差异即失败。

## 新收据与审查

在独立目录记录修正脚本、构造测试和独立预审，冻结原FREEZE/ALL/SCORES/VERIFICATION/SUMMARY/EXECUTION及POST_RUN_REVIEW、源锁、输入归档摘要、完整原失败链与本次核验代码。输出单独的新VERIFICATION/SUMMARY/EXECUTION，明确verification_of_sealed_results且original_verdict=failed，不能改称原批首次通过。相同数值须原样复现，不能据已知测试表现放宽阈值或改变候选。

构造测试必须证明只允许上述限定路径的三计时值差异，同时拒绝错误phase、非法计时值、任意真实指标/候选/cache/事件/配置变化及摘要链漂移。独立后审核验修正后的技术与科学结论后，才考虑对应的后续执行方案。

本阶段不占用尚未启动的development预算；不允许自动重跑失败训练。修正验证拟限制在600秒只读计算，优先使用原2880秒墙钟截止内的剩余时间，绝不自行重置原BUDGET。若原截止不再允许完成，须明确说明并在执行前确定新的只读核验边界。

## 后续阶段边界

原completion_development目录、其入口及共享harness已被原历史FREEZE绑定，不得原地改gate或启动绕过原failed。若新的独立核验成立，另用新目录明确绑定原失败与修正成功的两条证据链，保持同41源锁、同配置、同16对矩阵、同32系统/输出、原16/16完整接受和其余所有门槛，以及尚未使用的原单批2880秒预算。必须先审查新的接续gate；该准备工作不授权修改算法或扩大计算。

verify_sealed.py及构造测试已实现：初版tests_01.log为25 passed；独立预审发现旧评分表整体分母检查遗漏，新增四例先复现失败，恢复原len/唯一键/EXPECTED/all-complete检查后tests_02.log为29 passed。全部初版与失败日志保留。尚未执行正式核验修正，独立预审仍在核对最终修复，development未启动；生产默认和发布状态不变。
