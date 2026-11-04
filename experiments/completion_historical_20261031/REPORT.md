# 通用复合补充历史批：完成运行，原核验未通过

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-31
- Verification Status: 原正式核验FAILED；独立后审确认原失败及本批观察
- Version Label: completion_historical_v1

34个系统均正常结束，全部模型及失败分母按原合同封存。外监督865.0849244594574秒、worker返回0、无强杀；从BUDGET到核验859.7208552360535秒，在原2880秒预算内。正式VERIFICATION只有17/34系统通过：17个候选通过，17个control均在旧模型身份比较失败。因此原technical_pass=false、adoption_pass=false，外入口exit1，development没有启动。

## 原封存评分观察

| 指标 | 原precision_repair控制臂 | compositional_completion |
|---|---:|---:|
| 完成系统 | 17/17 | 17/17 |
| 完整精度与结构接受 | 11/17 | 13/17 |
| 平均干净测试NRMSE | 0.0022923970183390385 | 0.0017540199993836482 |
| 平均共享图规模 | 21.176470588235293 | 19.235294117647058 |

34/34配对输出满足原保护，17/17 foundation身份相同，原11个接受全部保留，两个指定紧凑目标均达原规格。新增接受为rational_n1/702与rational_variant/711。候选仍有四个未完整接受：damped_n0/702（图和精度）、damped_n1/701（图和参数）、damped_n1/702（图）、damped_variant/711（图、参数和干净精度）。逐输出数值与每项失败原因完整保留于SCORES及SUMMARY，不据此重选或更改算法。

这张表是原封存评分观察，不能抵销正式技术身份失败，也不是新任务或盲验证。

## 实际补充和匹配

新补充17次调用，146次缓存外完整MIMO评价、142有效、18通过固定父训练保护并在验证留存。线性问题总计177247，其中旧implicit子集61974、新复合变换513；implicit是旧总线性的一部分，不能重复相加。返回根中implicit177个、compositional41个；含新复合根的实际系统41次、40有效、6保护/验证留存。来源事件不支持把整体收益归因于单一构件。

候选matching为390新评价/390有效/127保护和验证留存。原SUMMARY把control matching汇总为0，是因为该臂在完整单元核验的最后身份比较失败后，失败行不保留先前算出的统计；该0不是“控制臂没有执行matching”的运行证据。后审需明确区分原失败收据和只读补充核查。

## 身份失败的只读定位

独立目录 ../completion_historical_identity_audit_20261031 保留PLAN、diagnose.py和DIAGNOSIS。主代理完整比较17对model/result/delivery/pool及score：在原normalizer之后，每个model仅有history四个epoch的epoch_allowance_seconds、neural_until_seconds、gp_until_seconds三类字段，共12处差异；其余文件和分数字段一致。所有差异及原值均已落盘。独立后审确认这三类字段确有阶段调度语义，本次保存的阶段截止裕量为正，其他归一保存轨迹一致；原冻结白名单未包含它们，故原失败仍成立。

这项定位没有修改原白名单、FREEZE、模型、指标、VERIFICATION或SUMMARY，也没有新增拟合/搜索/评分。原failed状态保持，尚未写接收通过收据。即使独立审计确认差异仅为计时，development仍被原固定gate阻断；不能把只读解释当自动改变执行边界的许可。

## 冻结与可复现状态

运行前归档923份已有输入；运行时生成的PREFLIGHT、BUDGET和41份源副本另核，共966冻结输入，所有原字节均保存在frozen_inputs。374份ALL封存运行工件摘要核对无漂移，FROZEN_INPUT_ARCHIVE记录完整清单。FREEZE SHA256为65b5993fe09e6590811d61ff2e93f6d49df11d7fe6b41b951b6a08ab224b51ec。

独立POST_RUN_REVIEW已完成（SHA aa0135137a3fa290cdcc91e7d5e1cfdfff0e0c7422cfa22dd4b1e48fb2dd0b5b），主代理以POST_REVIEW_RECEIVED接收，未写准入收据。独立重建17对数据、复算34所选及161补充/784匹配事件，确认表内观察和原失败；控制真实matching为394/394/115/115。全部神经阶段24提案完成且神经/GP保存截止裕量为正，其他归一保存轨迹相同。未独立复算全部5508候选、Rademacher或初始化枚举，亦未把未保存失败cache/RNG状态称为已证明相同。生产默认不变，1.7.0未提交、未发布。当前全目标仍缺采用门槛的有效技术接收、已观察开发批、新锁定任务验证、打包安装态及最终独立发布审查。
