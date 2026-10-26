# 第四轮：三角函数种子与共享结构发现

## Material Passport

已知开发任务4×新种子5×搜索组2=40系统；两组使用完全相同的guarded选择器。
**通过本轮候选版本门槛**。
独立核验 40/40，正式阶段耗时 889.05 秒（上限3000秒）。

|搜索组|完成核验|平均测试NRMSE|平均复杂度|墙钟秒|
|---|---:|---:|---:|---:|
|control|20/20|4.9203735e-05|24.273241|391.186|
|trig_seeds|20/20|4.2835128e-16|18.832661|390.343|

差值为trig_seeds减control；绝对差≤1e-8为持平。

|任务|有效配对|平均差|改善/持平/退化|通过输出保护|复杂度比|
|---|---:|---:|---|---:|---:|
|全部|20/20|-4.9203735e-05|5/15/0|40/40|0.77586101|
|shared_parameter|5/5|-0.00019681494|5/0/0|10/10|0.38929442|
|scales|5/5|3.5124485e-16|0/5/0|10/10|0.97814672|
|partial|5/5|5.9124148e-17|0/5/0|10/10|0.96200839|
|boundary|5/5|2.6842618e-16|0/5/0|10/10|1.0007547|

## shared_parameter结构恢复

|组|含两类参数根的最终池|含共享频率的最终池|最终选择共享频率|
|---|---:|---:|---:|
|control|0/5|0/5|0/5|
|trig_seeds|5/5|5/5|5/5|

## 预注册门槛

|门槛|通过|
|---|---|
|all_40_verified|True|
|all_40_full_mechanisms|True|
|all_40_domains_enclosed|True|
|hashes_order_sources_and_traces|True|
|both_arms_guarded|True|
|intervention_observed|True|
|budget_respected|True|
|root_coverage_at_least_4_of_5|True|
|shared_recovery_at_least_3_of_5|True|
|more_shared_recovery_than_control|True|
|mean_reduction_at_least_10_percent|True|
|absolute_reduction_above_1e_8|True|
|all_40_test_outputs_protected|True|
|mean_complexity_ratio_at_most_1_5|True|

完整逐输出数据见metrics_per_output.csv；全部40单元、失败分母、资源和原始轨迹见SUMMARY/VERIFICATION及runs。
这是固定选择器条件下的种子覆盖干预；没有增加单次常数拟合预算，但新增初始化评价会改变后续随机搜索路径。
直接模式仅识别显式共享频率参数身份，允许幅值及乘法换位；不是穷尽代数等价检验。
验证保护相对各自原选择结果，不保证组间测试无损。已知开发任务不构成未见任务或真实数据盲评。enclosed仅为有限浮点数值筛查。
初始化参照公式未单独序列化，约束按冻结参照误差重算；选择本身及候选验证误差从表达式独立重建。
