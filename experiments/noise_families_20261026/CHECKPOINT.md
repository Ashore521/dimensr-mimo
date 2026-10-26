# 当前检查点

- 正式goal仍active、无token预算；总目标未完成。用户授权按顺序自主推进，不重复审批；不清空仓库、不派子代理、不更新memory。
- 原生API版本1.1.0已提交推送a28bbd72eba670e2b78b326b8f41c347755e2379；85测试、6/6等价/物理交付、wheel隔离导入通过。GitHub Actions运行40349025015已成功，Windows/Ubuntu×Python3.10/3.12四项全过。
- 本批36/36系统及核验通过，651.39秒；legacy/dimensional/独立分别3/12、3/12、0/12全门槛达标，平均干净NRMSE约0.04026/0.02161/0.03424；新初始化均值降46.3%，23/24输出保护，性能推荐失败，默认不变。
- 阻尼/有理任务池内无绝对精度可行解；含噪多项式702不满足联合门槛；含噪多项式701第一个输出相对退化但仍绝对达标。禁止据本批测试调阈值/重跑选结果。
- 下一步先完成已登记的Concrete Slump物理单位历史回归，再诊断通用复合项发现。该数据旧使用记录明确，不可称新盲测。
- 新目录experiments/slump_physical_20261026已有PROTOCOL.md、data.py、DATA_MANIFEST.json；官方zip/成员哈希与根目录历史v3相同，61训练/21验证/21测试行精确还原；原始数组仅在outputs/slump_physical_20261026忽略缓存内。尚未写训练/监督/评分脚本，尚未拟合。
- 旧dataset_manifest中的split_id_sha256是行ID用换行连接且末尾再加换行的SHA256，不是JSON哈希。训练/验证序列等于default_rng(20261023).permutation(103)的前61/后21；测试为剩余21且哈希已核对。
- Slump已预定9系统：legacy/dimensional/independent×801/802/803；每系统1200评价、60秒，独立每输出400评价/20秒。注意旧first_round监督器硬编码2输出、每输出30秒，不能直接用于本批，需按事件allowance_seconds落实20秒上限。
- Slump物理SI输入kg/m³，输出m/m/Pa；训练坐标偏移与缩放，量纲保持。NRMSE研究门槛每输出0.5，搜索图36/9/45/6，物理图80/29/140/10；非工程容差。完整物理域及常数量纲目录见PROTOCOL。
- 当前没有运行中的实验进程；noise模型、评分、独立核验及源码副本已封存，脚本不可覆盖更改。保留原DimenSR干净HEAD=7fbcefadd105fe6266c8149cc05420de8ca74a76。
