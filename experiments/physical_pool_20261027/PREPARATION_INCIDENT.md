# 冻结前路径校正

首次执行study.py run在freeze()读取来源文件哈希时停止：数据清单错误地使用`runs/concrete_slump_s801_legacy/model.json`，而已有Slump运行目录名为`runs/slump_s801_legacy/model.json`。现场列出了六个实际目录并核对，已将清单改为实际路径；选用的模型、种子和方法不变。

该次失败发生在BUDGET.json、FREEZE.json和任何派生模型创建之前；没有开始候选池处理，没有读取测试数组。这里保留准备阶段事件，随后对全部24个来源文件先验证存在性，再首次正式冻结。它不是对失败实验单元的补跑或替换。
