# 1.7.0 独立最终发布审查

2026-11-04。What this change does: 本候选公开默认关闭的闭包、隐式初值、精度修复和复合补充入口，并交付其冻结研究证据、wheel和sdist。它保留legacy/random默认，未承诺新任务性能改善。

**结论：支持下列固定41源、固定两个安装包和候选索引的opt-in发布；本次检查没有发现未处理的实质工程阻断。新输出组合的科学采用仍为false。主代理可继续正常提交、推送及该精确提交的四矩阵CI核验；远端CI尚未执行，本报告不声称发布已完成或CI已通过。最终接受由主代理决定。**

## 固定版本和交付物

生产41源逐字节对应 `experiments/compositional_completion_20261031/CANDIDATE_LOCK.json`，SHA256 `51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452`。与本人的兼容性预审及既有CORE_REVIEW源码快照相同；本次保留原COMPATIBILITY_REVIEW及其检查收据原字节，没有修改生产源码、测试、文档、包或旧结果。

| 工件 | SHA256 |
|---|---|
| `dist/dimensr_mimo-1.7.0-py3-none-any.whl` | `ed4d43070310e369bb8071bbac7265c2bec7266d56167014fbec299f171d7a88` |
| `dist/dimensr_mimo-1.7.0.tar.gz` | `93e71c0dfc90ad24a1533d3778ae94fc3f0c0791e4f6562e623be73ed02caaf8` |
| `docs/RELEASE_1.7.0.zh-CN.md` | `2e7161258ee0826df3a240fa3581e2238a0cbfb7c07c05a75b868dfda30b1386` |
| `README.md` 的实际Git blob内容 | `0e8d13e6156664b3687dc25c6a268289b7c3377bb2758d0a4944b60829b1aad2` |
| `pyproject.toml` | `28018c0cc8c60eed3aa5a6c6d5277246dd729232cebef06ec98f1ea2e01b0501` |
| `MANIFEST.in` | `7cc5512fcc421897926f34ff5a2e8ec5d28416aa388ecb747e14c4f7a23639f8` |
| `.github/workflows/ci.yml` | `ad7076b9c4c8a0ba7867b58d6bd4018dffa50bf8178925824a7394300503f4da` |
| `RELEASE_VALIDATION.json` | `66509ef1326e6a18e66c5767d4110e90af7d3b6393599e3968bfaec6a23ead54` |
| `PACKAGING.json` | `57c55db4e55a8a778106042755aac3e026edda05b7dfa8c6a23b61d7e27e38aa` |
| `INSTALLATION_MANIFEST.json` | `eba5af87e7eec70a18886cc78991cbd55f51dcb1c1194cfcb0b427f3a7c009c6` |
| `PUBLICATION_AUDIT.json` | `60fcbf8618230277a6f8190a2a498d2a726dfe2d65581618cb6b01e87f1c54da` |

上述四份JSON位于本目录。候选HEAD仍为 `652b05884269c7563b101e0a14a9b15f749bbc6f`。本次由 `git ls-files --stage -z` 读取25990条索引记录，在内存按Git树编码重算所得树为 `2446f0a1796c82ac59030e0f807edf1ab655619e`，与主代理指定树相同；检查前后索引记录完全相同。最终报告尚不在该树内，主代理须核对后续仅加入获授权的审查工件及对应收据。

## 实际完成的核验

1. **包内容及元数据。** 独立打开完整wheel和tar目录，不提取或改写原包。wheel49成员、sdist367成员，无重名、路径穿越、tar链接及所列私人文档/数组/环境/outputs路径；wheel只含生产包和dist-info。两个包各41份Python源码的SHA256映射与锁完全相同。逐项验证wheel全部RECORD摘要/长度和完整成员覆盖，METADATA的name/version为dimensr-mimo/1.7.0，其UTF-8描述与当前README文本相同。sdist的README、最终含Windows说明的版本文档、pyproject、MANIFEST及两份conftest与当前工作树字节一致，PKG-INFO版本正确。

2. **安装证据真实绑定。** 阅读最终check_installation.py和原安装日志，核其SHA256为 `da70036e3a06c575c67b38fcb1731e3725bdae799b6fb2b8b9a799b713576ea4`；manifest、公开PACKAGING、RELEASE_VALIDATION及原始INSTALLATION_CHECK相互绑定。原始收据SHA256 `81fa115f80a45f9d25aa0fafc49cb41be2d6b733d556bfd3e62062d72d142715`，实际import路径为隔离target，41个安装源文件本次再次逐字节核锁。135组manifest记录与原收据模型表顺序/摘要一致，270份native/physical文件及一份旧native当前字节均与manifest相符。主代理的135组保存重载及12个固定种子独立坐标点预测相等结果可以据该绑定引用；本审查没有重复运行全部135组。

3. **独立安装抽查。** 另以 `python -I`、阻止experiments导入、显式禁止fit的方法，从上述安装target加载新任务的log_affine_ratio_n0/10241 completion及其physical模型。7个seed1701独立坐标点上，native、physical和原保存图逐元素精确相同；两个模型再存再载完整字典相同。另加载1.1.0 native旧模型并重存再载，新增字段均为False。所有实际加载的dimensr_mimo模块路径都位于target，默认仍legacy/random。该检查没有训练、初始化、测试标签访问或科学评分，结果在 `release_review_validation/installed_probe.json`。

4. **配置、迁移和原guard。** 复用未漂移41源对应的本人兼容性预审：旧字段默认完全相同，1152配置组合和10个严格bool拒绝通过，旧模型先核原来源再迁移、篡改选择/base来源拒绝、原逐输出训练/验证保护保留。新入口未绕过原单位、原图域/有限值、联合拟合、全局预算或冻结后验证边界。新log/sqrt逆变换/参数顺序/域边界等算法核心使用已独审核心证据，本次没有重做其枚举或全部数值试验。训练/验证保护不是未来测试保证，enclosed不是连续域证明。

5. **源码和sdist测试证据。** 已读并重核RELEASE_VALIDATION所列全部日志摘要。`ci_source_tests_06.log`为507 passed/90.83s，`sdist_tests_02.log`为最终sdist实际src路径导入及62 passed/26.62s；final_seal_guards另为2 passed。这些是主代理保留的本地执行证据，本审查没有独立重复上述测试。CI仍为Ubuntu/Windows×Python3.10/3.12，checkout局部core.longpaths、importlib模式和tests路径适配保留原守卫，未删断言。之前sandbox TEMP失败和文档候选包/日志被保留，不作为成功构建替代物。

6. **公开索引与精确冻结字节。** 本次独立读取69个关键索引blob并核对其内容，覆盖全部41源、最终公开文档/配置、包收据、两个批次的封存/评分/核验/摘要/独审及新任务接收收据。唯一工作树与索引字节差异是非冻结README的CRLF/LF，文本完全相同；其工作树SHA256为 `1f37ee5dbc7316a901224166a9ae5efcee13db35009f596f237e564d3c6bf94d`。生产源和本次所核冻结工件没有进行这种归一化。索引全路径另通过私人文档/数组/环境/outputs等指定路径模式筛查。全部19401个冻结/运行路径、2867个独特blob、凭据内容模式及100MiB检查引用主代理PUBLICATION_AUDIT；本次核其原audit/whitespace日志摘要，未重复全体blob内容审计，未声称不存在任何未知敏感信息。

7. **空白问题不是未处理工程失败。** 已读原whitespace_audit：git diff --check exit2，2306项/118路径，仅.log/.diff/.md，生产Python为0；主代理审计指出这些是保留的历史失败日志、记录diff和冻结旧审查文本。其原日志摘要与公开审计相符。不得为消除该诊断修改冻结历史字节，本审查没有这样做。

## 科学结论和公开表述

开发批SUMMARY、独立POST_RUN_REVIEW及8键 `accepted_for_release_validation` 收据均逐hash核对：32/32技术、7/16→16/16完整接受、32/32输出保护、科学采用true；开发证据的适用范围保持为已观察矩阵。

新任务SUMMARY、后审SHA256 `2cb6726781fb3dc11678678ca20147274e8e9eef45825d9fa062a4a529a1b940`、checks SHA256 `9db58b8d076435ae65204a74e9664a7aadc8f6d1d2444cd07458f4888da7a7ef`及POST_REVIEW_RECEIVED均匹配。32/32技术与输出保护成立，但两臂均5/16接受、均值clean NRMSE均0.021925807049642965、图均值22.625，11失败和adoption=false仍保留。原独审确认16/16最终图、float64常数字节及best载荷完全相同；已重算32所选、983事件、384返回根，没有重算5836全池、Rademacher或initializer枚举，本发布审查沿用该明示范围，不重复科学后审。

README开头及最终版本说明清楚呈现新任务失败、两臂持平和默认关闭。未宣传普适收益、外部盲验证、真实数据有效性或默认推广；历史原失败和新隔离核验合同仍分开。四个功能说明中的旧阶段状态由最终说明解释为各自签署时状态。已知的负结果不阻断经授权的显式可选软件入口发布，也不能被发布判定改写成科学采用通过。

## 执行记录与尚待完成

本次实际平台turn_context为 `2026-11-04T15:29:38.109Z`，turn `01a1a72e-15e6-7243-ac2f-d9ed23b393e6`，实际gpt-6.1-sol/high，同既有子代理日志。所有新增输出仅在获分配的RELEASE_REVIEW、release_review_checks及release_review_validation中；没有派生、搜索、训练、正式实验重跑、阈值/预算/模型选择变更、提交或发布。

有一项执行偏离：首次批量读取误带 `git write-tree --missing-ok --prefix=nonexistent/`，返回 `prefix nonexistent/ not found`，未成功返回树；已立即告知主代理。随后只读内存重构确认索引内容仍为指定2446f0a树，前后读取相同，未见内容变化；后续Git对象/索引/提交操作均交主代理。另自身首两轮审计因METADATA解码和README换行断言过严终止，只修正自有审计脚本并保留两次日志；最终artifact_audit_03通过，这两次不是产品缺陷。

发布前主代理仍须核对本报告/收据加入索引后的有限增量，正常提交推送；随后读取远端main精确SHA和该SHA的全部四矩阵CI结果。若CI失败，保留日志并按影响范围复核工程修正，不能以本地507/62通过替代远端成功。当前报告没有未决的科学范围或预算授权请求，亦不授权进一步优化；用户要求发布核验后暂停优化仍有效。

未检查：远端提交/CI、Python3.10或Linux的本机运行、外部真实任务、全部研究池的独立重算、全部凭据内容模式的独立扫描。未发现阻断只覆盖上述实际证据，不是全面算法/科学接受。

Verdict: 支持固定源码与包的opt-in发布；精确提交远端CI及最终发布完成状态由主代理继续核验。
