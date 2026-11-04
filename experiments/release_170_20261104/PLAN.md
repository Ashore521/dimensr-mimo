# DimenSR-MIMO 1.7.0 发布检查

用户于 2026-11-04 授权继续至发布 1.7.0，随后暂停优化。发布目标为既有公开仓库 Ashore521/dimensr-mimo 的 main，与既往版本发布方式一致。

## 内容与验收

发布源码版本为 pyproject.toml 的 1.7.0，候选源码精确对应 compositional_completion_20261031 的 41 文件锁。功能提供 closure、implicit、precision_repair 和 compositional_completion 的显式入口。README 和版本文档以已封存结果区分技术核验、科学采用和已知失败，默认配置为 legacy/random。

完成已定开发批与锁定候选新任务批的封存、原字节归档、报告及独立后审。各自科学门槛按预定口径报告，软件发布以可选功能的完整工程核验、准确结果文档和独立审查为条件。

源码根 tests 全套执行，新增stage构造测试另行记录。首次源码全套397通过、1失败的日志已留在私有outputs/release_170_20261104；失败来自已完成历史批的存在与旧pre-seal测试前提冲突。新增仓库根conftest.py只给该原测试提供空的隔离HERE，原tests/conftest.py的torch单线程设置及所有冻结测试字节保留。主代理曾将适配误写至原tests/conftest.py，独立审查由原冻结链检出后，已从原归档精确恢复并迁移适配位置。相关失败、修复和最终回归日志完整留存。

构建 wheel 和 sdist，逐份核41源码摘要与锁一致，检查不含私人Word/PDF、原始数组、outputs、环境凭据和临时目录。wheel安装至新的项目内隔离target，使用check_installation.py在阻止experiments导入的条件下读取已封存native/physical模型，验证精确预测、保存重载、版本与默认配置。sdist解包后对新增算法和旧模型迁移运行相关核心测试，解释器实际导入路径须位于该解包源码目录。

完整历史与失败留存本地，公开内容由明确文件范围暂存并核对Git实际blob摘要。源码审查、实验证据和安装态检查稳定后交独立发布审查。提交推送已获本轮授权，推送后读取远端main精确SHA和该提交的GitHub Actions全部Windows/Ubuntu、Python3.10/3.12结果。CI失败先保留日志并修复本次发布范围内的工程问题，再核验最终提交。

最终交付提交链接、CI链接、可用wheel/sdist及校验摘要、功能和科学结果边界。发布核验完成后将完整研究目标设为paused，并保存检查点和未完成优化项。
