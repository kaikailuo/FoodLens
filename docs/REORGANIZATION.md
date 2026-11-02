# GitHub 协作前目录整理

整理日期：2026-11-02。已在项目根目录初始化本地 Git 仓库，默认分支为 `main`；未暂存、提交、推送或配置远程仓库。

后续按用户确认清理并整理了 `docs/report/`：最终报告在 `report/research_proposal.pdf`，正文在 `sections/`，文献在 `bibliography/`，图片在 `figures/`，绘图程序在 `tools/`。下文为首次迁移时的历史记录，旧报告版本、检查产物及旧路径已删除；当前结构以项目根 README 和 `report/README.md` 为准。清理和路径映射清单分别保存在本地 `outputs/report-cleanup/plan.json`、`outputs/report-structure/moves.json`。

| 原路径 | 当前路径 |
| --- | --- |
| `相关工作综述/` | `docs/report/` |
| `郝天乐调研记录/` | `docs/research/郝天乐调研记录/` |
| `张嘉荣调研记录/` | `docs/research/张嘉荣调研记录/` |
| `相关工作调研建议.pdf` | `docs/research/相关工作调研建议.pdf` |
| `开题报告（GPT生成版）.md` | `docs/proposal/开题报告（GPT生成版）.md` |
| `data/` 下两个 notebook 及 PaddlePaddle 示例说明 | `notebooks/` 下同名文件 |

工程代码、数据、虚拟环境和原实验产物位置保持不变。报告内部结构、历史版本、正式 PDF 和原始附件保留；批量检查截图与编译缓存留在本地，由忽略规则排除。

新增根 README、协作约定、忽略规则、换行符约定和数据来源记录。更新工程实施依据链接、报告命令、图片附件清单路径及提取/核验脚本。PowerShell 报告构建入口改为通过 PATH 或 `TEX_BIN` 查找 TeX。

本地 `.agents/` 绘图技能与外部工具核验副本保留在磁盘，默认不提交项目仓库。

`outputs/reorganization-before.json` 保存移动前 775 个文件的原路径、目标路径和 SHA-256，供本地核对；不进入版本管理。文本路径和构建脚本发生必要修改，原始材料、模型代码、报告正文及交付 PDF 保留。已有历史哈希记录仍用于核查原交付。

报告历史绘图辅助程序中的 Windows 字体、Sharp 和 PDF 渲染依赖未统一打包；它们属于可选工具，现有本机环境说明继续保留。第一轮候选图的 README 和 build_trial.ps1 被后续版本按字节校验，保留原样，其中的旧路径属于历史环境。报告目录在 `.gitattributes` 中关闭自动换行转换，以免 Git 检出后历史哈希失效；新的工程文件使用统一换行规则。

自动 CI 后续按数据和运行平台约定配置，本轮没有引入会依赖私有数据而失败的工作流。

## 验证结果

- 775 个迁移文件全部存在；正式交付和原始附件哈希核查通过。
- 现有 21 项工程测试全部通过；PowerShell 构建脚本语法检查通过。
- 28 项 Git 忽略规则检查通过，数据与缓存排除，正式 PDF、图源和检查脚本可提交。
- 在 `outputs/build-review/` 临时副本中完成合编报告（16 页）、技术方案预览（9 页）和综述 V3（8 页）的完整编译；未覆盖正式 PDF。
