# 目录整理历史记录

整理日期：2026-11-02。本文用于追溯迁移，不描述当前 Git 提交、远程仓库或工作区状态，也不是复现入口。当前 agent 指引见根 [AGENTS.md](../AGENTS.md)。

后续按用户确认清理并整理了 `docs/report/`：最终报告在 `docs/report/research_proposal.pdf`，正文在 `sections/`，文献在 `bibliography/`，图片在 `figures/`，绘图程序在 `tools/`。下文为首次迁移时的历史记录，旧报告版本、检查产物及旧路径已删除；当前结构以[项目根 README](../README.md)和[报告 README](report/README.md)为准。清理和路径映射清单分别保存在本地 `outputs/report-cleanup/plan.json`、`outputs/report-structure/moves.json`；这些被忽略的文件不随 Git 交付，也不是新成员的必需输入。

| 原路径 | 当前路径 |
| --- | --- |
| `相关工作综述/` | `docs/report/` |
| `郝天乐调研记录/` | `docs/research/郝天乐调研记录/` |
| `张嘉荣调研记录/` | `docs/research/张嘉荣调研记录/` |
| `相关工作调研建议.pdf` | `docs/research/相关工作调研建议.pdf` |
| `开题报告（GPT生成版）.md` | `docs/proposal/开题报告（GPT生成版）.md` |
| `data/` 下两个 notebook 及 PaddlePaddle 示例说明 | `notebooks/` 下同名文件 |

工程代码、数据、虚拟环境和原实验产物位置在首次迁移时保持不变。后续报告清理只保留最终 PDF、源文件、必要图件及原始附件，旧版和候选图不再作为交付；批量检查截图与编译缓存留在本地，由忽略规则排除。

新增根 README、协作约定、忽略规则、换行符约定和数据来源记录。更新工程实施依据链接、报告命令、图片附件清单路径及提取/核验脚本。PowerShell 报告构建入口改为通过 PATH 或 `TEX_BIN` 查找 TeX。

本地 `.agents/` 绘图技能与外部工具核验副本保留在磁盘，默认不提交项目仓库，也不作为新成员复现依赖。通用 agent 指引由根 `AGENTS.md` 提供。

`outputs/reorganization-before.json` 保存移动前 775 个文件的原路径、目标路径和 SHA-256，供本地核对；不进入版本管理。文本路径和构建脚本发生必要修改，原始材料、模型代码、报告正文及交付 PDF 保留。已有历史哈希记录仍用于核查原交付。

早期绘图曾依赖本机字体、Sharp 或 PDF 渲染工具；第一轮候选图及旧 `build_trial.ps1` 等入口已随后续清理移除，不应再寻找或引用。当前绘图依赖和入口见[架构图资源说明](report/figures/architecture/README.md)。报告目录在 `.gitattributes` 中关闭自动换行转换，以免 Git 检出后历史哈希失效；新的工程文件使用统一换行规则。

自动 CI 后续按数据和运行平台约定配置，本轮没有引入会依赖私有数据而失败的工作流。

## 当时的验证结果

- 首次迁移的 775 个文件存在性、正式交付和原始附件哈希核查通过；775 不是后续清理后的当前文件总数。
- 现有 21 项工程测试全部通过；PowerShell 构建脚本语法检查通过。
- 28 项 Git 忽略规则检查通过，数据与缓存排除，正式 PDF、图源和检查脚本可提交。
- 在 `outputs/build-review/` 临时副本中完成合编报告（16 页）、技术方案预览（9 页）和综述 V3（8 页）的完整编译；未覆盖正式 PDF。

上述结果属于整理时的记录，不表示当前检出已重新检查。技术方案预览与综述 V3 是历史检查对象，当前不提供独立交付文件。新环境按工程及报告 README 验证并记录本次结果。
