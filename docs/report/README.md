# FoodLens 报告

**看报告：** 打开 [research_proposal.pdf](research_proposal.pdf)。这是当前唯一的最终报告，包含第四章综述与第五章技术方案，共 16 页。

**改正文：** 修改 [sections/related_work.tex](sections/related_work.tex) 或 [sections/technical_plan.tex](sections/technical_plan.tex)。

**重新编译：** 在项目根目录执行 `& .\docs\report\build.ps1`。

第五章是后续实施设计，描述的正式训练与实验不等于已经完成；实际实现以[工程说明](../../foodlens_engineering/README.md)和运行证据为准。课程交付见[原始要求整理稿](../COURSE_REQUIREMENTS.md)，本 PDF 仅第四、五章，不能替代课程要求的完整开题或结题材料。当前训练平台以 Linux GPU 服务器为主，第五章约 300 元为早期估算，不作为实施预算上限。本目录不需要原作者本地 `.agents/` 绘图技能即可编译。

## 目录怎么读

```text
docs/report/
├── research_proposal.pdf       最终报告，直接阅读这个
├── research_proposal.tex       总入口：排版设置、章节顺序和参考文献
├── build.ps1              一键编译
├── README.md              本说明
├── sections/              正文：第四章和第五章
├── bibliography/          参考文献库 .bib 和引用样式 .bst
├── styles/                报告共用的图表样式
├── figures/
│   ├── literature/        综述引用的原论文图片、排版片段和来源清单
│   └── architecture/      本项目的四张架构图及可编辑资源
│       └── math/          图中公式的 TeX 源文件、PDF/SVG 和映射表
├── tools/                 架构图生成程序；日常改正文通常不需要动
└── build/                 编译与绘图临时文件，运行时生成，不提交 Git
```

## 常见修改

| 想做什么 | 修改哪里 |
| --- | --- |
| 改综述文字 | `sections/related_work.tex` |
| 改技术方案文字 | `sections/technical_plan.tex` |
| 添加或修改文献 | `bibliography/references.bib` |
| 改整体排版或章节顺序 | `research_proposal.tex` |
| 调整架构图 | 查看 [架构图资源说明](figures/architecture/README.md) |
| 核对原论文图来源 | `figures/literature/image_manifest.json` |

报告构建需要 Windows 字体集，以及带中文支持的 TeX Live。将 `xelatex`、`bibtex` 加入 PATH，或设置 `$env:TEX_BIN`。脚本把中间文件写入 `build/latex/`，全部步骤成功后才更新根目录 PDF。

构建验收：脚本退出码为 0，生成的 `docs/report/research_proposal.pdf` 可打开，目录、图表、公式和参考文献无缺失；当前交付参考为 16 页，正文修改后页数可能变化。检查 PDF 后再提交源文件及 PDF。TeX 与绘图依赖独立于 Python CPU 环境，运行模型测试不需要安装它们；其他操作系统的字体适配尚未验证。

`literature/` 中的原论文图片保留作者署名、来源与许可记录。旧版目录名 `redesign/round2/` 已移除，后续版本通过 Git 管理。
