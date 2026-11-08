# FoodLens 报告

**看完整报告：** 打开 [research_proposal.pdf](research_proposal.pdf)，共 29 页，包含封面、目录、第一至九部分和参考文献。总入口已按顺序接入全部正文，页码、图表编号及交叉引用由统一编译生成。

**第三、六、九部分独立稿：** [sections_3_6_9.pdf](sections_3_6_9.pdf) 是此前独立编译的 4 页章节稿，供单独查阅；这些章节也已纳入完整报告。独立入口为 [sections_3_6_9.tex](sections_3_6_9.tex)，其中的表号保留原独立稿设置，完整报告以总入口自动生成的编号为准。

**改正文：** 修改 `sections/` 下的对应章节；章节顺序与封面、目录在 [research_proposal.tex](research_proposal.tex) 中维护。

**重新编译：** 在项目根目录执行 `& .\docs\report\build.ps1`。

本 PDF 为开题报告，描述的正式训练与实验属于计划，不等于已经完成，也不替代后续结题材料；实际实现以[工程说明](../../foodlens_engineering/README.md)和运行证据为准。课程交付见[原始要求整理稿](../COURSE_REQUIREMENTS.md)。当前训练平台以 Linux GPU 服务器为主，第五章约 300 元为早期估算，不作为实施预算上限。本目录不需要原作者本地 `.agents/` 绘图技能即可编译。

## 目录怎么读

```text
docs/report/
├── research_proposal.pdf       最终报告，直接阅读这个
├── research_proposal.tex       总入口：排版设置、章节顺序和参考文献
├── build.ps1              一键编译
├── README.md              本说明
├── sections/              正文：第一至九部分
├── bibliography/          参考文献库 .bib 和引用样式 .bst
├── styles/                报告共用的图表样式
├── figures/
│   ├── branding/          封面使用的学校标识
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

构建验收：脚本退出码为 0，生成的 `docs/report/research_proposal.pdf` 可打开，目录、图表、公式和参考文献无缺失；2026-11-08 完整合稿为 29 页，正文修改后页数可能变化。此次在 Windows / TeX Live 2026 上执行 `& .\docs\report\build.ps1`，并检查逐页渲染；整合基于 `56e22e3`，构建日志和页面检查图放在本地 `outputs/git-cleanup-20261108/`。检查 PDF 后再提交源文件及 PDF。TeX 与绘图依赖独立于 Python CPU 环境，运行模型测试不需要安装它们；其他操作系统的字体适配尚未验证。

`literature/` 中的原论文图片保留作者署名、来源与许可记录。旧版目录名 `redesign/round2/` 已移除，后续版本通过 Git 管理。

## 编译第三、六、九部分

在项目根目录启动 PowerShell，确认 `xelatex` 已加入 PATH（字体要求同上），然后执行：

```powershell
Push-Location docs/report
try {
    New-Item -ItemType Directory -Path build/sections -Force | Out-Null
    1..2 | ForEach-Object {
        xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build/sections sections_3_6_9.tex
        if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX compilation failed' }
    }
    Copy-Item build/sections/sections_3_6_9.pdf sections_3_6_9.pdf -Force
} finally {
    Pop-Location
}
```

该章节稿保留原文内容和表 5、表 6 编号，修正了一处开引号方向，并将表号改为交叉引用；风险表使用可跨页的三线表与重复表头。此入口不引用参考文献，无须 BibTeX。编译验收为两次 XeLaTeX 均退出 0、生成 4 页 PDF、无未定义引用及缺字或溢出警告。已在 Windows / TeX Live 2026 上实际编译并逐页检查。

