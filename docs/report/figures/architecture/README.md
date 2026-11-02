# 架构图资源

本目录保存最终报告使用的 4 张架构图：

| 文件名前缀 | 内容 |
| --- | --- |
| `figure06_system` | 系统与任务关系 |
| `figure07_caption` | 图像描述模型 |
| `figure08_hybrid` | 食材与营养混合预测 |
| `figure09_qwen` | Qwen 适配方案 |

每张图的 `.pdf` 用于报告编译，`.png` 用于快速预览，`.svg` 和 `.drawio` 用于编辑，`.scene.json` 记录可编辑场景。draw.io 对象的 `formulaSource` 相对本目录指向 `math/sources/`；公式的 PDF/SVG 及映射表也在 `math/`。

日常修改报告正文不需要重新生成图件。

## 通过程序修改图件

- `../../tools/scene_base.py`：共用绘图对象和基础场景。
- `../../tools/build_figures.py`：最终图的布局、连线和公式排版。
- `math_spec.json`：文字标签到 LaTeX 公式的映射。
- `../../tools/math_renderer.py`：公式 PDF/SVG 生成。

在项目根目录运行：

```powershell
python .\docs\report\tools\build_figures.py
& .\docs\report\build.ps1
```

Python 环境需要 reportlab、pypdf、pypdfium2、Pillow。公式生成还需要 TeX Live 的 latex、pdflatex 和 dvisvgm（通过 PATH 或 `TEX_BIN` 查找）。图形使用 Windows Arial 字体。

图件写回本目录，临时文件和审计记录写入 `../../build/figures/`，不会混在正文和最终图件中。
