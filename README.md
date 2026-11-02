# FoodLens

深度学习课程项目：食物图像描述生成，以及食材、重量与整餐营养信息的结构化预测。

目前已完成研究综述、技术方案和可运行的 CPU 工程基线；正式 Caption/GPU 训练、Qwen 适配与完整实验仍待实现。小样本 smoke 只验证工程链路，不能代表模型效果。

## 目录与入口

| 路径 | 内容 |
| --- | --- |
| [foodlens_engineering/](foodlens_engineering/README.md) | Python 代码、配置、测试、依赖及运行说明 |
| [docs/report/](docs/report/README.md) | 最终报告源文件、必要图片及参考文献 |
| [research_proposal.pdf](docs/report/research_proposal.pdf) | 最终第四、五章合编报告；不是完整第一至五章报告 |
| [technical_plan.tex](docs/report/sections/technical_plan.tex) | 最终第五章技术方案，工程实施依据 |
| [docs/research/](docs/research/) | 成员调研记录、原附件和调研建议 |
| [docs/proposal/](docs/proposal/) | 原始开题材料 |
| [notebooks/](notebooks/README.md) | 课程示例与探索笔记 |
| [data/](data/README.md) | 本地数据；获取和目录约定见说明 |
| outputs/ | 本地结果、检查日志与整理前校验清单，默认不提交 |

工程内现有 `artifacts/` 继续保存原实验产物，运行命令不变；新的临时整理结果放在 `outputs/`。

## Python 环境与验证

需要 Python 3.12。在项目根目录用 PowerShell 执行：

```powershell
Set-Location foodlens_engineering
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock-cpu.txt
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整测试需要先按 [数据说明](data/README.md) 准备任务二数据；仅运行不依赖数据的检查可用：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v -k ModelTests
```

更多 smoke、预测和评价命令见 [工程 README](foodlens_engineering/README.md)，当前实现边界见 [IMPLEMENTATION_STATUS.md](foodlens_engineering/IMPLEMENTATION_STATUS.md)。

## 构建报告

当前报告使用 Windows 字体集。在 Windows 上安装包含中文支持的 TeX Live，将 `xelatex` 和 `bibtex` 加入 PATH。在项目根目录执行（其他系统需先配置对应字体）：

```powershell
& .\docs\report\build.ps1
```

也可设置 `$env:TEX_BIN` 为 TeX 可执行文件目录。最终 PDF 位于 `docs/report/research_proposal.pdf`；正文在 `sections/`，图件在 `figures/`，编译缓存统一写入被 Git 忽略的 `build/`。完整结构见报告目录 README。

## 协作

分支、PR、数据与产物管理约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。本地整理的迁移映射见 [docs/REORGANIZATION.md](docs/REORGANIZATION.md)。
