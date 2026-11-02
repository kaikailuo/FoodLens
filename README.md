# FoodLens

深度学习课程项目：食物图像描述生成，以及食材、重量与整餐营养信息的结构化预测。

目前已完成研究综述、技术方案和可运行的 CPU 工程基线；正式 Caption/GPU 训练、Qwen 适配与完整实验仍待实现。小样本 smoke 只验证工程链路，不能代表模型效果。

**给同学和 agent：** 先读 [AGENTS.md](AGENTS.md)，再按[工程 README](foodlens_engineering/README.md)复现。当前唯一 CLI 提供 `smoke`、`predict`、`evaluate`；没有正式训练命令。复现不需要原作者的虚拟环境、历史产物或本地 `.agents/` 技能。

## 目录与入口

| 路径 | 内容 |
| --- | --- |
| [课程原始要求](docs/COURSE_REQUIREMENTS.md) | 用户提供的教师材料整理稿：任务、指标、分组、阶段交付及日期说明 |
| [foodlens_engineering/](foodlens_engineering/README.md) | Python 代码、配置、测试、依赖及运行说明 |
| [docs/report/](docs/report/README.md) | 最终报告源文件、必要图片及参考文献 |
| [research_proposal.pdf](docs/report/research_proposal.pdf) | 最终第四、五章合编报告；不是完整第一至五章报告 |
| [technical_plan.tex](docs/report/sections/technical_plan.tex) | 最终第五章技术方案，工程实施依据 |
| [docs/research/](docs/research/) | 成员调研记录、原附件和调研建议 |
| [docs/proposal/](docs/proposal/) | 原始开题材料 |
| [notebooks/](notebooks/README.md) | 课程示例与探索笔记 |
| [data/](data/README.md) | 本地数据；获取和目录约定见说明 |
| outputs/ | 本地结果、检查日志与整理前校验清单，默认不提交 |

工程输出位于 `foodlens_engineering/artifacts/`，整理检查位于 `outputs/`，均默认不提交。干净克隆不包含这些历史文件；工程 README 会生成复现所需的 manifest、checkpoint 和预测。

## 课程范围与运行平台

- 任务一：至少新实现两种指定结构，使用至少三种指定指标；当前方案为 A/B 与 BLEU-4、METEOR、ROUGE-L、CIDEr-D。
- 任务二：比较任务一模型适配与微调多模态模型，报告文本指标、F1 和 MSE；MAE、结构有效率等作为补充。
- 选做三选二：当前方案选择 SCST 和 Judge，两个方向均需交付，先完成必选任务再推进。
- 中期需要原型；结题报告使用 Notebook/Markdown，另交幻灯片或 2—5 分钟演示视频。课程未强制 Web 界面。当前第四、五章合编 PDF 不能替代包含全部要求的开题报告或结题报告。

正式开发、训练和完整实验以 **Linux GPU 服务器为主**，三位组员的 **Windows 本机用于轻量测试**。数据统一位于项目根 `data/`，任务二由组员从教学云平台自行下载并按数据说明归档。第五章约 300 元是早期估算，不作为实施预算上限；实验按课程要求和实际硬件推进。

## Python 环境与验证

Python 基线为 3.12。Linux 服务器的 CPU 预检查命令见[工程 README](foodlens_engineering/README.md#linux-服务器的-cpu-预检查)，正式 CUDA 训练器及 GPU 依赖尚未完成，CPU 锁定文件不能用于配置正式 CUDA 环境。

下面是已验证的 Windows PowerShell/CPU 小测试入口。先确认 `python --version` 为 3.12.x；如果系统有多个 Python，应改用正确的 3.12 解释器。命令从项目根目录开始，已有 `.venv` 时无需重复创建：

```powershell
Set-Location foodlens_engineering
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock-cpu.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v -k ModelTests
```

以上 `ModelTests` 不依赖课程数据，也不下载模型权重，预期 7 项通过。每条命令成功后再执行下一条；已有环境缺少 pip 或创建时 ensurepip 失败，按[工程环境说明](foodlens_engineering/README.md#1-创建环境与无数据检查)使用外部 pip 管理目标环境。完整测试需先按[数据说明](data/README.md)准备并校验任务二数据，然后在 `foodlens_engineering/` 执行：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

更多 smoke、预测和评价命令见 [工程 README](foodlens_engineering/README.md)，当前实现边界见 [IMPLEMENTATION_STATUS.md](foodlens_engineering/IMPLEMENTATION_STATUS.md)。

Linux/CUDA 正式训练尚未验证。Python 工程复现不需要 TeX；报告字体与构建目前仍面向 Windows。

## 构建报告

当前报告使用 Windows 字体集。在 Windows 上安装包含中文支持的 TeX Live，将 `xelatex` 和 `bibtex` 加入 PATH。在项目根目录执行（其他系统需先配置对应字体）：

```powershell
& .\docs\report\build.ps1
```

也可设置 `$env:TEX_BIN` 为 TeX 可执行文件目录。最终 PDF 位于 `docs/report/research_proposal.pdf`；正文在 `sections/`，图件在 `figures/`，编译缓存统一写入被 Git 忽略的 `build/`。完整结构见报告目录 README。

## 协作

分支、PR、数据与产物管理约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。本地整理的迁移映射见 [docs/REORGANIZATION.md](docs/REORGANIZATION.md)。
