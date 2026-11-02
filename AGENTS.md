# FoodLens agent 工作入口

本文件适用于整个仓库。目标是让其他同学及其 agent 能从干净检出理解当前状态、复现已有链路并继续实施。

## 先读什么

1. [README.md](README.md)及[课程原始要求整理稿](docs/COURSE_REQUIREMENTS.md)：项目现状、任务范围与提交要求。
2. [工程 README](foodlens_engineering/README.md)：环境、命令、产物和验收条件。
3. [实现记录](foodlens_engineering/IMPLEMENTATION_STATUS.md)：已有验证的时间、范围和限制。
4. [数据说明](data/README.md)及 [sources.json](data/sources.json)：获取方式、字段、版本和校验。
5. [第五章技术方案](docs/report/sections/technical_plan.tex)：后续实现依据。
6. [协作约定](CONTRIBUTING.md)：分支、验证与产物管理。

## 信息冲突如何处理

- 当前用户明确指令优先。课程要求以用户提供的 `docs/COURSE_REQUIREMENTS.md` 为依据，小组模型设计以第五章为依据；当前功能是否存在，以代码、测试和实际运行证据为准。课程要求和小组自定方案分开表述。
- 方案中的“采用”“计划”“最小交付”不表示功能已经实现。发现代码与方案不同，记录差异，不把 smoke 的简化配置当作正式架构。
- `docs/proposal/`、`docs/research/` 和 `notebooks/` 是历史材料或参考资料，不覆盖当前方案，也不增加已确认的交付范围。
- `docs/REORGANIZATION.md` 是历史迁移记录；其中本地产物不属于新成员必须取得的输入。
- 课程要求任务一至少新实现两种指定结构、采用至少三种指定指标；当前方案为 A/B 与 BLEU-4、METEOR、ROUGE-L、CIDEr-D。任务二比较任务一模型适配与微调多模态模型，并报告任务一文本指标、F1、MSE。
- 选做任务是三选二，当前方案选择 SCST 和 Judge，均纳入交付；B-global 等消融属于小组实验设计。中期需要原型，结题需要 Notebook/Markdown 报告及幻灯片或 2—5 分钟视频，不强制 Web 界面。各成员分工与检查点需可度量，汇报/提交形式和日期按课程说明及教师最新通知执行。
- 向用户确认的问题直接在聊天中提出，不在仓库建立待确认清单或写入未答复的问题。已经明确的项目约定写入相关文档。

## 当前可做与不可声称的结果

- 当前 CLI 只有 `smoke`、`predict`、`evaluate`。已提供 CPU tiny 工程闭环，没有正式 Caption/GPU 训练入口。
- A/B/B-global/Hybrid 模块及部分适配接口已存在；Caption 四指标后端、Qwen 训练/推理、SCST、Judge 尚未完成。
- CPU smoke 使用随机 tiny 骨干；不得称为预训练 ResNet 基线效果、完整实验或 GPU 资源测试。
- 仓库不附完整数据、虚拟环境、权重和历史 `artifacts/`。按工程 README 创建本次产物，不依赖原作者本机文件。
- 主要开发、正式训练和完整实验面向 Linux GPU 服务器；三位组员的 Windows 本机用于轻量测试。现有完成验证的记录仍为 Windows/Python 3.12/CPU，Linux 示例和 CUDA 训练不能标成已经实测。

## 执行与验证

- Python CLI 工作目录为 `foodlens_engineering/`，默认数据根为 `../data`；报告构建从项目根运行。
- 无数据时可运行工程 README 的 `ModelTests`；数据齐全再运行全部测试与 smoke。缺数据应记录缺项，不能造数据冒充真实数据验证。
- 原始任务二划分不变；推理只读 RGB、ID/路径和固定指令，不传入 target、target_text 或含答案的 metadata。词表、数值统计和奖励统计仅由训练集拟合。
- 任务二数据由组员自行从教学云平台下载。项目根 `data/source/` 保存原始压缩包，解压后的标注放 `data/annotations/`、图片放 `data/images/`，不另建含个人路径的数据入口；完整目录见数据 README。
- Food500 必须通过可追溯 manifest 映射图片与 caption，检查哈希及跨划分重复组，禁止按行号拼接。来源尚未锁定时先完成审计。
- 评价保持全样本分母，保留缺失、解析失败与截断；不得只报告成功样本或把训练集 smoke 指标当作测试效果。
- 修改模型、依赖或评价协议时更新配置和说明，运行相关已有检查；交付记录实际命令、代码版本、数据哈希和限制。仅改文档时检查链接和受影响的命令即可。
- 正式训练前由实施 agent 固定模型/Processor revision、数据 manifest、指标版本与单位、模型选择容差及早停配置，记录试跑批量、精度、显存和吞吐。第五章约 300 元是早期估算，不作为实施预算上限，也不以费用为由省略课程要求的实验；依据实际硬件验证配置，不假定已有 GPU 结果。

## 产物与交接

- 本次实验输出放工程 `artifacts/` 的独立子目录，整理检查放根 `outputs/`。两者默认被 Git 忽略，不覆盖原实验记录。
- 共享实验摘要可新建于 `docs/experiments/`，注明权重/大文件的获取位置；该目录当前尚无实验报告。
- 不修改原始数据；不提交数据集、环境、权重、密钥和个人绝对路径。
- 日常编译报告不需要重新生成架构图；已有 PDF/SVG/drawio 等资源已提交。修改正式 PDF 时同步源文件并检查渲染结果。
- 本地 `.agents/` 不随 Git 交付，也不是复现依赖。通用 agent 指引保留在本文件。

实施顺序为数据审计 → Caption 训练与指标 → 任务二迁移 → Qwen 配对实验 → SCST 与 Judge。每阶段先交付可运行入口及验收记录，再进入下一阶段。遇到需要用户决定的问题在聊天中说明，继续完成不依赖该答案的工作。
