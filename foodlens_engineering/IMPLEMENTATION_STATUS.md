# CPU 工程验证记录（2026-11-01）

本文保存 2026-11-01 的历史验证范围与结果，文中的耗时、loss 和本地产物位置仅指该次运行。当前复现步骤以[工程 README](README.md)为准，实现状态变化后应另追加带日期的记录，不把历史验证等同于新环境已通过。

工程位于仓库的 `foodlens_engineering/`。该次未修改报告、原始 data 或已有审计文件。Python 3.12 隔离venv安装CPU torch 2.9.1+cpu / torchvision 0.24.1+cpu；无预训练权重下载，无GPU训练。

文中 `artifacts/` 均相对工程目录，日志、checkpoint、原始 freeze 不随 Git 交付，干净克隆不存在这些文件；它们是历史证据位置，不是复现输入。可提交的依赖锁定文件是 `requirements-lock-cpu.txt`，新运行按 README 生成自己的 manifest 和产物。

## 实际通过的验证

- **21/21 unittest通过**，执行测试0.344秒，不含Python/torch导入启动时间。日志 `artifacts/unit-tests.txt`。
- 真实Torchvision ResNet-18，**weights=None**，真实224×224输入，layer4输出(1,512,7,7)、GAP(1,512)、投影(1,256)；冻结BN统计不动；layer4选择解冻。
- tiny随机卷积视觉骨干下，A、B、B-global、Hybrid实际CPU前向/反向，GRU/视觉投影/网格投影/回归头梯度非零；tiny隐层32与正式256明确区分。
- B的未来token改变不影响过去logits；过去PAD位置的嵌入人为改为1000不影响最后有效token，独立验证key padding mask；二维位置不混淆行列。
- BOS/name1/name2/SEP/EOS右移后的SEP索引为2，回归取产生SEP logits的状态；推理脚本也验证该位置，不取下一步。
- 非PAD token数量归一化CE、空食材重量损失为0、非负回归、无重量总和等式约束、未闭合末项丢弃。
- A→Hybrid迁移保留原ID并精确复制旧词行和GRU，新词行保持原随机初始化；checkpoint模型/优化器/调度器/Python和torch RNG恢复。
- Schema拒绝额外字段、bool、负数、NaN/Inf、空名称、重复JSON键；非法名称作为FP保留可读重量；缺失和无法解析的预测保留全样本误差。
- 配置实际加载及字段校验；输入隔离、路径越界拒绝、分组跨split拒绝、训练统计/词表拒用非训练split；151个训练名称全部往返一致。

首次运行有3项因Windows sandbox的系统临时目录权限失败；已将测试临时文件放在工程 `artifacts/test-temp`，重新运行全部通过，保留首次日志 `artifacts/unit-tests-first-run.txt`。不需要提升测试运行权限。

## 真实图片的小样本闭环

使用原任务二训练集前4张真实RGB图像，tiny骨干、隐层32、3步AdamW（lr=0.0003）、seed42、CPU两线程。词表和统计仅拟合完整训练标注，验证与测试未参与。

| 步 | 总损失 | 序列 | 重量 | 整餐 |
|---|---:|---:|---:|---:|
| 1 | 5.893153 | 5.060433 | 0.581489 | 0.251231 |
| 2 | 5.867357 | 5.034922 | 0.581456 | 0.250980 |
| 3 | 5.841678 | 5.009534 | 0.581428 | 0.250715 |

保存再加载checkpoint后，logits/states/weights/totals逐张量完全一致。另启动RGB-only CLI逐张预测，与原batch预测的4条record及raw token ID全部一致。独立缺图manifest测试生成包含原样本ID的失败记录，未直接丢样本。

原始运行配置、tiny覆盖、数据及配置SHA256、词表、统计、逐步loss在 `artifacts/smoke/run.json`；checkpoint为 `artifacts/smoke/smoke.pt`；预测和同4样本评价为该目录下 `predictions.jsonl`、`metrics.json`。

**这些只是工程烟雾测试。** 4条tiny预测全部达到32-token上限，未闭合名称被丢弃，食材列表为空、名称F1为0。组装JSON/Schema合法率为1不意味着识别成功。无模型质量结论、无正式训练效果、无GPU成本/显存结论。

另将仅4条预测交给**完整2338条训练标签**评价，实际统计samples=2338、覆盖4/2338、其余2334条缺失预测全部作为空集合和数值0计入，证明评价不会只统计成功样本。该结果位于 `artifacts/all_train_metrics.json`，仅用于检验分母约定，不能作为任务效果分数。没有进行测试集推理。

## 数据和依赖证据

`artifacts/data_contract_check.json`：训练/验证/测试2338/414/506；训练151个全名；固定通用标点规则后全名往返差异0。训练标注SHA256为 `29a3742405361c00f8f8908bec7c5944a762b3e5529e5fd9e84d6bc95265c856`。

本地Food500 JSON实际19880/4940条，训练数字与官方README所述19760存在差异，原因尚未核实；尚无可用图片唯一映射manifest。该次只读取JSON计数/结构，没有全量图片或parquet审计。

`pip check`：No broken requirements found。原始freeze在 `artifacts/environment-freeze.txt`；跨目录安装使用 `requirements-lock-cpu.txt`。官方CPU wheel SHA256 `a4e06b4f441675d26b462123c8a83e77c55f1ec8ebc081203be2db1ea8054add`，与HTTP官方checksum一致。pip初次网络下载不出进度，使用curl官方URL及Range续传完成，未使用第三方镜像。

## 如何重跑

按[工程 README 的环境与复现步骤](README.md#环境与复现)执行，包含标准环境安装、无数据/完整测试、RGB manifest 生成、4 样本 smoke、独立预测、同子集评价及可选全训练集分母检查。新产物默认位于 `artifacts/repro/`，不覆盖上文的历史输出。

下一阶段完整清单和历史官方接口调研链接在 README。主要未完成项：Food500唯一manifest及caption正式训练/四指标后端；正式ResNet预训练权重；完整GPU训练/早停/参数组/混合精度；Qwen revision/Processor/assistant-only mask和LoRA显存试跑；SCST、Judge/人评均仅保留接口。当前交付与平台约定见[项目 README](../README.md)，历史 CPU 结果不代表 Linux GPU 训练已验证。

## 历史环境排障

该次使用的 bundled Python 初始化 ensurepip 失败，实际由带 pip 的 Python 通过 `python -m pip --python .venv\Scripts\python.exe install -r requirements-lock-cpu.txt` 安装到隔离环境。该方式要求调用者已有可用且支持 `--python` 的 pip，不是新成员标准复现的前置条件。原始 freeze 含本地 wheel 路径，不能直接当作可移植安装文件。

## 文档命令复核（2026-11-02）

本次完善文档后，在现有 Windows 环境以工程 venv 的 Python 3.12.14 重跑，使用已有课程数据；未从空环境重新下载安装依赖，也未进行 Linux/CUDA 或正式模型训练验证。

- 按[数据 README](../data/README.md)中的原始代码块校验：三个划分的标注哈希、条数、split、全局 ID 唯一性和所有引用图片路径通过，共 3,258 个样本。此检查未做图片内容重复和完整解码审计。
- 完整 unittest 21/21 通过；无数据 `ModelTests` 7/7 通过。
- 现有 venv 没有 pip 模块；使用系统 pip 的 `python -m pip --python .venv\Scripts\python.exe check` 通过，无依赖冲突。工程 README 已补相应替代步骤。
- 从新建的 `artifacts/repro/` 开始执行工程 README 中 manifest 生成、4 样本/3 步 smoke、独立预测、同子集评价和可选完整训练集分母检查，命令均成功。
- manifest 严格只有 `id/image`；独立预测与 smoke 的 4 条结果逐条完全一致，checkpoint 精确往返通过，训练名称 token 往返差异为空。同子集评价为 4 条，完整分母为 2,338 条，缺失的 2,334 条保留为解析失败样本。
- Markdown 本地文件/目录链接检查通过；外部论文链接未在本次重新核验。正式 PDF 和报告正文未修改，也未重新编译。

本次仍是工程复核，名称 F1 为 0、4 条生成均截断，不产生模型质量结论。复核输出只保留本地；新成员仍按 README 自行生成。
