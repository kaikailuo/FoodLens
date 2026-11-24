# CPU 工程验证记录（2026-11-01）

本文保存 2026-11-01 的历史验证范围与结果，文中的耗时、loss 和本地产物位置仅指该次运行。当前复现步骤以[工程 README](README.md)为准，实现状态变化后应另追加带日期的记录，不把历史验证等同于新环境已通过。

**当前状态请结合文末 2026-11-24 Hybrid V2 实施记录及 [Qwen 第三轮整改记录](QWEN_ROUND3.md)；11-20 Hybrid 为旧 GRU/SEP 实现，Qwen 为旧自主训练器，旧任务二文本仅含名称。** 下列历史结果及未完成项保留，不能据此判断新代码能力。

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

## 代码建设与合成工程验收（2026-11-20）

工作基于提交 a60c248693b99ba05da8f05677a4b31cc0cac98e，改动保留在工作区，FoodLens 版本为 0.2.0。执行前已有根 README、报告 README/PDF、context README 及学生资源包修改；这些已有内容未撤销或覆盖，报告/PDF 本次未编辑。根 README 仅同步本次工程入口。运行元数据另存实际 Python 源码树 SHA256，父提交不代表本次未提交代码的完整版本。

本次范围是可运行代码、必要测试和交接，不获取/整理真实数据，不下载大模型，不运行正式长训练或付费服务。基于 Git `a60c248693b99ba05da8f05677a4b31cc0cac98e` 的未提交工作区，工程版本为 0.2.0；run.json 记录实际 Python 源码 SHA256、Git revision、tracked diff SHA256 和依赖。用户原有 README/报告/PDF/资源包修改予以保留，本次未编译或改写报告/PDF。

### 已实现并接通

- 公共配置、CPU/CUDA/精度控制、随机种子、可选依赖预检查、实验日志；v2 自主 checkpoint 保存模型/优化器/调度器/scaler/epoch/update/选模及 Python、NumPy、Torch、CUDA RNG。支持完整 epoch 恢复，不支持中途 batch/未更新梯度恢复。
- Caption A/B/B-global 的 collator、右移监督、长度/截断/UNK 审计、正式训练入口、骨干冻结/按配置解冻 layer4、低学习率组、累积/裁剪/混合精度、验证早停、best/last 和独立 RGB 推理。
- BLEU-4/METEOR/ROUGE-L 标准后端及固定 CIDEr-D scorer，版本、word_v1 分词、原始单位、离线 DF 与训练奖励 DF 协议明确区分；正式 Caption CIDEr-D 选模不降级。METEOR 连接代码已实现，Java 实际评分未执行。
- Hybrid 两种迁移、仅训练文本扩词且保留旧 ID、SEP 原状态重量回归、五维整餐回归、自由生成和严格 JSON 组装；完整标签超长拒绝截断。统一任务二文本/名称 F1/数值误差/结构有效率及全样本失败分母。
- Qwen3-VL 模型/Processor revision、显式 fast/slow、模板/Tokenizer/图像配置指纹、assistant-only collator、真实语言 Q/K/V/O LoRA、训练/恢复/保存和原始/微调配对推理。不可变 epoch bundle + SHA256 指针避免半写 checkpoint；基础权重外置。本地 Qwen 不需要 API Key。
- SCST 真实采样与贪心基线、固定训练 DF、stop-gradient advantage、EOS/PAD log probability、CE 混合及奖励/梯度追踪；等步数 CE 继续训练为独立配置。
- Judge 冻结本地 VLM 与独立环境变量 API 后端、严格四维评分/证据解析、原始响应/失败/截断记录、稳定匿名 ID、候选随机化、双人人评导入导出、Spearman、按图像 Bootstrap 和 quadratic weighted Cohen kappa。每种相关性用其自身完整配对并报告样本覆盖，不填失败评分。

来源固定版本及许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。自主架构和研究对照仍按第五章，工程优化不构成准确率/显存/吞吐收益结论。

### 本次实际环境及结果

Windows 11、Python 3.12.14、Torch 2.9.1+cpu、Torchvision 0.24.1+cpu、NumPy 2.2.6、Pillow 11.3.0、SciPy 1.15.3、pycocoevalcap 1.2、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.10.1；CUDA 不可用。完整安装快照为 requirements-lock-test-cpu.txt。

最初默认 Anaconda/Python 3.13.5 环境出现 OpenMP 重复运行库错误；没有采用 KMP_DUPLICATE_LIB_OK 绕过。改在本次独立 Python 3.12 venv 安装 CPU 依赖，pip check 无冲突。只下载了代码依赖及指标包，没有下载真实数据或预训练模型。

| 验证 | 实际结果 | 本地日志/产物（均被忽略，不是复现输入） |
| --- | --- | --- |
| 新增 tests/test_engineering.py | 16/16 通过，无 skip；约 9 秒，导入启动时间另计 | 根 outputs/goal-20261120/engineering-final.log |
| 既有 ModelTests | 7/7 通过 | 根 outputs/goal-20261120/model-final.log |
| 既有 test_baseline.py 全组 | 19 通过、2 项因缺 data/annotations/train.jsonl 报 FileNotFoundError，退出码 1 | 根 outputs/goal-20261120/baseline-final.log |
| 独立 CLI 合成验收 | 24 条命令全部成功 | 工程 artifacts/goal-20261120/cli-final/commands.json、acceptance.json；根 outputs/goal-20261120/cli-final.log |
| 依赖及静态检查 | pip check、compileall、Git whitespace 检查通过；文档链接/配置检查另存检查日志 | 根 outputs/goal-20261120/ |

从工程目录运行的实际测试命令（本次解释器位于 artifacts/goal-20261120/venv/Scripts/python.exe）：

```powershell
& artifacts/goal-20261120/venv/Scripts/python.exe -m unittest discover -s tests -p test_engineering.py -v
& artifacts/goal-20261120/venv/Scripts/python.exe -m unittest discover -s tests -p test_baseline.py -k ModelTests -v
& artifacts/goal-20261120/venv/Scripts/python.exe -m unittest discover -s tests -p test_baseline.py -v
& artifacts/goal-20261120/venv/Scripts/python.exe -m scripts.synthetic_acceptance --out artifacts/goal-20261120/cli-final
```

Qwen 测试启用 HF_HUB_OFFLINE=1：构造随机 tiny Qwen、合成 byte-level 词表和真实 Processor，保存为本地小型 HF 文件，再通过实际 load_qwen 重新加载。验证图像前向/反向、语言 LoRA 梯度/视觉冻结、assistant/EOS/PAD mask、完整训练与中断恢复后的 adapter 权重完全相同、实际原始/微调配对推理及冻结 Judge 解析失败的保留。适配器指针还测试了文件损坏拒绝加载。这不是 Qwen3-VL-4B 官方权重或能力验证。

其他新增测试实际检查了全窗口 token 分母、Caption 右移截断、CPU 恢复模型/优化器/调度器及 RNG 一致性、视觉迁移对照、真实 CIDEr-D 奖励及与 pycocoevalcap 等价、SCST 奖励差/梯度方向/EOS 后 PAD、匿名人评往返、缺失服务配置不返回默认分、逐图 Bootstrap 和加权 kappa。只有外部服务的响应边界使用明确标记的测试后端，不替代奖励、梯度或统计算法。

CLI 合成 manifest SHA256 为 `2a0af8cb19a1adbda65f3b9ca7ff2856ec2e042ea0de67d98950e36c8c2d3b44`，共 8 张程序生成 RGB、train/val/test=4/2/2。推理清单另有一条故意缺图项，全部 3 个 ID 保留输出。A/B/B-global、两种 Hybrid、CE 继续/SCST、恢复、三文本指标及匿名工具均通过公共 CLI 调用。原始 fixture、模型与结果均在独立 artifacts 子目录，明确标为 SYNTHETIC_ONLY，不能当作课程测试集或模型效果。

原有两项数据检查没有伪造真实输入或改为成功。合成数据只在测试/验收专用目录生成，不写入 data/。本次没有训练集/验证集/测试集真实效果报告。

### 尚未验证及接收人需要的资源

1. 真实 Food500 图文映射、来源 revision、哈希、近重复组/划分，以及学校任务二数据；执行 README 的 audit 再训练，不依赖历史本机文件。
2. ImageNet ResNet18 和固定 revision 的 Qwen3-VL-4B-Instruct/Processor 本地权重；Qwen 默认 local_files_only=true。完整官方权重、原始 Chat Template 与真实图像 Token 分布仍需实测，tiny 不代表 4B 兼容性已验证。
3. Linux CUDA 机器：BF16 LoRA、可选 NF4/bitsandbytes、实际批量/显存/吞吐/确定性和正式训练均未运行。取得后先短试跑并记录资源，再执行完整配置。
4. Java METEOR 运行环境：本次只验证标准 BLEU-4、ROUGE-L、CIDEr-D 和 Java 缺失时的清晰错误。正式四指标不得省略或静默替代 METEOR。
5. 真实冻结 Judge（本地权重或兼容 API 的 Key/地址/模型固定版本）及两名独立人评；API 环境变量可空，尚无真实 Judge 分数/人评一致性结论。

各方法的具体安装、训练、恢复、推理及评价命令均在 [README](README.md)，例配置位于 configs/，无密钥示例为 .env.example。报告/PDF、正式实验、可视化结果和课程提交材料不属于本次代码验收结果。

## Caption 修复与可选缓存验收（2026-11-24）

基于 `htl_1120` 的 `ebde49a35b77be285d6343343e0efa96dc74712a`，在 `codex/caption-cache-fixes` 实施；按用户指令完成后提交到本地 main，不执行远程 push。保留用户原有报告 README/PDF、context README、学生资源包及原始 Caption 交接文档，不纳入修复提交。上面的 11 月 20 日记录保持历史含义。

### 实现范围

A/Hybrid 词嵌入 Dropout 生效；B/B-global 不计算无用的条件投影，保留旧参数及优化器布局。新增可选 GRU hidden 和真实每层 self/cross KV 缓存，默认关闭；接通训练验证、CLI 独立预测、恢复覆盖、配置示例与协议元数据。A SCST 的 GRU hidden 不 detach，奖励、loss 及参数梯度等价已用真实张量和 CIDEr-D 验证；B 的 teacher forcing 不走 KV 路径。旧 checkpoint 可恢复，但旧 A/Hybrid 非零 Dropout 的训练轨迹改变，会警告并记录版本。详细边界、开关和 GPU 命令见 [CAPTION_CACHE_FIXES.md](CAPTION_CACHE_FIXES.md)。

### 实际环境与运行证据

Windows 11、Python 3.12.14、Torch 2.9.1+cpu、Torchvision 0.24.1+cpu；其余版本同 `requirements-lock-test-cpu.txt`。Codex bundled Python 更新后创建本次独立 venv，复用本地已有的锁定依赖，没有下载数据/模型。CUDA 不可用。

本次下列命令从工程目录执行，解释器为 `artifacts/caption-cache-20261124/venv/Scripts/python.exe`；新成员按 README 自建环境，不依赖这个被忽略的本机目录。

```powershell
python -m unittest discover -s tests -p test_caption_cache.py -v
python -m unittest discover -s tests -p test_engineering.py -v
python -m unittest discover -s tests -v
python -m scripts.synthetic_acceptance --out artifacts/caption-cache-20261124/cli-acceptance
```

| 检查 | 本次实际结果 | 根目录下的日志/产物 |
| --- | --- | --- |
| 新增缓存与修复测试 | 17 项运行，16 通过，1 项 CUDA 跳过 | outputs/caption-cache-20261124/new-tests.log |
| 既有工程测试 | 16/16 通过，含真实随机 tiny Qwen/PEFT 恢复、SCST 和 Judge 统计 | outputs/caption-cache-20261124/engineering-regression.log |
| 全部 unittest | 54 项，51 通过、2 错误、1 CUDA 跳过；退出码 1 | outputs/caption-cache-20261124/all-tests.log |
| 独立进程 CLI 合成验收 | 24 条命令成功，输出保留故意缺图项；非正式效果 | outputs/caption-cache-20261124/cli-acceptance.log；foodlens_engineering/artifacts/caption-cache-20261124/cli-acceptance/acceptance.json、commands.json |
| 串行缓存基准 | A/B/B-global 两模式 token 一致，CPU 中位延时 61.02→10.67 / 95.09→43.38 / 88.84→41.82 ms | foodlens_engineering/artifacts/caption-cache-20261124/benchmark-serial-{A,B,B-global}.json |
| 静态/依赖/文档复核 | compileall、pip check、git diff --check 通过；本地链接/锚点、全部训练配置及 baseline、32 条 CLI 示例解析通过 | outputs/caption-cache-20261124/compileall.log、pip-check.log、docs-config-check.log |

全组的两项错误为既有 `test_input_isolation_and_qwen_boundary`、`test_training_name_roundtrip_including_parentheses` 缺少 `../data/annotations/train.jsonl`，不是断言失败；未补造课程数据，不能称全量通过。新增的输入隔离与 CLI 测试使用独立合成 fixture。全组也重跑了全部新增测试和既有 ModelTests 7 项。

本次核心 Python 源码树 SHA256 为 `c34e5b8af616fac79f431b8426f32b366b3c1886bbe9eeed65aeb59669672c2e`，基准/CLI 元数据同时记录父提交及 tracked diff SHA256。首次单组测试的 Git 子进程收到宿主不完整的 GIT_CONFIG 环境变量，源码哈希仍记录；后续全组、串行基准及 CLI 用进程内 `GIT_CONFIG_COUNT=0` 运行，Git revision 正常记录，没有修改仓库或全局 Git 配置。CLI 合成 manifest SHA256 仍为 `2a0af8cb19a1adbda65f3b9ca7ff2856ec2e042ea0de67d98950e36c8c2d3b44`。

串行基准参数：tiny hidden=32/layers=2/FF=128、CPU FP32、2 threads、batch=4、RGB=224、64 tokens、预热 3、交替顺序各测 12 次，随机 EOS bias 明确压低以测满长度。仅可用于该合成配置延时比较。CUDA 峰值为 null，不能填写估算值。原始多次延时/p95 及资源接入命令在缓存交接说明；初次并发基准保留为调试产物，不用于报告结果。

### 尚未执行

未运行 Linux/CUDA、GPU 峰值显存/吞吐、CUDA FP32/BF16/FP16 缓存等价、真实预训练 ResNet/正式权重的生成及真实数据训练/评价。GPU 测试已提供且因资源缺失明确 skip；目标机器上先运行新增测试和 benchmark，再选择适用缓存。CE 的 teacher forcing 不使用这两个缓存，不能据此声称训练显存下降。Java METEOR、真实 Qwen 权重及外部 Judge 的历史未验证边界不变。无需新增 API Key 即可使用本次 Caption 修复和缓存。


## 2026-11-24：Hybrid V2 实施与 CPU 验收

基准分支 `htl_1120`，HEAD `ebde49a35b77be285d6343343e0efa96dc74712a`，本轮改动尚未提交。需求及审查始于 11-23，产物目录沿用 `hybrid-v2-20261123`。原始数据、划分、历史审查报告及历史产物未修改；没有提交、推送或合并。架构与协议见 [HYBRID_V2.md](HYBRID_V2.md)，它替代第五章旧 Hybrid 的 GRU/SEP 方案；报告 PDF/旧图件未在本轮重制。

### 实现范围与静态检查

- `models.py:Hybrid` 独立于 Caption A，使用 49 网格投影和 2 层类别级 Transformer Decoder，没有视觉 Transformer Encoder。正式维度 256、4 头、FFN 1024、Dropout 0.1；保留原紧凑 Weight/Total Head、Softplus 和 Train 统计标准化 Smooth L1。
- 新增 `ingredients.py:IngredientVocabulary`、`serialize_ingredients`；Train 全名映射、规则版本及 SHA-256 随 run.json 和 Checkpoint 保存。同名重量确定性求和，类别/重量成对排序。`data.py` 兼容旧序列化，新权重 gather 使用输入位置 1…K。未知类别仅用于验证诊断占位，不参与类别 CE，评价仍用未经改变的原始全名实体。
- `decode.py:predict_hybrid_v2` 消费每个生成类别后回归重量，最后一项不因 EOS 或预算边界丢失；约束 ID 严格递增，允许首步 EOS，保存截断、失败与 Schema 检查结果。
- `training.py` 不再强制 GRU 初始化；可直接预训练 ResNet-18 初始化，或从 Caption A 严格迁移 `visual.features.*`。`transfer_visual_backbone` 记录所有复制键，不复制投影/GRU/词表行/新头。V2 选模增加 Union 重量 MAE，优先级为 F1→重量→标准化五项 MAE→Schema。Qwen 维持原选优。
- `checkpoint.py` 保存架构标记与完整类别映射。V2 epoch 恢复核对数据、配置、类别映射；旧 Hybrid 显式走 LegacyHybrid 推理，禁止作为 V2 恢复点。旧 smoke 显式保留 LegacyHybrid；Caption A/B 代码未改变。
- `metrics.py` 补充 Micro Precision/Recall，保持原有名称键匹配、Matched/Union、全样本数值与失败分母。`losses.py` 无须修改，继续使用原三个损失；不新增 TF-set、SCST 或重量和约束。
- 更新 `configs/train_hybrid.json`、`scripts/synthetic_acceptance.py`、`tests/test_baseline.py`、`tests/test_engineering.py`，新增 `tests/test_hybrid_v2.py`。根 README、工程 README、本记录及 HYBRID_V2.md 同步说明。`git diff --check` 无错误；无模型静态阻断项。

### 实际命令与结果

工作目录为 `foodlens_engineering/`。PowerShell 在调用前设置 `$env:GIT_CONFIG_COUNT='0'`，避免执行环境中不完整的 Git 配置计数导致版本采集失败；不修改仓库 Git 配置。

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_hybrid_v2.py -v
.venv/Scripts/python.exe -m scripts.synthetic_acceptance --out artifacts/hybrid-v2-20261123-cli
```

- 全量：47 项，45 通过、2 跳过、0 失败，40.175 秒；两个跳过项需要本机未安装的 Transformers/PEFT（Qwen），不将其称为通过。日志 `artifacts/hybrid-v2-tests.log`。
- V2 专项：10 项全部通过、0 失败、0 跳过，33.905 秒；覆盖稳定映射/同名合并/顺序置换/输入目标移位、首末重量及 PAD/BOS mask、因果与 padding、网格 Cross-Attention、各头梯度、Teacher Forcing/自回归状态一致、EOS/重复/截断/非法回归、未知类全样本评价、选优、保存加载/逐张量一致的 epoch 恢复、旧模型兼容及真实数据。日志 `artifacts/hybrid-v2-focused.log`。
- CLI 合成端到端：21 条公共 CLI 命令均退出 0，涵盖 A/B/B-global、V2 仅视觉迁移训练/推理/评价、Caption 恢复及原 CE/SCST/匿名人评工具回归。日志 `artifacts/hybrid-v2-cli.log`，逐命令 stdout/stderr 为 `artifacts/hybrid-v2-20261123-cli/commands.json`，环境与摘要为该目录 `acceptance.json`。V2 合成配置为 2 层、16 维、4 头、FFN 32、随机 tiny；不是正式 ResNet 效果。
- 正式尺寸另用随机 ResNet-18（不下载权重）确认 49×256 Memory、2 层/4 头/1024 FFN、Weight 512→256→1 和 Total 768→256→5。没有以此证明预训练效果。

### 真实数据预处理

本机确有真实课程数据。逐项复核摘要保存于根 `outputs/hybrid_v2_20261124/preprocessing.json`，仅为本次本地产物，不是复现输入。Train 映射 SHA-256 为 `00fe7d8fac7b5436abd8265afe2460f8faec854649bfcae6901b6c7c86dda5c0`。三份 JSONL SHA-256 均与 data/sources.json 一致。Train 2338 条、151 类、8368 项；全部食材及重量 ID→名称往返一致、反序后序列/重量完全一致。Validation 414 条含 5 个未见类；Test 506 条含 3 个未见类。它们未扩充类别表，未作为 UNK 真值合并计分。三个划分共 3258 张 RGB 均通过既有读取与 224×224 预处理，无缺图/解码失败。本项不训练、不选模、不改变原划分。

### 环境、限制与后续训练风险

包内 Python 源码摘要为 `8b66c8f31ccad0c953f38bd801927d72b82feec7968df01886d45391b413feb5`，实际依赖快照为 `artifacts/hybrid-v2-freeze.txt`。实际为 Windows 11 / Python 3.12.14 / torch 2.9.1+cpu / torchvision 0.24.1+cpu / NumPy 2.4.6 / Pillow 13.0.0 / scipy 1.15.3 / pycocoevalcap 1.2，CUDA 不可用；与 11-20 锁定环境不同。只补装了仓库已有声明的 six、标准指标和 scipy 依赖，没有引入新依赖。最初的依赖缺失已修复后重跑；不覆盖历史结果。

尚未验证真实 ImageNet 权重迁移后的收敛、真实数据训练与模型效果、Linux/CUDA/BF16 显存及吞吐、Java METEOR。闭集对未见食材必然漏检；长尾重量（如 2991g）、RGB 份量歧义、训练真实前缀与自由生成的差异以及既有同餐跨划分疑点仍会影响实验可信度。正式实验前需按仓库约定固定权重、数据、指标与资源试跑；本轮不修改异常标签，不扩展模型研究范围。


## 2026-11-24：Qwen 第三轮 Trainer 与 JSON 评价整改

起点 htl_1120 / f5a2014，保留已合入的 Hybrid V2。详情、文件清单、官方参考、协议变化及命令见 [QWEN_ROUND3.md](QWEN_ROUND3.md)。Qwen 训练基础设施改用 HF Trainer，保留 Assistant-only Token 分母、语言 LoRA、全量验证三层选模；原生 checkpoint/Callback 状态恢复替代自主管理训练状态。任务二统一完整规范化 JSON 四文本指标；失败数值惩罚保持，修复没有 raw/record 的失败行被人造 null 计入 JSON 合法率的边界。

最终 Windows CPU 全量 58 项通过、0 失败、0 跳过（70.858 秒，artifacts/qwen-round3-complete.log）；公共 CLI 21 条、tiny Qwen CLI 5 条退出 0。真实课程 3258 张 RGB 通过 QwenDataset 读取，最长目标经合成 tiny Processor 完整编码。Java METEOR、官方 Processor/4B、Linux/CUDA/BF16/NF4 与正式效果均未验证。模块仅在静态与 CPU 工程边界暂时冻结。

GPU Issue #3 已核对，但补充发布被自动审批拒绝，远端未修改；拟发布正文在本地 outputs/qwen_round3/issue3_appendix.md。无需因该外部发布阻塞继续阅读本地验收和代码。

## Main 合并复核（2026-11-24）

本地 Caption 修复提交 `91075d3` 与远程 main `d7cbcd2d066ecb9129ff496b6d391b5fcab40a1c` 整合。保留远程报告提交、Hybrid V2、完整 JSON 文本评价及 HF Trainer，不以旧 GRU Hybrid/旧 Qwen 手工训练器覆盖它们；同时保留 A Dropout、B 无用投影清理、两种默认关闭的 Caption 缓存。历史验收段落分别来自不同环境，本次不改写其数据可用性或分数。

六处冲突按功能整合：decode 同时保留 V2 类别生成和 Caption 缓存；原生 training metadata 同时保留类别表/架构/选模及缓存配置；Qwen 训练器和独立配置保留远程版本，Caption generation 不加入 QwenTrainConfig；配置生成脚本保留独立 Qwen 配置及三套缓存示例；两侧实现记录全部保留。修复测试中的旧 Hybrid 明确引用 LegacyHybrid，并新增实际 V2 训练/恢复测试，确认它不会收到旧 GRU Dropout 的轨迹变化警告。

本次实际仍为 Windows 11 / Python 3.12.14 / Torch 2.9.1+cpu / Torchvision 0.24.1+cpu / NumPy 2.2.6 / Pillow 11.3.0 / Transformers 4.57.1 / PEFT 0.17.1 / Accelerate 1.10.1。没有获取真实课程数据/权重、使用 GPU 或调用外部服务。解释器使用本地 `artifacts/caption-cache-20261124/venv/Scripts/python.exe`；新成员按 README 建环境。

从工程目录执行，检查日志放根 `outputs/main-merge-20261124/`，合成产物放 `artifacts/main-merge-20261124/`，均不提交：

```powershell
$env:GIT_CONFIG_COUNT='0'
$env:OPENBLAS_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
python -m unittest discover -s tests -v
python -m scripts.synthetic_acceptance --out artifacts/main-merge-20261124/cli-acceptance
python -m compileall -q foodlens scripts tests
```

本次全组 76 项：71 通过、2 错误、3 跳过，退出码 1（tests-final.log，17.460 秒）；所有实际代码断言通过，不能称全量通过。两项错误仍为旧 baseline 缺 `../data/annotations/train.jsonl`；Hybrid V2 和 Qwen 各一项真实数据检查因缺数据跳过，另一项为缓存 CUDA 测试跳过。本地 18 项缓存/修复测试包含在全组中，17 通过、1 CUDA 跳过；还覆盖远程 Hybrid V2 和 Qwen Trainer 的新增合成测试，包括梯度/监督、恢复、选模及输入隔离。

本次报告/PDF 的暂存内容来自远程已有提交；用户自己的未提交报告改动保存在 `local-report-before-main-sync` stash，不混入合并提交。Linux/CUDA、真实数据/权重与服务等未验证范围保持不变。

最终结果：全组 **76 项，71 通过、2 个缺数据错误、3 跳过**（17.460 秒，`tests-final.log`，退出码 1）。两项错误仍是旧 baseline 缺 `data/annotations/train.jsonl`；跳过为 CUDA 检查、Hybrid V2 和 Qwen 两项真实数据检查。没有代码断言失败，不能称全量通过。当前合成 CLI **21 条命令成功**（V2 只保留一条视觉迁移配置，历史 24 条记录不改写），见 `cli-acceptance.log` 和工程 `artifacts/main-merge-20261124/cli-acceptance/commands.json` / `acceptance.json`。配置、独立副本执行的配置生成器、32 条文档 CLI 参数和本地文件/锚点检查通过，见 `static-final.log`；compileall 与 pip check 通过。核心源码 SHA256 为 `8c9e12e3ad5edc66912452eec6904ccd660d31095b74bee3a578f337828c562d`，与 CLI 元数据一致。合并检查保留远程已有报告文件的原字节，不为消除其既有 CRLF/空白提示改写报告。
