# FoodLens 工程准备

实施基线为[第五章技术方案](../docs/report/sections/technical_plan.tex)。所有代码、虚拟环境和实验产物均在本目录；未改报告或原始数据。当前交付是可运行的工程基础，CPU 小样本验证不能代表模型效果。

## 环境与复现

Windows PowerShell，从本目录运行：

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -v
& $py -m foodlens.cli smoke --data-root ..\data --samples 4 --steps 3 --out artifacts\smoke
& $py -m foodlens.cli predict --checkpoint artifacts\smoke\smoke.pt --data-root ..\data --manifest artifacts\rgb_manifest.jsonl --samples 4 --out artifacts\prediction_only.jsonl
& $py -m foodlens.cli evaluate --annotations ..\data\annotations\train.jsonl --predictions artifacts\prediction_only.jsonl --out artifacts\all_train_metrics.json
```

`artifacts/rgb_manifest.jsonl` 只含 `{"id":"...","image":"images/train/...png"}`。`predict` 不接收 target、target_text 或 metadata。最后的评价示例故意使用完整训练标签，缺失的预测仍进入全部样本分母；它不是验证/测试结果。

另建环境时使用 Python 3.12：`python -m venv .venv`，再 `python -m pip --python .venv\Scripts\python.exe install -r requirements-lock-cpu.txt`。本机 bundled Python 的 ensurepip 初始化失败，所以实际使用其 pip 的 `--python` 安装到隔离 venv；这不影响项目运行。原始依赖freeze见 `artifacts/environment-freeze.txt`，包含本地wheel路径；可移植版本锁定见 `requirements-lock-cpu.txt`。本工程不下载任何模型权重。

实际21项测试和小样本运行的结果、限制与日志位置见 `IMPLEMENTATION_STATUS.md`。

## 已实现模块

| 文件 | 责任 |
|---|---|
| `foodlens/data.py` | 原划分任务二数据、RGB-only 推理、Caption 显式映射 manifest、图像预处理、路径约束、训练统计、右移序列及 SEP 位置 |
| `foodlens/config.py` | 基线配置加载和字段/维度/优化器/生成参数校验；CLI记录tiny覆盖与配置哈希 |
| `foodlens/text.py` | 保留标点的词级切分、训练词扩充、固定旧 ID |
| `foodlens/models.py` | A / B / B-global / Hybrid、实际 ResNet-18 layer4、烟雾测试 tiny encoder、按词迁移 |
| `foodlens/losses.py` | 非 PAD 计数归一化 CE、边界重量及五维标准化 Smooth L1 |
| `foodlens/decode.py` | RGB-only 贪心解码、合法边界限制、SEP 原位置回归、未闭合项丢弃、失败追踪 |
| `foodlens/schema.py` | 严格 JSON/Schema、名称规范化、重复项合并、一位小数交付 |
| `foodlens/metrics.py` | 全样本精确名称 Micro/Macro F1、匹配/并集重量误差、整餐误差及 JSON/Schema 有效率 |
| `foodlens/checkpoint.py` | 模型/优化器/调度器/词表/统计/配置/版本/随机状态 |
| `foodlens/cli.py` | 真实任务二图片 tiny CPU 训练、checkpoint 精确往返、无标签预测及全样本评价 |
| `foodlens/extensions.py` | Qwen 固定指令及训练消息适配；Caption指标/SCST/Judge的待实现边界 |

默认正式架构保持 d=256、GRU 输入512、两层4头Transformer、FF1024、Dropout0.1。B 的网格模式有49个视觉Token和2D正弦位置；B-global有1个视觉Token。tiny smoke 明确将隐层降至32并使用随机小卷积；它只检验链路。真实 ResNet 测试使用 `weights=None`，检验224输入产生512×7×7及冻结BN；正式ImageNet权重尚未下载或验证。

回归头始终先输出原单位 Softplus 非负数，再按训练统计变换两侧计算损失。可选偏置初始化为训练均值的 inverse-softplus，使大尺度数值有合理起点；不改变数学目标。未强制食材重量之和等于整餐重量。骨干默认冻结；解冻layer4接口保留固定BN，正式训练时应为其设置主学习率的十分之一。

## 数据和评价边界

任务二 `annotations/*.jsonl` 的 image 相对 data 根；监督只来自 target。词表和数值统计仅由训练标签拟合。CPU smoke 使用前4张真实训练图像，词表/统计仍从完整训练标签拟合（没有训练验证/测试图像的混合）；它不是对小数据过拟合能力或模型质量的试验。

Food500 官方 JSON 是列表，其记录字段为 cat/filename/caption。本地实际读取 finetune_data.json 为19,880条、evaluation_data.json 为4,940条；本地训练条数与论文19,760存在差异，来源版本及差异原因需在manifest阶段核实，不能将本地计数直接当作清洗后计数。社区 parquet 与官方 caption 的唯一映射、哈希与重复分组 manifest 尚未建立；本轮没有按行号拼接、没有重新全量图片审计。Caption loader 要求显式的 id/image/caption/official_filename/sha256/duplicate_group/split，校验图片哈希，另有跨划分重复组检查器。正式训练应先执行该检查器并锁定排除清单和最终词表。

评价以 ground-truth ID 为全集，额外预测 ID 单独报告；预测缺失或JSON不可解析使用空食材/整餐零预测。语法可读但Schema失败时保留可读名称和非负有限数值，其余数值按0；非法名称保留为互不匹配的FP并计其可读重量误差。此约定在代码和测试中固定。cleanup item failures 即便最终JSON合法，也作为FP额外进入指标。非有限整餐输出保留为JSON null并记Schema失败，评价该字段按0。无匹配项时匹配误差为 null。有效子集额外给覆盖率和同套误差，不替代全样本结果。

词级 detokenize 使用固定标点连接规则（开括号前保留词间空格，括号内去多余空格）。已为全部151个训练全名加入往返回归测试，包含 spinach (cooked)/spinach (raw)；训练名称的逐项 roundtrip 差异写入 smoke 的 run.json。开放名称的标点空格仍由固定规则处理，不能用测试食材词典修复或全名白名单约束。

## 正式实现和GPU训练清单

1. 完成并锁定 Food500 caption↔图片唯一manifest、来源revision、文件哈希、重复分组/排除清单；测试归属优先，剩余训练按组分层seed42划验证。补词表覆盖率和最大64-token截断统计。
2. 保持训练集锁定的标点规则，审查开放名称的标点空格与UNK失败；验证未见全名/组成词分层。不在解码时查训练全名白名单。
3. 接入正式Caption数据collator/训练器、版本固定的BLEU-4/METEOR/ROUGE-L/CIDEr-D实现及单位，按验证CIDEr-D选A/B；当前只有指标协议接口，不能声称Caption分数已实现。
4. 加载明确版本的ImageNet ResNet18权重，先冻结再视验证解冻layer4；实现参数组低学习率、早停、混合精度/梯度累积、CUDA随机状态复现、恢复训练命令。当前CLI仅tiny CPU工程smoke，不是完整GPU训练器。
5. 跑A/B/B-global相同贪心协议。任务二从最佳A按旧ID和字符串迁移，比较仅视觉与视觉+GRU；按MicroF1、五维标准化MAE、有效率的固定规则选模型。训练/自由生成分别留追踪。
6. 固定任务二长度/项数上限并报告训练标签截断覆盖率；分层统计未见名称、长列表、大份量、疑似同餐及重量不守恒标记，标记不作输入。
7. Qwen固定4B revision/Processor/Chat Template、语言Q/K/V/O实际路径和可训练清单；单样本检查assistant-only loss及结束标记；先BF16 LoRA显存试跑，必要时NF4。4B不合则同一2B成对重跑。模型及配套依赖尚未下载。
8. 实测24GB单卡显存、吞吐、参数量、实际费用；总预算300元，先必选实验再任选。不得从CPU smoke推断GPU资源或效果。
9. SCST固定训练参考CIDEr-D DF、EOS概率/PAD mask、采样/贪心一致模式及stop-gradient，另跑等步数CE对照。Judge锁定原始冻结revision、匿名协议、约60图像ID及双人评分/Bootstrap；目前均仅接口。

待确认只有实施资源/策略：最终GPU和批量、正式模型revision/权重下载、Caption manifest来源和本地训练条数差异。没有需要修改第五章架构的分歧。

## 官方接口依据

2026-11-01 定向查阅，具体运行API以锁定的本地版本为准：

- [Torchvision ResNet18源码](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)：显式weights参数，layer4后再avgpool/fc。
- [PyTorch GRU](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html)：batch_first只改变输入输出排列，h0仍为层×批×隐层。
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html)：batch_first、布尔mask的True表示屏蔽、key padding mask形状。
- [PyTorch Transformer源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)：使用标准Encoder/Decoder层，显式因果mask，逐层重新初始化。
