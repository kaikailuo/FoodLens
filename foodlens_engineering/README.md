# FoodLens 工程与 CPU 复现

实施基线为[第五章技术方案](../docs/report/sections/technical_plan.tex)。agent 先读根 [AGENTS.md](../AGENTS.md)。当前交付是可运行的工程基础，CPU 小样本验证不能代表模型效果。2026-11-01 的历史运行记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

| 能力 | 当前状态 |
| --- | --- |
| A / B / B-global / Hybrid 模块 | 已实现，并有 CPU 前向、反向等测试 |
| CLI | 仅 `smoke`、`predict`、`evaluate`，均为当前 CPU 路径 |
| Caption 数据 | 有显式 manifest loader 及重复组校验函数；正式 manifest、collator、训练器、四指标后端未完成 |
| 正式 ResNet/GPU 训练 | 未完成预训练权重验证、训练入口和完整恢复训练流程 |
| Qwen | 仅固定指令及消息构造；未接入实际 Processor、模型推理或 LoRA 训练 |
| SCST / Judge | 仅协议接口；无可报告的实验结果 |

## 环境与复现

主要开发和正式训练在 Linux GPU 服务器进行；Windows 本机用于轻量检查。当前 CLI 仍只有 CPU 链路，以下 Linux 命令用于服务器上的 CPU 预检查，不是 CUDA 训练配方。已有实测记录来自 Windows，Linux 命令尚未在 Linux 主机执行验证。

Python 基线为 3.12，数据根统一为项目根 `data/`。各系统分别创建 venv，不跨系统复制环境；`requirements-lock-cpu.txt` 仅用于 CPU 检查，不应覆盖已有 GPU 环境。正式 CUDA 依赖和训练入口需在实现时另行锁定并验证。

### Linux 服务器的 CPU 预检查

从项目根目录开始，在 Bash 中执行。已有本工程 CPU `.venv` 时跳过创建行：

```bash
set -e
cd foodlens_engineering
python3.12 -m venv .venv
py=./.venv/bin/python
"$py" -m pip install -r requirements-lock-cpu.txt
"$py" -m pip check
"$py" -m unittest discover -s tests -v -k ModelTests
```

无数据时到此为止，预期 7 项测试通过。取得任务二数据并按[数据说明](../data/README.md)校验后，保持工作目录为 `foodlens_engineering/`，在同一 Bash 会话中执行下面的完整 CPU 闭环。输出与下文 Windows 步骤相同，重复运行会覆盖本次 `artifacts/repro/`：

```bash
"$py" -m unittest discover -s tests -v
"$py" - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("../data/annotations/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:4]
assert len(rows) == 4, "need four training samples"
out = Path("artifacts/repro")
out.mkdir(parents=True, exist_ok=True)
def write_jsonl(name, records):
    (out / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
write_jsonl("rgb_manifest.jsonl", [{"id": row["dish_id"], "image": row["image"]} for row in rows])
write_jsonl("annotations_4.jsonl", rows)
print("prepared 4 RGB-only entries and 4 evaluation annotations")
PY
"$py" -m foodlens.cli smoke --config configs/baseline.json --data-root ../data --samples 4 --steps 3 --out artifacts/repro/smoke
"$py" -m foodlens.cli predict --checkpoint artifacts/repro/smoke/smoke.pt --data-root ../data --manifest artifacts/repro/rgb_manifest.jsonl --samples 4 --out artifacts/repro/prediction_only.jsonl
"$py" -m foodlens.cli evaluate --annotations artifacts/repro/annotations_4.jsonl --predictions artifacts/repro/prediction_only.jsonl --out artifacts/repro/metrics_4.json
```

完整测试预期 21 项通过；产物、4 样本验收条件和失败分母含义见下文步骤 3—5。如另开 Bash 会话，先进入 `foodlens_engineering/` 并重新设置 `py=./.venv/bin/python`。正式 GPU 训练实现前不要自行拼接 `train` 或 `--device cuda` 参数。

### Windows 本机的 CPU 检查

以下步骤使用 PowerShell。**所有命令的工作目录都是 `foodlens_engineering/`**；如在项目根，先执行 `Set-Location foodlens_engineering`。逐块执行，确认每条命令退出码为 0 后继续，避免上一步失败却读取旧产物。

### 1. 创建环境与无数据检查

确认 `python --version` 为 3.12.x，已有本工程 `.venv` 时跳过创建命令。无需激活环境，始终通过 venv 解释器运行：

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-lock-cpu.txt
& $py -m pip check
& $py -m unittest discover -s tests -v -k ModelTests
```

预期 `pip check` 无依赖冲突，`ModelTests` 的 7 项测试通过。这些测试不依赖完整数据、不下载权重。安装需要访问 PyPI 和官方 PyTorch CPU wheel 源；`requirements-cpu.txt` 只锁直接依赖，复现使用完整的 `requirements-lock-cpu.txt`。这份锁定用于当前 CPU 环境，不是 GPU 依赖文件。

如 `venv` 创建或 pip 初始化失败，先确认使用正常安装的 Python 3.12，记录具体错误。已有环境若报 `No module named pip`，而系统 Python 有支持 `--python` 的 pip，可通过外部 pip 管理目标 venv：

```powershell
python -m pip --version
python -m pip --python $py install -r requirements-lock-cpu.txt
python -m pip --python $py check
```

这会安装到 `$py` 指定的环境；外部 pip 所属 Python 与目标 venv 可不同，但目标仍须为 3.12。若仅 ensurepip 初始化失败，可用已确认的 Python 3.12 执行 `python -m venv --without-pip .venv`，再用上述外部 pip 安装。系统 Python 也没有可用 pip 时，应先修复 Python/pip 安装。两种安装方式都需完成依赖检查和后续测试；本机历史处理另见实现记录。

### 2. 准备数据并运行完整测试

按[数据说明](../data/README.md)取得任务二数据，并从项目根执行其中的校验命令，再回到本目录。不需要 Food500 数据即可运行以下闭环。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -v
```

当前预期 21 项测试通过。缺少真实训练标注或图像会使部分测试失败；不要把缺数据报告成模型实现失败，也不要直接跳过后声称全部通过。

### 3. 生成本次 RGB manifest 和评价子集

下面只提取原训练集前 4 条的 ID 和图片路径供推理，另存同 4 条标注供离线评价。原始文件不变，监督不进入推理 manifest。

输出统一放 `artifacts/repro/`。该目录不存在时自动创建；重复执行会覆盖本次复现产物。如需保留多次实验，请先将代码块及后续命令中的 `artifacts/repro`（含反斜线写法）一起改成新的子目录。

```powershell
@'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("../data/annotations/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:4]
assert len(rows) == 4, "need four training samples"
out = Path("artifacts/repro")
out.mkdir(parents=True, exist_ok=True)
def write_jsonl(name, records):
    (out / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
write_jsonl("rgb_manifest.jsonl", [{"id": row["dish_id"], "image": row["image"]} for row in rows])
write_jsonl("annotations_4.jsonl", rows)
print("prepared 4 RGB-only entries and 4 evaluation annotations")
'@ | & $py -
```

`rgb_manifest.jsonl` 每行严格只有 `id`、`image`，其中 ID 对应标注的 `dish_id`，图片路径相对 `data/`。`predict` 拒绝包含 target、target_text 或 metadata 的 manifest。

### 4. tiny smoke、独立预测和同子集评价

```powershell
& $py -m foodlens.cli smoke --config configs\baseline.json --data-root ..\data --samples 4 --steps 3 --out artifacts\repro\smoke
& $py -m foodlens.cli predict --checkpoint artifacts\repro\smoke\smoke.pt --data-root ..\data --manifest artifacts\repro\rgb_manifest.jsonl --samples 4 --out artifacts\repro\prediction_only.jsonl
& $py -m foodlens.cli evaluate --annotations artifacts\repro\annotations_4.jsonl --predictions artifacts\repro\prediction_only.jsonl --out artifacts\repro\metrics_4.json
```

成功条件：三条命令退出码为 0；smoke 输出 `samples=4`、`steps=3`、`checkpoint_roundtrip_exact=true`；独立预测恰好 4 行且 ID 与 manifest 一致；评价 `samples=4`、`prediction_extras=[]`。`run.json` 的 `train_name_token_roundtrip_failures` 应为空。名称 F1、Schema 有效率和 loss 的具体值不作为工程验收门槛。

| 输出（相对本目录） | 用途 |
| --- | --- |
| `artifacts/repro/rgb_manifest.jsonl` | 无标签推理清单，由步骤 3 生成 |
| `artifacts/repro/annotations_4.jsonl` | 同 4 条训练标注，仅供评价 |
| `artifacts/repro/smoke/smoke.pt` | 本次 tiny checkpoint |
| `artifacts/repro/smoke/run.json` | 配置、tiny 覆盖、哈希、词表、统计、loss 和精确往返结果 |
| `artifacts/repro/smoke/predictions.jsonl`、`metrics.json` | smoke 内部预测与同子集评价 |
| `artifacts/repro/prediction_only.jsonl`、`metrics_4.json` | 独立预测与同子集评价 |

这些文件均由命令生成且被 Git 忽略，干净检出无需附带历史 `artifacts/`。tiny smoke 的骨干、维度、dropout 和生成长度有显式覆盖，应以 `run.json` 为准；它只执行 3 步且可能产生空食材或全部截断，不代表模型效果。

### 5. 可选：检查完整分母约定

```powershell
& $py -m foodlens.cli evaluate --annotations ..\data\annotations\train.jsonl --predictions artifacts\repro\prediction_only.jsonl --out artifacts\repro\all_train_metrics.json
```

本步故意以完整训练标签评价仅 4 条预测，预期 `samples=2338`，缺失的 2334 条仍按空食材/整餐零预测进入统计；输出当前没有独立的“预测条数覆盖率”字段，可由 manifest 和预测 ID 数核对。它只检验评价分母，不是验证/测试集效果。`valid_subset_coverage` 表示 Schema 有效样本覆盖率，与收到多少条预测含义不同。

当前 `predict --samples` 默认为 4，不能靠省略参数进行全量推理；后续全量评价时显式传入相应正整数，并核对预测 ID 全集。当前 CLI 没有 `train`、`resume`、`--device cuda` 等正式训练选项，不要从方案推导出不存在的命令。

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

Food500 官方 JSON 是列表，其记录字段为 cat/filename/caption。本地读取 finetune_data.json 为19,880条、evaluation_data.json 为4,940条；本地训练条数与上游说明19,760存在差异，来源版本及差异原因需在manifest阶段核实，不能将本地计数直接当作清洗后计数。社区 parquet 与官方 caption 的唯一映射、哈希与重复分组 manifest 尚未建立。Caption loader 要求显式的 id/image/caption/official_filename/sha256/duplicate_group/split，读取图片时校验哈希；`foodlens.data.validate_manifest_splits(rows)` 是跨划分重复组校验函数，尚无对应独立 CLI。正式训练入口应先对完整 manifest 调用该函数，并锁定排除清单和最终词表。

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
8. 在 Linux GPU 服务器实测显存、吞吐、参数量与训练配置；24GB 单卡是方案的起始配置参考，按实际硬件验证。约300元为早期估算，不作为实施预算上限，不以费用为由省略必选任务或当前选择的 SCST/Judge；不得从CPU smoke推断GPU资源或效果。
9. SCST固定训练参考CIDEr-D DF、EOS概率/PAD mask、采样/贪心一致模式及stop-gradient，另跑等步数CE对照。Judge锁定原始冻结revision、匿名协议、约60图像ID及双人评分/Bootstrap；目前均仅接口。

课程范围见[原始要求整理稿](../docs/COURSE_REQUIREMENTS.md)，小组运行平台与交付约定见[项目 README](../README.md)。选做为三选二，现有方案选择 SCST 和 Judge；两者均属计划交付。任务二还须补齐任务一的文本评测指标，固定 JSON/食材文本的序列化、Tokenizer、指标版本与单位，同时保留名称 F1、数值 MSE 和全样本结构评价，文本指标不替代这些结构化指标。

模型 revision、批量和数据计数差异由实施 agent 调查、试跑并记录。正式训练前固定模型选择容差、五维标准化 MAE 的聚合规则、早停和指标版本等可执行细则；上述清单不代表已有完整训练配方。需要用户决定的问题在聊天提出，不向项目添加待确认清单。

## 官方接口依据

2026-11-01 定向查阅，具体运行API以锁定的本地版本为准：

- [Torchvision ResNet18源码](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)：显式weights参数，layer4后再avgpool/fc。
- [PyTorch GRU](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html)：batch_first只改变输入输出排列，h0仍为层×批×隐层。
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html)：batch_first、布尔mask的True表示屏蔽、key padding mask形状。
- [PyTorch Transformer源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)：使用标准Encoder/Decoder层，显式因果mask，逐层重新初始化。
