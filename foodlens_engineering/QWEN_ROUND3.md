# Qwen3-VL 第三轮整改记录（2026-11-24）

整改起点：本地 `htl_1120`，HEAD `f5a2014`（完整提交以 Git 为准），已包含 Hybrid V2；上一审查点为 `ebde49a`。本次分支 `fix/qwen-trainer-json-evaluation`。原有未跟踪的 docs/reviews/、学生资源包/、数据应急包/ 保留。未重建或更改 Caption A/B、Hybrid V2 的架构和训练策略；自主训练文件仅替换公共文本评价调用。

结论：Qwen 模块可在**静态审查与 CPU 工程边界内暂时冻结**，进入下一轮模型审查；正式 GPU 训练前必须完成 [Issue #3](https://github.com/kaikailuo/FoodLens/issues/3)。本轮没有官方 4B 的训练结果、模型效果、真实显存或 GPU 吞吐结论。

## 方法对照与职责迁移

| 项目 | 修改前 | 修改后及理由 |
| --- | --- | --- |
| 成熟方法 | 自主训练循环 | 阅读 [Qwen train_qwen.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py)、[data_processor.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_processor.py)、[trainer.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/trainer.py)，对照 [Transformers 4.57.1 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)、[PEFT k-bit 准备](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization)与安装源码。采用公开 API，不复制官方训练工程 |
| 训练基础设施 | 自管 AdamW、AMP/scaler、累计、裁剪、StepLR、循环与日志 | Trainer / TrainingArguments 接管。明确采用原生 constant Scheduler 替代旧 StepLR，不声称旧优化轨迹完全等价；不新增调度消融 |
| Loss | 模型平均 CE × 当前监督 Token 数 / 全累计窗口 Token 数 | compute_loss_func：移位 Assistant CE 求和 / Trainer 提供的窗口 num_items_in_batch；不再除 accumulation。首位置标签强制忽略，使 Trainer 未移位计数等于实际移位计数 |
| Checkpoint | UUID epoch 目录、自管 training_state.pt、两套保存/恢复、带重复文件哈希的指针 | 原生 checkpoint-N、resume_from_checkpoint、优化器/Scheduler/AMP/RNG/TrainerState。初训和恢复均先相同的 inject_lora，NF4 先做相同 k-bit prepare，再由 Trainer 唯一加载 Adapter/训练状态 |
| 选模、早停 | 自管 progress/best/bad_epochs | 轻量 StructuredSelection，状态由 Trainer 原生 ExportableState 保存/恢复。标量 EarlyStoppingCallback 与 load_best_model_at_end 不能表达带容差的三层排序，因此保留此必要增量 |
| 全量生成评价 | 每 epoch 自管验证 | 仅覆盖公开 evaluate 方法：完整 Validation Teacher Forcing Loss、RGB-only 生成、公共结构/文本评价；不缓存全语料 logits，不接触测试集 |
| 保留增量 | 数据划分、Mask、LoRA 范围、固定协议、失败留存 | 均保留；动态获取结束 Token；RGB Dataset/官方 Chat Template/Processor；监督不截断；语言 Q/K/V/O 精确路径；原始和微调模型同一配置/输入/贪心协议 |
| 删除增量 | 手写优化与训练状态系统、重复 Adapter 保存、旧指针完整性系统、Qwen 中自主模型字段 | 不再保存第二份训练状态；best.json/last.json 只作为 CLI 相对路径别名。独立 QwenTrainConfig，移除 model/family/transfer/layer4/SCST 等无效字段；旧公共配置兼容投影，非空 updates_per_epoch 显式拒绝 |

本次读取的官方文件 blob 分别为 `f67d56963a4615e8b6f3a26b19b112888dc92af6`、`754838ca808cc44f01737878ec67c2fdd016bd18`、`a5fe3ea1f44a0545148e0c3d577e39e5be791062`（上表顺序）。官方 data_processor 的硬编码 Token ID、监督截断、替换失败样本逻辑，以及 trainer.py 的打包/FlashAttention/优化器 monkey patch 均未照搬。FoodLens 保留严格前缀与动态 Token ID Mask、超限报错、不替换样本。未引入视频、多 GPU、DeepSpeed 或数据打包。

依赖选择沿用已知兼容组合并在本轮实测：Transformers **4.57.1**、PEFT **0.17.1**、Accelerate **1.10.1**、Tokenizers **0.22.2**；不是声称它们是当前最新版。完整本轮 CPU 快照见 [requirements-lock-qwen-cpu.txt](requirements-lock-qwen-cpu.txt)。GPU 安装另行按硬件选择 Torch/CUDA/bitsandbytes，不能套用 CPU 锁。

## 评价协议修复

`canonical_record_text` 在严格 Schema 校验后才执行名称 NFKC/小写/空白合并、同名重量求和、字典序排序；求和使用确定性的 fsum。根字段为 items、total_weight_g、total_calories_kcal、protein_g、fat_g、carbohydrate_g；项字段为 name、weight_g。所有数值一位小数、紧凑 UTF-8 JSON。非法预测在正式文本指标中为空假设，不修复 JSON，不删除参考。

`score_record_texts` 是离线 evaluate、Hybrid Validation、Qwen Validation、原始/微调配对评价的唯一文本协议入口。四项指标现在包含食材、重量和全部整餐值；输出附序列化版本。结构数值指标继续使用原交付值，不将文本格式化用于改写原始记录。Qwen 训练仍监督输入标注的完整 JSON，并保留原始生成文本/Token/截断和解析状态。

当前第五章末段曾允许食材序列或 JSON 辅助文本；本轮用户明确要求完整 JSON 为正式四项文本评价，按本轮指令统一。历史名称文本得分失去与新协议直接比较的条件，须对保存的原预测重新评价。没有重建历史开题 PDF。

CIDEr-D 离线 DF 来自**同一完整评价参考语料**，含失败预测所对应的参考；比较模型时固定全部 ID、参考、Tokenizer、scorer、sigma=6 与缩放（CIDEr-D 原始 ×10；其余原始 0..1）。Validation 只用 Validation 参考，Test 只在冻结系统后的离线评价使用 Test 参考；Test DF 不用于训练、奖励、选模或生成。SCST 固定训练缓存不变。有效子集不用于替换主分数，也不能给不同模型各自成功子集计算 DF 后直接排名。

现有失败规则核查后保留，并写入输出 protocol：

| 情况 | 全样本计分 | valid_subset |
| --- | --- | --- |
| 缺失、不可解析或无输出 | 空食材、五维零预测；仍计全部真值分母 | 不包含 |
| JSON 可解析但 Schema 失败 | 可读名称/有限非负重量继续评价；非法名称计 FP；可读整餐字段保留，其余置零 | 不包含 |
| Schema 合法 | 正常评价 | 包含并报告覆盖率 |
| 截断 | 单列截断率；合法的完整 JSON 仍照常计分 | 仅按 Schema 决定是否包含 |

inference_failed 与 parse_failed 是可重叠诊断，不能相加当失败总数；缺失也计入 parse_failed。非法名称清理项已从 Hybrid 最终 items 移除，其失败日志补计 FP，不重复计入 items。无匹配项的匹配重量误差、空有效子集误差为 null。**全样本 MSE 是包含失败处理的端到端误差，不是纯数值预测精度**；应同时报告有效 JSON 子集误差和覆盖率。

另修复一处边界：仅含失败状态、没有 raw/record 的预测行，旧实现会通过 json.dumps(None) 人造出 `null`，错误增加 JSON 语法合法率。现在按无正文计解析失败；真正生成字符串 `null` 仍计语法合法、Schema 失败。未改变零预测数值惩罚或样本分母，新增断言区分两者。

## 选模与恢复约定

正式 Qwen：Micro F1 大优 → mean(五维 MAE/train_std) 小优 → Schema 有效率大优；逐级容差为配置 tolerance（默认 1e-6），std 仅训练集拟合，下界 1e-6。保持原三项规则，未引入 Hybrid V2 的额外重量优先级。每 epoch 全量 Validation 生成，结果保存 validation-N/；best_step、best、bad_epochs 存原生 Callback 状态。load_best_model_at_end=false，内存保留 last，配对推理必须传 best.json 或所选 checkpoint 目录。工程 val_loss 仅限 engineering=true。

每个 checkpoint 保存 run.json/protocol.json（数据 manifest SHA-256、模型/Processor revision、配置/种子、训练统计、参数范围和实际环境）、Adapter、Processor 与原生训练状态。恢复只允许原目录、相同数据/协议/运行源码及相关环境版本，总 epochs 可增；不因无关文档或 Git 提交变化单独拒绝。早停后不能直接续训。原 UUID 自主管理 checkpoint 不支持直接迁入 Trainer 恢复；旧 Adapter 可通过 epoch 目录推理。只加载可信本地训练状态。

当前限制为单进程单 GPU（或 CPU），num_workers=0，epoch 保存；中断后从最后已完整保存 checkpoint 重跑。保留所有 checkpoint，不自动清理 best 所指目录。单卡 CUDA/NF4 的实际兼容性仍须目标环境验收。

## 实际验收与边界

工作目录 foodlens_engineering/。Windows 11 / Python 3.12.14 / Torch 2.9.1+cpu / Torchvision 0.24.1+cpu / NumPy 2.4.6 / Pillow 13.0.0，CUDA 不可用。日志和产物均在被忽略的 artifacts/，不是新成员复现输入。

```powershell
$env:GIT_CONFIG_COUNT='0'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_qwen_round3.py -v
python -m pip --python .venv/Scripts/python.exe check
```

- 最终全量：**58 项通过，0 失败，0 跳过**，70.858 秒，日志 artifacts/qwen-round3-complete.log。包含原 Qwen/配对推理、Caption A/B、Hybrid V2、SCST、Judge 回归，以及末轮非有限梯度保护、无正文失败行计数修复。最终包内源码 SHA-256 为 `d6e257d709b3d8593b08f0ec0538ea219771fbcf5dca253b2253d22759058226`，环境保存在 artifacts/qwen-round3-final-environment.json。
- 专项阶段：**10 项通过，0 失败，0 跳过**，31.085 秒，日志 artifacts/qwen-round3-focused-final.log。之后新增第 11 个专项测试（非有限梯度保护），所有专项测试均纳入上述最终全量。NF4 调用只做静态核查，不能称为执行通过。
- 固定种子 tiny Qwen：真实 HF Processor/Tokenizer、随机 tiny Qwen3-VL、语言 LoRA、Forward/Backward、完整 Trainer 短训练、梯度累计/末尾短窗口、Gradient Checkpointing 开关、保存/恢复、早停状态、原始/微调配对。连续与恢复 Adapter 逐张量相同，Callback 进度一致。不同长度 JSON 的自定义 CE 与模型内部平均 CE 相符；旧 Token 加权 SGD、Trainer 累计、合并大 Batch 的更新在 rtol=1e-5/atol=1e-7 内相符，无重复累计缩放。合并 Batch 的浮点归约顺序可导致末位差异；Trainer AdamW 的默认分组与旧单参数组、Scheduler 改变也意味着不承诺旧训练全轨迹相同。
- 真实数据处理：校验 sources.json 中三份标注哈希；通过 QwenDataset 读取 train 2338、val 414、test 506，共 3258 张 RGB，严格 Schema 和完整 JSON 文本规范化均通过。每个划分字符最长目标再经**合成 tiny Processor**完整编码；不是官方 Processor 长度/视觉 Token 验收。未拟合测试统计、未训练或按测试选模。
- 对应测试覆盖排列等价、同名合并、重量/营养敏感性、字段顺序、精度、非法 JSON/Schema、公共入口一致、完整参考 DF、全样本/有效子集分母、缺失/推理失败/截断，以及 Processor 附带 token_type_ids 时训练和生成均不转发。
- pip check：无依赖冲突。源码与实际依赖快照保存在各 CLI 的 acceptance/run 文件及 artifacts/qwen-round3-freeze.txt；共享锁文件已去除本机 wheel 绝对路径。
- Qwen CLI：5 条命令退出 0，覆盖 tiny 训练、原生恢复、原始/微调配对与两侧离线三指标评价。artifacts/qwen-round3-tiny-cli/ 保存完整配置、随机 tiny 本地模型、commands.json、逐命令 stdout/stderr、acceptance.json 和预测；命令可从 commands.json 核对，临时模型路径仅留在被忽略产物中。
- 可选依赖隔离：模拟 transformers/peft/accelerate 不可发现时，测试发现成功，无导入错误。一次并行附加检查遭遇 OpenBLAS 内存分配失败；设置 OPENBLAS_NUM_THREADS=1 串行重跑成功，不计为模型测试通过。一次早期根目录运行因工作目录错误导入失败，改用上述工程目录后重新执行。
- 公共 CLI 合成验收：`.venv/Scripts/python.exe -m scripts.synthetic_acceptance --out artifacts/qwen-round3-20261124-cli-final`，**21 条命令退出 0**，产物含 commands.json / acceptance.json，日志 artifacts/qwen-round3-cli-final.log。覆盖 A/B/B-global、Hybrid V2、CE/SCST、预测/评价和 Judge 准备。早期并行尝试未得到完整验收记录，不计通过；此项在独立新目录重新完成。该脚本不包括官方 4B 或外部 Judge。
- CLI 产物记录各自执行时的源码哈希；最后的非有限梯度保护与无正文边界修复由最终全量覆盖，不把此前 CLI 快照标成最终源码的逐字节验证。
- Java/METEOR 本轮未运行（无 Java），已验证缺失时明确报错，无替代指标。工程 CLI 使用显式三指标，不等于正式四指标实验验收。

仍待真实 GPU/官方 4B 验证：固定权重与官方 Processor 加载、实际多模态 Forward/Backward、BF16 LoRA 一步、官方模板 Mask/结束符、Gradient Checkpointing、峰值显存与吞吐、固定原始/微调生成、Trainer checkpoint 恢复、NF4/双重量化/prepare_model_for_kbit_training 与恢复连续性。清单维护在 Issue #3；所有项目保持未勾选。Linux/CUDA、正式数据训练与收敛、模型收益均未验证。

Issue 更新状态：已读取并核对原清单；向 Issue #3 发布补充正文被自动审批拒绝，理由是分支/提交、测试结果和验收计划等项目细节未获该审批器认可的具体公开授权。**远端 Issue 未修改**。可审阅的精简补充正文保存在本地 outputs/qwen_round3/issue3_appendix.md，未随代码交付，不把拒绝当作更新成功。

## 修改文件

- 核心：foodlens/qwen_training.py、foodlens/qwen.py、foodlens/commands.py。
- 公共评价：foodlens/schema.py、foodlens/caption_metrics.py、foodlens/metrics.py、foodlens/cli.py；foodlens/training.py 仅更新 Hybrid 文本评价调用。
- 测试：tests/test_qwen_round3.py、tests/test_engineering.py（原 Checkpoint 断言转为原生状态，保留逐张量恢复和配对验证，补严格 Assistant 文本断言）。
- 配置依赖：configs/train_qwen.json、scripts/make_configs.py、requirements-qwen.txt、requirements-lock-qwen-cpu.txt。
- 文档：根 README.md、工程 README.md、IMPLEMENTATION_STATUS.md、本记录。

真实课程标注快照（均与 data/sources.json 匹配；图片全集哈希未锁定）：

| 划分 | 条数 | SHA-256 |
| --- | ---: | --- |
| train | 2338 | 29a3742405361c00f8f8908bec7c5944a762b3e5529e5fd9e84d6bc95265c856 |
| val | 414 | d0ea314b21c6d97dd3a592f444ac9143024cc91a131facc9343bf42bea086ccc |
| test | 506 | 8e7f3ad1666c344b600fcd8799dc3f61a6cfb466affe9a82ea0e994b25f63b34 |
