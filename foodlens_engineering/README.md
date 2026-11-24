# FoodLens 工程入口

依据[课程要求](../docs/COURSE_REQUIREMENTS.md)和[第五章](../docs/report/sections/technical_plan.tex)，已接通 Caption A/B/B-global、Hybrid、Qwen3-VL LoRA、SCST 与 Judge 的代码入口。**2026-11-20 的实际验证为 Windows/Python 3.12/CPU 合成工程测试，没有正式训练、真实数据效果或 GPU 性能结论。** 历史与本次记录分别见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)，来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 环境及无数据验收

<a id="环境与复现"></a>

以下命令的工作目录均为 `foodlens_engineering/`。Python 基线为 3.12，各系统独立创建环境。Windows PowerShell：

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-lock-test-cpu.txt
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_engineering.py -v
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_baseline.py -k ModelTests -v
& .venv/Scripts/python.exe -m scripts.synthetic_acceptance --out artifacts/my-synthetic-check
```

Linux Bash 使用 `.venv/bin/python` 替换解释器路径，其余参数相同：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock-test-cpu.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -p test_engineering.py -v
.venv/bin/python -m unittest discover -s tests -p test_baseline.py -k ModelTests -v
.venv/bin/python -m scripts.synthetic_acceptance --out artifacts/my-synthetic-check
```

新增测试使用临时合成 RGB/标签及真实张量运算；Qwen 使用真实 HF Tokenizer/Processor、随机 tiny Qwen3-VL 和 PEFT，只有模型获取和 Judge 服务边界被隔离。不会下载预训练权重。synthetic_acceptance 要求新的输出目录，通过独立进程调用 CLI；其显式三指标配置省略 Java METEOR，不能代替正式四指标验收。

最小 CPU 自主模型依赖为 `requirements-cpu.txt`；指标/统计增加 `requirements-metrics.txt`，Qwen 增加 `requirements-qwen.txt`。完整 CPU 锁定文件 `requirements-lock-test-cpu.txt` 来自本次实际安装环境。旧 `requirements-lock-cpu.txt` 保留旧基线版本，补充 scorer 所需 six。venv 的 ensurepip 不可用时，可用外部 pip 管理：`python -m pip --python .venv/Scripts/python.exe install -r requirements-lock-test-cpu.txt`。

正式 Linux/CUDA 先按硬件安装兼容的 PyTorch/Torchvision，再装可选依赖并保存 freeze；不要把 CPU 锁定文件覆盖到 GPU 环境。BF16、FP16、NF4、CUDA 确定性及显存/吞吐均须目标机器试跑。METEOR 另需 PATH 上的 Java 和 pycocoevalcap 自带 jar/英文 paraphrase 资源；缺项明确报错，没有替代分数。Linux/CUDA、Java METEOR 未在本次实测。

完整 `python -m unittest discover -s tests -v` 还包含依赖真实课程数据的旧测试。2026-11-20 曾因缺数据报错；2026-11-23 本机已有课程数据，本轮结果见 IMPLEMENTATION_STATUS。新增 V2 检查：`python -m unittest discover -s tests -p test_hybrid_v2.py -v`，无数据时仅跳过 RealDataTests。

## 数据契约与审计

不修改原始数据。取得资源后按 [data/README.md](../data/README.md) 归档。

- Caption manifest JSONL：每行 `id/image/caption/official_filename/sha256/duplicate_group/split`，可附来源 revision。image 相对 data 根；split 为 train/val/test，ID 全局唯一，哈希与重复组不跨划分。先完成可追溯映射、排除清单和官方测试优先的划分，禁止按行号拼接。
- Nutrition JSONL：`dish_id/image/split/target`，保留学校原划分，target 符合严格 Schema。target_text/metadata 不进入推理。
- 推理 JSONL：严格只有 `id/image`，多余字段拒绝。**省略 predict 的 --samples 即全量推理**。缺图或失败保留同 ID 失败行。
- Qwen 读取原 RGB。自主模型采用短边 256、中心裁剪 224、ImageNet 标准化，训练额外水平翻转。词表、数值统计与奖励 DF 只拟合 train。

```bash
python -m foodlens.cli audit --kind caption --inputs ../data/caption_manifest.jsonl --data-root ../data --check-images --out artifacts/audit/caption.json
python -m foodlens.cli audit --kind nutrition --inputs ../data/annotations/train.jsonl ../data/annotations/val.jsonl ../data/annotations/test.jsonl --data-root ../data --check-images --out artifacts/audit/nutrition.json
```

不传 check-images 时只检查标注/ID/划分/重复组及路径契约；传入后验证图片解码和 Caption SHA256。失败退出码非零并保存 ID。它不自动解决近重复图、同餐关系或来源不明的映射。

## Caption：A/B/B-global

A 使用 ResNet18 全局条件 GRU，B 使用 49 网格、二维位置编码和 Transformer Encoder/Decoder，B-global 保留 B 配置但只有一个全局视觉 Token。正式配置使用 ImageNet 权重；资源应先在目标环境准备。工程配置 `configs/engineering_cpu.json` 明确使用随机 tiny 和 val_loss，不静默降级正式配置。

```bash
python -m foodlens.cli train --config configs/train_caption_a.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-a
python -m foodlens.cli train --config configs/train_caption_b.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-b
python -m foodlens.cli train --config configs/train_caption_b_global.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-b-global
python -m foodlens.cli train --config configs/train_caption_a.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-a --resume artifacts/caption-a/last.pt
python -m foodlens.cli predict --checkpoint artifacts/caption-a/best.pt --manifest artifacts/test_rgb.jsonl --data-root ../data --device cuda --out artifacts/caption-a/test.jsonl
python -m foodlens.cli evaluate-caption --manifest ../data/caption_manifest.jsonl --split test --predictions artifacts/caption-a/test.jsonl --out artifacts/caption-a/test_metrics.json
```

RGB 清单由最终测试 manifest 提取 id/image；评价独立读取参考。右移 CE 按全批非 PAD 数归一化，EOS 参与；训练上限保留 EOS，记录原长度/截断/UNK。推理输出文本、token ID、EOS/截断和失败状态。正式验证使用 CIDEr-D 选模并早停；只有 engineering=true 才允许 val_loss 选模。

四指标共享 `foodlens.word_v1`：NFKC、小写、词/标点切分后空格连接。**它不是 COCO PTB，不能直接与采用 PTB 的发表分数比较。** BLEU-4/METEOR/ROUGE-L 调用 pycocoevalcap 1.2 的标准代码/原始 METEOR jar，CIDEr-D 调用固定上游 scorer，sigma=6。前三项保留 0—1，CIDEr-D 保留标准 scorer 自带 ×10，不再 ×100。离线 DF 取整个被评价参考语料；SCST 使用独立训练缓存。单图语料的 CIDEr-D 可能退化为零，应保留语料级评价。

## Hybrid：迁移、自由生成与评价

```bash
python -m foodlens.cli train --config configs/train_hybrid.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --initialize artifacts/caption-a/best.pt --out artifacts/hybrid
python -m foodlens.cli predict --checkpoint artifacts/hybrid/best.pt --manifest artifacts/nutrition_test_rgb.jsonl --data-root ../data --device cuda --out artifacts/hybrid/test.jsonl
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/hybrid/test.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/hybrid/test_metrics.json
```

2026-11-23 起新训练采用 [Hybrid V2](HYBRID_V2.md)：ResNet-18 的 49 个网格直接投影到 256 维，接 2 层、4 头、FFN 1024、Dropout 0.1 的 Transformer Decoder，无视觉 Transformer Encoder。Train 全名类别按规范化名称排序，独立于 Caption 词表。先合并同名重量，再成对排序；输入 BOS,c1,…,cK，目标 c1,…,cK,EOS。重量取已消费类别的 h1,…,hK，PAD 不监督。完整监督超长直接报错，不能截断重量配对。

`--initialize` 可省略，此时使用配置中的 ImageNet ResNet-18 初始化。提供 Caption A 时，`transfer=visual` 只复制 `visual.features.*`，具体参数名记录在 run.json/Checkpoint；全局投影、网格投影、Transformer、类别 Embedding、输出层和回归头均重新初始化，回归输出偏置用 Train 均值初始化。不支持 visual_gru。旧 Hybrid Checkpoint 由 `restore_model()` 显式构建 LegacyHybrid，可推理但不能作为 V2 恢复点；旧 smoke 保留原 SEP 链路。Caption A/B 架构与权重加载不变。

自由生成仅允许大于前一类别的 ID 或 EOS，允许空集合。选出类别后先消费并回归重量，再预测下一类；达到 Token/项数上限仍保留最后完整项并记录截断。非有限 logits/重量保留失败记录，非法整餐值用 null 标记 Schema 失败；JSON 组装保留一位小数，不约束食材重量和等于整餐重量。

Hybrid V2 正式验证依次比较 Micro F1（大优）、食材 Weight Union MAE（g，小优）、五维 `mean(MAE_j/train_std_j)`（小优）、Schema 有效率（大优）；逐级容差均为 1e-6。选模只用 Validation。Qwen 保留原三项优先级，不在本轮改变。五维依次为总重量(g)、热量(kcal)、蛋白质(g)、脂肪(g)、碳水(g)。

任务二正式四项文本指标统一使用**规范化完整 JSON**：名称 NFKC/小写/合并空白，同名重量求和，按名称字典序排序；字段固定为 items 及 TOTAL_FIELDS 顺序，项字段为 name、weight_g；全部数值一位小数，紧凑 JSON 序列化。名称、重量和整餐数值均参与文本评分；Schema 无效预测为空文本，保留全量参考与分母。结构 MAE/MSE 仍使用交付值，不以文本取整改变数值评价。除四文本指标外，报告精确名称 Micro F1、逐图 Macro F1、匹配/并集重量 MAE/MSE、五维 MAE/MSE、JSON/Schema 有效率及缺失/失败/截断。缺失侧重量置零，无匹配项误差为 null；完全解析失败为空食材/五维零预测；Schema 失败保留可读字段。非法名称及 Hybrid 清理失败仍计 FP。有效子集另报覆盖率，不能替代全样本结果。旧 evaluate 不带 text-metrics 时只运行结构评价，正式评价必须传四项。

## 公共训练与恢复边界

训练支持 AdamW、冻结骨干且 BN 保持 eval、unfreeze_layer4_epoch（从 0 起）、骨干低学习率倍率、梯度裁剪/累积和 CUDA BF16/FP16。累积窗口分别按全窗口 token/食材/样本分母归一化，末个短窗口仍更新。

自主 best.pt/last.pt 原子替换，包含模型、优化器、调度器、AMP scaler、epoch/update、best/bad_epochs、词表/训练统计、Python/NumPy/Torch/CUDA RNG、配置及版本。仅加载可信本地产物。恢复从**上个完整保存 epoch 的下一 epoch**开始，num_workers=0，seed+epoch 重建 shuffle，不恢复未保存 batch 或梯度。仅总 epochs 可增加，其他配置和数据哈希须一致。中断 epoch 会重跑，按 update 识别重复日志。相同 CPU/软件已验证精确恢复；跨设备/依赖版本/CUDA 不保证位级相同。已早停状态不能直接恢复追加训练，须初始化新研究对照。

## Qwen3-VL：本地 LoRA 与原始/微调配对

本模块已迁移到 Hugging Face `Trainer`，参见[第三轮整改记录](QWEN_ROUND3.md)。train_qwen.json 使用独立 QwenTrainConfig，只含生效的训练和选模参数；模型、LoRA 和图像参数以 qwen.json 为准。兼容读取旧公共 TrainConfig，忽略自主模型字段；非空 updates_per_epoch 显式拒绝。

configs/qwen.json 固定 Qwen3-VL-4B-Instruct 模型及 Processor revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。默认 local_files_only=true，不自动下载大模型。组员先准备该 revision 的本地 HF cache，或把 model_id 指向已准备目录并保留 revision 和文件校验记录。本地 Qwen 无需 API Key。

```bash
python -m foodlens.cli qwen-train --qwen-config configs/qwen.json --config configs/train_qwen.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --out artifacts/qwen-lora
python -m foodlens.cli qwen-train --qwen-config configs/qwen.json --config configs/train_qwen.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --out artifacts/qwen-lora --resume artifacts/qwen-lora/last.json
python -m foodlens.cli qwen-pair --qwen-config configs/qwen.json --adapter artifacts/qwen-lora/best.json --manifest artifacts/nutrition_test_rgb.jsonl --data-root ../data --out artifacts/qwen-pair
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/qwen-pair/original.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/qwen-pair/original_metrics.json
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/qwen-pair/finetuned.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/qwen-pair/finetuned_metrics.json
```

独立 qwen-predict 参数相同，省略 adapter 为原始模型。配对固定同一配置/manifest/指令/贪心协议，保留原始文本和全部失败，不重试或修复答案。collator 检查图像展开后完整对话与 generation prefix 严格一致，再只监督 assistant 回答和 im_end；图像/user/header/PAD/尾部换行不监督。超长要求减少像素或增大限制，不截断 JSON。

LoRA 只匹配 `model.language_model.layers.N.self_attn.{q,k,v,o}_proj` 实际完整路径，在 run.json 保存 scope/可训练参数清单并校验无视觉层。默认 BF16 LoRA；独立配置 quantization=nf4 才启用 QLoRA，还需目标 CUDA 环境 bitsandbytes。真实 4B/Processor、NF4 算子及显存均未实测。

processor_use_fast 在配置中显式固定，避免框架默认值变更；协议保存 Chat Template、Tokenizer 与图像 Processor 配置哈希，加载适配器时校验这些指纹。

Qwen 使用 Trainer 原生 `checkpoint-<global_step>`，每个完整 epoch 验证并保存 Adapter、Processor、优化器、Scheduler、AMP（启用时）、RNG 和 trainer_state.json。run.json/protocol.json 补充数据哈希、统计、版本、参数范围及生成协议；best.json/last.json 只保存相对路径，不再维护额外训练状态和重复权重哈希。恢复通过 resume_from_checkpoint，同一输出目录，只有总 epochs 可增加；数据、配置、处理协议及环境/源码须一致。旧自定义 epoch 包不能作为 Trainer 恢复点，旧 Adapter 可通过 epoch 目录推理。只加载可信本地产物。

Trainer 管理 AdamW、常数学习率、AMP、梯度累计、裁剪和日志。原 StepLR 显式替换为常数调度，不声称与旧优化过程位级等价。公开 compute_loss_func 使用移位 Assistant CE 总和除以累计窗口监督 Token 总数，不再额外除以 accumulation；末个短窗口同样归一化。Gradient Checkpointing 使用 use_reentrant=false。

每 epoch 全量 Validation 生成，按 Micro F1 → 五维 mean(MAE/train_std) → Schema 有效率逐级比较（1e-6 容差），std 仅从训练集拟合。StructuredSelection 的最优值/步数/早停计数由 Trainer 原生 Callback 状态保存与恢复；原生标量 EarlyStoppingCallback 无法表达此规则，因此保留轻量回调。load_best_model_at_end=false，训练结束的内存模型为 last，**配对推理显式传 best.json**。不使用测试集选模。工程模式可显式选择 val_loss。保留全部 checkpoint 供 best 路径和审计使用。

当前仅支持单进程单卡；多卡机器先用 CUDA_VISIBLE_DEVICES 选择一张卡。CPU 验证固定 Transformers 4.57.1 / PEFT 0.17.1 / Accelerate 1.10.1，完整快照为 requirements-lock-qwen-cpu.txt；不是最新版本声明，也不是 GPU 依赖锁。无权重/无 GPU 时运行：`python -m unittest discover -s tests -p test_qwen_round3.py -v`。未安装 Qwen 可选依赖时对应测试显式跳过。

CIDEr-D 离线评价（含 Validation）从**本次完整参考语料**计算 DF。比较模型时必须使用同一划分、样本 ID、完整 JSON 参考、Tokenizer 和 scorer 版本，不随预测成功率筛选语料。不把测试 DF 用于训练、奖励或选模；SCST 仍单独使用训练缓存。历史名称文本分数不能与整改后 JSON 分数直接比较。

全样本 MSE 包含失败处理，不能解释为成功输出时的纯数值精度：缺失/解析失败为五维零预测；Schema 失败保留有限非负的可读字段，其余字段置零。valid_subset 只要求 JSON 通过 Schema；合法但标记截断的 JSON 仍在子集中，截断率单列。inference_failed 与 parse_failed 可重叠，不能相加为失败总数；缺失也计入 parse_failed。空子集误差为 null，子集始终报告覆盖率。详见整改记录及输出 protocol。

## SCST 与等步数 CE 对照

```bash
python -m foodlens.cli build-df --manifest ../data/caption_manifest.jsonl --out artifacts/scst/train_df.json
python -m foodlens.cli train --config configs/train_ce_continue.json --manifest ../data/caption_manifest.jsonl --data-root ../data --initialize artifacts/caption-a/best.pt --out artifacts/ce-continue
python -m foodlens.cli train --config configs/train_scst.json --manifest ../data/caption_manifest.jsonl --data-root ../data --initialize artifacts/caption-a/best.pt --df-cache artifacts/scst/train_df.json --out artifacts/scst
```

原始 CE 权重保持原样；两个追加对照都从同一验证选定 CE 初始化。示例相同低学习率、10 epochs、每 epoch 最多 100 次更新，patience=11，核对实际 update 保证追加步数相等。三组共用 predict/evaluate-caption。

缓存为 JSON，固定训练 ID、manifest SHA256、分词/标准 scorer 版本及 n-gram DF；ID/哈希不符拒绝训练。采样/贪心均 eval 模式、同长度，采样保留梯度，贪心和奖励无梯度。stop-gradient advantage 乘以序列 log probability 之和；EOS 参与、EOS 后 PAD 排除，再按样本平均。scst_weight 显式混合 CE，日志保存描述、奖励差及梯度异常。

## Judge、双人人评与统计

在查看候选效果前锁定约 60 张 RGB 图；需要类别/困难样本分层时，先准备分层 population 再锁定抽样。人评者各自独立评分，第三人协商仅用于案例解释，原始评分仍用于均分与一致性。

```bash
python -m foodlens.cli judge-lock --manifest artifacts/judge_population_rgb.jsonl --count 60 --out artifacts/judge/locked.jsonl
python -m foodlens.cli judge-prepare --manifest artifacts/judge/locked.jsonl --predictions artifacts/caption-a/test.jsonl artifacts/caption-b/test.jsonl artifacts/scst/test.jsonl --out artifacts/judge/blind
python -m foodlens.cli judge-run --config configs/judge_local.json --candidates artifacts/judge/blind/candidates.jsonl --data-root ../data --out artifacts/judge/results.jsonl
python -m foodlens.cli human-export --candidates artifacts/judge/blind/candidates.jsonl --rater R1 --out artifacts/judge/R1.csv
python -m foodlens.cli human-export --candidates artifacts/judge/blind/candidates.jsonl --rater R2 --out artifacts/judge/R2.csv
python -m foodlens.cli human-import --candidates artifacts/judge/blind/candidates.jsonl --csv artifacts/judge/R1.csv --out artifacts/judge/R1.jsonl
python -m foodlens.cli human-import --candidates artifacts/judge/blind/candidates.jsonl --csv artifacts/judge/R2.csv --out artifacts/judge/R2.jsonl
python -m foodlens.cli judge-analyze --candidates artifacts/judge/blind/candidates.jsonl --judge artifacts/judge/results.jsonl --human artifacts/judge/R1.jsonl artifacts/judge/R2.jsonl --raters R1 R2 --out artifacts/judge/analysis.json
```

先导出 CSV，分别查看 RGB 后填写 1—5 整数，再导入；附 rubric.json。顺序按 rater+seed 固定，隐藏 candidate_id、模型身份与 Judge 响应。private_mapping.json 不交给人评者。Judge 后端仅接收 RGB/候选/固定规则，隐藏参考答案。四维语义正确性、细节覆盖、事实可靠性、语言流畅度各需整数 1—5 及一句图像证据，可靠性高分表示更少幻觉；拒绝围栏 JSON、缺项、bool 和默认评分。

默认 local 加载冻结原始 Instruct，不加载微调 adapter。API 使用 configs/judge_api.json，先把 service_revision 改为供应商固定版本，再设置 .env.example 中三个系统环境变量；代码不自动读取 .env。Key 和地址不写日志/快照/checkpoint。服务格式为 OpenAI-compatible chat/completions 的图像+文本请求，部署兼容性后续实测。缺变量/空响应报错，不产生分数；本次未启动付费服务。

保存原始响应、解析/截断/后端失败、版本及全样本成功率。Spearman 使用完整配对，另报覆盖率；按图像 Bootstrap 保留同图所有候选，输出 95% 区间/有效重复次数，常数向量为 undefined/null。两名原始人评的 Cohen kappa 为固定 1—5 分类的 quadratic 权重。可选 `--traditional scores.jsonl`，每行 `sample_id/image_id/scores`（metric 到逐候选原始分数的字典），接入传统指标相关性；不能复制单个语料级分数充当逐样本值。

## 资源与交接

还需：锁定 Caption manifest/课程任务二数据、ImageNet 和固定 Qwen 权重、Linux CUDA 机器、Java METEOR、真实 Judge 后端及两名人评。取得后按审计 → Caption → Hybrid → Qwen → SCST → Judge 的顺序验收，记录批量/精度/显存/吞吐、数据及代码哈希。CPU 合成通过不表示真实资源验证通过。

旧 smoke 仍可用：`python -m foodlens.cli smoke --config configs/baseline.json --data-root ../data --out artifacts/new-smoke`，它依赖真实任务二训练图片，不属于本次无数据测试。不要覆盖历史 artifacts；新输出用独立子目录，检查日志放根 outputs，均默认不提交。报告/PDF 和原始数据不被本次实现修改。
