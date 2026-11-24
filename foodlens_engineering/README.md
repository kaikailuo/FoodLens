# FoodLens 工程入口

依据[课程要求](../docs/COURSE_REQUIREMENTS.md)和[第五章](../docs/report/sections/technical_plan.tex)，已接通 Caption A/B/B-global、Hybrid、Qwen3-VL LoRA、SCST 与 Judge 的代码入口。**截至 2026-11-24 的实际验证为 Windows/Python 3.12/CPU 合成工程测试，没有正式训练、真实数据效果或 GPU 性能结论。** 历史与本次记录分别见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)，来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。Caption 修复、缓存开关与基准结果见 [CAPTION_CACHE_FIXES.md](CAPTION_CACHE_FIXES.md)。

## 环境及无数据验收

<a id="环境与复现"></a>

以下命令的工作目录均为 `foodlens_engineering/`。Python 基线为 3.12，各系统独立创建环境。Windows PowerShell：

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-lock-test-cpu.txt
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_engineering.py -v
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_caption_cache.py -v
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_baseline.py -k ModelTests -v
& .venv/Scripts/python.exe -m scripts.synthetic_acceptance --out artifacts/my-synthetic-check
```

Linux Bash 使用 `.venv/bin/python` 替换解释器路径，其余参数相同：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock-test-cpu.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -p test_engineering.py -v
.venv/bin/python -m unittest discover -s tests -p test_caption_cache.py -v
.venv/bin/python -m unittest discover -s tests -p test_baseline.py -k ModelTests -v
.venv/bin/python -m scripts.synthetic_acceptance --out artifacts/my-synthetic-check
```

新增测试使用临时合成 RGB/标签及真实张量运算；Qwen 使用真实 HF Tokenizer/Processor、随机 tiny Qwen3-VL 和 PEFT，只有模型获取和 Judge 服务边界被隔离。不会下载预训练权重。synthetic_acceptance 要求新的输出目录，通过独立进程调用 CLI；其显式三指标配置省略 Java METEOR，不能代替正式四指标验收。

最小 CPU 自主模型依赖为 `requirements-cpu.txt`；指标/统计增加 `requirements-metrics.txt`，Qwen 增加 `requirements-qwen.txt`。完整 CPU 锁定文件 `requirements-lock-test-cpu.txt` 来自本次实际安装环境。旧 `requirements-lock-cpu.txt` 保留旧基线版本，补充 scorer 所需 six。venv 的 ensurepip 不可用时，可用外部 pip 管理：`python -m pip --python .venv/Scripts/python.exe install -r requirements-lock-test-cpu.txt`。

正式 Linux/CUDA 先按硬件安装兼容的 PyTorch/Torchvision，再装可选依赖并保存 freeze；不要把 CPU 锁定文件覆盖到 GPU 环境。BF16、FP16、NF4、CUDA 确定性及显存/吞吐均须目标机器试跑。METEOR 另需 PATH 上的 Java 和 pycocoevalcap 自带 jar/英文 paraphrase 资源；缺项明确报错，没有替代分数。Linux/CUDA、Java METEOR 未在本次实测。

完整 `python -m unittest discover -s tests -v` 还包含依赖真实课程数据的旧测试。当前缺少 data/annotations/train.jsonl，旧测试 19 项通过、2 项因缺数据报错；不能称全量通过。

## 数据契约与审计

本次不获取或配置真实数据。取得资源后按 [data/README.md](../data/README.md) 归档。

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

### Caption 缓存与修复

A/Hybrid 的 `model.dropout` 现作用于词嵌入，训练开启、eval 关闭；B/B-global 只提取网格/GAP，不执行 A 的条件投影，保留原参数以兼容旧权重和优化器。恢复旧 A/Hybrid checkpoint 时，非零 Dropout 现在实际生效，会警告训练轨迹变化；不能据此复现修复前的训练结果。

训练 JSON 新增 `generation.gru_cache_enabled`、`generation.transformer_kv_cache_enabled`，均为布尔值且默认 false。开启示例为 `configs/train_caption_a_cache.json`、`train_caption_b_cache.json`、`train_caption_b_global_cache.json`。CLI 可以覆盖 JSON 或预测 checkpoint 中的开关：

```bash
python -m foodlens.cli train --config configs/train_caption_a.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-a-cache --gru-cache-enabled
python -m foodlens.cli train --config configs/train_caption_b.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-b-cache --transformer-kv-cache-enabled
python -m foodlens.cli train --config configs/train_caption_b.json --manifest ../data/caption_manifest.jsonl --data-root ../data --out artifacts/caption-b-cache --resume artifacts/caption-b-cache/last.pt --no-transformer-kv-cache-enabled
python -m foodlens.cli predict --checkpoint artifacts/caption-b-cache/best.pt --manifest artifacts/test_rgb.jsonl --data-root ../data --device cuda --out artifacts/caption-b-cache/pred-kv.jsonl --transformer-kv-cache-enabled
python -m foodlens.cli predict --checkpoint artifacts/caption-b-cache/best.pt --manifest artifacts/test_rgb.jsonl --data-root ../data --device cuda --out artifacts/caption-b-cache/pred-full.jsonl --no-transformer-kv-cache-enabled
```

开关影响生成：A 复用 GRU hidden（包括 SCST 的采样/贪心），B/B-global 复用每层 self-attention K/V 和视觉 cross-attention K/V。CE teacher forcing 不使用缓存；开关不能用于降低其前向/反向显存。Hybrid 保持已有逐步 GRU 生成，Qwen 由自身生成配置控制，这两个 Caption 开关对它们无效。缓存可能增加常驻张量，不能保证显存下降；应在目标 GPU 比较后决定。训练日志按 epoch 记录训练/验证显存峰值及验证时间，CPU 显存字段为 null。run.json/checkpoint 和预测旁的 `.protocol.json` 记录申请/实际生效开关，恢复时可切换。

无数据基准及 CUDA/真实 checkpoint 对比命令见 [缓存交接说明](CAPTION_CACHE_FIXES.md)。当前验证生成使用 FP32；训练 precision 只控制损失前向，独立 predict 沿用 FP32。基准脚本可显式测试 CUDA BF16/FP16，尚未实测。

## Hybrid：迁移、自由生成与评价

```bash
python -m foodlens.cli train --config configs/train_hybrid.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --initialize artifacts/caption-a/best.pt --out artifacts/hybrid
python -m foodlens.cli predict --checkpoint artifacts/hybrid/best.pt --manifest artifacts/nutrition_test_rgb.jsonl --data-root ../data --device cuda --out artifacts/hybrid/test.jsonl
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/hybrid/test.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/hybrid/test_metrics.json
```

主配置 transfer=visual_gru 复制视觉/投影/GRU 及旧词行；复制配置改为 transfer=visual、另选 out 完成仅视觉对照。保留所有旧词 ID，仅以训练文本追加词和 SEP。完整食材监督超长直接报错，不能截断重量配对。训练和自由生成都在产生 SEP logits 的状态回归重量；空项不回归。追踪空名/UNK/非有限值，丢弃未闭合项，合并重复名称，一位小数组装及严格校验 JSON，不约束食材重量之和等于整餐重量。

Hybrid/Qwen 正式验证按 Micro F1 选模，容差内比较五维 `mean(MAE_j/train_std_j)`，再比较 Schema 有效率；默认容差 1e-6。五维依次为总重量(g)、热量(kcal)、蛋白质(g)、脂肪(g)、碳水(g)。

任务二文本固定为规范化、重复合并、字典序排序的食材名，用 ` ; ` 连接，不含数字；Schema 无效预测为空文本。除四文本指标外，报告精确名称 Micro F1、逐图 Macro F1、匹配/并集重量 MAE/MSE、五维 MAE/MSE、JSON/Schema 有效率及缺失/失败/截断。缺失侧重量置零，无匹配项误差为 null；完全解析失败为空食材/五维零预测；Schema 失败保留可读字段。非法名称及 Hybrid 清理失败仍计 FP。有效子集另报覆盖率，不能替代全样本结果。旧 evaluate 不带 text-metrics 时只运行结构评价，正式评价必须传四项。

## 公共训练与恢复边界

训练支持 AdamW、冻结骨干且 BN 保持 eval、unfreeze_layer4_epoch（从 0 起）、骨干低学习率倍率、梯度裁剪/累积和 CUDA BF16/FP16。累积窗口分别按全窗口 token/食材/样本分母归一化，末个短窗口仍更新。

自主 best.pt/last.pt 原子替换，包含模型、优化器、调度器、AMP scaler、epoch/update、best/bad_epochs、词表/训练统计、Python/NumPy/Torch/CUDA RNG、配置及版本。仅加载可信本地产物。恢复从**上个完整保存 epoch 的下一 epoch**开始，num_workers=0，seed+epoch 重建 shuffle，不恢复未保存 batch 或梯度。总 epochs 可增加，generation 中两个缓存执行开关可切换，其余配置和数据哈希须一致。中断 epoch 会重跑，按 update 识别重复日志。修复后的相同 CPU/软件及配置已验证精确恢复；旧 A/Hybrid 的非零 Dropout 训练轨迹不复现，跨设备/依赖版本/CUDA 不保证位级相同。已早停状态不能直接恢复追加训练，须初始化新研究对照。

## Qwen3-VL：本地 LoRA 与原始/微调配对

train_qwen.json 复用公共训练与任务二选模配置；模型、LoRA 和图像处理参数以 qwen.json 为准，公共配置中的自主 model/transfer/layer4 字段不参与 Qwen 构建。

configs/qwen.json 固定 Qwen3-VL-4B-Instruct 模型及 Processor revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。默认 local_files_only=true，不自动下载大模型。组员先准备该 revision 的本地 HF cache，或把 model_id 指向已准备目录并保留 revision 和文件校验记录。本地 Qwen 无需 API Key。

```bash
python -m foodlens.cli qwen-train --qwen-config configs/qwen.json --config configs/train_qwen.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --out artifacts/qwen-lora
python -m foodlens.cli qwen-train --qwen-config configs/qwen.json --config configs/train_qwen.json --train-annotations ../data/annotations/train.jsonl --val-annotations ../data/annotations/val.jsonl --data-root ../data --out artifacts/qwen-lora --resume artifacts/qwen-lora/last.json
python -m foodlens.cli qwen-pair --qwen-config configs/qwen.json --adapter artifacts/qwen-lora/best.json --manifest artifacts/nutrition_test_rgb.jsonl --data-root ../data --out artifacts/qwen-pair
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/qwen-pair/original.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/qwen-pair/original_metrics.json
python -m foodlens.cli evaluate --annotations ../data/annotations/test.jsonl --predictions artifacts/qwen-pair/finetuned.jsonl --text-metrics BLEU-4 METEOR ROUGE-L CIDEr-D --out artifacts/qwen-pair/finetuned_metrics.json
```

独立 qwen-predict 参数相同，省略 adapter 为原始模型。配对固定同一配置/manifest/指令/贪心协议，保留原始文本和全部失败，不重试或修复答案。collator 检查图像展开后完整对话与 generation prefix 严格一致，再只监督 assistant 回答和 im_end；图像/user/header/PAD/尾部换行不监督。超长要求减少像素或增大限制，不截断 JSON。

LoRA 只匹配 `model.language_model.layers.N.self_attn.{q,k,v,o}_proj` 实际完整路径，输出 scope.json/可训练参数清单并校验无视觉层。默认 BF16 LoRA；独立配置 quantization=nf4 才启用 QLoRA，还需目标 CUDA 环境 bitsandbytes。真实 4B/Processor、NF4 算子及显存均未实测。

processor_use_fast 在配置中显式固定，避免框架默认值变更；协议保存 Chat Template、Tokenizer 与图像 Processor 配置哈希，加载适配器时校验这些指纹。

Qwen 先写不可变 epoch 子目录，再原子更新 best.json/last.json 指针。指针验证 adapter/训练状态 SHA256，避免半更新恢复；子目录保存 adapter、Processor、模板哈希、协议、优化器/调度器/scaler/RNG/进度，基础权重外置。可传指针或 epoch 目录，协议不一致拒绝加载；旧 epoch 保留供归档复核。恢复粒度同自主模型。

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
