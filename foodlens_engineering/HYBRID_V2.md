# Hybrid V2 工程协议（2026-11-23）

本轮用户确定的架构替代第五章旧 Hybrid 的 GRU/词级 SEP 方案；Caption A/B 不变。原审查报告保留于 [食材与重量审查](../docs/reviews/HYBRID_INGREDIENT_WEIGHT_REVIEW_20261123.md)。这里描述实现协议，不代表真实训练效果。

## 架构与监督

RGB 预处理沿用 Resize(256)、CenterCrop(224)、ImageNet 标准化，Train 加水平翻转。正式模型为 ImageNet ResNet-18，输出 7×7×512；49 个 Grid Token 经独立 Linear(512,256) 和二维正弦位置编码直接成为 Cross-Attention Memory，无视觉 Transformer Encoder。GAP 得 v512，再由 Linear(512,256)+tanh 得 z256。

食材 Decoder 使用 PyTorch `nn.TransformerDecoder`，固定 2 层、256 维、4 头、FFN 1024、Dropout 0.1，含因果 Self-Attention、视觉 Cross-Attention、残差、LayerNorm 和正弦序列位置编码。工程测试可显式使用 tiny/小隐藏维度；它们不构成正式模型效果。

`IngredientVocabulary.fit()` 只接受 Train，名称沿用 `nfkc-lower-whitespace-v1`（NFKC、小写、合并空白）。规范化全名稳定排序后分配 ID；PAD/BOS/EOS/UNK 分别为 0/1/2/3，真实食材从 4 开始。同名重量先确定性求和，再按类别 ID 成对排序。不分词，不用 TF-set、Hungarian 或集合损失。

输入为 `[BOS,c1,…,cK]`，目标为 `[c1,…,cK,EOS]`，CE 忽略 PAD。`boundaries` 字段为兼容批处理 API 而保留，V2 的含义是输入类别位置 `[1,…,K]`，与旧 SEP 索引不同。h0 预测 c1；hj 已消费 cj，预测下一类（hK 预测 EOS），但重量监督属于 cj：

`weight(cj) = Softplus(Linear256→1(ReLU(Linear512→256(concat(hj,z)))))`。

整餐头为 `concat(v512,z256) → Linear(768,256) → ReLU → Linear(256,5) → Softplus`，输出依次为总重量 g、热量 kcal、蛋白质 g、脂肪 g、碳水 g。三损失为 CE、Train 统计标准化 Smooth L1 重量及整餐，初始系数均 1。回归输出偏置由训练均值初始化。没有重量和一致性约束。

## 生成与评价

预测接口只接收 RGB 张量、冻结类别表和固定生成配置。可生成 EOS（包括首步），或严格大于已生成 ID 的食材。每选出一个类别即送回 Decoder，取已消费状态回归重量；即使达到最后一个 Token/项数预算也完成这项回归。EOS 后停止；未见 EOS 而触及预算则记截断。raw_token_ids、stop_reason、failures、invalid_total_fields、schema_errors 随预测留存。非有限 logits 按样本终止并保留失败；非法重量不进入最终 items，仍以失败 FP 进入评价；非法整餐值为 null，Schema 无效且相应数值按零计误差。组装值保留一位小数。

Validation/Test 不扩表。诊断 Teacher Forcing 时未知类别仅作 UNK 输入，未知类别 CE 置 PAD 忽略；对应重量诊断在 UNK 条件下计算，不作为未见类识别成绩。实际选模用自由生成结果和未修改的原始 target；每个未知名称保留为独立 FN，重量也保留在 Union 分母，不把它们合并成 UNK 真值。

评价沿用规范化名称集合与键匹配，顺序无关；报告 Micro Precision/Recall/F1、Macro F1、Matched/Union 重量 MAE/MSE、匹配覆盖率、五字段全样本 MAE/MSE、JSON/Schema 有效率和截断率。漏检/多检重量另一侧置零。解析和推理失败不从分母排除。

V2 只在 Validation 按以下优先级选模：Micro F1 最大 → Weight Union MAE(g) 最小 → 五维 `mean(MAE_j/train_std_j)` 最小 → Schema 有效率最大，各级容差 1e-6。全空真值且空预测的 Union 无项时选模重量项记 0。Test 不参与选模；Qwen 保持原有三项选优规则。

## 权重与恢复

新 Hybrid Checkpoint 标识 `architecture_version=hybrid_v2`，保留格式 v2 的 epoch 恢复协议，并保存完整类别 tokens、规范化版本及 SHA-256；加载校验类别映射，恢复还核对当前 Train 映射、数据哈希及配置。模型、优化器、Scheduler、Scaler、RNG 和 Early Stopping 进度均恢复。

Caption A 初始化是可选项，仅复制严格匹配的 `visual.features.*` 参数和 buffer，全部名称写入 transfer.parameters。它不复制视觉全局投影、类别 Embedding、输出行、GRU 或任何新头。未提供 Caption A 时，正式配置直接使用预训练 ResNet-18。下载与恢复分离，Checkpoint 恢复不重新下载权重。

无 V2 标记的旧 Hybrid 显式构造 `LegacyHybrid` 并走原词级 SEP 推理；不静默迁入 V2，`run_training --resume` 拒绝将旧 Hybrid 当作 V2。旧 smoke 仍使用 LegacyHybrid。Caption A/B 原实现和 strict 权重加载保持可用。

## 验证入口

工作目录 `foodlens_engineering/`，Windows 用 `.venv/Scripts/python.exe`，Linux 用 `.venv/bin/python`：

```bash
python -m unittest discover -s tests -p test_hybrid_v2.py -v
python -m unittest discover -s tests -v
python -m scripts.synthetic_acceptance --out artifacts/my-hybrid-v2-acceptance
```

输出目录必须新建。真实数据检查读取原始 Train/Val/Test、比对 sources.json 哈希，检查 Train 全量往返、反序一致、未见类数，并逐张执行现有 RGB 预处理。具体执行数量、环境与日志见 [实现状态](IMPLEMENTATION_STATUS.md)。预训练效果、GPU 资源与正式收敛仍需后续实验。
