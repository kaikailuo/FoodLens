# FoodLens Hybrid 食材预测与重量监督审查

> 历史设计审查：下文“当前实现”均指审查基准提交。建议的类别级 Transformer 已由后续 [Hybrid V2](../../foodlens_engineering/HYBRID_V2.md) 实施；当前代码能力与验证边界以该说明及 [实现记录](../../foodlens_engineering/IMPLEMENTATION_STATUS.md) 为准。

审查日期：2026-11-23。代码基准：`ebde49a35b77be285d6343343e0efa96dc74712a`。本次只新增报告和 `outputs/hybrid_review_20261123/` 下的临时审计产物，未修改模型、训练器、数据或划分，未开展训练。

**结论：优先采用 Train 全名类别 ID、固定升序、普通自回归 CE 的小型 Transformer Decoder，保留现有紧凑整餐回归头。固定排序足以解决训练序列与重量配对，不需要 Hungarian Matching；它不能解决训练集未见类别，也不会自动保证隐藏状态索引正确。** TF-set 是有依据的集合预测方法，但逐食材重量并不在原论文中，直接接上当前按位置取标签的重量头会发生错误配对。

## 1. 真实数据审计结果

### 1.1 来源、范围和复现

本次实际读取项目根 `data/annotations/{train,val,test}.jsonl`，通过现有 [data.py](../../foodlens_engineering/foodlens/data.py) 的 `read_jsonl`、`safe_image_path`，以及 [schema.py](../../foodlens_engineering/foodlens/schema.py) 的 `normalize_name`、`canonical_items`、`validate_record` 审计。没有从历史记录抄录样本数，也没有使用 fixture。三个文件 SHA-256 与 [sources.json](../../data/sources.json) 一致：

| 划分 | 实际条数 | 标注 SHA-256 |
| --- | ---: | --- |
| Train | 2,338 | `29a3742405361c00f8f8908bec7c5944a762b3e5529e5fd9e84d6bc95265c856` |
| Validation | 414 | `d0ea314b21c6d97dd3a592f444ac9143024cc91a131facc9343bf42bea086ccc` |
| Test | 506 | `8e7f3ad1666c344b600fcd8799dc3f61a6cfb466affe9a82ea0e994b25f63b34` |

全部 3,258 个 ID 全局唯一，`split` 与文件一致，目标 Schema 全部通过。所有引用图片均在 `data/` 内、存在并完成 PIL RGB 解码；缺图及解码失败均为 0。没有进行图片内容去重或同一餐盘增量拍摄组的泄漏审计，不能从 ID 唯一推断不存在此类泄漏。

临时脚本：[audit.py](../../outputs/hybrid_review_20261123/audit.py)；完整结果：[audit.json](../../outputs/hybrid_review_20261123/audit.json)；Train 类别表：[class_map.json](../../outputs/hybrid_review_20261123/class_map.json)；逐类别重量表：[per_class_weights.csv](../../outputs/hybrid_review_20261123/per_class_weights.csv)。`outputs/` 被 Git 忽略，这些是本机可复核附件，不是新成员隐含必备输入；本节保留关键结果和标注哈希。

本机 Windows / Python 3.12.14，使用现有工程环境，从项目根运行：

```powershell
& foodlens_engineering/.venv/Scripts/python.exe outputs/hybrid_review_20261123/audit.py
```

脚本只读原始文件；输出写入脚本所在目录。分位数统一为排序后在 `(n−1)p` 处线性插值，所以离散计数的 P95 可能不是整数。规范化沿用 NFKC、转小写、合并空白，不擅自合并同义词。

### 1.2 食材数、类别与异常

| 统计 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 全名类别数（原始 / 规范化） | 151 / 151 | 100 / 100 | 118 / 118 |
| 食材标注条目数 | 8,368 | 1,777 | 1,766 |
| 每图食材数最小值 | 1 | 1 | 1 |
| 最大值 | 16 | 15 | 12 |
| 均值 | 3.5791 | 4.2923 | 3.4901 |
| P50 | 3 | 4 | 3 |
| P95 | 9 | 9 | 8.75 |
| P99 | 11 | 11 | 11 |
| 同一列表内原始名称重复样本 | 0 | 0 | 0 |
| 同一列表内规范化名称重复样本 | 0 | 0 | 0 |
| 空食材列表 | 0 | 0 | 0 |
| 零 / 负 / 非有限 / 缺失重量条目 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

跨三个划分也没有“不同原始名称规范化成同一名称”的冲突。原始与 `canonical_items` 后的每图食材数完全一致。当前数据不需要重复实例匹配；未来若出现同名多条，可沿用现有同名重量求和语义，必须在排序前完成。

| 训练集未见类别 | 类别数 | 涉及样本数 / 全划分 | 样本比例 | 类别及出现样本数 |
| --- | ---: | ---: | ---: | --- |
| Validation | 5 | 32 / 414 | 7.7295% | pepperoni 2；pilaf 4；sandwiches 5；toast 1；tortilla 22 |
| Test | 3 | 9 / 506 | 1.7787% | bread 1；pilaf 3；tuna salad 5 |

Validation 各类别涉及样本数之和为 34，大于 32，因为有样本同时含多个未见类别。两划分合计有 7 个不同未见类别。**不能把它们全部映射为一个 UNK 后计算实体 F1，否则会合并不同真值实体。** 不用 Validation/Test 扩展训练词表或训练统计。

### 1.3 同一食材跨样本重量分布

先按规范化食材名称分组；以下为 Train 的示例，完整三个划分、全部类别的 n/min/max/mean/std/P50/P95/P99 见 CSV 和 JSON。统计为每次食材出现时的克数，不含未出现样本的零填充。

| 食材 | n | 最小 g | 最大 g | 均值 g | 标准差 g | P50 g | P95 g | P99 g |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chicken | 210 | 13.1 | 381.5 | 125.737 | 84.370 | 103.0 | 317.17 | 381.5 |
| broccoli | 259 | 3.1 | 163.9 | 57.529 | 32.922 | 54.0 | 129.0 | 163.9 |
| white rice | 195 | 2.4 | 2991.0 | 41.654 | 214.044 | 13.4 | 97.0 | 132.66 |
| onions | 5 | 30.0 | 56.0 | 47.8 | 10.107 | 55.0 | 55.8 | 55.96 |

| 全部食材出现的重量统计 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 最小 g | 2.1 | 2.0 | 2.0 |
| 最大 g | 2991.0 | 557.0 | 570.0 |
| 均值 g | 54.2249 | 52.3729 | 51.1150 |
| P50 g | 42.0 | 37.6 | 36.1 |
| P95 g | 148.0 | 149.4 | 143.0 |
| P99 g | 217.0 | 264.0 | 233.0 |

Train 每类出现次数最少 1、最多 311、中位数 22，151 类中 28 类不超过 5 次。重量变化明显，类别平均重量不能替代图像条件重量预测。`dish_1560974769` 的 white rice 为 2991.0 g，总重量 3051.0 g，热量 3943.3 kcal：这是实际标注中的高值，Schema 合法，但应列为来源复核对象，不能未经依据改成 299.1 或删除。全局重量标准差易受这类长尾影响；本次不清洗数据，建议保留现有稳健回归损失并记录异常。

### 1.4 食材总和与整餐重量

定义 `Δ = sum(items.weight_g) − total_weight_g`；逐图计算，无样本过滤。

| 统计（g） | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Δ 最小 | −679.6 | −685.6 | −714.9 |
| Δ 最大 | 3.0 | 0.0 | 2.0 |
| Δ 均值 | −19.5184 | −18.4647 | −20.5018 |
| Δ P50 | −0.85 | −7.7 | −0.9 |
| Δ P95 / P99 | 0 / 0 | 0 / 0 | 0 / 0 |
| 平均绝对差 | 19.5281 | 18.4647 | 20.5097 |
| 绝对差 P50 | 0.9 | 7.7 | 0.9 |
| 绝对差 P95 | 72.445 | 48.455 | 71.0 |
| 绝对差 P99 | 400.5 | 196.591 | 462.9 |
| 绝对差 > 1 g 样本数 | 1159 | 289 | 250 |
| 绝对差 > 5 g 样本数 | 977 | 242 | 204 |

具体证据：Train `dish_1566835701` 只有 sweet potato 64.4 g，但总重量为 744.0 g；Validation `dish_1565972429` 为 145.4 / 831.0 g；Test `dish_1566920365` 为 156.1 / 871.0 g。仅凭标注不能确定这些差异来自未列食材、筛选过程还是其他来源问题。**不能将总重量强制设为食材之和，也不能用等式一致性损失迫使模型满足与标签冲突的约束。** 继续分别监督，并报告差异。

## 2. 食材顺序问题结论

| 问题 | 结论与必要条件 |
| --- | --- |
| 仅 Train 建稳定映射？ | 可以。规范化后全名去重、按字符串稳定排序、一次分配连续 ID，保存规则、表和哈希。不能按首次出现顺序或 Python set 迭代赋 ID。当前为 151 类；保留 PAD/BOS/EOS，UNK 仅作输入异常占位。 |
| 固定 ID 排序能保证食材与重量对应？ | 可以，前提是排序 `(class_id, weight_g)` 成对记录，随后从同一排列同时拆出类别与重量。不能各自独立排序两个数组。Train 全部 2338 条经 ID→名称及重量往返检查，无误配。 |
| 评价能否忽略顺序？ | 可以。以类别为键比较集合；重量以键做 join。无需求解匹配。未见真值名称保留为独立的评价键，不能丢弃或合并成 UNK。 |
| 固定排序足够用于普通自回归 CE？ | 足够构造唯一监督：`BOS,c1,…,cK` 输入，`c1,…,cK,EOS` 目标；PAD 忽略。固定排序选择一个规范序列，并不使 CE 成为集合损失。 |
| TF-set 是否仍需固定标签顺序？ | 集合 BCE 不需要；多热真值与食材数不随排列改变。可以为存储固定顺序，但不能再按真值位置给自由生成步骤分配重量。 |
| 是否需要 Hungarian？ | 当前同类唯一且有稳定类别标识，不需要。类别键查表即可；只有匿名槽位且身份未知等不同问题才可能需要额外匹配。 |

### 2.1 评价协议

`TP=|P∩G|，FP=|P\G|，FN=|G\P|`；micro Precision 为 `TP/(TP+FP)`、Recall 为 `TP/(TP+FN)`、F1 为 `2TP/(2TP+FP+FN)`。零分母约定应写入协议并统一；沿用现有全空集合 F1=1 的语义，非空真值且空预测时 Recall/F1=0。

重量保留两种口径：交集上 MAE/MSE 诊断数值头，并报告 matched coverage；主结果保留并集误差，将缺失预测或多预测一侧置 0，单位分别为 g / g²，避免只评价识别成功的食材。整餐五项继续对全样本计算，不用成功子集替代总分母。

现有 [metrics.py:14](../../foodlens_engineering/foodlens/metrics.py) 的 `evaluate_records` 已按规范化名称构建 map、累加 TP/FP/FN，并分别返回 `weight_matched`、`weight_union`、覆盖率和全样本整餐误差；类别 ID 只需映射回名称即可复用。它尚未显式输出 Precision/Recall，可从 TP/FP/FN 增补，不能改变现有分母。

实际验证：将全部 3258 条真实 target 的 `items` 反转后作为评价输入，micro/macro F1 均为 1、11911 个食材的交集/并集重量 MAE/MSE 均为 0。另检查现有 Train `serialize_items` 的 SEP 目标索引与重量，无错误。记录见 [order_check.json](../../outputs/hybrid_review_20261123/order_check.json)。这是标签重排和序列化检查，没有模型推理或效果结论。

### 2.2 隐藏状态与重量：推荐的明确索引约定

令输入位置 0 为 BOS，位置 j 为 `c_j`，Decoder 返回 `h_0,…,h_K`。`h_{j−1}` 负责分类预测 `c_j`，但推荐使用**已消费 `c_j` 的 `h_j`** 联合图像条件 z 回归该类重量：`w_j=Softplus(MLP([h_j,z]))`。这样第一项重量也能看到自身类别；类别 CE 与重量监督的状态有一位偏移，必须显式实现。

训练输入 `BOS,c1,…,cK` 包含最后一个食材，不包含 EOS；食材重量 gather 索引为 `1,…,K`。`h_K` 虽负责输出 EOS，同时也负责最后食材重量；不能用“分类目标是否 EOS”的掩码把最后重量删掉。BOS、PAD 没有重量，空列表重量损失为零。

推理在选出 `c_j` 后把它送回 Decoder，取得该位置 `h_j`，先为已选 `c_j` 写入重量，再判断下一类别是否 EOS。不能遇到 EOS 就提前跳过最后重量。输出的每对 `(class_id,weight)` 绑定后一起排序/组装，不能排序名称而保留旧重量位置。

当前 [data.py:90](../../foodlens_engineering/foodlens/data.py) 与 [decode.py:18](../../foodlens_engineering/foodlens/decode.py) 的旧约定是“产生 SEP logits 的隐藏状态”，训练和推理一致，本次没有发现旧 SEP 偏移错误。类别 token 后已无 SEP，不能直接照搬旧 boundary 计算。上述消费类别状态的设计是 **FoodLens 的重量监督扩展，不是 Inverse Cooking 原有方法**。

### 2.3 固定排序的边界

固定排序充分解决当前已知类别的确定性监督、成对重量和顺序无关评价，但完整系统还需：冻结 Train 类别表；规范化/同名合并；正确移位和 mask；解码禁止 PAD/BOS/UNK、禁止重复或更小 ID、EOS 停止及截断记录；明确未见类别政策。

本次优先方案采用闭集 151 类，保留所有原始评价真值；未见类按独立 FN 计入，重量并集误差保留其真实克数。已知类可以用 ID，OOV 在评价器用规范化名称作为独立键，**不新增训练输出行**。这可形成诚实、可比较的闭集基线，但不能宣称覆盖任意食材。若要求预测未见类别，必须另行提供不依赖 Val/Test 的预先固定完整本体或开放词汇机制；固定排序自身没有这种能力。

## 3. 两篇参考论文与当前实现对照

### 3.1 Inverse Cooking：论文与官方实现分开核对

核对的是 [论文 v2 §3.2–3.3、§4.2](https://arxiv.org/html/1812.06164v2)，以及官方 `master` 的 [model.py](https://github.com/facebookresearch/inversecooking/blob/master/src/model.py)、[transformer_decoder.py](https://github.com/facebookresearch/inversecooking/blob/master/src/modules/transformer_decoder.py)、[args.py](https://github.com/facebookresearch/inversecooking/blob/master/src/args.py)。论文食材网络使用 ResNet-50 空间卷积特征及自回归 Transformer，比较 list CE 与 set 目标；食材为全名类别，不是名称分词。论文报告食材 Decoder **4 层、512 维、2 头（每头 256）**；指令 Decoder 则是 16 层、8 头。不要混淆两者。[论文 §4.2](https://arxiv.org/html/1812.06164v2)

| 检查点 | 官方实现证据与含义 |
| --- | --- |
| 实际参数传递 | `args.py:get_parser` 默认 embed_size=512、transf_layers_ingrs=4、n_att_ingrs=4；`model.py:get_model` 将其传入食材 Decoder。**官方默认 4 头与论文 2 头不同**，不能把默认值写成论文原配置。构造函数自身的默认 8 层/16 头也被调用参数覆盖。 |
| 图像与解码层 | `encoder.py:EncoderCNN` 去掉池化和 FC，1×1 Conv 投影后保留空间 token；`TransformerDecoderLayer` 为因果 self-attention、图像 cross-attention、FFN、残差与 LayerNorm。其 fc1/fc2 均为 d→d，并非默认 4d FFN。 |
| 位置编码 | `get_model` 的食材 Decoder 设置 `pos_embeddings=False`；因果结构和自回归历史仍存在，不能解释成无历史的并行分类器。 |
| 真正训练路径 | `InverseCookingModel.forward` 调用 `ingredient_decoder.sample(...greedy=True,replacement=False)`，自生成的 argmax ID 回馈并 detach；该离散选择不做可微匹配。普通 `forward` 的 teacher forcing 接口并不意味着 TF-set 使用 teacher forcing。 |
| 集合目标 | `label2onehot` 转多热真值；各步 softmax 概率按真实食材数对应的前 K 步掩码后，对时间轴 max pooling，再做带 label smoothing 的 BCE。K 来自标签长度，不来自预测 EOS。原实现 EOS 通道也进入 BCE 张量，多热 EOS 真值置 0。 |
| 去重 | `sample` 的 `replacement=False` 将已经选出的非 EOS 类 logits 置 −∞；不需要不同预测槽位之间的 Hungarian。 |
| EOS Loss | EOS 是 ID 0，也作为启动值。单独取每步 EOS 概率做 BCE；真值食材步为 0、EOS 为 1，实际只对食材段和真实 EOS 位置计损失，两组各占 1/2；PAD 段忽略。防止集合 pooling 丢失停止位置。 |
| Cardinality Penalty | 对 pooled 概率 q 与多热 y 计算 `abs(sum(q*y)−sum(y)) + abs(sum(q*(1−y)))`。第一项惩罚真类概率质量不足，第二项惩罚假类质量；是软概率惩罚，不是预测整数数量的独立 Count Head，也不是重量一致性约束。 |
| 损失启用 | `args.py` 默认 loss_weight 为 `[1,0,0,0]`，并未默认启用食材训练；官方 README 的 im2ingr 命令显式传 `[0,1000,1,1]`。不能照搬默认值或将 1000 当作 FoodLens 必需系数。 |

上述细节可分别定位于 [官方 model.py 的 get_model/forward](https://github.com/facebookresearch/inversecooking/blob/master/src/model.py)、[DecoderTransformer.sample 与 TransformerDecoderLayer](https://github.com/facebookresearch/inversecooking/blob/master/src/modules/transformer_decoder.py)、[参数](https://github.com/facebookresearch/inversecooking/blob/master/src/args.py)、[官方训练命令](https://github.com/facebookresearch/inversecooking/blob/master/README.md)。本机只读快照及文件哈希在 `outputs/hybrid_review_20261123/inverse_*` 和 `reference_snapshots.json`，它们是读取时快照，不宣称已锁定官方 Git commit。

TF-set 的集合标签排列不影响上述多热真值、K、pooling 与 EOS 位置；普通 CE 逐步要求命中规范位置的类别。二者的关键差别是监督目标及训练时的历史来源，不是“是否把列表排序”。TF-set 自回归计算仍具有步骤依赖，不能说整个网络对任意生成历史置换都相同。

如果将 TF-set 扩展到重量，应以实际生成类别查 `truth_weight_by_class`，只在命中类上监督对应状态并报告监督覆盖；错误类不能拿当前位置真值重量，漏预测类也没有自然的步骤状态。另一点是 pooled 某类概率的峰值步骤未必就是该类的采样步骤，不能无说明地把 max pooling 索引当作重量匹配。这些额外选择是 FoodLens 新增设计；本次不优先引入。

| 内容 | 归属及是否迁移 |
| --- | --- |
| CNN 空间特征、因果 Decoder、跨注意力、完整食材类别 | 论文/官方已有，建议复用思想。 |
| TF-set BCE + EOS + 软数量惩罚 | 论文已有集合方案；有价值，但当前重量任务优先普通 CE，不把它列为必须实现。 |
| 逐食材 Weight Head、五项 JSON、g/kcal 监督 | FoodLens 自行设计；Inverse Cooking 不预测逐食材克数。 |
| Recipe1M 名称聚类、稀有类过滤、1488 类词表、最多 20 项 | 不能直接用来替换课程名称和原划分，不因参考论文而删除课程样本。 |
| 烹饪指令 Decoder、食材到指令编码器 | 与当前任务二 JSON 数值输出不同，不迁移。 |
| 官方旧 attention/cache/ByteTensor 实现 | README 环境为 Python 3.6 / PyTorch 0.4.1；适合参考逻辑，优先现有现代 PyTorch 组件重新实现，复制代码须保留许可及归属。 |

当前 [models.py:143](../../foodlens_engineering/foodlens/models.py) 的 `Hybrid(CaptionA)` 实际仍是 ResNet-18 + **单层 GRU**；配置中的 layers=2、heads=4、feedforward=1024 没有用于 Hybrid 的 GRU。其食材名称仍由 [text.py:7](../../foodlens_engineering/foodlens/text.py) 分词，标签经 [data.py:90](../../foodlens_engineering/foodlens/data.py) 逐词加 SEP。不能因为配置有 heads 就称当前 Hybrid 已采用 Transformer。

### 3.2 Nutrition5k：整餐回归的适用范围

原方法使用 JFT-300M 预训练 InceptionV2，输入 256×256；mixed5c 后接 3×3、stride 2、valid 平均池化，再接**两层共享 4096 FC**。每个标量预测再有独立的 4096→1 两层 FC，即共享表示后任务专属分支。损失为 `L_macro + L_calorie + L_mass`，其中 `L_macro=(|e_protein|+|e_fat|+|e_carb|)/3`，各项使用绝对误差；不是 MSE。这里应参考 RGB 直接预测版本，逐食材重量头不属于该网络。见 [Nutrition5k §4.2–4.3、Fig.6、Eq.2](https://arxiv.org/html/2103.03375v2)。

| 对照项 | FoodLens 当前实现与审查判断 |
| --- | --- |
| 视觉编码器 | ResNet-18 512 通道网格、GAP 得 v，投影得 z。共享视觉表示适用；换成当前编码器是适配，不能称复现 InceptionV2/JFT 结果。 |
| Total Head | `concat(v512,z256)→Linear(768,256)→ReLU→Linear(256,5)→Softplus`，见 models.py:149、167。只有一层共享隐藏层，五个输出有各自线性权重，没有各自非线性隐藏分支。 |
| 五输出是否“不算多任务”？ | 仍是有效的联合多目标回归；最后 256→5 等价五个独立的 256→1 线性输出。缺少独立隐藏分支是容量选择，不是结构错误。 |
| 损失 | losses.py:9 的 hybrid_loss 使用 Train 均值/标准差归一化后的 Smooth L1，五字段整体平均；与论文 MAE、三组任务加权不同。该差异应明确记录，不冒称原论文损失。 |
| 视觉冗余 | z 是 v 的非线性投影，拼接不提供额外模态；但也不是目标泄漏。暂保留有助于控制修改范围。 |
| 是否照搬大 FC？ | 不建议。仅 4096→4096 一层已约 1678 万权重，独立大分支继续增加参数；本数据 Train 仅 2338 张且长尾明显，额外容量缺乏当前数据证据。 |
| 可迁移的核心 | 共享视觉编码与整餐多任务回归、分字段报告误差。保留紧凑头；没有发现必须通过加深头部修复的结构错误。 |

原论文每克营养版本评价时会乘真实总质量，深度/体积版本需额外输入；都不应作为 FoodLens RGB-only 自动推理的可直接照用结果。[Nutrition5k §4.3–4.4](https://arxiv.org/pdf/2103.03375)

## 4. 唯一优先推荐架构

推荐 **ResNet-18 空间网格 + 类别级自回归 Transformer（固定排序 CE）+ 消费类别状态的 Weight Head + 保留当前 Total Head**。这是 FoodLens 适配方案，非两篇论文的原样拼接。

1. 保留 224×224 RGB 预处理与 ResNet-18，得到 7×7×512 网格；线性投影为 49×256 作为 Decoder memory。不增加新的视觉 Transformer Encoder。
2. 食材 Decoder 采用 **2 层、hidden 256、4 头、FFN 1024、dropout 0.1**，因果 self-attention 和图像 cross-attention；使用确定性位置编码表达规范序列。参数取现有工程配置尺度，属于小数据适配判断，不是论文 4 层/512 维的复现，也没有训练验证其最优性。
3. Train 的 151 个全名类别及特殊 ID 替代词级名称与 SEP。按 ID 升序、teacher forcing、带 EOS 的 CE。推理按同一升序约束，只允许比上次更大的食材 ID 或 EOS；这同时防止重复。保留 max_items=32、max_tokens=128，Train 最大需求仅 16 项+EOS=17 步，无需因 Val/Test 修改上限；达上限仍标记截断。
4. Weight Head 沿用 `512→256→1→Softplus`，其输入变为 `[h_j,z]`；按第 2.2 节的消费类别位置监督。重量尺度仅从 Train 拟合，保留标准化 Smooth L1。
5. Total Head 保持当前 `768→256→5→Softplus` 与 Train 标准化 Smooth L1。独立五项回归，不与预测食材求和绑定，也不输入真值名称、重量或 metadata。
6. JSON 名称从固定 ID 表还原，保持当前 Schema、规范化及顺序无关评价；主结果保留 OOV 真值和全样本分母。当前闭集方案不能输出那 7 个未见类别，必须明示覆盖限制。

可行性依据是实际类别规模、每图项数、没有同类多实例歧义，以及成熟 Decoder/现有头的接口可接合；**不是精度、收敛或 GPU 成本证据**。主要风险为 28 个低频类别、未见类漏检、长尾重量、RGB 的尺度/遮挡歧义、CE 的训练/推理历史差异，以及多任务梯度竞争。回归采用稳健损失、冻结骨干起步与现有验证集早停即可，不先增加层数或匹配算法。

还有一项工程和课程定位风险：[training.py:283、309、350](../../foodlens_engineering/foodlens/training.py) 正式 Hybrid 当前强制从已选 A checkpoint 初始化，`transfer_caption_a` 要求保留全部旧词 ID 并复制 GRU。新的类别词表及 Transformer **不兼容旧 GRU/词行迁移**。可以保留已训练视觉编码器迁移，但必须明确“视觉迁移”范围，更新 checkpoint 类型和技术方案，不能声称迁移了 GRU 或完成了全部原有任务一模型适配。这里仅指出接口影响，不扩展实施其他模型。

## 5. 修改工作量与影响范围

以下均为后续实施建议，本次没有执行这些修改。

| 文件 / 函数或环节 | 必需改动 | 范围 |
| --- | --- | --- |
| foodlens/text.py，新增 IngredientVocabulary | 独立全名类别表，Train-only 拟合、稳定 ID、双向映射、版本/哈希；与原词级 Vocabulary 分离 | 中 |
| foodlens/data.py：serialize_items、collate_nutrition、collate_hybrid | 成对规范化/排序，类别序列、右移监督、消费类别重量索引、item mask、长度及 OOV 审计 | 中 |
| foodlens/models.py：Hybrid.__init__/forward，VisualEncoder 接口 | 接空间 memory 与小型 Decoder；返回 states；改重量 gather；保留 Total Head。避免通过继承误保留无用 GRU | 中至大 |
| foodlens/decode.py：predict_hybrid | 类别自回归、升序 mask、EOS/空集/截断、最后项重量、ID→名称；RGB-only 入参保持严格 | 中 |
| foodlens/losses.py：hybrid_loss | CE 目标与新 weight mask 对接，保持回归单位和 Train-only 统计；不添加 set loss、数量头或一致性等式 | 小 |
| foodlens/training.py：build_model、restore_model、run_training、validate | 新 family/类别表构造和恢复；视觉迁移替代旧 GRU 迁移；审计/选模/日志记录 OOV 与覆盖；测试标签不进训练 | 中至大 |
| foodlens/checkpoint.py：save_checkpoint/load_checkpoint | 保存 ingredient map、规范化版本、Decoder 类型、回归索引协议；旧 checkpoint 显式识别，禁止静默当新结构恢复 | 中 |
| foodlens/metrics.py：evaluate_records | 尽量保留名称键评价，增显式 P/R；OOV 保留名称身份、FN/并集误差；顺序置换不变 | 小 |
| foodlens/commands.py、config.py、configs/train_hybrid.json | 模型类型路由、兼容参数校验、初始化范围和 checkpoint 契约；不改现有 Caption 实现 | 中 |
| tests/test_baseline.py、test_engineering.py | 新类别级序列/索引/推理协议测试；保留旧分词与迁移测试用于旧格式，不简单删掉防回归测试 | 中 |
| 工程 README、IMPLEMENTATION_STATUS、第五章及架构说明 | 明确论文/适配差别、迁移范围、类别覆盖和实际验证；实施时按仓库规则同步源与正式图件 | 小至中 |

整体属于**中等偏大的接口迁移**：Decoder 核心成熟，主要工作在类别表、移位重量、解码、恢复与迁移协议；不能当成只把 GRU 替换一行的修改。本次不估计未经目标 GPU 验证的训练时长或费用。

后续必要验收测试：

- Train 输入打乱后类别表完全相同；Val/Test 不改变表/统计；多词名是单类别；不同 OOV 不合并。
- 所有 2338 条 Train 类别/重量成对往返；输入 items 任意置换后序列和监督不变；规范化同名项按既定规则求和。
- 两项不同重量验证首项、末项、BOS/EOS/PAD 的精确索引；`h_K` 不因其 CE 目标为 EOS 被错误屏蔽；改变未来输入不影响过去状态。
- 训练与逐步推理状态一致，最后一项在 EOS 前完成重量，类别升序且无重复；空集、截断、失败全部可见，不依赖真值约束解码。
- 重排预测不改变 P/R/F1 或重量误差；漏类/多类/OOV/缺失样本进入并集与全样本分母。
- Weight Head/Decoder 梯度有效、有限，非负回归；Train-only 统计与真实长尾输入保持数值稳定。
- 新 checkpoint 完整往返，旧格式清晰拒绝或走旧恢复路径；视觉迁移复制正确且不误复制 GRU/词行。

本次已经完成的只有真实数据只读审计、图片解码、现有序列化索引检查、真实标签顺序置换评价检查，以及论文/官方代码静态核对。没有运行新架构前向/反向、正式训练、GPU 验收或模型效果比较。
