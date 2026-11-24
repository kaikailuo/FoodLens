# FoodLens 必选任务数据与 Manifest 最终验收（2026-11-24）

> 历史检查快照：本报告的任务一阻塞结论已由同日后续的 [Food500 映射修复与最终验收](FOOD500_MANIFEST_FINAL_20261124.md) 更新。下文保留当时发现与执行证据，不能作为当前 Food500 就绪状态；任务二风险和当时验证范围仍需结合后续记录阅读。

**DATA NOT READY**

任务一未具备正式训练条件，正式 Manifest 不存在且映射、来源和去重尚未闭合。任务二的原始数据资产、严格 Schema、全量图片解码和 CPU 数据接口通过，可按原课程划分上传；仍有近重复候选需要正式实验前复核。本结论只覆盖本次数据验收，不包含模型训练或 GPU/官方 Qwen Processor 验证。

代码基线：`main` / `8d501a9794099c9cb9efdd0abaf53a76504bafe7`。本轮未创建分支、Commit、PR 或推送，未改模型代码、训练配置、依赖锁或原始数据。开始时已有未跟踪的 `docs/reviews/` 和两个中文资源目录，未覆盖。主要检查环境为 Windows、Python 3.12.14、Torch 2.9.1+cpu、CPU；Parquet 独立检查使用 Python 3.13、PyArrow 26.0.1、Pillow 13.0.0，小型读取依赖仅安装到被忽略的输出目录，没有下载数据集或模型。

## 1. 数据资产清单

路径均相对项目根；CLI 的常规工作目录为 `foodlens_engineering/`、data-root 为 `../data`。

| 资产 | 实际条数 | SHA-256 | 与 sources.json 比较 |
| --- | ---: | --- | --- |
| `data/annotations/train.jsonl` | 2338 | `29a3742405361c00f8f8908bec7c5944a762b3e5529e5fd9e84d6bc95265c856` | PASS |
| `data/annotations/val.jsonl` | 414 | `d0ea314b21c6d97dd3a592f444ac9143024cc91a131facc9343bf42bea086ccc` | PASS |
| `data/annotations/test.jsonl` | 506 | `8e7f3ad1666c344b600fcd8799dc3f61a6cfb466affe9a82ea0e994b25f63b34` | PASS |
| `data/Food500-Cap/finetune_data.json` | 19880 | `b69439dfdca82b2d930c6da3d23a9538351459ac6fd85b18c510aaa7cef712e2` | PASS（仅标注） |
| `data/Food500-Cap/evaluation_data.json` | 4940 | `8218b582f4fbbadcff0566ada5ea6e807a86c404ab9cdf89166035b335e9cfe8` | PASS（仅标注） |

任务二 `data/images/train/val/test/` 分别有 2338 / 414 / 506 个非 AppleDouble 普通文件，与 3258 条引用路径一一对应。本轮所有引用图片均存在、可完整解码并转换为 RGB。`data/annotations/` 的三个 JSONL 本身就是监督数据入口，不要求另造含答案的训练 Manifest；测试推理清单已派生到 [nutrition_test_rgb.jsonl](../../outputs/data_manifest_acceptance_20261124/nutrition_test_rgb.jsonl)，506 行，每行严格只有 `id/image`。

任务一原始标注有 19880 / 4940 个唯一 filename，各文件内 filename 重复数为 0；这不是上游 README 所写的 19760 / 4940。官方标注仓库本地 HEAD 与 sources.json 一致：`a32dfe90c01801231aad92c59f6bebe8a8e0dfe6`。社区 Parquet 实际为 train **19877**、test **4938** 行；没有 Val，也没有 `data/caption_manifest.jsonl`。因此正式 train/val/test 图片数、Caption 数和 Manifest 哈希均为 **BLOCKED**；原始数量不能当成清洗后的正式数量。

本地社区副本位于 `data/Food500-Cap/Food500Cap-HF/data/`。九个分片的大小及 SHA-256 均与本地 HF 缓存树逐项一致：

| 分片 | bytes | SHA-256 | 本地缓存对照 |
| --- | ---: | --- | --- |
| test-00000-of-00002-032e07b311d1db77.parquet | 295757170 | `4dd72492866388f1937d7a0f8e202a82b7f93188507aa5cfd47a48d070f4a3a5` | PASS |
| test-00001-of-00002-d55c6542fe70f451.parquet | 293856844 | `4ca6368721c84e2516d07c9477ce4ed88969241a3a22a3132908b955ce74d2fa` | PASS |
| train-00000-of-00007-9de56bcdafdee2d8.parquet | 398641250 | `c797764dd18ff7b2293e346541a5852797746f5b8bb04b90654679501b22c8ad` | PASS |
| train-00001-of-00007-d4acaf169b202bde.parquet | 371182555 | `72a0b647e2ee6e988dcb261df6383d99047a6e761d70c18f18875f6d63d8a015` | PASS |
| train-00002-of-00007-3eeb547f5ee2bc77.parquet | 337133702 | `d79073169ef4654d218c5da4a80c75744e9c4a01c3e402fe9e3d37522adf43f2` | PASS |
| train-00003-of-00007-db06d49ad5d281c2.parquet | 357558494 | `7217dc6f2abdec2c382116462dfef82cb16d49a3f01acaad3a528b76d370266f` | PASS |
| train-00004-of-00007-34cdb0916951598d.parquet | 290325153 | `1fdec820cd981fce3d285041a9934ce7821c3b2cc38eb124418c81f4fd95d831` | PASS |
| train-00005-of-00007-25f227541c143e60.parquet | 317233056 | `bfe697b66f20ebe5992053e55dde30c3ee060a34ba94d233a8ba579f44a3eabd` | PASS |
| train-00006-of-00007-84404baa7f02576e.parquet | 339022377 | `d57f80438949f1a8d965cadc708f18e39521bc1fb70514a472bc3d48fead892c` | PASS |

HF 缓存元数据记录 revision `1b2e0cb098ef3d58584f80bde015207c5a3d00c7`，但目录无社区仓库 README/远程仓库标识，缓存树也不能证明仓库 ID。这是本地一致性证据，**不是已独立核实的图片来源锁定**。残留 `.incomplete` 和 `._*` 文件不计入分片/图片；九个实际 Parquet 完整读取成功。

其他本轮新算的本地资产指纹如下；两个压缩包未有来源方校验和供独立比对，不能把新算哈希写成来源方验证通过。

| 文件 | bytes | SHA-256 |
| --- | ---: | --- |
| data/sources.json | 1606 | `d888bda1732391c5bba2a78e0cf924a396f502f591e7d4c5e061615969b67f11` |
| data/source/annotations.zip | 233058 | `d877e9da81fe0a589220bbf93a218331c678d501611bced476e8cc2a6a743cdf` |
| data/source/images.zip | 1282104125 | `18dc27890d698dba9f44a32ceaa156b850b48684278aea739cde90000f54a1a8` |

## 2. 数据质量结果

### 2.1 Food-500 Cap

| 检查项 | 状态 | 数量与证据 |
| --- | --- | --- |
| 官方 JSON 解析、cat/filename/caption 为非空字符串 | PASS | 24820 条；空值/类型异常 0，见 [caption_assets.json](../../outputs/data_manifest_acceptance_20261124/caption_assets.json) |
| Caption 内容启发式检查 | PASS | 空文本/占位词/无字母/非法控制字符 0；训练长度43—377字符、测试44—298字符，完全相同 filename+caption 行重复0；不代表人工语义核验 |
| 官方标注文件 SHA-256 | PASS | 两个文件均与 sources.json 一致 |
| 正式 Manifest 的字段、类型、ID、相对路径、已有逐图哈希、duplicate_group | BLOCKED | Manifest 不存在，不能冻结 |
| 社区图像 bytes 全量解码为 RGB | PASS | 24815 条；解码失败 0，见 [parquet_full.json](../../outputs/data_manifest_acceptance_20261124/parquet_full.json) |
| 原划分 + 完整 filename + Caption 显式核对 | FAIL | 精确匹配 24036 条；779 条未能同时匹配（train 634 / test 145） |
| 官方 filename 在对应社区划分中逐字匹配 | FAIL | 官方 train 579 / test 142 个键没有精确命中；含 Unicode/名称转写，不能一律称为缺图 |
| 仅以同划分唯一相同 Caption 诊断别名 | 未冻结 | 25 条候选可增加对应证据；754 条社区记录仍未解，759 条官方标注仍未对应，见 [alias_full.json](../../outputs/data_manifest_acceptance_20261124/alias_full.json) |
| 原始 train/test filename 交集 | FAIL | 88 个 filename，见 [caption_assets.json](../../outputs/data_manifest_acceptance_20261124/caption_assets.json) 的 cross_official_filenames |
| 社区字节/像素重复 | FAIL（正式划分尚未处理） | 两种检查各 140 组、每组 2 行，其中 126 组跨 train/test、14 组在单划分内 |
| 既定测试优先、按重复组划约 10% Val（seed 42） | BLOCKED | 尚无排除清单、冻结重复组及最终 Val |



社区副本按原划分统计如下；同哈希是唯一图片计数依据，尚未执行正式清洗：

| 原划分 | 图像记录数 | Caption 数 | 唯一图片字节哈希数 | 划分内重复组/涉及行 | 涉及跨划分重复组 |
| --- | ---: | ---: | ---: | --- | ---: |
| train | 19877 | 19877 | 19863 | 14 / 28 | 126 |
| test | 4938 | 4938 | 4938 | 0 / 0 | 126 |

合计 24815 条图文记录、24675 个唯一字节哈希；合法多描述与跨划分重复需依来源和分组分别处理。

严格映射采用 `(原划分, cat + '/' + image.path)` 与原标注 filename 比较，并核对 Caption；没有采用行号或文件排列。名称转写例子：官方 `Æbleflæsk/Æbleflæsk_0001.jpg` 对应社区候选 `AEbleflaesk/AEbleflaesk_0001.jpg`，Caption 中食物名也有变化。未经来源证明，不把转写算法产生的键直接当成可靠映射。779 条中，716 条社区 filename 在对应官方划分不存在，另 63 条 filename 命中但 Caption 不同。**24820−24815=5 只是净条数差，不能断言只缺五张图。**

官方未精确命中路径示例（全量 ID 在 [mapping_full.json](../../outputs/data_manifest_acceptance_20261124/mapping_full.json) 的 missing；全部社区异常 ID 在同文件 issues）：

- train: `Abalone/Abalone_0062.jpg`
- train: `Arròs_negre/Arròs_negre_0001.jpg`
- train: `Arròs_negre/Arròs_negre_0003.jpg`
- train: `Arròs_negre/Arròs_negre_0004.jpg`
- train: `Arròs_negre/Arròs_negre_0005.jpg`
- test: `Arròs_negre/Arròs_negre_0081.jpg`
- test: `Arròs_negre/Arròs_negre_0083.jpg`
- test: `Arròs_negre/Arròs_negre_0085.jpg`
- test: `Arròs_negre/Arròs_negre_0086.jpg`
- test: `Arròs_negre/Arròs_negre_0087.jpg`

Parquet 解码警告 2 条，均单独保留，不以 EXIF 警告直接判坏图：

- train: `Crudo/Crudo_0039.jpg`：Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.
- train: `Lemon_chicken/Lemon_chicken_0022.jpg`：Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.

### 2.2 Nutrition5k 课程数据

| 检查项 | 状态 | 异常数量/具体 ID |
| --- | --- | --- |
| JSONL 每个物理行严格解析，拒绝重复键和 NaN/Infinity | PASS | 3258 行，解析错误/空行 0 |
| dish_id 非空字符串、全局唯一；split 与文件一致 | PASS | 错误 0 |
| image 路径相对 data 根且不越界、存在、完整 RGB 解码 | PASS | 错误 0，3258/3258 |
| target 严格字段集合、items/name/weight_g、五个整餐数值 | PASS | Schema 错误 0；bool、非法类型、非有限/负数均未发现 |
| 空食材、同名食材合并、零重量 | PASS | 实际空食材 0、需合并同名样本 0、零食材重量样本 0 |
| 标注 SHA-256/条数与 sources.json 一致 | PASS | 三个划分全部相符 |
| 跨划分 ID、引用路径、文件哈希、RGB 像素重复 | PASS | 四种确定性重复均 0（包括划分内也未发现） |
| Train 类别与统计只用训练集拟合 | PASS | 151 个训练类别 + 4 个特殊 Token；Val/Test 拟合保护实际拒绝 |
| Val/Test 未见类 | 合法边界 | Val 5 类/32 图；Test 3 类/9 图，保留原标签，不扩充词表 |
| 每图食材重量和与整餐重量差异 >0.1g | 保留 | Train 1214 / Val 296 / Test 260；不作为 Schema 错误，不改数据 |

Schema 规则直接复用 `foodlens/schema.py`：target 仅含 items 与五个 TOTAL_FIELDS，item 仅含 name/weight_g；数值为有限、非负的 int/float（bool 不合法）。无非法标注样本 ID；完整检查结果和所有重量差异样本 ID 见 [nutrition_full.json](../../outputs/data_manifest_acceptance_20261124/nutrition_full.json)。训练类别词表及训练统计分别保存在 [ingredient_vocabulary.json](../../outputs/data_manifest_acceptance_20261124/ingredient_vocabulary.json)、[train_statistics.json](../../outputs/data_manifest_acceptance_20261124/train_statistics.json)，均是本次读取原 Train 后生成的审计快照。

各划分规范化食材类别数：Train 151、Val 100、Test 118。原始食材条数与规范化合并后的条数分布相同：

| 每图食材数 | Train | Val | Test |
| ---: | ---: | ---: | ---: |
| 1 | 770 | 78 | 175 |
| 2 | 244 | 57 | 55 |
| 3 | 293 | 49 | 62 |
| 4 | 295 | 59 | 61 |
| 5 | 204 | 40 | 44 |
| 6 | 169 | 35 | 31 |
| 7 | 137 | 38 | 32 |
| 8 | 107 | 26 | 20 |
| 9 | 56 | 13 | 8 |
| 10 | 29 | 13 | 11 |
| 11 | 15 | 4 | 5 |
| 12 | 10 | 1 | 2 |
| 13 | 6 | 0 | 0 |
| 14 | 1 | 0 | 0 |
| 15 | 1 | 1 | 0 |
| 16 | 1 | 0 | 0 |

未见类的计数是“含该类别的样本数”，多类别可能共现，不能相加当作图数：

**val**：类别及含该类别的样本数 `{"pepperoni": 2, "sandwiches": 5, "tortilla": 22, "toast": 1, "pilaf": 4}`；共 32 个样本。ID：

- `dish_1564168695`
- `dish_1564168739`
- `dish_1564169197`
- `dish_1564169254`
- `dish_1564169298`
- `dish_1564169340`
- `dish_1564169878`
- `dish_1564169932`
- `dish_1564169958`
- `dish_1564170010`
- `dish_1564170283`
- `dish_1562777566`
- `dish_1562784388`
- `dish_1562787783`
- `dish_1562787817`
- `dish_1562787848`
- `dish_1561747075`
- `dish_1561748274`
- `dish_1561749392`
- `dish_1561752449`
- `dish_1561752481`
- `dish_1561752509`
- `dish_1561752535`
- `dish_1561752563`
- `dish_1561752970`
- `dish_1561752996`
- `dish_1561753021`
- `dish_1561753075`
- `dish_1561753371`
- `dish_1561753400`
- `dish_1561753432`
- `dish_1561753477`

**test**：类别及含该类别的样本数 `{"bread": 1, "pilaf": 3, "tuna salad": 5}`；共 9 个样本。ID：

- `dish_1562085672`
- `dish_1562788536`
- `dish_1562788568`
- `dish_1562788601`
- `dish_1568401167`
- `dish_1568401199`
- `dish_1568401233`
- `dish_1568401261`
- `dish_1568401302`

## 3. 数据泄漏审计

**确定性重复。** 任务一社区副本有 140 个相同字节 SHA-256 组，RGB 像素哈希给出相同的组；126 组跨官方 train/test，因此原社区划分不能直接作为正式训练划分。不能只检查 filename：如同一图可能使用不同类别路径。示例：

- SHA-256 `4a34f961d5d7c55e748da4ecc4ec6b47cca91355ac6c342aa5cccda5f37ce64d`：test: `Apple_cobbler/Apple_cobbler_0089.jpg`; train: `Cobbler_food/Cobbler_food_0039.jpg`
- SHA-256 `1466a53e27f1a9200ea2105e3e1b0e64b0252304a3f9786cde6f3886ad6555a9`：test: `Bacon_and_eggs/Bacon_and_eggs_0096.jpg`; train: `bacon_and_eggs/bacon_and_eggs_0096.jpg`
- SHA-256 `02334111a51df2433dc80afa2dc5b5052339d3f6174f397b5e42d3d3167eef42`：test: `Bacon_and_eggs/Bacon_and_eggs_0097.jpg`; train: `bacon_and_eggs/bacon_and_eggs_0097.jpg`
- SHA-256 `d59bcf32656af72e65f94b51c7700d37567554fde254f1de7dbb27feec4ea63a`：test: `Bacon_and_eggs/Bacon_and_eggs_0098.jpg`; train: `bacon_and_eggs/bacon_and_eggs_0098.jpg`
- SHA-256 `59e103612602e39aaec73c2fc7c70ae6d1d1788ddf96a9c4c51ec5f6ea857050`：test: `Bacon_and_eggs/Bacon_and_eggs_0099.jpg`; train: `bacon_and_eggs/bacon_and_eggs_0099.jpg`

全部重复组成员、原始所属划分见 [parquet_full.json](../../outputs/data_manifest_acceptance_20261124/parquet_full.json) 的 duplicate_sha256 / duplicate_rgb_pixels。此次没有自动删除、重新分配或生成 duplicate_group。任务一需要在明确映射后落实测试归属优先、测试重复组稳定代表、训练重复组隔离验证，以及可审查排除清单。正式图片数和描述数必须分别统计。

任务二 3258 张图没有相同文件哈希或相同 RGB 像素；同 target 的 138 组中有 60 组跨划分，**相同营养标签不证明同一图像或同一餐**。跨划分标签相同示例：

- train: `dish_1559149615`; test: `dish_1559578551`
- train: `dish_1565119418`; test: `dish_1562963246`
- train: `dish_1568320296`; test: `dish_1566851549`
- train: `dish_1558547068`; test: `dish_1558546434`; test: `dish_1559157777`
- train: `dish_1558548666`; test: `dish_1558549605`

**可疑近重复。** 对任务二全量图像执行 64-bit 灰度 dHash（9×8、LANCZOS），跨划分 Hamming 距离 ≤4 得到 536 对、涉及 348 个 ID：Train/Test 495、Train/Val 22、Val/Test 19。此方法受餐盘背景、构图和菜品相似影响，未人工判定同餐或拍摄序列；既不能当作 536 个泄漏，也不能保证漏检率为零。完整候选 ID 见 [near_duplicates.json](../../outputs/data_manifest_acceptance_20261124/near_duplicates.json)。距离最小的示例：

- `dish_1559844506` ↔ `dish_1559844490`，dHash 距离 0。
- `dish_1558717886` ↔ `dish_1558717604`，dHash 距离 0。
- `dish_1558110245` ↔ `dish_1558110297`，dHash 距离 0。
- `dish_1558628921` ↔ `dish_1558718610`，dHash 距离 1。
- `dish_1558629398` ↔ `dish_1558641253`，dHash 距离 1。

任务一未执行感知近重复扫描/人工复核，状态 **BLOCKED（尚无可冻结映射）**。任务二缺少可证明餐次/拍摄序列关系的来源表，此层泄漏 **无法判断**；保留原官方划分和所有候选标记，不删除主评价样本。

**合法多 Caption。** 相同图像的多条描述可在同一划分内共享 image/sha256/duplicate_group、使用不同的样本 id。当前 `audit_caption_rows` 允许同划分共享图像哈希；不应把“多描述”自动认定为无效重复。两个官方标注文件各自没有重复 filename；跨划分 88 个同名键和跨划分图像哈希组仍必须隔离，不能以多 Caption 为由放行。现有 Caption collator 每样本返回一个 reference；以后引入多参考数据时还需显式固定评价聚合协议，本轮没有改变协议。

## 4. 工程兼容性与 CPU 验收

**Caption A/B/B-global。** `training.run_training` 的三个 Caption family 共用 `CaptionManifestDataset` 与 `collate_caption`。Dataset 要求七个非空 provenance 字段、全局唯一 id、合法 split、64 位小写十六进制 SHA-256；取样时校验实际图片 SHA-256，再按短边 256/中心裁剪 224、训练水平翻转和 ImageNet 归一化读取。Collator 输出 images/inputs/targets/references/truncated，默认 64 Token，词表只由 Train 拟合。正式数据接口目前被缺失 Manifest 阻塞，不能直接读取 Parquet。

仅为接口测试，从四条显式文件名和 Caption 均精确匹配官方 Train 的真实 Parquet 记录复制图片 bytes 到 outputs，使用项目根作为该诊断 fixture 的 data-root。此 fixture **不是** `data/caption_manifest.jsonl`，没有新增 Val、没有替代全量训练清单。CPU Dataset/Collator 与该诊断清单的 CLI 审计均成功，图片批形状 `[4, 3, 224, 224]`，输入形状 `[4, 16]`，无截断。ID：`Abalone/Abalone_0002.jpg`, `Abalone/Abalone_0003.jpg`, `Abalone/Abalone_0004.jpg`, `Abalone/Abalone_0005.jpg`。未实例化 A/B/B-global 模型或执行训练。

**Hybrid V2。** 复用 `IngredientVocabulary.fit(train.rows)`、`TrainStatistics.fit(train.rows)`、`serialize_ingredients`、`NutritionDataset` 和 `collate_hybrid`。先规范化同名重量求和，再成对按类别 ID 排序；BOS,c1,… 对应 c1,…,EOS，重量 boundaries 为 1…K，PAD 不监督，输入 OOV 为 UNK、对应 CE 忽略且原始评价真值保留。Train 全量类别—重量往返及反序输入不变性已有真实数据测试通过；Val/Test 的已知类别配对也全量核对。最大监督长度 Train/Val/Test 为 17/16/13，低于正式配置 max_tokens=128，不会截断配对。

每划分选首条、末条及食材最多样本执行 CPU 真实数据加载/Collate，全部张量有限：

| 划分 | 样本 ID | images | inputs | weights | totals | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| train | dish_1559149615, dish_1562963704, dish_1562959599 | [3, 3, 224, 224] | [3, 17] | [3, 16] | [3, 5] | PASS |
| val | dish_1561567053, dish_1564778057, dish_1561060703 | [3, 3, 224, 224] | [3, 16] | [3, 15] | [3, 5] | PASS |
| test | dish_1556575327, dish_1568401302, dish_1562099157 | [3, 3, 224, 224] | [3, 13] | [3, 12] | [3, 5] | PASS |

现有 `test_hybrid_v2.RealDataTests.test_all_train_roundtrip_permutations_and_split_oov` 实际通过 **1 test / 25.206s**，还逐张调用真实 NutritionDataset 验证全部 3258 图的 `(3,224,224)` 输出，无训练。日志见 [hybrid_data_test.log](../../outputs/data_manifest_acceptance_20261124/hybrid_data_test.log)。

**Qwen3-VL。** `QwenDataset` 直接读取原图 RGB PIL，不走自主模型裁剪。真实 Train 首末两个样本 `dish_1559149615, dish_1562963704` 均为 640×480 RGB，target 与原始完整 JSON 对象相等。复用现有测试辅助函数构造本地合成 tokenizer/template 和真实 HF Processor（随机 tiny 模型构造后未执行前向/训练），对真实 RGB/target 运行 QwenCollator：input_ids/labels 形状 `[2, 548]`，监督文本解码回 JSON 与两个完整 target 均相等，含 assistant 结束 token；image/user/header/PAD 不监督。未加载官方 4B 权重，也没有官方 Processor 缓存的实测，因此官方模板、像素展开、全语料 max_length 和 Token 长度仍 **BLOCKED**；此项应在正式实验前用固定 revision 的 Processor 单独预检，不能把本测试当作官方协议验收。

**Train/Val/Test 边界。** 静态核对 `training.py`：仅 train Dataset 进入训练 loader；`validate` 仅接收 Val；Caption 词表只由 train Caption 扩展；Hybrid 类别表、数值统计只拟合 Train。`qwen_training.py` 明确 train_dataset=train、eval_dataset=val、TrainStatistics.fit(train.rows)，不使用 Test 选模。实际用 Val/Test 调用类别表与统计 fit 均抛 ValueError，四项保护通过。没有启动训练来验证选模效果。

**推理隔离。** `RGBManifestDataset` 和 QwenDataset(inference=True) 在真实 Test 派生清单上返回字段均严格为 id/image；含 target 的诊断输入均实际拒绝。静态检查 `qwen_messages`：推理消息只含原 RGB 和固定指令，训练时才额外构造完整 assistant target。target_text/metadata 不进入两个 Dataset 的返回样本或推理消息。

**CLI 审计退出状态。** 原始任务二全量 `audit --check-images` 退出 0、passed=true、image_failures=[]。以下负例均只修改 outputs 中的诊断副本，不计入原数据异常数：

| 诊断输入 | 实际退出码 | 是否写 JSON 报告 | 结果 |
| --- | ---: | --- | --- |
| 一条真实合法 Nutrition 标注 | 0 | 是 | passed=true |
| 改为不存在的图片路径 | 1 | 是 | passed=false；保留 dish_1559149615 / FileNotFoundError |
| fat_g 改为 bool | 1 | 否 | Schema 异常直接抛出，日志没有结构化样本 ID |
| 无效 JSON 物理行 | 1 | 否 | JSONDecodeError，未生成异常样本清单 |
| 缺失的正式 Caption Manifest | 1 | 否 | FileNotFoundError，正式验收 BLOCKED |
| 四条真实 Caption 诊断清单 | 0 | 是 | 哈希/图片检查通过 |

详见 [cli_cases.json](../../outputs/data_manifest_acceptance_20261124/cli_cases.json) 和对应 cli_*.log。现有 audit 只对图像错误保存 ID；JSON/Schema/全局划分前置校验抛异常时不写 out，safe_image_path 的异常也位于捕获块外。此外 nutrition audit 不计算跨图哈希、不强制所属文件名对应 split，Pillow verify 不等于完整 RGB 解码。本轮通过补充检查覆盖这些数据项，**未声称 CLI 单独提供全部验收能力，也未修改工具代码**。

## 5. 问题清单与最小修复方案

| 级别 | 问题 | 最小修复/关闭条件 |
| --- | --- | --- |
| P0 | 任务一正式 Manifest 缺失；尚无清洗后 Train/Val/Test | 完成来源和逐条映射后，生成七字段 Manifest、逐图哈希、duplicate_group、排除清单；按既定测试优先和 seed42 分组验证规则生成并审查计数，再跑全量审计 |
| P0 | 社区图片来源未锁定，779 条严格映射异常，净条数差 5 不能定位为五个缺图 | 提供社区仓库 ID+revision 的来源证据/转换记录，逐条核对 Unicode/名称和 Caption 差异及未配对记录；补齐可靠对应输入或由组员明确有证据的排除规则。禁止按顺序配对、猜标签或静默丢样本 |
| P0 | 任务一 126 个相同图像组跨 train/test，14 组在单划分内 | 映射确认后按既定协议隔离整组、记录稳定代表/训练侧排除及多 Caption 保留规则；保留原图原标注，派生清单不能覆盖原划分 |
| P1 | 任务二 536 对近重复候选、60 组跨划分相同标签的餐次关系未判定 | 正式实验前查看候选 RGB 与来源餐次/拍摄信息并标记结论；沿用课程原划分、保留主评价全样本，可另做敏感性说明，不能据 dHash 删除/改划分 |
| P1 | Qwen 官方 Processor 与全语料长度未实测 | 准备配置中固定 revision 的 Processor 后做 CPU 图像展开/完整 Target 长度审计，无需加载 4B 权重；确认 max_length 不截断 JSON |
| P1 | audit 的 JSON/Schema/路径异常分支无统一结构化 ID/行号报告 | 后续最小修复统一失败报告与非零退出，并用本次负例复验；本轮数据通过由独立严格解析/全量检查佐证，不因工具抛异常改标签 |
| P2 | Task2 OOV 和真实重量差异需要随实验解释 | 固定151类训练映射，保留 Val/Test 未见类真值、OOV CE 行为和全样本分母；不使用 Val/Test 扩词，不要求食材和等于整餐 |
| P2 | 任务一来源/图片哈希、派生产物还未写入可冻结来源记录 | 来源及映射获证后更新来源记录和生成工具文档，发布审计证据及清单哈希；本轮不为通过验收而改 sources.json |

## 6. 最终结论

- **任务一：不具备正式训练数据条件（BLOCKED）。** 原始标注及社区图像可读，但正式 Manifest 缺失、来源/映射不闭合、确定性跨划分重复未按既定规则处理、Val 未生成。
- **任务二：具备原课程数据的读取与正式训练数据基础条件。** 2338/414/506、全部严格标签及图片检查、train-only 词表统计和 CPU 数据接口通过；现阶段可原样上传这部分。未证明不存在近重复/同餐关联，候选复核和 Qwen 官方 Processor 预检需在正式实验前完成。
- **Manifest 不能整体冻结。** Caption 无可冻结 Manifest；任务二监督标注可按本报告 SHA-256 锁定，Test RGB 清单为本次派生审计产物，应连同哈希/来源记录保存。
- **上传 AutoDL 前必须完成：** 关闭任务一三个 P0，导出可直接读取的逐图文件与正式 Manifest，保留来源、排除清单和重复组；在本机对最终清单重跑本报告全量检查并核对所有哈希。任务二可先上传，但应附候选标记与检查报告，不以当前整体结论放行任务一 GPU 训练。

整体门禁：**DATA NOT READY**。本轮未运行正式训练、微调、模型推理、任选任务或 GPU 适配，没有模型效果/显存/吞吐结论。

## 7. 实际命令、证据与复现

已阅读根 AGENTS/README、COURSE_REQUIREMENTS、CONTRIBUTING、数据 README/sources、工程 README/实现记录、第五章数据规则及现有 data.py/schema.py/commands.py/ingredients.py/training.py/qwen.py/qwen_training.py 与相关数据测试；仓库内未发现可生成真实 Food500 Manifest 的现成脚本。优先复用现有 CLI、Schema、Dataset、Collator；outputs 下的脚本只补充验收，不是第二套训练入口。

项目根实际命令（PowerShell；本机已有 venv，无需重新安装训练依赖）：

```powershell
git status --short
git rev-parse HEAD
git branch --show-current
git -C data/Food500-Cap rev-parse HEAD
python -m pip install --target outputs/data_manifest_acceptance_20261124/deps pyarrow --disable-pip-version-check --retries 0
python -m pip install --target outputs/data_manifest_acceptance_20261124/deps pillow --disable-pip-version-check --retries 0
& foodlens_engineering/.venv/Scripts/python.exe outputs/data_manifest_acceptance_20261124/check.py
& foodlens_engineering/.venv/Scripts/python.exe outputs/data_manifest_acceptance_20261124/near_check.py
python -X utf8 outputs/data_manifest_acceptance_20261124/parquet_check.py
python -X utf8 outputs/data_manifest_acceptance_20261124/mapping_check.py
python -X utf8 outputs/data_manifest_acceptance_20261124/alias_check.py
# 在 parquet_check 完成后执行，使用其精确匹配的四条真实诊断图片：
& foodlens_engineering/.venv/Scripts/python.exe outputs/data_manifest_acceptance_20261124/cli_cases.py
```

从 `foodlens_engineering/` 实际执行：

```powershell
& .venv/Scripts/python.exe -m foodlens.cli audit --kind nutrition --inputs ../data/annotations/train.jsonl ../data/annotations/val.jsonl ../data/annotations/test.jsonl --data-root ../data --check-images --out ../outputs/data_manifest_acceptance_20261124/nutrition_cli.json
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_hybrid_v2.py -k test_all_train_roundtrip_permutations_and_split_oov -v
# 实际失败，退出1；并没有获得正式 Caption 审计通过：
& .venv/Scripts/python.exe -m foodlens.cli audit --kind caption --inputs ../data/caption_manifest.jsonl --data-root ../data --check-images --out ../outputs/data_manifest_acceptance_20261124/caption_missing_cli.json
```

负例的完整 argv/退出状态在 cli_cases.json；原始 CLI 全量结果在 nutrition_cli.json。新环境应使用新的输出子目录或先归档本次输出；deps 是 Python 3.13 的独立小型依赖，不能并入 Python 3.12 训练环境。`mapping_check.py` 只用于独立投影核对；正式图像解码与诊断 fixture 由 parquet_check 生成。主检查脚本、near_check、Parquet/别名脚本和完整 JSON 都保存在 `outputs/data_manifest_acceptance_20261124/`，默认 Git 忽略；交接此审计时须随报告携带该证据目录（可不携带 deps，按上述版本安装），不能假定干净检出包含本机产物。

执行过程曾遇到 PyArrow 首次安装被沙箱网络拦截，随后获准安装小型读取依赖成功；首次补充脚本误读 `._` AppleDouble 元数据以及非 UTF-8 控制台打印失败，均已修正并重跑，以最终 JSON/退出状态为准。未将这些执行错误写成真实数据异常。
