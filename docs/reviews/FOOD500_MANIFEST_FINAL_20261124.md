# Food-500 Cap 映射修复与正式 Manifest 最终验收（2026-11-24）

**FOOD500 READY FOR GPU UPLOAD**

本轮已实际生成正式 `data/caption_manifest.jsonl` 和规范图片资产，并通过全量 CLI、独立图片/标注/排除账本验收及第二次完整构建的一致性检查。状态覆盖本报告冻结的社区图片版本与派生子集，不代表官方 24820 条原样全集或 GPU 训练已经验证。上一轮报告中的任务一 P0 已关闭；任务二不在本轮修改范围。

## 1. 交付与冻结指纹

| 资产 | 实际结果 |
| --- | --- |
| 正式 Manifest | [data/caption_manifest.jsonl](../../data/caption_manifest.jsonl)，24689 行、17835290 bytes |
| Manifest SHA-256 | `9d85e824a84a87e3c790dec34e8df8064586d1d368b3dc6d8a0ffc7d8d417485` |
| 图片 | `data/caption_images/sha256/`，24675 个实际文件、2989710653 bytes |
| 图片格式 | JPEG 24662、PNG 11、MPO 2，全部保持社区源 bytes，无转码 |
| 每张图 SHA-256 / RGB 哈希 / 大小 / 格式 | [image_inventory.jsonl](../../outputs/food500_manifest_build/image_inventory.jsonl)；该清单 SHA-256 `03992c031ab5c81ac61ce856175fdfd2c9cc92ab3859463ee89a570e26c89a77` |
| 可携带哈希校验表 | [caption_sha256sums.txt](../../data/caption_sha256sums.txt)，含 Manifest 与全部24675张图片；SHA-256 `2b8c73b586e586c2cf2c2b1a776dd044d2dc1727f1caa95a99956bbfc5d806bf` |
| 冻结元数据 | [caption_data_lock.json](../../data/caption_data_lock.json)、[sources.json](../../data/sources.json) |
| 构建入口 | [build_food500_manifest.py](../../foodlens_engineering/scripts/build_food500_manifest.py) |
| 独立验收入口 | [verify_food500_manifest.py](../../foodlens_engineering/scripts/verify_food500_manifest.py) |
| 机器可读最终结论 | [final_acceptance.json](../../outputs/food500_manifest_build/final_acceptance.json) |

工程代码基线为 `main` / `8d501a9794099c9cb9efdd0abaf53a76504bafe7`，新数据工具为本轮未提交的工作区文件，不能将其当成已存在于该 Commit。构建脚本 SHA-256 `5e38e3126ec8af9729fb086887d252eacd02bf8281dc0e9103877259f28eb440`，独立验收脚本 SHA-256 `3d7a0bc570556f934b46d66f4342e4195b6ca25a4ecd4321fd7850d7a24cfb22`，来源锁 SHA-256 `44db3d91638a805887ccdfdfaa32aeb99f1557a2117d4fa2763583d71626ccdf`。没有新建分支、Commit、PR 或推送。

## 2. 来源和版本证据

社区图片来源已锁定为 [advancedcv/Food500Cap 固定版本](https://huggingface.co/datasets/advancedcv/Food500Cap/tree/1b2e0cb098ef3d58584f80bde015207c5a3d00c7)，revision `1b2e0cb098ef3d58584f80bde015207c5a3d00c7`。公开 [HF API](https://huggingface.co/api/datasets/advancedcv/Food500Cap/revision/1b2e0cb098ef3d58584f80bde015207c5a3d00c7?blobs=true) 的 repo ID、commit SHA、九个文件名/大小/LFS SHA-256，与本地缓存和重新计算的实际文件哈希全部相符。

原始标注来源为 [aaronma2020/Food500-Cap](https://github.com/aaronma2020/Food500-Cap/tree/a32dfe90c01801231aad92c59f6bebe8a8e0dfe6)，revision `a32dfe90c01801231aad92c59f6bebe8a8e0dfe6`。训练 JSON 19880 条、SHA-256 `b69439dfdca82b2d930c6da3d23a9538351459ac6fd85b18c510aaa7cef712e2`；测试 JSON 4940 条、本地实际 SHA-256 `8218b582f4fbbadcff0566ada5ea6e807a86c404ab9cdf89166035b335e9cfe8`。本地测试 JSON 的序列化与 Git blob 不同：Git blob SHA-256 为 `401b615984ec06c2da431d12c44e1ad5f9efd220f92f7fe4e2b19ca3f6e02e63`，但两文件 json.loads 后完整对象序列完全相同；训练 JSON 连字节也相同。没有为匹配校验和而改写本地标注。构建器现同时接受这两个固定测试JSON字节版本，且必须通过相同的规范化完整JSON内容哈希；已用实际Git blob及5项测试验证。因此干净检出不需要取得本机排版副本，实际Git对象兼容性结果见 [clean_checkout_compatibility.json](../../outputs/food500_manifest_build/clean_checkout_compatibility.json)。证据见 [source/official_git_verification.json](../../outputs/food500_manifest_build/source/official_git_verification.json)。

| 固定 Parquet | bytes | 本地重算且与公开 LFS 一致的 SHA-256 |
| --- | ---: | --- |
| test-00000-of-00002-032e07b311d1db77.parquet | 295757170 | 4dd72492866388f1937d7a0f8e202a82b7f93188507aa5cfd47a48d070f4a3a5 |
| test-00001-of-00002-d55c6542fe70f451.parquet | 293856844 | 4ca6368721c84e2516d07c9477ce4ed88969241a3a22a3132908b955ce74d2fa |
| train-00000-of-00007-9de56bcdafdee2d8.parquet | 398641250 | c797764dd18ff7b2293e346541a5852797746f5b8bb04b90654679501b22c8ad |
| train-00001-of-00007-d4acaf169b202bde.parquet | 371182555 | 72a0b647e2ee6e988dcb261df6383d99047a6e761d70c18f18875f6d63d8a015 |
| train-00002-of-00007-3eeb547f5ee2bc77.parquet | 337133702 | d79073169ef4654d218c5da4a80c75744e9c4a01c3e402fe9e3d37522adf43f2 |
| train-00003-of-00007-db06d49ad5d281c2.parquet | 357558494 | 7217dc6f2abdec2c382116462dfef82cb16d49a3f01acaad3a528b76d370266f |
| train-00004-of-00007-34cdb0916951598d.parquet | 290325153 | 1fdec820cd981fce3d285041a9934ce7821c3b2cc38eb124418c81f4fd95d831 |
| train-00005-of-00007-25f227541c143e60.parquet | 317233056 | bfe697b66f20ebe5992053e55dde30c3ee060a34ba94d233a8ba579f44a3eabd |
| train-00006-of-00007-84404baa7f02576e.parquet | 339022377 | d57f80438949f1a8d965cadc708f18e39521bc1fb70514a472bc3d48fead892c |

公开版本 [README](https://huggingface.co/datasets/advancedcv/Food500Cap/blob/1b2e0cb098ef3d58584f80bde015207c5a3d00c7/README.md) 记录 train=19877、test=4938；[12条提交历史](https://huggingface.co/datasets/advancedcv/Food500Cap/commits/1b2e0cb098ef3d58584f80bde015207c5a3d00c7) 仅见分片/README等上传，未提供历史转换脚本。快照/API/提交记录保存于 `outputs/food500_manifest_build/source/`，请求 URL 在 [source/requests.json](../../outputs/food500_manifest_build/source/requests.json)，仓库身份和来源锁对照结果在 [source_verification.json](../../outputs/food500_manifest_build/source_verification.json)。

**已证明的是固定镜像身份、文件完整性及逐条标识/文本等价；没有证明历史制作者使用了哪个转换脚本，也没有官方 ISIA 图片原始字节可比，不能声称社区图片从未在上游重编码。** 本轮导出严格不转码，以锁定社区版本为图片资产依据。

## 3. 779 条异常的实际解决

匹配键是“原始官方 split + 完整文件标识”，不是 Parquet 行号、排序位置或 Caption 模糊相似度。构建器先对官方 filename 建立固定 Unidecode 1.4.0 + casefold 路径索引，要求该键唯一，再核对类别及**整个 Caption 经 Unidecode 后逐字符相等**；全文比较保留空白和标点，不使用编辑距离。只因 Caption 相似而存在的其他候选不会进入 Manifest；路径多候选时也不会按 Caption 选择第一个。来源锁和完整输入哈希在扫描前验证。

| 处理规则 | 匹配总数 | 最终纳入 | 因 Test 优先排除 Train |
| --- | ---: | ---: | ---: |
| case_path_full_caption_unidecode | 1 | 1 | 0 |
| exact | 24036 | 23925 | 111 |
| same_path_full_caption_unidecode | 63 | 63 | 0 |
| transliterated_path_full_caption_unidecode | 715 | 700 | 15 |

之前 716 个文件名异常可细分为 715 个字符转写、1 个大小写差异；另63条文件名已相同、Caption 仅在固定转写后全文一致。779 条**全部唯一确认**，未解决社区记录 **0**。其中764条最终纳入，15条虽已确认映射，仍因与Test完全相同图像而排除训练侧；不是把不确定匹配强行纳入。

例：官方 `Æbleflæsk/Æbleflæsk_0001.jpg` 与社区 `AEbleflaesk/AEbleflaesk_0001.jpg` 的文件标识、类别和完整文字转写均一致；官方 `Bacon_explosion/Bacon_explosion_0015.jpg` 与社区 `Bacon_explosion/Bacon_Explosion_0015.jpg` 为唯一的路径大小写情形。中文标点、弯引号、零宽字符和 `丝`→`Si ` 等差异也由固定版本的通用转写函数逐字符验证，没有逐条猜标签或把社区 Caption 替换成“更好”的标签。

**正式 Manifest 一律使用官方原始 filename 和 Caption**，不是转写后的文本。[mapping_evidence.jsonl](../../outputs/food500_manifest_build/mapping_evidence.jsonl) 每行保留官方 filename/caption/cat/split、社区 image.path/caption/cat/split、源分片、字节及RGB哈希、候选唯一性、规则、最终状态与排除原因。行号只作为 Parquet 内部迭代过程，没有作为身份、输出路径或匹配证据。

现在可从完整一一对应证明，固定社区版本未覆盖以下5条官方标注；这不同于上一轮仅凭净条数差的推测。没有伪造图片、补Caption或移到其他划分：

| 原官方划分 | 未有可确认社区图像的官方 filename | 固定排除 ID |
| --- | --- | --- |
| test | Corn_dog/Corn_dog_0093.jpg | food500_test_dd9265d955015a38293a3503 |
| test | Corn_dog/Corn_dog_0097.jpg | food500_test_a49c7519e2e9bcb7331f51bf |
| train | Abalone/Abalone_0062.jpg | food500_train_95bb0cdecdb0f97740ebdca3 |
| train | Corn_dog/Corn_dog_0047.jpg | food500_train_c6f1a913bc761f40915a2607 |
| train | Hummus/Hummus_0033.jpg | food500_train_d8085efdd59d7fb726c6124b |

原官方标注总24820条 = 最终纳入24689条 + 跨划分冲突训练侧126条 + 无对应社区图像5条。社区24815条 = 唯一确认24815条，所有社区行均在证据账本有最终状态。排除清单共131条，见 [exclusions.jsonl](../../outputs/food500_manifest_build/exclusions.jsonl)。

## 4. 确定性重复处理

重新读取所有原始图像 bytes，以 SHA-256 和主帧解码后的 RGB 像素哈希任一相同建立连通组（支持传递合并）。RGB 哈希协议为 SHA-256(`ASCII("RGB:<width>x<height>\0") + RGB像素bytes`)，头部末尾是一个NUL字节，不旋转EXIF、不缩放。组 ID 为 `dg_` + 组内最小字节 SHA-256，输入顺序变化不会改变组 ID。

- 140个非单例组，每组2条；126组跨Train/Test、14组仅在原Train。
- 126组跨划分：全部保留可靠Test记录，126条Train均排除，图片内容没有从Test迁回Train或Val。
- 14组原Train内部相同图片保留全部28条合法描述，统一分组：11组在Train、3组在Val；不因共享图片误删Caption。
- 官方Train/Test的88个同名filename均核实为字节相同，属于上述126组，训练侧均有排除原因。完整对照见 [official_filename_overlaps.json](../../outputs/food500_manifest_build/official_filename_overlaps.json)。
- Test内没有重复图像组，不需要额外删Test代表；Test保留4938条可靠图文记录。

完整成员、源划分、最终划分与排除原因见 [duplicate_groups.json](../../outputs/food500_manifest_build/duplicate_groups.json)。最终 SHA-256、RGB像素哈希、duplicate_group、官方filename 跨 Train/Val/Test 的重叠数量均为 **0**。没有把类别相同、Caption相似或视觉近似当成确定性重复。

## 5. 正式划分与图片资产

| 最终划分 | Caption / 样本ID数 | 唯一图片字节哈希数 | duplicate_group数 |
| --- | ---: | ---: | ---: |
| test | 4938 | 4938 | 4938 |
| train | 17758 | 17747 | 17747 |
| val | 1993 | 1990 | 1990 |

排除冲突后，训练侧19751条Caption、19737个图片组。仅在这些原Train组上进行约10% Val划分：每组主类别取组内**官方Train类别字典序最小项**，按类别和组ID稳定排序；Python `random.Random(42)` 按固定类别顺序逐层shuffle；每类别取 `int(n*0.1+0.5)` 个组，n>1时约束在1到n−1，n=1时不抽验证组。实际Val1990组（约10.08%）、1993条Caption（约10.09%）；稀有多类别重复组仍整体保留。每类别组数见 [stratification.json](../../outputs/food500_manifest_build/stratification.json)。Test只用于既定去重优先归属，不用于训练词表、验证分层、参数选择或模型指标决策。样本ID中的train/test表示原官方来源，最终划分以split字段为准，所以Val样本ID保留train前缀。

图片路径为 `caption_images/sha256/<哈希前两位>/<完整SHA-256>.<实际格式>`，相对data-root，无绝对路径、原始Unicode路径冲突或运行时行号依赖。同bytes共享同一物理文件。24662个JPEG、11个PNG、2个MPO均原样写出；MPO源虽名为jpg，按真实格式导出mpo，沿用Dataset读取主帧行为；没有重新编码或把后续帧传入模型。

所有文件的字节哈希与源Parquet bytes相同，且独立重新读取导出文件验证。图片目录实际文件集合与Manifest引用、image_inventory集合完全一致，无缺失或多余图片。训练和推理不需要原Parquet/HF缓存。

## 6. 全量验收与工程兼容性

| 实际检查 | 结果 |
| --- | --- |
| 固定源JSON/九分片的实际哈希 | PASS；所有值与来源锁一致 |
| 现有 Caption CLI audit --check-images | PASS，退出0，image_failures=[]；[JSON](../../outputs/caption_manifest_final_audit.json) |
| 独立主帧RGB完整解码及SHA/RGB哈希复算 | PASS，24675张唯一图片全量 |
| 全局ID、原始Caption、路径、三划分、重复组隔离 | PASS；无空Caption、无ID碰撞、无跨划分确定性重复 |
| 官方与社区账本/所有排除 | PASS；24820条官方和24815条社区全部有去向 |
| A/B/B-global共用正式Dataset/Collator | PASS；三个split各3条真实样本，images=(3,3,224,224)，未实例化模型 |
| Train-only词表 | PASS；仅17758条Train拟合，含特殊Token共4210，哈希`600ce72fa7cd880808a8074ed247eb1e3a14e61c9a6d64e031bdee8ec0a6fa4c` |
| 构建工具测试 | PASS；5项测试、0失败，无训练/下载 |
| 第二次完整离线构建 | PASS；Manifest、来源锁、全部映射/排除/图片索引/划分/summary/三份RGB清单逐字节一致 |

独立结果见 [independent_audit.json](../../outputs/food500_manifest_build/independent_audit.json)，复现对照见 [reproducibility.json](../../outputs/food500_manifest_build/reproducibility.json)。两个构建均重新校验全部原JSON/Parquet并重新解码图像；第二次仅复用经过实际字节哈希复核的内容寻址图片。每个模型使用同一个Manifest哈希，未从Val/Test扩词。派生 train/val/test_rgb.jsonl 严格只有id/image，可供后续推理；参考Caption留在监督Manifest，不能直接传为无标签推理输入。

两条非阻塞EXIF元数据警告来自 `Crudo/Crudo_0039.jpg` 和 `Lemon_chicken/Lemon_chicken_0022.jpg`，完整RGB像素可读，未为消除警告重编码。详见 [decode_warnings.json](../../outputs/food500_manifest_build/decode_warnings.json)。

## 7. 已知限制与对实验的影响

**P0：0项。** 来源、779条映射、确定性重复隔离、正式Val、图片导出与全量审计均已完成。

**P1（训练记录）：4条Train Caption长度超过现有max_tokens=64。** 训练最长含EOS为74，4/17758约0.0225%；Val最长63、Test最长64，均不超限。官方原文完整保留在Manifest；现有Collator会保留EOS并标记CE截断。本轮未修改超参数、未截短存储标签；正式实验需保留并报告既有长度审计/截断记录，若另作加长配置应作为后续实验配置变更。

| 官方filename | 样本ID | 含EOS长度 |
| --- | --- | ---: |
| Strawberry_cheesecake/Strawberry_cheesecake_0045.jpg | food500_train_1715aee16d130de7eb2036ce | 67 |
| Hainanese_chicken_rice/Hainanese_chicken_rice_0038.jpg | food500_train_1d30ed4c21ac4fde419befd3 | 66 |
| Cebolada/Cebolada_0066.jpg | food500_train_7bef59b2354b1390a21466d0 | 66 |
| Gołąbki/Gołąbki_0031.jpg | food500_train_8c36e8b746feea3efdafadb9 | 74 |

完整数据见 [caption_length_exceptions.json](../../outputs/food500_manifest_build/caption_length_exceptions.json)。Val/Test的UNK内容Token分别136/477，训练为0；这是Train-only词表的正常结果，不能用Val/Test扩词来消除。

**P2（版本/协议）：** 上游README训练计数19760与固定JSON快照19880不一致，已按实际版本哈希和样本身份冻结，不声称复现其19760条设置。本次Test分母为冻结的4938条可靠记录，不能当作原始4940条测试全集或与不同测试清单的文献分数直接对比。五条无对应记录和126条冲突训练记录已固定排除，不根据模型效果调整。

社区历史转换代码/图片重编码历史未公开，保留此来源说明；本轮建立的是可验证逐条对应，不捏造历史过程。未进行感知近重复或同源拍摄关系的人工审查，不声称排除了所有视觉近重复。后续若需此类清洗，应发布新清单版本与独立敏感性结果，不能静默覆盖已冻结清单。Windows CPU通过不代表Linux CUDA、ImageNet权重、四指标后端、正式训练或模型效果已通过。

## 8. 实际命令、退出状态与复现

本机从项目根开始，复用上轮独立数据工具依赖目录（Python3.13.18 / PyArrow26.0.1 / Pillow13.0.0），仅新增安装Unidecode1.4.0，小型依赖235kB，无数据集或模型下载。构建工具可从零在独立环境安装 [requirements-data.txt](../../foodlens_engineering/requirements-data.txt) 后使用；CPU Dataset 验收使用既有Python3.12.14工程venv。

| 实际操作 | 退出状态/证据 |
| --- | --- |
| git status/rev-parse，读取AGENTS/既有报告/数据契约/上轮产物 | 0；起始main及既有未跟踪目录不变 |
| 公开HF API、tree、README、commit元数据小文件读取 | 0；source/requests.json四个ok=true，无Parquet下载 |
| python -m pip install --target outputs/data_manifest_acceptance_20261124/deps Unidecode==1.4.0 --disable-pip-version-check --retries 0 | 0 |
| 映射探查：NFKD诊断、固定Unidecode完整匹配 | 0；probe.json与unidecode_probe.json |
| 首次构建 | 1；原始MPO格式未在导出后缀表支持，未生成正式Manifest；保留build_first_failure.log |
| 最小修复MPO原bytes导出后的完整构建 | 0；build.log、build_summary.json |
| 5项build工具unittest | 0；unit_tests.log |
| 全量CLI审计 | 0；cli_audit.log、caption_manifest_final_audit.json |
| 独立全量验收 | 0；independent_audit.log/json |
| 第二次完整构建 | 0；rebuild.log、rebuild/；全部冻结数据文件哈希一致 |

读取外部标注仓库Git对象时出现AppleDouble `._*.idx`的非单调索引警告；真实Git对象仍成功读取，逐对象JSON比较有结果，本轮未删除该仓库文件，也未用警告绕过标注哈希。

从 `foodlens_engineering/` 实际执行的主体命令（PowerShell）：

```powershell
# 本机用--deps-dir复用隔离的数据工具依赖；新环境安装requirements-data.txt后可省略该参数。
python -X utf8 -m scripts.build_food500_manifest --data-root ../data --out ../outputs/food500_manifest_build --manifest ../data/caption_manifest.jsonl --deps-dir ../outputs/data_manifest_acceptance_20261124/deps
$env:PYTHONPATH='../outputs/data_manifest_acceptance_20261124/deps'
python -X utf8 -m unittest discover -s tests -p test_food500_manifest.py -v
# 实际验收命令使用独立进程，不继承数据工具PYTHONPATH：
Remove-Item Env:PYTHONPATH
& .venv/Scripts/python.exe -m foodlens.cli audit --kind caption --inputs ../data/caption_manifest.jsonl --data-root ../data --check-images --out ../outputs/caption_manifest_final_audit.json
& .venv/Scripts/python.exe -m scripts.verify_food500_manifest --data-root ../data --manifest ../data/caption_manifest.jsonl --evidence ../outputs/food500_manifest_build --out ../outputs/food500_manifest_build/independent_audit.json
python -X utf8 -m scripts.build_food500_manifest --data-root ../data --out ../outputs/food500_manifest_build/rebuild --manifest ../outputs/food500_manifest_build/rebuild/caption_manifest.jsonl --deps-dir ../outputs/data_manifest_acceptance_20261124/deps
```

本轮测试以单独进程设置PYTHONPATH，未改变其他工具进程或训练环境。新重建不要覆盖历史审计目录，选新的out目录后比较相同Manifest SHA-256；若源/规则变化导致既有Manifest内容不同，脚本会拒绝覆盖。固定来源锁、脚本、Python/依赖版本、原始JSON和九分片是从零重建所需输入；不依赖本机的临时导出fixture或其他个人绝对路径。

## 9. 上传清单与最终结论

可以上传AutoDL的运行资产：`data/caption_manifest.jsonl`、完整 `data/caption_images/`、`data/caption_data_lock.json`、`data/caption_sha256sums.txt`、`data/sources.json`、工程代码及本次审计目录中的JSON/JSONL/来源快照和报告。训练读取只需Manifest与图片；独立验收原始Caption/账本时还需两个原始官方JSON；从零重建才需要九个原始Parquet。无需上传Windows venv、临时deps目录或本轮rebuild副本；正式Linux CUDA依赖另行配置。

上传后从服务器 `data/` 目录执行 `sha256sum -c caption_sha256sums.txt`，再按工程CPU可用环境运行全量CLI审计。校验表在本机由已全量复验的Manifest和图片索引生成；本轮没有上传服务器，因此不声称服务器端校验已执行。

本轮实际修改仅为数据构建/验收脚本、来源锁、数据工具依赖说明、5项工具测试、数据来源记录和说明文档；新增派生Manifest、原bytes图片和审计产物。官方JSON、社区Parquet、Nutrition原始数据、模型架构、训练代码、超参数及评价指标均未修改。未执行训练、GPU测试或Qwen加载。

**FOOD500 READY FOR GPU UPLOAD** — 可冻结并上传本报告明确版本、分母与排除规则下的数据资产。
