# 本地数据准备

完整数据、图片、原始压缩包和外部数据仓库不进入本项目 Git 历史。干净检出只有本说明和 [sources.json](sources.json)，不会自带可训练数据。后者记录来源与本地文件指纹；Food500 的固定源分片还由[构建来源锁](../foodlens_engineering/scripts/food500_source_lock.json)记录。课程图片压缩包的来源方校验和仍未取得。

## 任务二：课程营养数据

组员自行从**教学云平台**下载 `annotations.zip` 和 `images.zip`。在 Linux 服务器和 Windows 本机都使用项目根目录下的 `data/`；原始压缩包统一保存到 `data/source/`，标注解压到 `data/annotations/`，图片解压到 `data/images/`。不要直接下载到工程代码目录，也不要多嵌套一层文件夹：

```text
data/
├── annotations/
│   ├── train.jsonl      # 2338 条
│   ├── val.jsonl        # 414 条
│   └── test.jsonl       # 506 条
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── source/             # 下载的原始压缩包：annotations.zip、images.zip
```

每条 JSONL 的 `image` 都相对 `data/`，例如 `images/train/...png`。不要修改原始划分或把标签内容放入推理输入。完整工程测试及 smoke 需要这些数据。

教学云平台入口由组员按课程通知访问，无需在仓库保存个人访问链接或账号。不能用其他 Nutrition5k 版本直接替代此课程数据包并声称复现成功。没有数据时可先运行根 README 的无数据测试。

### 字段约定

| 字段 | 含义及使用位置 |
| --- | --- |
| `dish_id` | 样本 ID；推理 manifest 的 `id` 来自此字段 |
| `split` | `train`、`val` 或 `test`，与原文件划分一致 |
| `image` | 相对 `data/` 的图片路径，不能越界 |
| `target.items` | 食材对象列表，每项仅含 `name`、`weight_g` |
| `target` 的整餐字段 | `total_weight_g`、`total_calories_kcal`、`protein_g`、`fat_g`、`carbohydrate_g` |
| `target_text`、`metadata`（如存在） | 不进入推理输入，不能用于提供答案或修复预测 |

质量单位为 g，热量单位为 kcal；目标数值有限且非负。严格结构规则以[实现](../foodlens_engineering/foodlens/schema.py)为准。食材重量之和与整餐总重量不强制相等。

### 校验本地数据

从**项目根目录**选择对应系统的代码块执行。只用 Python 标准库，无需下载权重；校验标注哈希、条数、split、ID 唯一性及全部引用图片的路径与存在性，不修改原始数据。

Linux / Bash：

```bash
python3.12 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("data").resolve()
sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
seen = set()
for entry in sources["nutrition"]["splits"]:
    path = root / entry["path"]
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"], f"hash mismatch: {path}"
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == entry["records"], f"record count mismatch: {path}"
    for row in rows:
        assert row["split"] == path.stem, f"split mismatch: {row['dish_id']}"
        assert row["dish_id"] not in seen, f"duplicate ID: {row['dish_id']}"
        seen.add(row["dish_id"])
        relative = Path(row["image"])
        image = (root / relative).resolve()
        assert not relative.is_absolute() and image.is_relative_to(root), f"image path escapes data root: {relative}"
        assert image.is_file(), f"missing image: {image}"
    print(f"OK {entry['path']}: {len(rows)} records, SHA-256, IDs and image paths")
print(f"OK total: {len(seen)} unique samples")
PY
```

Windows / PowerShell：

```powershell
@'
import hashlib
import json
from pathlib import Path

root = Path("data").resolve()
sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
seen = set()
for entry in sources["nutrition"]["splits"]:
    path = root / entry["path"]
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"], f"hash mismatch: {path}"
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == entry["records"], f"record count mismatch: {path}"
    for row in rows:
        assert row["split"] == path.stem, f"split mismatch: {row['dish_id']}"
        assert row["dish_id"] not in seen, f"duplicate ID: {row['dish_id']}"
        seen.add(row["dish_id"])
        relative = Path(row["image"])
        image = (root / relative).resolve()
        assert not relative.is_absolute() and image.is_relative_to(root), f"image path escapes data root: {relative}"
        assert image.is_file(), f"missing image: {image}"
    print(f"OK {entry['path']}: {len(rows)} records, SHA-256, IDs and image paths")
print(f"OK total: {len(seen)} unique samples")
'@ | python -
```

预期三个划分均为 `OK`，总计 3,258 个唯一 ID。该检查不验证图片能否解码、标签 Schema 或图片内容重复；完整测试和后续图片审计各自补充。哈希不一致时先检查来源、解压层级和文件是否被编辑，不要直接修改 `sources.json` 让检查通过。

## Food500 Caption 数据

在项目根目录运行以下命令可获取已记录版本的标注仓库（已有目录时无需重复）：

```powershell
git clone https://github.com/aaronma2020/Food500-Cap.git data/Food500-Cap
git -C data/Food500-Cap checkout a32dfe90c01801231aad92c59f6bebe8a8e0dfe6
```

这里的 `.git` 属于外部标注仓库，整个目录由根 `.gitignore` 排除，无需删除其 Git 历史，也不作为本项目 submodule。

图片获取说明见该仓库 README。本地 `Food500-Cap/Food500Cap-HF/` 已核对为社区数据集 [advancedcv/Food500Cap](https://huggingface.co/datasets/advancedcv/Food500Cap/tree/1b2e0cb098ef3d58584f80bde015207c5a3d00c7)，revision 为 `1b2e0cb098ef3d58584f80bde015207c5a3d00c7`。九个 Parquet 的本地 SHA-256 与该 revision 的公开 LFS 元数据一致。不要改用同名数据集的其他版本。

本地官方 `finetune_data.json` / `evaluation_data.json` 分别为 19880 / 4940 条；上游 README 写的训练数量为 19760，与固定 JSON 快照不一致。社区副本分别为 19877 / 4938 条。构建时以官方 filename、类别和完整 Caption 建立唯一对应：24,036 条完全精确，779 条经固定 Unidecode 1.4.0 转写后同时精确匹配文件名、类别和全文。该转换的逐条等价已经验证，但社区未发布历史生成脚本，不能声称已找到其原始转换代码。正式清单保留官方原始 Caption，不使用社区改写文本，也不按行号配对。

本地 `evaluation_data.json` 的序列化格式与固定 Git blob 不同，但完整 JSON 内容相同。来源锁同时接受两个已知字节哈希，且另验规范化完整 JSON 内容哈希；干净克隆可直接使用固定 Git 版本，不需要取得本机排版副本。脚本不重写任一原始标注。

完整构建规则、五条缺图标注、跨划分重复隔离和本次实际最终数量见[最终验收报告](../docs/reviews/FOOD500_MANIFEST_FINAL_20261124.md)。缺图标注与冲突训练侧只写入派生排除清单；官方 JSON、Parquet 及课程任务二数据不变。

正式 Caption 数据就绪至少应包含：图片来源仓库 ID 与不可变 revision、原始标注哈希、逐图片哈希、显式映射、重复组、排除清单和最终划分计数。manifest 每行包含 `id/image/caption/official_filename/sha256/duplicate_group/split`；其中 `image` 相对 `data/`。遵循第五章测试归属优先、剩余训练按重复组划验证的规则，词表和统计仅从最终训练集拟合。

### 离线构建与上传

从 `foodlens_engineering/` 执行，先在独立数据工具环境安装 `requirements-data.txt`（不需要 Torch 或模型权重），确认官方 JSON 与九个固定 Parquet 位于上述目录：

```bash
python -m pip install -r requirements-data.txt
python -m scripts.build_food500_manifest --data-root ../data --manifest ../data/caption_manifest.jsonl --out ../outputs/food500_manifest_build
```

脚本先校验源锁，再按文件标识及完整文本双重匹配，按图片字节与 RGB 像素建立连通重复组。Test 优先隔离训练侧；Train 内合法多 Caption 保留在同一组。剩余训练组以官方类别的稳定主类别分层、seed=42、约 10% 组数生成 Val。图片原始 bytes 导出为 `caption_images/sha256/<前两位>/<完整哈希>.<实际格式>`，无转码。部分原 `.jpg` 实际为 MPO，导出 `.mpo` 并按现有 Dataset 行为读取主帧。已有导出图片须与内容哈希一致；已有不同内容的 Manifest 会被拒绝覆盖。

然后使用工程 CPU 环境独立验收（其中验证脚本不需要 PyArrow，但需要 `Food500-Cap/finetune_data.json` 和 `Food500-Cap/evaluation_data.json` 两个固定官方标注；仅有导出图片和 Manifest 时可运行 CLI 审计，不能完成原始标注账本验收）：

```bash
python -m foodlens.cli audit --kind caption --inputs ../data/caption_manifest.jsonl --data-root ../data --check-images --out ../outputs/caption_manifest_final_audit.json
python -m scripts.verify_food500_manifest --data-root ../data --manifest ../data/caption_manifest.jsonl --evidence ../outputs/food500_manifest_build --out ../outputs/food500_manifest_build/independent_audit.json
```

上传时至少携带 `data/caption_manifest.jsonl`、整个 `data/caption_images/`、`data/caption_data_lock.json`、`data/caption_sha256sums.txt` 和 `outputs/food500_manifest_build/` 中的 JSON/JSONL 审计证据（排除 `rebuild/` 重复副本）；代码、来源锁和 `sources.json` 通过 Git 同步。训练与推理不依赖原 Parquet 或临时 HF 缓存；服务器若需运行独立原始标注验收，还应携带上述两个官方 JSON，保留 `data/Food500-Cap/` 层级；从零重建另需原始九个分片。三个 Caption 模型使用同一 Manifest，推理使用审计目录下只含 `id/image` 的 `test_rgb.jsonl`。不要将本机独立数据工具环境或 CPU 训练环境上传为 Linux CUDA 环境。

来源锁的原始字节 SHA-256 已写入冻结记录，`.gitattributes` 因此单独禁止该文件的换行转换；不要重新排版或归一化它来修复校验错误。数据工具依赖与训练环境分离，`requirements-data.txt` 不替代 Linux/CUDA 训练依赖。

## 运行约定

从 `foodlens_engineering/` 执行 CLI，数据参数使用 `--data-root ../data`。无标签推理 manifest 与同子集评价标注的生成步骤见[工程 README](../foodlens_engineering/README.md)。少量可共享的自动测试样例将来可放到工程的 `tests/fixtures/`；目前没有随仓库提供的课程数据 fixture。
