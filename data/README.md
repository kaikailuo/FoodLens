# 本地数据准备

完整数据、图片、原始压缩包和外部数据仓库不进入本项目 Git 历史。干净检出只有本说明和 [sources.json](sources.json)，不会自带可训练数据。后者记录已有来源、标注条数和标注文件 SHA-256；图片全集及压缩包的哈希尚未锁定。

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

图片获取说明见该仓库 README。现有 `Food500-Cap/Food500Cap-HF/` 是本地社区图片副本，其来源 revision 尚未核实，不能当作已经锁定的数据版本。正式训练前需完成图片与 caption 的唯一 manifest、哈希、重复组及划分校验。

当前本地 `finetune_data.json` / `evaluation_data.json` 分别为 19880 / 4940 条；上游 README 写的训练数量为 19760，差异仍待核实。不要仅按行号把图片和 caption 拼接。

现有 `sources.json` 的 Food500 哈希是已记录本地标注快照的证据，尚未证实社区图片与这些标注一一对应。克隆官方仓库只获取该仓库提供的内容，不等于已经取得可直接训练的全部图文对。

正式 Caption 数据就绪至少应包含：图片来源仓库 ID 与不可变 revision、原始标注哈希、逐图片哈希、显式映射、重复组、排除清单和最终划分计数。manifest 每行包含 `id/image/caption/official_filename/sha256/duplicate_group/split`；其中 `image` 相对 `data/`。遵循第五章测试归属优先、剩余训练按重复组划验证的规则，词表和统计仅从最终训练集拟合。

## 运行约定

从 `foodlens_engineering/` 执行 CLI，数据参数使用 `--data-root ../data`。无标签推理 manifest 与同子集评价标注的生成步骤见[工程 README](../foodlens_engineering/README.md)。少量可共享的自动测试样例将来可放到工程的 `tests/fixtures/`；目前没有随仓库提供的课程数据 fixture。
