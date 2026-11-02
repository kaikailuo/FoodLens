# 本地数据准备

完整数据、图片、原始压缩包和外部数据仓库不进入本项目 Git 历史。当前已有文件保留原位；新成员需要自行准备。已记录的版本、文件计数与 SHA-256 见 [sources.json](sources.json)。

## 任务二：课程营养数据

向课程或项目成员获取原始 `annotations.zip` 和 `images.zip`，分别解压到本目录，检查不要多嵌套一层文件夹：

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
└── source/             # 可选：原始压缩包备份
```

每条 JSONL 的 `image` 都相对 `data/`，例如 `images/train/...png`。不要修改原始划分或把标签内容放入推理输入。完整工程测试及 smoke 需要这些数据，下载渠道尚无公开地址记录。

## Food500 Caption 数据

在项目根目录运行以下命令可获取已记录版本的标注仓库（已有目录时无需重复）：

```powershell
git clone https://github.com/aaronma2020/Food500-Cap.git data/Food500-Cap
git -C data/Food500-Cap checkout a32dfe90c01801231aad92c59f6bebe8a8e0dfe6
```

这里的 `.git` 属于外部标注仓库，整个目录由根 `.gitignore` 排除，无需删除其 Git 历史，也不作为本项目 submodule。

图片获取说明见该仓库 README。现有 `Food500-Cap/Food500Cap-HF/` 是本地社区图片副本，其来源 revision 尚未核实，不能当作已经锁定的数据版本。正式训练前需完成图片与 caption 的唯一 manifest、哈希、重复组及划分校验。

当前本地 `finetune_data.json` / `evaluation_data.json` 分别为 19880 / 4940 条；上游 README 写的训练数量为 19760，差异仍待核实。不要仅按行号把图片和 caption 拼接。

## 运行约定

从 `foodlens_engineering/` 执行 CLI，数据参数使用 `--data-root ../data`。少量可共享的自动测试样例将来放到工程的 `tests/fixtures/`，避免 CI 依赖完整数据包。
