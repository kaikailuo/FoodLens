# FoodLens 迁移前 Git 整理与轻量验收（2026-11-25）

基准为 `main` / `8d501a9794099c9cb9efdd0abaf53a76504bafe7`，整理分支为 `chore/prepare-autodl-migration`。本次仅整理本地 Git 交付，不推送、上传、下载或重新构建数据，不修改原始数据和冻结划分；测试只使用临时合成样例。

## 交付内容

- 纳入 Food500 离线构建器、独立验收器、5 项工具测试、独立数据依赖清单和固定来源锁。
- 更新数据说明、来源记录、工程入口，保留数据初审、Food500 最终验收和 Hybrid 历史审查；旧结论增加后续记录入口，避免把历史状态误当作当前能力。
- 忽略课堂原始资源包、macOS 附属文件、常见本地模型/数据/缓存目录及凭据文件；课堂 Notebook 原样保留本地，不纳入正式工程交付。
- 来源锁保留原始 CRLF 字节，使用单文件 `.gitattributes` 例外禁止换行转换，维持冻结 SHA-256 `44db3d91638a805887ccdfdfaa32aeb99f1557a2117d4fa2763583d71626ccdf`。其他 JSON 继续遵循项目 LF 约定。

所有提交均显式列出文件暂存；数据、图片、模型、环境、临时检查结果和审计大附件不提交。忽略规则不代替后续提交时的暂存清单检查，也不会删除本地文件。

## 本次验证

工作目录为 `foodlens_engineering/`，无网络下载：

| 检查 | 命令或方法 | 结果 |
| --- | --- | --- |
| Food500 工具 | `python -X utf8 -B -m unittest discover -s tests -p test_food500_manifest.py -v` | 5 项通过，无模型或真实数据构建 |
| 自主模型基础 | `.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_baseline.py -k ModelTests -v` | 7 项通过，CPU 合成/随机权重 |
| 工程回归 | `.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_engineering.py -v` | 16 项通过，含 tiny Qwen、恢复和输入隔离 |
| 两个新 CLI | `python -m scripts.build_food500_manifest --help`、`python -m scripts.verify_food500_manifest --help` | 参数入口通过，不执行构建或全量数据验收 |
| 静态一致性 | 3 个新增 Python 文件 AST、训练 JSON 配置、来源锁 revision/原字节哈希、受影响文档本地文件链接 | 通过 |
| Git 交付 | `git diff --cached --check`、显式文件清单、忽略规则探针、暂存内容检查 | 通过，暂存来源锁字节哈希与冻结记录一致，无数据/模型/环境混入 |
| 无数据导出检查 | 从暂存 Git blob 导出工具、测试和来源记录至临时空目录，复用已安装的数据工具依赖 | 5 项工具测试再次通过，不复制数据或 venv |

Food500 工具测试使用现有 Python 3.13 及隔离数据工具依赖；模型/工程回归使用现有 Python 3.12.14 CPU venv。新环境分别按数据依赖和目标 CUDA 依赖安装，不依赖这些本地环境目录。

首次工程测试断言全部通过，但版本采集子进程遇到宿主不完整的 `GIT_CONFIG_COUNT` 环境。随后仅对测试进程设置 `GIT_CONFIG_COUNT=0` 并开启 HF 离线模式，重跑 16 项全部通过；没有修改仓库或全局 Git 配置。最终回归日志位于本地 `outputs/git-migration-prep-20261125/engineering.log`，不作为干净检出的必备输入。没有运行全量测试套件、正式训练、Linux/CUDA 或真实 4B 加载。

## 远端资源与边界

代码可从包含本次提交的 Git 版本检出。运行数据仍需另行迁移：Food500 Manifest、图片、锁、校验表和精选证据；任务二原课程图片与三份标注；固定 Qwen 本地模型文件。

仅训练和 CLI Caption 审计不需要原始 Parquet。若服务器运行 `scripts.verify_food500_manifest`，还必须提供 `data/Food500-Cap/finetune_data.json` 和 `evaluation_data.json`，合计 5,028,566 bytes，以及构建证据；二者不纳入 Git。从零重建才另需九个固定 Parquet。

当前具备代码交付和迁移准备条件，不能称为远程训练已验收。正式训练前仍须在目标容器确认磁盘/数据盘、GPU/CUDA/BF16、ImageNet 权重、官方 Qwen Processor/4B、Java METEOR 和相关后端，并记录试跑显存、吞吐及环境。已有数据与来源风险按各验收报告保留，不因本次 Git 整理而关闭。
