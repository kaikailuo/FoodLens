# Caption A/B 修复与缓存交接（2026-11-24）

基于 `htl_1120` 的 `ebde49a35b77be285d6343343e0efa96dc74712a` 完成。保留第五章架构、数据划分、损失与贪心协议；修复 A/Hybrid 原先未生效的 Dropout 会改变后续训练，这是有意修复。没有真实数据训练或准确率结论。未改写用户提供的原始交接文件或报告/PDF。

## 改动与兼容

| 项目 | 实现 | 兼容边界 |
| --- | --- | --- |
| A/Hybrid Dropout | 词嵌入后 `nn.Dropout(config.dropout)` | 无新增参数；train 随机失活，eval 关闭，p=0 无影响。旧权重/优化器可加载；旧非零 Dropout checkpoint 恢复发出警告并记录版本，无法复现过去未失活的轨迹 |
| B/B-global 视觉投影 | `VisualEncoder.extract_features()` 返回 grid/GAP，B 不调用条件 projection | 保留 projection 参数及注册顺序，旧模型/优化器布局不变；相同权重 eval logits 与旧实现精确相同 |
| A GRU 缓存 | 最新 token + 前次 hidden，每步仍注入视觉 z | 默认关；BOS/EOS/PAD/上限保持；A SCST 的采样分支保留 hidden 的完整梯度 |
| B/B-global KV 缓存 | 原 decoder 权重执行每层增量 self Q/K/V；缓存过去 K/V 和视觉 cross K/V | 默认关；绝对位置、PAD key mask、残差/LayerNorm/FFN 与原语义一致；teacher forcing 仍走原 PyTorch Decoder；仅 eval/inference 可调用 |

缓存对象仅属于一次生成请求，不注册成模型参数或缓冲区，也不保存到 checkpoint。KV step 拒绝跨模型/视觉 memory 复用。历史配置缺开关自动按 false；未知开关或非布尔值报错。旧 `caption_generation`/共享 `generation_config` checkpoint 均可读取。

训练 JSON 的形式如下（max_tokens 仍在训练配置顶层）：

```json
{
  "generation": {
    "gru_cache_enabled": false,
    "transformer_kv_cache_enabled": false
  }
}
```

该片段应合入完整训练配置。`configs/train_caption_{a,b,b_global}_cache.json` 分别开启适用缓存；普通配置均关闭。两个开关互不影响，不适用的模型忽略对应开关并记录 effective=false。CLI 的 `--gru-cache-enabled` / `--no-gru-cache-enabled`、`--transformer-kv-cache-enabled` / `--no-transformer-kv-cache-enabled` 覆盖配置；predict 未指定时读取 checkpoint。具体训练/预测/恢复命令见 [README](README.md#caption-缓存与修复)。

CE 训练只在验证生成时使用缓存；A SCST 另在两条策略生成时使用 GRU 缓存。Hybrid 原有生成已逐 token 保留 hidden，Qwen 的生成机制独立，均不受这两个 Caption 开关控制。缓存不是 CE teacher forcing 显存开关，也不保证总显存下降。

run.json、checkpoint metadata 记录 `caption_generation`、`effective_caption_cache`；恢复另记录原开关。预测写 `输出文件.protocol.json`（配置、输入及 checkpoint 哈希、环境）。epochs.jsonl 记录 `validation_seconds`、`cuda_train_peak_bytes`、`cuda_validation_peak_bytes`、`cuda_peak_bytes`。CPU 显存值为 null；CUDA 记录 allocated 峰值，不等于驱动总占用。验证生成及独立 predict 使用 FP32，SCST 策略在训练 loss 的 autocast 范围内执行。

## 已实际验证

Windows/Python 3.12.14、Torch 2.9.1+cpu、Torchvision 0.24.1+cpu；完整依赖版本同 `requirements-lock-test-cpu.txt`。所有输入为明确标记的合成图像/文本或真实随机张量，没有下载预训练权重。

新增 `tests/test_caption_cache.py`：17 项运行，16 项通过，1 项因 CUDA 不可用跳过。覆盖 Dropout/梯度、默认 ResNet 无权重 memory 形状、旧实现权重/优化器/调度器往返和 eval logits、GRU/KV 每步 logits、真实单 token 工作量及 K/V 复用、三模型多 batch/长度/混合 EOS/截断、跨请求隔离、旧配置/检查点及 CLI 恢复切换。A SCST 用真实 CIDEr-D 奖励与张量反传比较两模式奖励、loss 和各参数梯度。只有 EOS 排程等控制条件使用测试 hook，核心缓存/attention/GRU/梯度不使用 mock。

已有回归与 CLI 结果见 [实现记录](IMPLEMENTATION_STATUS.md#caption-修复与可选缓存验收2026-11-24)。新成员从工程目录运行，Windows 将 python 替换为 `.venv/Scripts/python.exe`，Linux 为 `.venv/bin/python`：

```bash
python -m unittest discover -s tests -p test_caption_cache.py -v
python -m unittest discover -s tests -p test_engineering.py -v
python -m unittest discover -s tests -p test_baseline.py -v
python -m scripts.synthetic_acceptance --out artifacts/my-cache-acceptance
```

最后两条涉及的边界：baseline 有两项需要 `../data/annotations/train.jsonl`；synthetic_acceptance 独立生成专用 fixture，不补造真实数据。目录必须是新的。

## 延时基准与 GPU 接入

本次串行运行三模型、CPU FP32、2 threads、随机 tiny hidden=32/layers=2/FF=128、batch=4、RGB=224、max_tokens=64。明确压低随机模型的 EOS bias，以测满长度，不反映真实描述长度。预热 3 次、交替顺序各测 12 次；包含视觉编码，每次检查两模式 token 完全一致。

| 模型 | 关缓存中位延时 ms | 开缓存中位延时 ms | 中位加速比 | CUDA 显存 |
| --- | ---: | ---: | ---: | --- |
| A | 61.02 | 10.67 | 5.72 | 未测/null |
| B | 95.09 | 43.38 | 2.19 | 未测/null |
| B-global | 88.84 | 41.82 | 2.12 | 未测/null |

原始每次延时、p95、环境及源码哈希在 `artifacts/caption-cache-20261124/benchmark-serial-{A,B,B-global}.json`，仅本地产物，不是复现输入。初次与测试并发的 benchmark 文件仅为历史调试，本表只使用串行结果。

无数据、无权重的可复现命令（每种 family 分别执行）：

```bash
python -m scripts.benchmark_caption_cache --family A --device cpu --precision fp32 --batch-size 4 --max-tokens 64 --image-size 224 --threads 2 --warmup 3 --repeats 12 --out artifacts/my-cache-benchmark/A.json
python -m scripts.benchmark_caption_cache --family B --device cpu --precision fp32 --batch-size 4 --max-tokens 64 --image-size 224 --threads 2 --warmup 3 --repeats 12 --out artifacts/my-cache-benchmark/B.json
python -m scripts.benchmark_caption_cache --family B-global --device cpu --precision fp32 --batch-size 4 --max-tokens 64 --image-size 224 --threads 2 --warmup 3 --repeats 12 --out artifacts/my-cache-benchmark/B-global.json
```

目标 CUDA 机器和可信 checkpoint 到位后执行，例如：

```bash
python -m unittest discover -s tests -p test_caption_cache.py -v
python -m scripts.benchmark_caption_cache --checkpoint artifacts/caption-b/best.pt --device cuda --precision fp32 --batch-size 4 --max-tokens 64 --warmup 3 --repeats 12 --out artifacts/gpu-cache-benchmark/B-fp32.json
python -m scripts.benchmark_caption_cache --checkpoint artifacts/caption-b/best.pt --device cuda --precision bf16 --batch-size 4 --max-tokens 64 --warmup 3 --repeats 12 --out artifacts/gpu-cache-benchmark/B-bf16.json
```

脚本每轮同步 CUDA、分别清零峰值，记录峰值 allocated bytes 及相对生成前 baseline 的增量；一旦 token 不一致即拒绝出具性能报告。checkpoint 模式保留原权重/EOS，仍使用合成 RGB 张量。真实图像与正式权重的输出、混合精度误差、真实 batch/长度延时、Linux/CUDA/峰值显存均待实测；不能把上表速度外推到它们。还需按 README 的 predict 命令在真实 RGB manifest 上对比开关结果。
