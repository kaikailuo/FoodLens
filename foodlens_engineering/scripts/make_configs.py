"""Regenerate reviewed method examples; no data/model access."""
from dataclasses import asdict, replace
from pathlib import Path
import json
from foodlens.models import ModelConfig
from foodlens.training import TrainConfig
from foodlens.qwen import QwenConfig


def write_json(path, value):
    # Portable LF config examples, independent of the Windows text newline setting.
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))

folder = Path(__file__).resolve().parents[1] / "configs"
base = TrainConfig(model=asdict(ModelConfig(pretrained=True)), device="cuda", precision="bf16")
for family, filename in (("A", "train_caption_a.json"), ("B", "train_caption_b.json"), ("B-global", "train_caption_b_global.json")):
    write_json(folder / filename, asdict(replace(base, family=family)))
    generation = {"gru_cache_enabled": family == "A", "transformer_kv_cache_enabled": family != "A"}
    write_json(folder / filename.replace(".json", "_cache.json"), asdict(replace(base, family=family, generation=generation)))
write_json(folder / "train_hybrid.json", asdict(replace(base, task="hybrid", family="hybrid", monitor="hybrid", max_tokens=128)))
for method in ("scst", "ce_continue"):
    write_json(folder / ("train_"+method+".json"), asdict(replace(base, method=method, lr=1e-5, epochs=10, patience=11, updates_per_epoch=100)))
write_json(folder / "qwen.json", asdict(QwenConfig()))
write_json(folder / "train_qwen.json", asdict(replace(base, backend="qwen", task="hybrid", family="hybrid", monitor="hybrid",
    lr=2e-4, batch_size=1, accumulation=16, epochs=3, max_tokens=128)))
write_json(folder / "judge_local.json", {"backend": "local", "qwen_config": "configs/qwen.json"})
write_json(folder / "judge_api.json", {"backend": "api", "service_revision": "REPLACE_WITH_PROVIDER_PINNED_MODEL_VERSION", "timeout": 120})
write_json(folder / "engineering_cpu.json", asdict(TrainConfig(engineering=True, monitor="val_loss", metrics=[],
    model=asdict(ModelConfig(backbone="tiny", hidden=16, embedding=16, layers=1, feedforward=32, dropout=.1)),
    epochs=2, batch_size=2, max_tokens=12)))
