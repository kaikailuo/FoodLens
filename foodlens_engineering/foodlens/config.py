"""Validated baseline configuration shared by smoke and future GPU trainers."""
from dataclasses import asdict
import json
from pathlib import Path
from .models import ModelConfig
from .decode import GenerationConfig


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seed", "model", "optimizer", "loss_weights", "caption_generation", "nutrition_generation"}
    if not isinstance(value, dict) or not required <= value.keys() or set(value) - required - {"note"}:
        raise ValueError("invalid baseline config fields")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    model = ModelConfig(**value["model"])
    if model.backbone not in {"resnet18", "tiny"}:
        raise ValueError("unknown visual backbone")
    for field in ("hidden", "embedding", "heads", "layers", "feedforward"):
        if type(getattr(model, field)) is not int or getattr(model, field) < 1:
            raise ValueError(f"model {field} must be a positive integer")
    if model.hidden % model.heads or model.hidden % 4 or not 0 <= model.dropout < 1:
        raise ValueError("incompatible model heads/position dimension/dropout")
    if type(model.pretrained) is not bool or type(model.frozen_backbone) is not bool:
        raise ValueError("backbone switches must be boolean")
    optimizer = value["optimizer"]
    if set(optimizer) != {"name", "lr", "layer4_lr_multiplier", "gradient_clip"} or optimizer["name"] != "AdamW":
        raise ValueError("invalid optimizer fields")
    import math
    for field in ("lr", "layer4_lr_multiplier", "gradient_clip"):
        if type(optimizer[field]) not in (int, float) or not math.isfinite(optimizer[field]) or optimizer[field] <= 0:
            raise ValueError(f"invalid optimizer {field}")
    weights = value["loss_weights"]
    if not isinstance(weights, list) or len(weights) != 3 or any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in weights) or not any(weights):
        raise ValueError("three finite nonnegative loss weights required")
    generation = GenerationConfig(**value["nutrition_generation"])
    if type(generation.max_tokens) is not int or type(generation.max_items) is not int:
        raise ValueError("generation lengths must be integers")
    caption = value["caption_generation"]
    if set(caption) != {"max_tokens", "strategy"} or caption["strategy"] != "greedy" or type(caption["max_tokens"]) is not int or caption["max_tokens"] < 1:
        raise ValueError("invalid caption generation")
    value["model"] = asdict(model)
    value["nutrition_generation"] = asdict(generation)
    return value
