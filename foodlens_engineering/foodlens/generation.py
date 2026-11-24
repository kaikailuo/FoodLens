"""Caption cache switches and normalization of pre-cache configuration/checkpoints."""
from dataclasses import asdict, dataclass


@dataclass
class CaptionCacheConfig:
    gru_cache_enabled: bool = False
    transformer_kv_cache_enabled: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("caption cache switches must be boolean")


def cache_options(value=None):
    if isinstance(value, CaptionCacheConfig):
        return asdict(value)
    if value is not None and not isinstance(value, dict):
        raise ValueError("generation must be an object of caption cache switches")
    return asdict(CaptionCacheConfig(**(value or {})))


def caption_generation(max_tokens=64, options=None):
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("caption max_tokens must be a positive integer")
    return {"max_tokens": max_tokens, "strategy": "greedy", **cache_options(options)}


def checkpoint_generation(metadata):
    """Read current metadata or the older shared nutrition generation record."""
    record = metadata.get("caption_generation")
    if record is not None:
        record = dict(record)
        maximum = record.pop("max_tokens", 64)
        if record.pop("strategy", "greedy") != "greedy":
            raise ValueError("caption checkpoint must use greedy decoding")
        return caption_generation(maximum, record)
    maximum = metadata.get("generation_config", {}).get("max_tokens", 64)
    options = metadata.get("train_config", {}).get("generation", {})
    return caption_generation(maximum, options)


def effective_cache_options(model_family, options=None):
    requested = cache_options(options)
    return {"gru_cache_enabled": model_family == "A" and requested["gru_cache_enabled"],
            "transformer_kv_cache_enabled": model_family in {"B", "B-global"} and requested["transformer_kv_cache_enabled"]}


def override_cache_options(options, args):
    result = cache_options(options)
    for field in result:
        value = getattr(args, field, None)
        if value is not None:
            result[field] = value
    return cache_options(result)


def add_cache_arguments(parser):
    import argparse
    parser.add_argument("--gru-cache-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Caption A generation only; override config/checkpoint (default: disabled)")
    parser.add_argument("--transformer-kv-cache-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Caption B/B-global generation only; override config/checkpoint (default: disabled)")
