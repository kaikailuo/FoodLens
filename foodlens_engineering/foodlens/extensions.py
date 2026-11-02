"""Reserved integration boundaries. These are deliberately not mock implementations."""
from typing import Protocol

FIXED_NUTRITION_INSTRUCTION = (
    "Describe this RGB meal as one JSON object with items (name, weight_g), total_weight_g, "
    "total_calories_kcal, protein_g, fat_g, carbohydrate_g. All numeric values must be finite "
    "and nonnegative; units are grams except calories in kcal. Return no extra fields.")


def qwen_messages(rgb_image, *, training_target=None):
    messages = [{"role": "user", "content": [{"type": "image", "image": rgb_image},
                  {"type": "text", "text": FIXED_NUTRITION_INSTRUCTION}]}]
    if training_target is not None:
        from .schema import validate_record
        import json
        if validate_record(training_target):
            raise ValueError("invalid supervised Qwen target")
        messages.append({"role": "assistant", "content": [{"type": "text", "text":
            json.dumps(training_target, ensure_ascii=False, separators=(",", ":"))}]})
    return messages


class CaptionMetricBackend(Protocol):
    """Pin tokenizer/version/scaling; CIDEr DF must be training-reference-only."""
    def score(self, references: dict, predictions: dict) -> dict: ...


class SCSTReward(Protocol):
    def __call__(self, sample_ids: list[str], captions: list[str]) -> list[float]: ...


class FrozenJudge(Protocol):
    """Input never includes model name, references, targets, or metadata."""
    def score(self, rgb_images, captions: list[str]) -> list[dict]: ...
