from dataclasses import dataclass
import math
import torch
from .schema import TOTAL_FIELDS, assemble
from .text import detokenize


@dataclass
class GenerationConfig:
    max_tokens: int = 128
    max_items: int = 32

    def __post_init__(self):
        if self.max_tokens < 1 or self.max_items < 1:
            raise ValueError("generation limits must be positive")


@torch.inference_mode()
def predict_hybrid(model, images, vocab, config=None):
    """Only RGB tensors and fixed decoding configuration enter prediction."""
    config = config or GenerationConfig()
    model.eval()
    _, vectors, conditions = model.visual(images)
    totals = model.total_head(torch.cat((vectors, conditions), -1))
    results = []
    sep = vocab.index["<SEP>"]
    for i in range(images.shape[0]):
        z = conditions[i:i+1]
        hidden, previous = z.unsqueeze(0), 1
        name_tokens, items, failures, raw_ids = [], [], [], []
        closures, ended = 0, False
        for _ in range(config.max_tokens):
            logits, states, hidden = model.decode(torch.tensor([[previous]], device=images.device), z, hidden)
            scores = logits[0, 0].clone()
            scores[0] = scores[1] = -torch.inf
            if not name_tokens:
                scores[sep] = -torch.inf
            if name_tokens or closures == 0:
                scores[2] = -torch.inf
            token = int(scores.argmax())
            raw_ids.append(token)
            if token == 2:
                ended = True
                break
            if token == sep:
                # Exactly the state that produced SEP logits, never next-step state.
                weight = float(model.weight_head(torch.cat((states[:, 0], z), -1)).item())
                name = detokenize([vocab.tokens[j] for j in name_tokens])
                if 3 in name_tokens or not name or not math.isfinite(weight):
                    failures.append({"reason": "unk_empty_or_nonfinite", "raw_name": name,
                                     "weight_g": weight if math.isfinite(weight) else None})
                else:
                    items.append({"name": name, "weight_g": weight})
                name_tokens = []
                closures += 1
                if closures >= config.max_items:
                    break
            else:
                name_tokens.append(token)
            previous = token
        numbers = totals[i].tolist()
        bad_totals = [key for key, value in zip(TOTAL_FIELDS, numbers) if not math.isfinite(value) or value < 0]
        record = assemble(items, [value if math.isfinite(value) and value >= 0 else 0.0 for value in numbers])
        for key in bad_totals:
            record[key] = None  # JSON-readable Schema failure; evaluator uses zero for this field.
        results.append({"raw_token_ids": raw_ids, "record": record, "invalid_total_fields": bad_totals,
                        "failures": failures, "truncated": not ended,
                        "discarded_unclosed_tokens": name_tokens})
    return results


@torch.inference_mode()
def predict_caption(model, images, *, max_tokens=64):
    model.eval()
    sequences = torch.ones(images.shape[0], 1, dtype=torch.long, device=images.device)
    finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
    if hasattr(model, "memory"):
        memory = model.memory(images)
        step = lambda prefix: model.decode(prefix, memory)
    else:
        _, _, z = model.visual(images)
        step = lambda prefix: model.decode(prefix, z)[0]
    for _ in range(max_tokens):
        logits = step(sequences)[:, -1].clone()
        logits[:, :2] = -torch.inf
        token = logits.argmax(-1).masked_fill(finished, 0)
        sequences = torch.cat((sequences, token[:, None]), 1)
        finished |= token.eq(2)
        if finished.all():
            break
    return sequences
