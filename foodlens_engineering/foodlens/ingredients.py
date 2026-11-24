"""Train-only whole-name ingredient classes, independent of Caption vocabulary."""
import hashlib
import json
import math
from .schema import canonical_items, normalize_name, validate_record
from .text import SPECIAL

NORMALIZATION_VERSION = "nfkc-lower-whitespace-v1"


class IngredientVocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        names = self.tokens[4:]
        if (self.tokens[:4] != list(SPECIAL) or names != sorted(set(names))
                or any(not n or normalize_name(n) != n or n in SPECIAL for n in names)):
            raise ValueError("invalid stable ingredient class mapping")
        self.index = {name: i for i, name in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def fit(cls, rows):
        if not rows or any(r.get("split") != "train" or validate_record(r["target"]) for r in rows):
            raise ValueError("ingredient mapping requires valid training rows only")
        return cls(list(SPECIAL) + sorted({i["name"] for r in rows for i in canonical_items(r["target"]["items"])}))

    def metadata(self):
        payload = {"kind": "ingredient_classes", "normalization_version": NORMALIZATION_VERSION,
                   "tokens": self.tokens}
        payload["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return payload

    @classmethod
    def restore(cls, payload):
        vocab = cls(payload["tokens"])
        if vocab.metadata() != payload:
            raise ValueError("ingredient mapping/version/hash mismatch")
        return vocab


def ingredient_items(items):
    grouped = {}
    for item in items:
        grouped.setdefault(normalize_name(item["name"]), []).append(float(item["weight_g"]))
    return [{"name": n, "weight_g": math.fsum(sorted(values))} for n, values in sorted(grouped.items())]


def serialize_ingredients(record, vocab):
    # Merge first, then sort paired IDs/weights. OOV stays an input UNK only;
    # its CE is ignored and raw truth remains untouched for entity evaluation.
    pairs = sorted(((vocab.index.get(i["name"], 3), i["name"], i["weight_g"])
                    for i in ingredient_items(record["items"])))
    return [1] + [p[0] for p in pairs] + [2], list(range(1, len(pairs)+1)), [p[2] for p in pairs]
