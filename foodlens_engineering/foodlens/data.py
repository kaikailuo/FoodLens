"""Supervision stays outside the RGB-only model prediction interface."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .schema import TOTAL_FIELDS, canonical_items, validate_record


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_image_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
        raise ValueError("image must be a nonempty path relative to data root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes data root")
    return path


def image_transform(train=False):
    steps = [transforms.Resize(256), transforms.CenterCrop(224)]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class NutritionDataset(Dataset):
    def __init__(self, annotation, data_root, *, split, limit=None, train_transform=False):
        self.rows = read_jsonl(annotation)
        if any(not isinstance(row.get("dish_id"), str) or not row["dish_id"].strip() for row in self.rows):
            raise ValueError("nutrition IDs must be nonempty strings")
        if len({row["dish_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate nutrition IDs")
        self.root = Path(data_root)
        if any(row.get("split") != split for row in self.rows):
            raise ValueError("annotation split mismatch")
        if train_transform and split != "train":
            raise ValueError("augmentation allowed only for training")
        self.rows = self.rows[:limit] if limit else self.rows
        self.transform = image_transform(train_transform)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if validate_record(row["target"]):
            raise ValueError(f"invalid target: {row['dish_id']}")
        with Image.open(safe_image_path(self.root, row["image"])) as image:
            rgb = self.transform(image.convert("RGB"))
        # Deliberately exclude target_text and metadata from returned batch.
        return {"id": row["dish_id"], "image": rgb, "target": row["target"]}


class RGBManifestDataset(Dataset):
    """Unlabeled inference manifest: exactly id/image, with no answer-bearing fields."""
    def __init__(self, manifest, data_root, *, limit=None):
        self.rows = read_jsonl(manifest)
        if any(set(row) != {"id", "image"} for row in self.rows):
            raise ValueError("RGB inference manifest must contain only id and image")
        if any(not isinstance(row["id"], str) or not row["id"].strip() for row in self.rows):
            raise ValueError("RGB manifest IDs must be nonempty strings")
        if len({row["id"] for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate RGB manifest IDs")
        self.rows = self.rows[:limit] if limit else self.rows
        self.root, self.transform = data_root, image_transform(False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_image_path(self.root, row["image"])) as image:
            return {"id": row["id"], "image": self.transform(image.convert("RGB"))}


def serialize_items(record, vocab):
    items = canonical_items(record["items"])
    sequence, boundaries, weights = [1], [], []
    sep = vocab.index["<SEP>"]
    for item in items:
        sequence.extend(vocab.encode(item["name"]))
        sequence.append(sep)
        # Output time index: inputs=sequence[:-1], targets=sequence[1:].
        boundaries.append(len(sequence) - 2)
        weights.append(item["weight_g"])
    sequence.append(2)
    return sequence, boundaries, weights


def collate_nutrition(samples, vocab):
    serialized = [serialize_items(sample["target"], vocab) for sample in samples]
    length = max(len(row[0]) - 1 for row in serialized)
    count = max(len(row[1]) for row in serialized)
    inputs = torch.zeros(len(samples), length, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    boundaries = torch.zeros(len(samples), count, dtype=torch.long)
    weights = torch.zeros(len(samples), count)
    item_mask = torch.zeros(len(samples), count, dtype=torch.bool)
    for i, (sequence, positions, values) in enumerate(serialized):
        inputs[i, :len(sequence)-1] = torch.tensor(sequence[:-1])
        targets[i, :len(sequence)-1] = torch.tensor(sequence[1:])
        boundaries[i, :len(positions)] = torch.tensor(positions, dtype=torch.long)
        weights[i, :len(values)] = torch.tensor(values)
        item_mask[i, :len(values)] = True
    return {"ids": [s["id"] for s in samples], "images": torch.stack([s["image"] for s in samples]),
            "inputs": inputs, "targets": targets, "boundaries": boundaries,
            "weights": weights, "item_mask": item_mask,
            "totals": torch.tensor([[s["target"][key] for key in TOTAL_FIELDS] for s in samples])}


@dataclass
class TrainStatistics:
    weight_mean: float
    weight_std: float
    total_mean: list
    total_std: list

    @classmethod
    def fit(cls, rows):
        if not rows or any(row.get("split") != "train" for row in rows):
            raise ValueError("statistics require nonempty training rows only")
        if any(validate_record(row["target"]) for row in rows):
            raise ValueError("statistics require valid training targets")
        weights = torch.tensor([item["weight_g"] for row in rows
                                for item in canonical_items(row["target"]["items"])])
        totals = torch.tensor([[row["target"][key] for key in TOTAL_FIELDS] for row in rows])
        return cls(float(weights.mean()) if weights.numel() else 0.0,
                   float(weights.std(unbiased=False).clamp_min(1e-6)) if weights.numel() else 1.0,
                   totals.mean(0).tolist(), totals.std(0, unbiased=False).clamp_min(1e-6).tolist())


class CaptionManifestDataset(Dataset):
    """Requires an explicit verified mapping; never joins parquet and captions by row."""
    def __init__(self, manifest, data_root, *, split):
        all_rows = read_jsonl(manifest)
        audit_caption_rows(all_rows)
        self.rows = [row for row in all_rows if row["split"] == split]
        self.root = data_root
        self.transform = image_transform(split == "train")
        ids = set()
        for row in self.rows:
            if not {"id", "image", "caption", "official_filename", "sha256", "duplicate_group", "split"} <= row.keys():
                raise ValueError("manifest lacks mapping provenance")
            if row["id"] in ids:
                raise ValueError("duplicate sample ID")
            ids.add(row["id"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = safe_image_path(self.root, row["image"])
        if sha256(path) != row["sha256"]:
            raise ValueError("manifest image hash mismatch")
        with Image.open(path) as image:
            return {"id": row["id"], "image": self.transform(image.convert("RGB")), "caption": row["caption"]}


def validate_manifest_splits(rows):
    groups = {}
    for row in rows:
        for key in ("sha256", "duplicate_group"):
            identifier = (key, row[key])
            previous = groups.setdefault(identifier, row["split"])
            if previous != row["split"]:
                raise ValueError(f"cross-split duplicate: {identifier}")


def audit_caption_rows(rows):
    if not rows:
        raise ValueError("empty caption manifest")
    required = {"id", "image", "caption", "official_filename", "sha256", "duplicate_group", "split"}
    for row in rows:
        if not required <= row.keys() or row["split"] not in {"train", "val", "test"}:
            raise ValueError("caption manifest lacks provenance or valid split")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
            raise ValueError("caption provenance fields must be nonempty strings")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("invalid image SHA256")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate caption IDs across splits")
    validate_manifest_splits(rows)
    return {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}


def collate_caption(samples, vocab, max_tokens=64):
    if not samples or max_tokens < 2:
        raise ValueError("nonempty caption batch and max_tokens>=2 required")
    encoded = [vocab.encode(sample["caption"]) for sample in samples]
    sequences = [[1] + ids[:max_tokens-1] + [2] for ids in encoded]
    length = max(len(seq)-1 for seq in sequences)
    inputs = torch.zeros(len(samples), length, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    for i, seq in enumerate(sequences):
        inputs[i, :len(seq)-1] = torch.tensor(seq[:-1])
        targets[i, :len(seq)-1] = torch.tensor(seq[1:])
    return {"ids": [s["id"] for s in samples], "images": torch.stack([s["image"] for s in samples]),
            "inputs": inputs, "targets": targets, "original_lengths": [len(x)+1 for x in encoded],
            "truncated": [len(x)+1 > max_tokens for x in encoded],
            "references": [[s["caption"]] for s in samples]}


def collate_hybrid(samples, vocab, max_tokens=128):
    batch = collate_nutrition(samples, vocab)
    if batch["targets"].shape[1] > max_tokens:
        raise ValueError("hybrid supervision exceeds max_tokens; increase limit after length audit")
    return batch


def length_audit(texts, vocab, max_tokens):
    encoded = [vocab.encode(text) for text in texts]
    lengths = [len(ids)+1 for ids in encoded]
    return {"samples": len(lengths), "max_length": max(lengths, default=0),
            "over_limit": sum(n > max_tokens for n in lengths),
            "over_limit_rate": sum(n > max_tokens for n in lengths)/len(lengths) if lengths else None,
            "max_tokens": max_tokens, "includes_eos": True,
            "mean_length": sum(lengths)/len(lengths) if lengths else None,
            "unknown_tokens": sum(ids.count(3) for ids in encoded),
            "content_tokens": sum(len(ids) for ids in encoded)}
