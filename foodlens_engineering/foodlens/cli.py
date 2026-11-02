"""CPU smoke loop over real training images; not a model-quality experiment."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import torch
from .checkpoint import load_checkpoint, save_checkpoint, seed_everything
from .config import load_config
from .data import NutritionDataset, RGBManifestDataset, TrainStatistics, collate_nutrition, read_jsonl, sha256
from .decode import GenerationConfig, predict_hybrid
from .losses import hybrid_loss
from .metrics import evaluate_records
from .models import Hybrid, ModelConfig
from .text import Vocabulary, detokenize, tokenize


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def smoke(args):
    torch.set_num_threads(args.threads)
    baseline = load_config(args.config)
    seed = args.seed if args.seed is not None else baseline["seed"]
    seed_everything(seed)
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root).resolve()
    train_path = root / "annotations" / "train.jsonl"
    train_rows = read_jsonl(train_path)
    vocab = Vocabulary()
    names = [item["name"] for row in train_rows for item in row["target"]["items"]]
    vocab.extend_training(names, split="train")
    vocab.add_sep()
    statistics = TrainStatistics.fit(train_rows)
    smoke_overrides = {"backbone": "tiny", "pretrained": False, "hidden": 32, "embedding": 32,
                       "heads": 4, "layers": 2, "feedforward": 128, "dropout": 0.0, "frozen_backbone": True}
    config = replace(ModelConfig(**baseline["model"]), **smoke_overrides)
    dataset = NutritionDataset(train_path, root, split="train", limit=args.samples)
    batch = collate_nutrition([dataset[i] for i in range(len(dataset))], vocab)
    model = Hybrid(len(vocab), config)
    model.initialize_regression(statistics)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=baseline["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    losses = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["images"], batch["inputs"], batch["boundaries"])
        parts = hybrid_loss(output, batch, statistics, baseline["loss_weights"])
        parts["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), baseline["optimizer"]["gradient_clip"])
        if not torch.isfinite(gradient):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        scheduler.step()
        losses.append({key: float(value.detach()) for key, value in parts.items()} | {"gradient_norm": float(gradient)})
    generation = GenerationConfig(max_tokens=32, max_items=4)
    metadata = {"purpose": "CPU engineering smoke only", "seed": seed,
                "annotation_sha256": sha256(train_path), "data_root": str(root),
                "sample_ids": batch["ids"], "split": "train", "model_family": "hybrid",
                "generation_config": asdict(generation), "baseline_config": baseline,
                "smoke_model_overrides": smoke_overrides,
                "config_sha256": sha256(args.config)}
    checkpoint_path = destination / "smoke.pt"
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, vocab=vocab,
                    statistics=statistics, metadata=metadata, step=args.steps)
    restored = Hybrid(len(vocab), config)
    load_checkpoint(checkpoint_path, restored)
    model.eval()
    restored.eval()
    with torch.no_grad():
        first = model(batch["images"], batch["inputs"], batch["boundaries"])
        second = restored(batch["images"], batch["inputs"], batch["boundaries"])
    assert all(torch.equal(first[key], second[key]) for key in first)
    predictions = predict_hybrid(restored, batch["images"], vocab, generation)
    indexed = dict(zip(batch["ids"], predictions, strict=True))
    with (destination / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for sample_id, prediction in indexed.items():
            stream.write(json.dumps({"id": sample_id, **prediction}, ensure_ascii=False, allow_nan=False) + "\n")
    truth = {dataset.rows[i]["dish_id"]: dataset.rows[i]["target"] for i in range(len(dataset))}
    metrics = evaluate_records(truth, indexed)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "run.json", {"metadata": metadata, "model_config": asdict(config),
        "statistics": asdict(statistics), "vocab": vocab.tokens, "losses": losses,
        "checkpoint_roundtrip_exact": True, "torch": str(torch.__version__), "python": platform.python_version(),
        "train_name_token_roundtrip_failures": sorted({n for n in names if detokenize(tokenize(n)) != n.lower()})})
    print(json.dumps({"out": str(destination), "steps": args.steps, "samples": len(dataset),
                      "last_loss": losses[-1]["loss"], "checkpoint_roundtrip_exact": True}))


def predict(args):
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary(checkpoint["vocab"])
    model = Hybrid(len(vocab), ModelConfig(**checkpoint["model_config"]))
    load_checkpoint(args.checkpoint, model)
    dataset = RGBManifestDataset(args.manifest, args.data_root, limit=args.samples)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for i in range(len(dataset)):
            sample_id = dataset.rows[i]["id"]
            try:
                sample = dataset[i]
                generation = GenerationConfig(**checkpoint["metadata"].get("generation_config", {}))
                prediction = predict_hybrid(model, sample["image"].unsqueeze(0), vocab, generation)[0]
            except (OSError, ValueError, RuntimeError) as error:
                prediction = {"raw": "", "inference_error": f"{type(error).__name__}: {error}"}
            stream.write(json.dumps({"id": sample_id, **prediction}, allow_nan=False) + "\n")
    print(str(destination.resolve()))


def evaluate(args):
    annotations = read_jsonl(args.annotations)
    truth = {row["dish_id"]: row["target"] for row in annotations}
    if len(truth) != len(annotations):
        raise ValueError("duplicate ground truth IDs")
    rows = read_jsonl(args.predictions)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate prediction IDs")
    result = evaluate_records(truth, {row["id"]: row for row in rows})
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("smoke")
    run.add_argument("--data-root", default="../data")
    run.add_argument("--out", default="artifacts/smoke")
    run.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/baseline.json"))
    run.add_argument("--steps", type=int, default=3)
    run.add_argument("--samples", type=int, default=4)
    run.add_argument("--seed", type=int)
    run.add_argument("--threads", type=int, default=2)
    run.set_defaults(function=smoke)
    infer = commands.add_parser("predict")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--data-root", default="../data")
    infer.add_argument("--manifest", required=True, help="JSONL containing only id/image")
    infer.add_argument("--out", required=True)
    infer.add_argument("--samples", type=int, default=4)
    infer.add_argument("--threads", type=int, default=2)
    infer.set_defaults(function=predict)
    score = commands.add_parser("evaluate")
    score.add_argument("--annotations", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--out", required=True)
    score.set_defaults(function=evaluate)
    args = parser.parse_args()
    if hasattr(args, "steps") and args.steps < 1:
        parser.error("steps must be positive")
    if hasattr(args, "samples") and args.samples < 1:
        parser.error("samples must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
