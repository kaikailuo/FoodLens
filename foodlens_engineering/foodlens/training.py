"""Caption/Hybrid/SCST training, validation selection and epoch-boundary resume."""
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from itertools import islice
import json
import math
from pathlib import Path
import time
import warnings
import torch
from torch.utils.data import DataLoader
from .caption_metrics import CiderDReward, score_record_texts, score_captions, check_metric_dependencies
from .checkpoint import load_checkpoint, save_checkpoint, seed_everything
from .data import (CaptionManifestDataset, NutritionDataset, TrainStatistics, collate_caption,
                   collate_hybrid, length_audit, sha256, serialize_items)
from .decode import GenerationConfig, predict_caption, predict_hybrid
from .losses import token_loss, hybrid_loss
from .metrics import evaluate_records
from .models import CaptionA, CaptionB, Hybrid, LegacyHybrid, ModelConfig, transfer_visual_backbone
from .ingredients import IngredientVocabulary
from .runtime import append_log, autocast, environment, reject_secrets, runtime, to_device, write_json, write_jsonl
from .scst import captions_from_tokens, scst_loss
from .text import Vocabulary
from .generation import cache_options, caption_generation, effective_cache_options


@dataclass
class TrainConfig:
    backend: str = "native"
    task: str = "caption"
    family: str = "A"
    model: dict = field(default_factory=lambda: asdict(ModelConfig()))
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"
    deterministic: bool = True
    threads: int = 2
    epochs: int = 30
    batch_size: int = 16
    accumulation: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    layer4_lr_multiplier: float = 0.1
    unfreeze_layer4_epoch: int | None = None
    gradient_clip: float = 1.0
    scheduler_step: int = 5
    scheduler_gamma: float = 0.5
    patience: int = 5
    tolerance: float = 1e-6
    monitor: str = "CIDEr-D"
    max_tokens: int = 64
    max_items: int = 32
    loss_weights: list = field(default_factory=lambda: [1., 1., 1.])
    metrics: list = field(default_factory=lambda: ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"])
    method: str = "ce"
    scst_weight: float = 0.5
    transfer: str = "visual"
    engineering: bool = False
    updates_per_epoch: int | None = None
    generation: dict = field(default_factory=cache_options)

    def __post_init__(self):
        reject_secrets(asdict(self))
        self.generation = cache_options(self.generation)
        if any(type(getattr(self, name)) is not bool for name in ("engineering", "deterministic")):
            raise ValueError("engineering/deterministic must be boolean")
        if self.backend not in {"native", "qwen"}:
            raise ValueError("unknown training backend")
        if self.task not in {"caption", "hybrid"} or self.family not in {"A", "B", "B-global", "hybrid"}:
            raise ValueError("unknown task/family")
        if (self.task == "hybrid") != (self.family == "hybrid"):
            raise ValueError("hybrid task requires hybrid family")
        for name in ("threads", "epochs", "batch_size", "accumulation", "scheduler_step", "patience", "max_tokens", "max_items"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens < 2 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("max_tokens>=2 and nonnegative integer seed required")
        for name in ("lr", "gradient_clip", "layer4_lr_multiplier", "scheduler_gamma"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        for name in ("tolerance", "weight_decay"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        if self.method not in {"ce", "ce_continue", "scst"} or not 0 <= self.scst_weight <= 1:
            raise ValueError("invalid CE/SCST method or mixture")
        if self.method != "ce" and (self.task != "caption" or self.family != "A"):
            raise ValueError("CE continuation/SCST protocol currently targets Caption A")
        if self.transfer not in {"visual", "visual_gru"}:
            raise ValueError("unknown transfer scope")
        if self.updates_per_epoch is not None and (type(self.updates_per_epoch) is not int or self.updates_per_epoch < 1):
            raise ValueError("updates_per_epoch must be positive")
        if self.unfreeze_layer4_epoch is not None and (type(self.unfreeze_layer4_epoch) is not int or self.unfreeze_layer4_epoch < 0 or self.model.get("backbone", "resnet18") != "resnet18"):
            raise ValueError("layer4 epoch requires ResNet18 and nonnegative epoch")
        m = ModelConfig(**self.model)
        if self.backend == "native" and self.task == "hybrid":
            if self.transfer != "visual":
                raise ValueError("Hybrid V2 supports visual backbone transfer only")
            if not self.engineering and (m.hidden, m.layers, m.heads, m.feedforward, m.dropout) != (256, 2, 4, 1024, 0.1):
                raise ValueError("formal Hybrid V2 requires 256/2/4/1024/0.1 decoder")
        if type(m.pretrained) is not bool or type(m.frozen_backbone) is not bool:
            raise ValueError("model pretrained/frozen_backbone must be boolean")
        if m.backbone not in {"resnet18", "tiny"} or any(type(x) is not int or x < 1 for x in (m.hidden, m.embedding, m.heads, m.layers, m.feedforward)):
            raise ValueError("invalid model dimensions/backbone")
        if m.hidden % 4 or m.hidden % m.heads or not 0 <= m.dropout < 1:
            raise ValueError("invalid position dimension/heads/dropout")
        if len(self.loss_weights) != 3 or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in self.loss_weights) or not any(self.loss_weights):
            raise ValueError("invalid loss weights")
        if set(self.metrics) - {"BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"}:
            raise ValueError("unknown metric")
        expected = "CIDEr-D" if self.task == "caption" else "hybrid"
        if self.monitor != expected and not (self.engineering and self.monitor == "val_loss"):
            raise ValueError(f"formal selection requires {expected}; val_loss requires explicit engineering=true")
        if not self.engineering and set(self.metrics) != {"BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"}:
            raise ValueError("formal runs require all four caption metrics")
        if self.monitor == "CIDEr-D" and "CIDEr-D" not in self.metrics:
            raise ValueError("CIDEr-D selection cannot omit CIDEr-D")
        if self.backend == "native" and not self.engineering and (m.backbone != "resnet18" or not m.pretrained):
            raise ValueError("formal runs require explicit pretrained ResNet18; random/tiny requires engineering=true")


def load_train_config(path):
    return TrainConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def resume_compatible_config(value):
    """Only epochs and equivalent generation execution switches may change on resume."""
    value = dict(value)
    cache_options(value.pop("generation", {}))  # validate old/new values before ignoring runtime-only switches
    value.pop("epochs", None)
    return value


def build_model(family, vocab_size, config, *, loading=False):
    original = ModelConfig(**config) if isinstance(config, dict) else config
    creation = replace(original, pretrained=False) if loading else original
    if family == "A":
        model = CaptionA(vocab_size, creation)
    elif family in {"B", "B-global"}:
        model = CaptionB(vocab_size, creation, global_only=family == "B-global")
    elif family == "hybrid":
        model = Hybrid(vocab_size, creation)
    else:
        raise ValueError("unknown model family")
    # Loading never downloads the original pretrained initialization again.
    model.config = original
    return model


def restore_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    family = checkpoint["metadata"].get("model_family", "hybrid")
    if family == "hybrid" and checkpoint.get("architecture_version") == "hybrid_v2":
        vocab = IngredientVocabulary.restore(checkpoint["ingredient_vocabulary"])
        if vocab.tokens != checkpoint["vocab"]:
            raise ValueError("checkpoint vocabulary mismatch")
        model = build_model(family, len(vocab), checkpoint["model_config"], loading=True).to(device)
    elif family == "hybrid":
        if checkpoint.get("architecture_version") is not None:
            raise ValueError("unsupported Hybrid architecture version")
        vocab = Vocabulary(checkpoint["vocab"])
        original = ModelConfig(**checkpoint["model_config"])
        model = LegacyHybrid(len(vocab), replace(original, pretrained=False)).to(device)
        model.config = original
    else:
        vocab = Vocabulary(checkpoint["vocab"])
        model = build_model(family, len(vocab), checkpoint["model_config"], loading=True).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, vocab, checkpoint


def parameter_groups(model, config):
    backbone, decoder = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("visual.features.") else decoder).append(parameter)
    return [{"params": decoder, "lr": config.lr, "name": "decoder"},
            {"params": backbone, "lr": config.lr * config.layer4_lr_multiplier, "name": "backbone"}]


def is_better(candidate, best, monitor, tolerance):
    if best is None:
        return True
    if monitor == "hybrid":
        directions = (1, -1, -1, 1) if len(candidate) == 4 else (1, -1, 1)  # existing Qwen protocol
        for new, old, direction in zip(candidate, best, directions, strict=True):
            if abs(new-old) > tolerance:
                return direction*(new-old) > 0
        return False
    return candidate > best + tolerance if monitor == "CIDEr-D" else candidate < best - tolerance


def _loader(dataset, collator, config, epoch=0, train=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=train, num_workers=0,
                      collate_fn=collator, generator=torch.Generator().manual_seed(config.seed + epoch))


def train_epoch(model, loader, optimizer, scaler, device, config, statistics, *, reward=None, step=0, log_path=None):
    model.train()
    iterator, updates = iter(loader), 0
    while window := list(islice(iterator, config.accumulation)):
        optimizer.zero_grad(set_to_none=True)
        token_count = sum(int(b["targets"].ne(0).sum()) for b in window)
        item_count = sum(int(b["item_mask"].sum()) for b in window) if config.task == "hybrid" else 0
        sample_count = sum(len(b["ids"]) for b in window)
        logged, reward_rows = {}, []
        for cpu_batch in window:
            batch = to_device(cpu_batch, device)
            with autocast(device, config.precision):
                if config.task == "hybrid":
                    parts = hybrid_loss(model(batch["images"], batch["inputs"], batch["boundaries"]), batch, statistics, config.loss_weights)
                    loss = (config.loss_weights[0]*parts["sequence"]*int(batch["targets"].ne(0).sum())/max(token_count, 1)
                          + config.loss_weights[1]*parts["weight"]*int(batch["item_mask"].sum())/max(item_count, 1)
                          + config.loss_weights[2]*parts["total"]*len(batch["ids"])/sample_count)
                else:
                    ce = token_loss(model(batch["images"], batch["inputs"]), batch["targets"])
                    loss = ce * int(batch["targets"].ne(0).sum()) / max(token_count, 1)
                    parts = {"ce": ce}
                    if config.method == "scst":
                        # Both policies use eval mode (dropout off), with gradients enabled only for sampling.
                        model.eval()
                        try:
                            rl, diagnostics = scst_loss(model, batch["images"], batch["ids"], reward, vocab=reward.vocab,
                                max_tokens=config.max_tokens, gru_cache_enabled=config.generation["gru_cache_enabled"])
                        finally:
                            model.train()
                        loss = (1-config.scst_weight)*loss + config.scst_weight*rl*len(batch["ids"])/sample_count
                        parts["scst"] = rl
                        reward_rows.extend(diagnostics)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss at update {step+1}")
            scaler.scale(loss).backward()
            for key, value in parts.items():
                logged[key] = logged.get(key, 0.) + float(value.detach()) / len(window)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        step, updates = step+1, updates+1
        if log_path:
            append_log(log_path, {"update": step, "loss_parts_batch_mean": logged, "gradient_norm": float(norm), "rewards": reward_rows})
        if config.updates_per_epoch and updates >= config.updates_per_epoch:
            break
    return step


@torch.inference_mode()
def validate(model, dataset, collator, config, vocab, statistics, device):
    model.eval()
    numerator = denominator = 0.
    predictions, truth = {}, {}
    for cpu_batch in _loader(dataset, collator, config):
        batch = to_device(cpu_batch, device)
        with autocast(device, config.precision):
            if config.task == "caption":
                loss = token_loss(model(batch["images"], batch["inputs"]), batch["targets"])
                count = int(batch["targets"].ne(0).sum())
            else:
                loss = hybrid_loss(model(batch["images"], batch["inputs"], batch["boundaries"]), batch, statistics, config.loss_weights)["loss"]
                count = len(batch["ids"])
        numerator += float(loss)*count
        denominator += count
        if config.task == "caption":
            generated = predict_caption(model, batch["images"], max_tokens=config.max_tokens, **config.generation)
            for sample_id, tokens in zip(batch["ids"], generated.tolist(), strict=True):
                predictions[sample_id] = caption_prediction(tokens, vocab)
            truth.update(dict(zip(batch["ids"], batch["references"], strict=True)))
        else:
            generated = predict_hybrid(model, batch["images"], vocab, GenerationConfig(config.max_tokens, config.max_items))
            predictions.update(dict(zip(batch["ids"], generated, strict=True)))
    if denominator == 0 or not math.isfinite(numerator):
        raise ValueError("empty or nonfinite validation")
    result = {"val_loss": numerator/denominator}
    if config.task == "caption":
        if config.metrics:
            result["text"] = score_captions(truth, {i: p["caption"] for i, p in predictions.items()}, metrics=config.metrics)
    else:
        truth_records = {r["dish_id"]: r["target"] for r in dataset.rows}
        result["structured"] = evaluate_records(truth_records, predictions)
        if config.metrics:
            result["text"] = score_record_texts(truth_records, predictions, metrics=config.metrics)
    return result, predictions


def caption_prediction(tokens, vocab):
    generated = tokens[1:]  # BOS is never part of delivered text.
    ended = 2 in generated
    if ended:
        generated = generated[:generated.index(2)]
    return {"caption": captions_from_tokens(torch.tensor([generated]), vocab)[0],
            "raw_token_ids": tokens, "eos": ended, "truncated": not ended,
            "unk_count": generated.count(3), "status": "ok"}


def selection_value(result, config, statistics):
    if config.monitor == "val_loss":
        return result["val_loss"]
    if config.monitor == "CIDEr-D":
        return result["text"]["scores"]["CIDEr-D"]
    metrics = result["structured"]
    from .schema import TOTAL_FIELDS
    normalized = sum(metrics["totals"][key]["mae"] / max(scale, 1e-6)
                     for key, scale in zip(TOTAL_FIELDS, statistics.total_std, strict=True))/5
    return [metrics["name_micro_f1"], metrics["weight_union"]["mae"] or 0., normalized, metrics["schema_valid_rate"]]


def run_training(config, *, data_root, out, manifest=None, train_annotations=None, val_annotations=None,
                 resume=None, initialize=None, df_cache=None):
    if config.backend != "native":
        raise ValueError("use qwen-train for the Qwen backend")
    check_metric_dependencies(config.metrics)
    if resume and initialize:
        raise ValueError("resume and initialize are mutually exclusive")
    if config.method != "ce" and not (resume or initialize):
        raise ValueError("SCST/CE continuation requires a selected CE checkpoint")
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    device, scaler = runtime(config.device, config.precision, config.deterministic)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "last.pt").exists() and not resume:
        raise ValueError("output already has last.pt; choose a new run directory or --resume")
    if config.task == "caption":
        if manifest is None:
            raise ValueError("Caption training requires --manifest")
        train = CaptionManifestDataset(manifest, data_root, split="train")
        val = CaptionManifestDataset(manifest, data_root, split="val")
        hashes = {"manifest": sha256(manifest)}
    else:
        if train_annotations is None or val_annotations is None:
            raise ValueError("Hybrid requires --train-annotations and --val-annotations")
        train = NutritionDataset(train_annotations, data_root, split="train", train_transform=True)
        val = NutritionDataset(val_annotations, data_root, split="val")
        if {r["dish_id"] for r in train.rows} & {r["dish_id"] for r in val.rows}:
            raise ValueError("nutrition train/validation IDs overlap")
        hashes = {"train": sha256(train_annotations), "val": sha256(val_annotations)}
    if not len(train) or not len(val):
        raise ValueError("nonempty train and validation required")
    source = source_vocab = None
    if initialize:
        source, source_vocab, initial = restore_model(initialize)
        if initial["metadata"].get("model_family") != "A":
            raise ValueError("initialization requires Caption A checkpoint")
        if not config.engineering and initial["metadata"].get("checkpoint_role") != "best":
            raise ValueError("formal transfer/continuation requires the validation-selected best CE checkpoint")
        if config.method != "ce" or config.task == "hybrid":
            source_method = initial["metadata"].get("train_config", {}).get("method")
            if source_method not in {"ce", "ce_continue"}:
                raise ValueError("SCST/CE control must start from a CE checkpoint")
            if config.task == "caption" and initial["metadata"].get("input_hashes", {}).get("manifest") != hashes["manifest"]:
                raise ValueError("CE initialization caption manifest changed")
        hashes["initial_checkpoint"] = sha256(initialize)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved:
        if config.task == "hybrid" and saved.get("architecture_version") != "hybrid_v2":
            raise ValueError("legacy Hybrid cannot resume as V2; use Caption A visual initialization or a new run")
        if saved.get("format_version") != 2 or not saved.get("progress"):
            raise ValueError("training resume requires format v2 epoch checkpoint")
        original_config = resume_compatible_config(saved["metadata"]["train_config"])
        requested_config = resume_compatible_config(asdict(config))
        if original_config != requested_config:
            raise ValueError("resume configuration differs (only total epochs and generation cache switches may change)")
        # Initialization hash persists on resume even without the original file.
        current_data = {k: v for k, v in hashes.items() if k != "initial_checkpoint"}
        saved_data = {k: v for k, v in saved["metadata"]["input_hashes"].items() if k != "initial_checkpoint"}
        if current_data != saved_data:
            raise ValueError("resume data hash mismatch")
        hashes = saved["metadata"]["input_hashes"]
        vocab = IngredientVocabulary.restore(saved["ingredient_vocabulary"]) if config.task == "hybrid" else Vocabulary(saved["vocab"])
        if config.task == "hybrid" and vocab.metadata() != IngredientVocabulary.fit(train.rows).metadata():
            raise ValueError("resume ingredient mapping differs from Train")
    elif config.task == "hybrid":
        vocab = IngredientVocabulary.fit(train.rows)
    else:
        vocab = Vocabulary(source_vocab.tokens if source_vocab is not None else None)
        if source is None:
            vocab.extend_training([r["caption"] for r in train.rows], split="train")
    statistics = TrainStatistics.fit(train.rows) if config.task == "hybrid" else None
    model = build_model(config.family, len(vocab), config.model, loading=bool(saved or source))
    transfer = saved["metadata"].get("transfer") if saved else None
    if source is not None:
        if config.task == "hybrid":
            transfer = transfer_visual_backbone(source, model)
        else:
            if source.config != model.config or source_vocab.tokens != vocab.tokens:
                raise ValueError("CE/SCST initialization architecture/vocab mismatch")
            model.load_state_dict(source.state_dict())
    if statistics:
        model.initialize_regression(statistics)
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, config), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_step, config.scheduler_gamma)
    progress = {"epoch": 0, "step": 0, "best": None, "bad_epochs": 0}
    if saved:
        load_checkpoint(resume, model, optimizer, scheduler, restore_rng=True, scaler=scaler)
        progress = saved["progress"]
    collator = partial(collate_caption if config.task == "caption" else collate_hybrid, vocab=vocab, max_tokens=config.max_tokens)
    if config.task == "caption":
        audit = {split: length_audit([r["caption"] for r in dataset.rows], vocab, config.max_tokens) for split, dataset in (("train", train), ("val", val))}
    else:
        audit = {split: {"max_length": max(len(serialize_items(r["target"], vocab)[0])-1 for r in dataset.rows),
                         "samples": len(dataset)} for split, dataset in (("train", train), ("val", val))}
        if any(row["max_length"] > config.max_tokens for row in audit.values()):
            raise ValueError("Hybrid target exceeds max_tokens; no silent truncation")
    reward = None
    if config.method == "scst":
        if not df_cache:
            raise ValueError("SCST requires --df-cache built from this training manifest")
        reward = CiderDReward(df_cache, {r["id"]: [r["caption"]] for r in train.rows}, manifest_sha256=hashes["manifest"])
        reward.vocab = vocab
        if saved and saved["metadata"].get("df_cache_sha256") != sha256(df_cache):
            raise ValueError("SCST resume reward cache changed")
    legacy_dropout_resume = bool(saved and isinstance(model, CaptionA) and model.config.dropout > 0
        and saved["metadata"].get("caption_a_dropout_version", 1) < 2)
    if legacy_dropout_resume:
        warnings.warn("Restoring a pre-fix A/Hybrid checkpoint: configured Dropout now takes effect; "
                      "weights/optimizer load normally, but the historical training trajectory is not reproduced.", UserWarning)
    metadata = {"model_family": config.family, "train_config": asdict(config), "input_hashes": hashes,
                "transfer": transfer, "length_audit": audit, "environment": environment(),
                "resume_granularity": "completed epoch, num_workers=0, same device/software; no mid-epoch continuation",
                "df_cache_sha256": sha256(df_cache) if df_cache else None,
                "generation_config": asdict(GenerationConfig(config.max_tokens, config.max_items)),
                "caption_generation": caption_generation(config.max_tokens, config.generation),
                "effective_caption_cache": effective_cache_options(config.family, config.generation),
                "caption_a_dropout_version": 2 if isinstance(model, CaptionA) else None,
                "legacy_dropout_resume": legacy_dropout_resume,
                "resume_generation_options": cache_options(saved["metadata"]["train_config"].get("generation", {})) if saved else None,
                "architecture_version": getattr(model, "architecture_version", None),
                "ingredient_vocabulary": vocab.metadata() if isinstance(vocab, IngredientVocabulary) else None,
                "selection_order": ["name_micro_f1:max", "weight_union.mae_g:min", "normalized_total_mae:min", "schema_valid_rate:max"] if config.task == "hybrid" else None}
    write_json(destination / "run.json", {**metadata, "vocab": vocab.tokens, "statistics": asdict(statistics) if statistics else None})
    if progress["bad_epochs"] >= config.patience:
        raise ValueError("checkpoint already early-stopped; initialize a new run for continuation")
    for epoch in range(progress["epoch"], config.epochs):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if config.unfreeze_layer4_epoch is not None and epoch >= config.unfreeze_layer4_epoch:
            model.visual.unfreeze_layer4()
        try:
            step = train_epoch(model, _loader(train, collator, config, epoch, True), optimizer, scaler, device,
                               config, statistics, reward=reward, step=progress["step"], log_path=destination / "train.jsonl")
            train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            validation_start = time.perf_counter()
            result, predictions = validate(model, val, collator, config, vocab, statistics, device)
            validation_seconds = time.perf_counter()-validation_start
            validation_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        except Exception as error:
            # Exception class only: external service messages/configuration must not leak credentials.
            append_log(destination / "events.jsonl", {"epoch": epoch, "status": "failed", "error_type": type(error).__name__})
            raise
        selected = selection_value(result, config, statistics)
        improved = is_better(selected, progress["best"], config.monitor, config.tolerance)
        progress = {"epoch": epoch+1, "step": step, "best": selected if improved else progress["best"],
                    "bad_epochs": 0 if improved else progress["bad_epochs"]+1}
        scheduler.step()
        write_json(destination / "validation.json", result)
        write_jsonl(destination / "validation_predictions.jsonl", ({"id": i, **p} for i, p in predictions.items()))
        append_log(destination / "epochs.jsonl", {"epoch": epoch+1, "update": step, "selection": selected,
            "improved": improved, "val_loss": result["val_loss"], "seconds": time.perf_counter()-start,
            "validation_seconds": validation_seconds,
            "effective_caption_cache": effective_cache_options(config.family, config.generation),
            "cuda_train_peak_bytes": train_peak, "cuda_validation_peak_bytes": validation_peak,
            "cuda_peak_bytes": max(train_peak, validation_peak) if device.type == "cuda" else None})
        for filename in (["best.pt", "last.pt"] if improved else ["last.pt"]):
            save_checkpoint(destination / filename, model, optimizer, scheduler, vocab=vocab, statistics=statistics,
                            metadata={**metadata, "checkpoint_role": filename.removesuffix(".pt")}, step=step, scaler=scaler, progress=progress)
        if progress["bad_epochs"] >= config.patience:
            break
    return progress
