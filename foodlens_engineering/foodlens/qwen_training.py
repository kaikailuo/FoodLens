"""Single-device Qwen LoRA training with complete epoch state and paired inference."""
from dataclasses import asdict
from itertools import islice
import json
import math
import os
from pathlib import Path
import time
import uuid
import torch
from torch.utils.data import DataLoader
from .caption_metrics import record_text, score_captions, check_metric_dependencies
from .checkpoint import random_state, restore_random_state, seed_everything
from .data import sha256, TrainStatistics
from .metrics import evaluate_records
from .qwen import (QwenConfig, QwenCollator, QwenDataset, generate_qwen, inject_lora,
                   load_qwen, parse_qwen_prediction, template_hash, processor_signature)
from .runtime import append_log, autocast, environment, runtime, to_device, write_json, write_jsonl
from .schema import TOTAL_FIELDS
from .training import TrainConfig, is_better


def load_qwen_config(path):
    return QwenConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def resolve_adapter(path):
    path = Path(path)
    if path.suffix == ".json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent.resolve()
        checkpoint = (root / pointer["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError("adapter pointer escapes run directory")
        for filename, field in (("training_state.pt", "state_sha256"), ("adapter_model.safetensors", "adapter_sha256")):
            if sha256(checkpoint / filename) != pointer[field]:
                raise ValueError("adapter checkpoint is incomplete or changed")
        path = checkpoint
    return path


def load_adapter(model, adapter, config, processor, *, trainable=False):
    from peft import PeftModel
    adapter = resolve_adapter(adapter)
    protocol = json.loads((Path(adapter) / "protocol.json").read_text(encoding="utf-8"))
    if protocol["qwen_config"] != asdict(config) or protocol["processor_signature"] != processor_signature(processor):
        raise ValueError("adapter/base/processor/generation protocol mismatch")
    return PeftModel.from_pretrained(model, adapter, is_trainable=trainable)


def qwen_predict_file(config, manifest, data_root, out, adapter=None):
    seed_everything(config.seed)
    model, processor = load_qwen(config)
    if adapter:
        model = load_adapter(model, adapter, config, processor)
    dataset = QwenDataset(manifest, data_root, split=None, inference=True)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Streaming preserves already-produced failures/results if a later item fails.
    with destination.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(dataset.rows):
            try:
                sample = dataset[index]
                prediction = parse_qwen_prediction(generate_qwen(model, processor, [sample["image"]], config)[0])
            except (OSError, RuntimeError, ValueError) as error:
                prediction = {"raw": "", "status": "inference_failed", "error_type": type(error).__name__, "truncated": False}
            stream.write(json.dumps({"id": row["id"], **prediction}, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
    write_json(str(destination)+".protocol.json", {"qwen_config": asdict(config), "chat_template_sha256": template_hash(processor),
        "processor_signature": processor_signature(processor),
        "manifest_sha256": sha256(manifest), "adapter": bool(adapter), "environment": environment()})


def run_qwen_training(qconfig, train_config, *, train_annotations, val_annotations, data_root, out, resume=None):
    check_metric_dependencies(train_config.metrics)
    if train_config.backend != "qwen" or train_config.task != "hybrid" or train_config.method != "ce":
        raise ValueError("Qwen trainer requires task=hybrid, method=ce selection protocol")
    if train_config.seed != qconfig.seed or train_config.device != qconfig.device or train_config.precision != qconfig.precision:
        raise ValueError("Qwen and training seed/device/precision must agree")
    seed_everything(qconfig.seed)
    torch.set_num_threads(train_config.threads)
    device, scaler = runtime(qconfig.device, qconfig.precision, train_config.deterministic)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "last.json").exists() and not resume:
        raise ValueError("Qwen output exists; use a new directory or --resume")
    train = QwenDataset(train_annotations, data_root, split="train")
    val = QwenDataset(val_annotations, data_root, split="val")
    if not len(train) or not len(val) or {r["dish_id"] for r in train.rows} & {r["dish_id"] for r in val.rows}:
        raise ValueError("nonempty disjoint Qwen train/validation sets required")
    hashes = {"train": sha256(train_annotations), "val": sha256(val_annotations)}
    statistics = TrainStatistics.fit(train.rows)
    model, processor = load_qwen(qconfig)
    processor.tokenizer.padding_side = "right"
    if resume:
        resume = resolve_adapter(resume)
        model = load_adapter(model, resume, qconfig, processor, trainable=True)
        if qconfig.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.config.use_cache = False
        scope = json.loads((Path(resume) / "scope.json").read_text(encoding="utf-8"))
    else:
        model, scope = inject_lora(model, qconfig)
    protocol = {"qwen_config": asdict(qconfig), "chat_template_sha256": template_hash(processor),
                "processor_signature": processor_signature(processor)}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_config.scheduler_step, train_config.scheduler_gamma)
    progress = {"epoch": 0, "step": 0, "best": None, "bad_epochs": 0}
    if resume:
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        previous, requested = dict(state["train_config"]), asdict(train_config)
        previous.pop("epochs"); requested.pop("epochs")
        if previous != requested or state["input_hashes"] != hashes or state["protocol"] != protocol:
            raise ValueError("Qwen resume configuration/data mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_random_state(state["rng"])
        progress = state["progress"]
    if progress["bad_epochs"] >= train_config.patience:
        raise ValueError("Qwen checkpoint already early-stopped")
    collator = QwenCollator(processor, qconfig.max_length)
    write_json(destination / "run.json", {**protocol, "train_config": asdict(train_config), "input_hashes": hashes,
        "scope": scope, "environment": environment(), "statistics": asdict(statistics),
        "resume_granularity": "complete epoch; single device; num_workers=0; unchanged software/hardware"})
    def loader(dataset, epoch=0, train=False):
        return DataLoader(dataset, batch_size=train_config.batch_size, collate_fn=collator,
            shuffle=train, num_workers=0, generator=torch.Generator().manual_seed(qconfig.seed+epoch))
    for epoch in range(progress["epoch"], train_config.epochs):
        start = time.perf_counter()
        model.train()
        iterator, updates = iter(loader(train, epoch, True)), 0
        step = progress["step"]
        while window := list(islice(iterator, train_config.accumulation)):
            optimizer.zero_grad(set_to_none=True)
            count = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in window)
            if count == 0:
                raise ValueError("Qwen batch has no supervised assistant targets")
            total_loss = 0.
            for cpu_batch in window:
                batch = to_device(cpu_batch, device)
                with autocast(device, qconfig.precision):
                    output = model(**batch)
                    loss = output.loss * int(batch["labels"][:, 1:].ne(-100).sum())/count
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite Qwen loss")
                scaler.scale(loss).backward()
                total_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], train_config.gradient_clip, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            step, updates = step+1, updates+1
            append_log(destination / "train.jsonl", {"update": step, "assistant_loss": total_loss, "gradient_norm": float(norm)})
            if train_config.updates_per_epoch and updates >= train_config.updates_per_epoch:
                break
        model.eval()
        numerator = denominator = 0
        with torch.inference_mode():
            for cpu_batch in loader(val):
                batch = to_device(cpu_batch, device)
                with autocast(device, qconfig.precision):
                    loss = model(**batch).loss
                count = int(batch["labels"][:, 1:].ne(-100).sum())
                numerator += float(loss)*count
                denominator += count
        if denominator == 0 or not math.isfinite(numerator):
            raise RuntimeError("invalid Qwen validation loss")
        predictions = {}
        for index, row in enumerate(val.rows):
            try:
                sample = val[index]
                # Target is deliberately not passed to the generation interface.
                prediction = parse_qwen_prediction(generate_qwen(model, processor, [sample["image"]], qconfig)[0])
            except (OSError, RuntimeError, ValueError) as error:
                prediction = {"raw": "", "status": "inference_failed", "error_type": type(error).__name__}
            predictions[row["dish_id"]] = prediction
        truth = {r["dish_id"]: r["target"] for r in val.rows}
        metrics = evaluate_records(truth, predictions)
        result = {"val_loss": numerator/denominator, "structured": metrics}
        if train_config.metrics:
            from .schema import strict_loads
            def prediction_text(prediction):
                try:
                    return record_text(strict_loads(prediction["raw"]))
                except (ValueError, TypeError):
                    return ""
            result["text"] = score_captions({i: [record_text(r)] for i, r in truth.items()},
                {i: prediction_text(p) for i, p in predictions.items()}, metrics=train_config.metrics)
        selected = result["val_loss"] if train_config.monitor == "val_loss" else [metrics["name_micro_f1"],
            sum(metrics["totals"][key]["mae"]/max(scale, 1e-6) for key, scale in zip(TOTAL_FIELDS, statistics.total_std))/5,
            metrics["schema_valid_rate"]]
        improved = is_better(selected, progress["best"], train_config.monitor, train_config.tolerance)
        progress = {"epoch": epoch+1, "step": step, "best": selected if improved else progress["best"],
                    "bad_epochs": 0 if improved else progress["bad_epochs"]+1}
        scheduler.step()
        write_json(destination / "validation.json", result)
        write_jsonl(destination / "validation_predictions.jsonl", ({"id": i, **p} for i, p in predictions.items()))
        append_log(destination / "epochs.jsonl", {"epoch": epoch+1, "update": step, "selection": selected,
            "improved": improved, "seconds": time.perf_counter()-start,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        # Complete an immutable epoch bundle before atomically updating the pointers.
        folder = destination / f"epoch-{epoch+1:04d}-{uuid.uuid4().hex[:8]}"
        folder.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(folder, safe_serialization=True)
        processor.save_pretrained(folder / "processor")
        write_json(folder / "protocol.json", protocol)
        write_json(folder / "scope.json", scope)
        temporary = folder / "training_state.pt.tmp"
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": random_state(), "progress": progress, "train_config": asdict(train_config),
            "input_hashes": hashes, "protocol": protocol}, temporary)
        os.replace(temporary, folder / "training_state.pt")
        pointer = {"checkpoint": folder.name, "state_sha256": sha256(folder/"training_state.pt"),
                   "adapter_sha256": sha256(folder/"adapter_model.safetensors")}
        for name in (["best.json", "last.json"] if improved else ["last.json"]):
            write_json(destination / name, pointer)
        if progress["bad_epochs"] >= train_config.patience:
            break
    return progress
