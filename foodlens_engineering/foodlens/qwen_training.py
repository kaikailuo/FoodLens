"""HF Trainer infrastructure and FoodLens structured validation selection."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .caption_metrics import score_record_texts, check_metric_dependencies
from .checkpoint import seed_everything
from .data import sha256, TrainStatistics
from .metrics import evaluate_records
from .qwen import (QwenConfig, QwenCollator, QwenDataset, generate_qwen, inject_lora,
                   load_qwen, parse_qwen_prediction, template_hash, processor_signature, load_qwen_config, QwenTrainConfig, training_config, load_qwen_train_config)
from .runtime import environment, runtime, write_json, write_jsonl
from .schema import TOTAL_FIELDS
from .training import is_better


def resolve_adapter(path):
    path = Path(path)
    if path.suffix == ".json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent.resolve()
        path = (root / pointer["checkpoint"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("adapter pointer escapes run directory")
        if "state_sha256" in pointer:
            raise ValueError("legacy custom-training checkpoint: inference requires its epoch directory; Trainer resume unsupported")
    if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
        raise ValueError("incomplete adapter checkpoint")
    return path


def load_adapter(model, adapter, config, processor, *, trainable=False):
    from peft import PeftModel
    adapter = resolve_adapter(adapter)
    protocol = json.loads((adapter / "protocol.json").read_text(encoding="utf-8"))
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


# These public HF extension points are the only training customizations.
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import ExportableState


def assistant_token_loss(outputs, labels, num_items_in_batch=None):
    """Sum shifted assistant CE / tokens in the entire accumulation window.

    Trainer supplies num_items_in_batch and handles accumulation scaling when a
    compute_loss_func is supplied. First-position labels must be masked, so its
    unshifted count equals the shifted count. Do not divide again by GA steps.
    """
    if not bool(labels[:, 0].eq(-100).all()):
        raise ValueError("first Qwen label must be ignored")
    targets = labels[:, 1:].contiguous()
    count = targets.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    if int(count) <= 0:
        raise ValueError("Qwen batch has no supervised assistant targets")
    loss = torch.nn.functional.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.size(-1)),
        targets.reshape(-1), ignore_index=-100, reduction="sum") / count
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite Qwen assistant loss")
    return loss


class StructuredSelection(TrainerCallback, ExportableState):
    """Native scalar EarlyStoppingCallback cannot express tolerance tie-breaks."""
    def __init__(self, patience=5, tolerance=1e-6, monitor="hybrid"):
        self.patience, self.tolerance, self.monitor = patience, tolerance, monitor
        self.best, self.bad_epochs, self.best_step = None, 0, None

    def state(self):
        return {"args": {"patience": self.patience, "tolerance": self.tolerance, "monitor": self.monitor},
                "attributes": {"best": self.best, "bad_epochs": self.bad_epochs, "best_step": self.best_step}}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        candidate = metrics["eval_loss"] if self.monitor == "val_loss" else [
            metrics["eval_name_micro_f1"], metrics["eval_normalized_mae"], metrics["eval_schema_valid_rate"]]
        if is_better(candidate, self.best, self.monitor, self.tolerance):
            self.best, self.bad_epochs, self.best_step = candidate, 0, state.global_step
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        if self.bad_epochs >= self.patience:
            raise ValueError("Qwen checkpoint already early-stopped")


class GenerationTrainer(Trainer):
    """Public evaluate override: full validation generation, no test labels."""
    def __init__(self, *args, qconfig, statistics, text_metrics, **kwargs):
        super().__init__(*args, **kwargs)
        self.qconfig, self.statistics, self.text_metrics = qconfig, statistics, text_metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        was_training = self.model.training
        predictions = {}
        self.model.eval()
        try:
            # Only scalar loss is retained; no corpus-sized logits allocation.
            with torch.inference_mode(), self.compute_loss_context_manager():
                numerator = denominator = 0
                for batch in self.get_eval_dataloader(dataset):
                    labels = batch["labels"]
                    count = int(labels[:, 1:].ne(-100).sum())
                    loss = assistant_token_loss(self.model(**{k: v for k, v in batch.items() if k != "labels"}), labels)
                    numerator += float(loss) * count
                    denominator += count
                for index, row in enumerate(dataset.rows):
                    try:
                        prediction = parse_qwen_prediction(generate_qwen(self.model, self.processing_class,
                            [dataset[index]["image"]], self.qconfig)[0])
                    except (OSError, RuntimeError, ValueError) as error:
                        prediction = {"raw": "", "status": "inference_failed", "error_type": type(error).__name__, "truncated": False}
                    predictions[row["dish_id"]] = prediction
            truth = {r["dish_id"]: r["target"] for r in dataset.rows}
            structured = evaluate_records(truth, predictions)
            result = {"val_loss": numerator / denominator, "structured": structured}
            if self.text_metrics:
                result["text"] = score_record_texts(truth, predictions, metrics=self.text_metrics)
            normalized = sum(structured["totals"][key]["mae"] / max(scale, 1e-6)
                for key, scale in zip(TOTAL_FIELDS, self.statistics.total_std)) / 5
            metrics = {f"{metric_key_prefix}_loss": result["val_loss"],
                f"{metric_key_prefix}_name_micro_f1": structured["name_micro_f1"],
                f"{metric_key_prefix}_normalized_mae": normalized,
                f"{metric_key_prefix}_schema_valid_rate": structured["schema_valid_rate"]}
            folder = Path(self.args.output_dir) / f"validation-{self.state.global_step}"
            write_json(folder / "metrics.json", result)
            write_jsonl(folder / "predictions.jsonl", ({"id": i, **p} for i, p in predictions.items()))
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        finally:
            self.model.train(was_training)


class ProtocolCheckpoint(TrainerCallback):
    def __init__(self, trainer, metadata):
        self.trainer, self.metadata = trainer, metadata

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        # Trainer clips gradients, but its default clipping does not raise on
        # nonfinite gradients. Preserve the old trainer's fail-fast safeguard.
        checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if checks and not bool(torch.stack(checks).all()):
            raise RuntimeError("nonfinite Qwen gradient before optimizer step")

    def on_save(self, args, state, control, **kwargs):
        root = Path(args.output_dir)
        folder = root / f"checkpoint-{state.global_step}"
        write_json(folder / "protocol.json", self.metadata["protocol"])
        write_json(folder / "run.json", self.metadata)
        selection = next(c for c in self.trainer.callback_handler.callbacks if isinstance(c, StructuredSelection))
        # Small compatibility aliases only. Trainer owns all training/callback state.
        write_json(root / "last.json", {"checkpoint": folder.name})
        write_json(root / "best.json", {"checkpoint": f"checkpoint-{selection.best_step}"})


def run_qwen_training(qconfig, train_config, *, train_annotations, val_annotations, data_root, out, resume=None):
    config = training_config(train_config)
    check_metric_dependencies(config.metrics)
    if (config.seed, config.device, config.precision) != (qconfig.seed, qconfig.device, qconfig.precision):
        raise ValueError("Qwen and training seed/device/precision must agree")
    seed_everything(config.seed)
    torch.set_num_threads(config.threads)
    device, _ = runtime(config.device, config.precision, config.deterministic)
    destination = Path(out)
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError("Qwen output exists; use a new directory or --resume")
    train = QwenDataset(train_annotations, data_root, split="train")
    val = QwenDataset(val_annotations, data_root, split="val")
    if not len(train) or not len(val) or {r["dish_id"] for r in train.rows} & {r["dish_id"] for r in val.rows}:
        raise ValueError("nonempty disjoint Qwen train/validation sets required")
    hashes = {"train": sha256(train_annotations), "val": sha256(val_annotations)}
    statistics = TrainStatistics.fit(train.rows)
    model, processor = load_qwen(qconfig)
    processor.tokenizer.padding_side = "right"
    # Identical fresh/resume setup, including NF4 k-bit preparation. Trainer loads
    # adapter, optimizer, scheduler and RNG exactly once via resume_from_checkpoint.
    model, scope = inject_lora(model, qconfig)
    protocol = {"qwen_config": asdict(qconfig), "chat_template_sha256": template_hash(processor),
                "processor_signature": processor_signature(processor)}
    metadata = {"format": "foodlens.hf_trainer.v1", "protocol": protocol, "train_config": asdict(config),
        "input_hashes": hashes, "scope": scope, "statistics": asdict(statistics), "environment": environment(),
        "loss": "sum shifted assistant CE / accumulation-window supervised tokens",
        "scheduler": "Trainer constant; replaces legacy epoch StepLR explicitly",
        "selection": "validation F1 -> train-std-normalized five-field MAE -> Schema, tolerance at each stage"}
    if resume:
        resume = resolve_adapter(resume)
        previous = json.loads((resume / "run.json").read_text(encoding="utf-8"))
        old, new = dict(previous["train_config"]), asdict(config)
        old.pop("epochs"); new.pop("epochs")
        if old != new or previous["input_hashes"] != hashes or previous["protocol"] != protocol or previous["format"] != metadata["format"]:
            raise ValueError("Qwen resume configuration/data/protocol mismatch")
        replay_keys = ("python", "platform", "torch", "torchvision", "numpy", "pillow",
                       "transformers", "peft", "accelerate", "python_source_sha256", "cuda_available", "gpu")
        if any(previous["environment"].get(key) != metadata["environment"].get(key) for key in replay_keys):
            # Check executable sources/runtime, not unrelated documentation/Git edits.
            raise ValueError("Qwen resume environment/source mismatch")
        if resume.parent.resolve() != destination.resolve():
            raise ValueError("resume within original run directory to preserve selected checkpoints")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("FoodLens Qwen training is single-process only")
    args = TrainingArguments(output_dir=str(destination), use_cpu=device.type == "cpu",
        num_train_epochs=config.epochs, per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size, gradient_accumulation_steps=config.accumulation,
        learning_rate=config.lr, weight_decay=config.weight_decay, max_grad_norm=config.gradient_clip,
        lr_scheduler_type="constant", optim="adamw_torch", bf16=config.precision == "bf16", fp16=config.precision == "fp16",
        gradient_checkpointing=qconfig.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=1,
        load_best_model_at_end=False, restore_callback_states_from_checkpoint=True,
        seed=config.seed, data_seed=config.seed, full_determinism=config.deterministic,
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
        dataloader_pin_memory=device.type == "cuda", report_to=[], disable_tqdm=True, save_safetensors=True)
    if (args.n_gpu > 1 or args.device.type != device.type or
            (device.index is not None and args.device.index != device.index)):
        raise ValueError("select one matching GPU with CUDA_VISIBLE_DEVICES; multi-GPU is unsupported")
    trainer = GenerationTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
        data_collator=QwenCollator(processor, qconfig.max_length), processing_class=processor,
        compute_loss_func=assistant_token_loss, callbacks=[StructuredSelection(config.patience, config.tolerance, config.monitor)],
        qconfig=qconfig, statistics=statistics, text_metrics=config.metrics)
    trainer.add_callback(ProtocolCheckpoint(trainer, metadata))
    write_json(destination / "run.json", metadata)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    selection = next(c for c in trainer.callback_handler.callbacks if isinstance(c, StructuredSelection))
    return {"epoch": trainer.state.epoch, "step": trainer.state.global_step,
            "best": selection.best, "bad_epochs": selection.bad_epochs}
