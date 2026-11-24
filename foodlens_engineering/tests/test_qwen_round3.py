"""Round-three protocol and CPU Trainer regression tests, no weights download."""
import copy
from dataclasses import replace
import inspect
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from foodlens.caption_metrics import record_text, score_captions, score_record_texts
from foodlens.schema import canonical_record_text, TOTAL_FIELDS, strict_loads
from foodlens.metrics import evaluate_records
from foodlens.qwen import QwenConfig, QwenCollator, QwenDataset, inject_lora, model_inputs, generate_qwen
HAS_QWEN = all(importlib.util.find_spec(p) for p in ("transformers", "peft", "accelerate"))
if HAS_QWEN:
    from foodlens.qwen_training import (assistant_token_loss, StructuredSelection, QwenTrainConfig,
        run_qwen_training, resolve_adapter, training_config)
from foodlens.checkpoint import seed_everything
from foodlens.runtime import write_jsonl
from test_engineering import tiny_qwen_components, record, fixtures, ART


class JSONProtocolTests(unittest.TestCase):
    def test_order_merge_fields_precision_and_numeric_sensitivity(self):
        a = record()
        a["items"] = [{"name": " ＲＩＣＥ ", "weight_g": 3.12}, {"name": "rice", "weight_g": 6.92},
                      {"name": "apple", "weight_g": 1}]
        b = dict(reversed(list(a.items())))
        b["items"] = list(reversed(a["items"]))
        text = canonical_record_text(a)
        self.assertEqual(text, canonical_record_text(b))
        self.assertEqual(list(strict_loads(text)), ["items", *TOTAL_FIELDS])
        self.assertEqual(strict_loads(text)["items"], [{"name": "apple", "weight_g": 1.0}, {"name": "rice", "weight_g": 10.0}])
        self.assertIn('"weight_g":1.0', text)
        self.assertIn('"protein_g":10.0', text)
        self.assertEqual(record_text(a), text)
        c = copy.deepcopy(a); c["protein_g"] = 11
        self.assertNotEqual(text, record_text(c))
        c = copy.deepcopy(a); c["items"][0]["weight_g"] += 1
        self.assertNotEqual(text, record_text(c))

    def test_invalid_json_never_repaired(self):
        self.assertEqual(record_text({**record(), "extra": 1}), "")
        with self.assertRaises(ValueError):
            canonical_record_text({**record(), "fat_g": True})
        for raw in ('{"items":[],"items":[]}', '```json {} ```', '{"protein_g":NaN}'):
            with self.assertRaises(ValueError):
                strict_loads(raw)

    def test_shared_call_paths_and_evaluated_df(self):
        from foodlens import training, cli
        self.assertIs(training.score_record_texts, score_record_texts)
        if HAS_QWEN:
            from foodlens import qwen_training
            self.assertIs(training.score_record_texts, qwen_training.score_record_texts)
        self.assertIn("score_record_texts", inspect.getsource(cli.evaluate))
        truth = {"a": record(), "b": record(12)}
        hybrid = score_record_texts(truth, {"a": {"record": record()}}, metrics=["CIDEr-D"])
        qwen = score_record_texts(truth, {"a": {"raw": json.dumps(record())}}, metrics=["CIDEr-D"])
        self.assertEqual(hybrid, qwen)
        # Corpus DF uses every reference, even when its hypothesis fails/missing.
        refs = {"a": [record_text(record())], "b": [record_text(record(12))]}
        a = score_captions(refs, {"a": record_text(record())}, metrics=["CIDEr-D"])
        b = score_captions(refs, {"a": record_text(record()), "b": ""}, metrics=["CIDEr-D"])
        self.assertEqual(a["scores"], b["scores"])
        self.assertEqual(a["scores"], hybrid["scores"])
        self.assertEqual(a["samples"], 2)
        self.assertIn("evaluated reference corpus", a["protocol"]["df"])

    def test_full_denominator_failure_penalty_and_valid_subset(self):
        truth = {i: record() for i in ("ok", "missing", "parse", "schema", "inference", "truncated")}
        partial = record(12); partial["fat_g"] = -1
        pred = {"ok": {"record": record(12)}, "parse": {"raw": "broken"},
            "schema": {"record": partial}, "inference": {"raw": "", "status": "inference_failed"},
            "truncated": {"record": record(), "truncated": True}, "extra": {"record": record()}}
        result = evaluate_records(truth, pred)
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertEqual(result["parse_failed"], 3)
        self.assertEqual(result["schema_failed"], 1)
        self.assertEqual(result["inference_failed"], 1)  # overlaps parse_failed
        self.assertEqual(result["truncation_rate"], 1/6)
        self.assertEqual(result["prediction_extras"], ["extra"])
        self.assertEqual(result["totals"]["protein_g"], {"mae": 34/6, "mse": 308/6, "count": 6})
        self.assertEqual(result["totals"]["fat_g"]["mse"], 404/6)
        self.assertEqual(result["valid_subset_coverage"], 2/6)
        self.assertEqual(result["valid_subset"]["totals"]["protein_g"], {"mae": 1., "mse": 2., "count": 2})
        self.assertEqual(result["name_micro_f1"], 2/3)
        empty = evaluate_records({"x": record()}, {})
        self.assertEqual(empty["valid_subset"]["samples"], 0)
        self.assertIsNone(empty["valid_subset"]["totals"]["protein_g"]["mse"])
        no_output = evaluate_records({"x": record()}, {"x": {"status": "inference_failed"}})
        self.assertEqual(no_output["json_valid_rate"], 0)
        self.assertEqual(no_output["parse_failed"], 1)
        self.assertEqual(no_output["inference_failed"], 1)
        literal_null = evaluate_records({"x": record()}, {"x": {"raw": "null"}})
        self.assertEqual(literal_null["json_valid_rate"], 1)
        self.assertEqual(literal_null["schema_failed"], 1)


@unittest.skipUnless(HAS_QWEN, "optional Qwen dependencies unavailable")
class TrainerProtocolTests(unittest.TestCase):
    def test_nonfinite_gradient_rejected(self):
        from foodlens.qwen_training import ProtocolCheckpoint
        guard = ProtocolCheckpoint(None, {})
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.zeros_like(model.weight)
        guard.on_pre_optimizer_step(None, None, None, model)
        model.weight.grad[0, 0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "nonfinite Qwen gradient"):
            guard.on_pre_optimizer_step(None, None, None, model)

    def test_selection_tolerance_and_serializable_state(self):
        from transformers import TrainerControl, TrainerState
        cb = StructuredSelection(2, .001)
        control, state = TrainerControl(), TrainerState(global_step=1)
        def evaluate(f1, mae, schema):
            cb.on_evaluate(None, state, control, {"eval_name_micro_f1": f1, "eval_normalized_mae": mae,
                                                "eval_schema_valid_rate": schema})
        evaluate(.5, 2., .9)
        state.global_step = 2
        evaluate(.5005, 1., .8)  # F1 tolerance -> MAE wins
        self.assertEqual(cb.best_step, 2)
        state.global_step = 3
        evaluate(.5, 1., .9)  # MAE tolerance -> Schema wins
        self.assertEqual(cb.best_step, 3)
        evaluate(.4, 0., 1.); evaluate(.4, 0., 1.)
        self.assertTrue(control.should_training_stop)
        restored = StructuredSelection.from_state(json.loads(json.dumps(cb.state())))
        self.assertEqual(restored.state(), cb.state())
        with self.assertRaisesRegex(ValueError, "already early-stopped"):
            restored.on_train_begin(None, state, control)

    def test_processor_fields(self):
        self.assertEqual(set(model_inputs({"input_ids": 1, "token_type_ids": 2})), {"input_ids"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            model_inputs({"unrecognized": 1})
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            model, processor = tiny_qwen_components(Path(tmp))
            real_template = processor.apply_chat_template
            def with_types(*args, **kwargs):
                batch = real_template(*args, **kwargs)
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
                return batch
            with patch.object(processor, "apply_chat_template", side_effect=with_types):
                image = Image.new("RGB", (16, 16))
                batch = QwenCollator(processor, 2048)([{"image": image, "target": record()}])
                self.assertNotIn("token_type_ids", batch)
                self.assertTrue(torch.isfinite(model(**batch).loss))
                generated = generate_qwen(model, processor, [image], QwenConfig(device="cpu", precision="fp32", max_length=2048, max_new_tokens=2))
                self.assertIn("raw", generated[0])

    def test_dataset_inference_rejects_answer_fields(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            write_jsonl(root/"leak.jsonl", [{"id": "x", "image": "synthetic_0.png", "target": record()}])
            with self.assertRaisesRegex(ValueError, "only id/image"):
                QwenDataset(root/"leak.jsonl", root, inference=True)
            dataset = QwenDataset(root/"train.jsonl", root, split="train")
            self.assertEqual(dataset[0]["image"].mode, "RGB")

    def test_accumulation_matches_old_weighted_ce_and_large_batch(self):
        from transformers import Trainer, TrainingArguments
        for checkpointing in (False, True):
            with self.subTest(gradient_checkpointing=checkpointing), tempfile.TemporaryDirectory(dir=ART) as tmp:
                root = Path(tmp)
                seed_everything(73)
                base, processor = tiny_qwen_components(root)
                qconfig = QwenConfig(device="cpu", precision="fp32", dropout=0., gradient_checkpointing=checkpointing)
                model, scope = inject_lora(base, qconfig)
                manual = copy.deepcopy(model)
                large = copy.deepcopy(model)
                target = record(); target["items"] += [{"name": "apple with long ingredient name", "weight_g": 20.}]
                samples = [{"id": "a", "image": Image.new("RGB", (16, 16)), "target": record()},
                           {"id": "b", "image": Image.new("RGB", (16, 16)), "target": target}]
                collator = QwenCollator(processor, 2048)
                micro = [collator([s]) for s in samples]
                counts = [int(b["labels"][:, 1:].ne(-100).sum()) for b in micro]
                self.assertNotEqual(*counts)
                manual.train()
                opt = torch.optim.SGD([p for p in manual.parameters() if p.requires_grad], lr=.01)
                for batch, count in zip(micro, counts):
                    output = manual(**batch)
                    custom = assistant_token_loss(output, batch["labels"])
                    torch.testing.assert_close(custom, output.loss, rtol=1e-6, atol=1e-6)
                    (output.loss * count / sum(counts)).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in manual.parameters() if p.grad is not None))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in manual.parameters()))
                opt.step()
                for candidate, batch_size, accumulation in ((model, 1, 4), (large, 2, 1)):
                    # accumulation=4 deliberately exercises a short final window of two.
                    optimizer = torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad], lr=.01)
                    args = TrainingArguments(output_dir=str(root/f"trainer-{batch_size}"), use_cpu=True,
                        num_train_epochs=1, per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
                        max_grad_norm=1e6, save_strategy="no", report_to=[], disable_tqdm=True,
                        remove_unused_columns=False, dataloader_pin_memory=False, seed=73,
                        gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
                    trainer = Trainer(model=candidate, args=args, train_dataset=samples, data_collator=collator,
                        compute_loss_func=assistant_token_loss, optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)))
                    trainer.train()
                    self.assertEqual(trainer.state.global_step, 1)
                    for (name, actual), (other, expected) in zip(candidate.named_parameters(), manual.named_parameters()):
                        self.assertEqual(name, other)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, msg=name)
                self.assertTrue(all(".self_attn." in p["name"] and ".lora_" in p["name"] for p in scope["trainable"]))

    def test_trainer_selection_checkpoint_and_resume_guards(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            base, processor = tiny_qwen_components(root)
            base.save_pretrained(root/"base"); processor.save_pretrained(root/"base")
            q = QwenConfig(model_id=str(root/"base"), device="cpu", precision="fp32", dropout=0.,
                gradient_checkpointing=True, max_length=2048, max_new_tokens=2, min_pixels=16, max_pixels=64)
            c = QwenTrainConfig(device="cpu", precision="fp32", engineering=True, metrics=["CIDEr-D"],
                epochs=3, patience=1, accumulation=3)
            common = dict(train_annotations=root/"train.jsonl", val_annotations=root/"val.jsonl", data_root=root, out=root/"run")
            result = run_qwen_training(q, c, **common)
            self.assertEqual(result["epoch"], 2)
            self.assertEqual(result["step"], 4)  # ceil(4 / 3) per epoch
            best, last = resolve_adapter(root/"run/best.json"), resolve_adapter(root/"run/last.json")
            self.assertEqual(best.name, "checkpoint-2")
            self.assertEqual(last.name, "checkpoint-4")
            for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "run.json", "protocol.json"):
                self.assertTrue((last/filename).is_file(), filename)
            state = json.loads((last/"trainer_state.json").read_text())
            self.assertEqual(state["stateful_callbacks"]["StructuredSelection"]["attributes"]["bad_epochs"], 1)
            with self.assertRaisesRegex(ValueError, "configuration/data/protocol mismatch"):
                run_qwen_training(q, replace(c, accumulation=2), resume=last, **common)
            with self.assertRaisesRegex(ValueError, "already early-stopped"):
                run_qwen_training(q, c, resume=last, **common)


@unittest.skipUnless(HAS_QWEN, "optional Qwen dependencies unavailable")
class RealDataQwenTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[2]/"data/annotations/train.jsonl").exists(), "real course data absent")
    def test_rgb_dataset_and_longest_targets_with_tiny_processor(self):
        from foodlens.data import sha256
        root = Path(__file__).resolve().parents[2]/"data"
        sources = json.loads((root/"sources.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            _, processor = tiny_qwen_components(Path(tmp))
            for entry in sources["nutrition"]["splits"]:
                annotation = root/entry["path"]
                self.assertEqual(sha256(annotation), entry["sha256"])
                dataset = QwenDataset(annotation, root, split=annotation.stem)
                self.assertEqual(len(dataset), entry["records"])
                for index in range(len(dataset)):
                    sample = dataset[index]
                    self.assertEqual(sample["image"].mode, "RGB")
                    self.assertEqual(set(sample), {"id", "image", "target"})
                    self.assertTrue(record_text(sample["target"]))
                longest = max(range(len(dataset)), key=lambda i: len(json.dumps(dataset.rows[i]["target"])))
                batch = QwenCollator(processor, 16384)([dataset[longest]])
                self.assertGreater(int(batch["labels"].ne(-100).sum()), 0)


if __name__ == "__main__":
    unittest.main()
