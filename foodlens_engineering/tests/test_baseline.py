import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest
from dataclasses import asdict
import torch
from torch import nn

from foodlens.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from foodlens.config import load_config
from foodlens.data import (NutritionDataset, RGBManifestDataset, TrainStatistics, collate_nutrition, safe_image_path,
                           serialize_items, validate_manifest_splits)
from foodlens.decode import GenerationConfig, predict_caption, predict_hybrid
from foodlens.extensions import qwen_messages
from foodlens.losses import hybrid_loss, token_loss
from foodlens.metrics import evaluate_records
from foodlens.models import (CaptionA, CaptionB, Hybrid, ModelConfig, VisualEncoder,
                             causal_mask, grid_positions, transfer_caption_a)
from foodlens.schema import TOTAL_FIELDS, assemble, canonical_items, normalize_name, strict_loads, validate_record
from foodlens.text import Vocabulary, detokenize, tokenize

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2] / "data"
TEST_TEMP = Path(__file__).resolve().parents[1] / "artifacts/test-temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


def record(items=None, value=10.0):
    return {"items": [{"name": "rice", "weight_g": 10.0}] if items is None else items,
            **dict.fromkeys(TOTAL_FIELDS, value)}


def vocabulary():
    vocab = Vocabulary()
    vocab.extend_training(["brown rice", "apple"], split="train")
    vocab.add_sep()
    return vocab


def tiny():
    return ModelConfig(backbone="tiny", hidden=32, embedding=32, heads=4, feedforward=128, dropout=0.0)


class TextSchemaTests(unittest.TestCase):
    def test_config_loading_validation(self):
        path = Path(__file__).resolve().parents[1] / "configs/baseline.json"
        value = load_config(path)
        self.assertEqual(value["model"]["hidden"], 256)
        self.assertEqual(value["loss_weights"], [1, 1, 1])
        for mutation in ({"typo": 1}, {"loss_weights": [1, float("nan"), 1]},
                         {"nutrition_generation": {"max_tokens": -1, "max_items": 1}}):
            bad = {**value, **mutation}
            with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
                file = Path(temp) / "bad.json"
                file.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises((ValueError, TypeError)):
                    load_config(file)

    def test_normalization_and_duplicate_merge(self):
        self.assertEqual(normalize_name("  ＲＩＣＥ\t Brown "), "rice brown")
        self.assertEqual(canonical_items([{"name": " Rice ", "weight_g": 2},
                                          {"name": "ＲＩＣＥ", "weight_g": 3}]),
                         [{"name": "rice", "weight_g": 5.0}])

    def test_strict_schema(self):
        self.assertFalse(validate_record(record()))
        for value in (True, -1, float("nan"), float("inf"), "10"):
            bad = record()
            bad["protein_g"] = value
            self.assertTrue(validate_record(bad))
        bad = record()
        bad["extra"] = 1
        self.assertTrue(validate_record(bad))
        self.assertTrue(validate_record(record([{"name": " ", "weight_g": 1}])))
        with self.assertRaises(ValueError):
            strict_loads('{"a":NaN}')
        with self.assertRaises(ValueError):
            strict_loads('{"a":1,"a":2}')

    def test_vocab_training_only_retains_ids(self):
        vocab = vocabulary()
        original = list(vocab.tokens)
        vocab.extend_training(["new food"], split="train")
        self.assertEqual(vocab.tokens[:len(original)], original)
        with self.assertRaises(ValueError):
            vocab.extend_training(["test secret"], split="test")
        self.assertEqual(vocab.encode("unseenword"), [3])

    def test_training_name_roundtrip_including_parentheses(self):
        for name in ("spinach (cooked)", "spinach (raw)", "brown rice"):
            self.assertEqual(detokenize(tokenize(name)), name)
        rows = [json.loads(line) for line in (ROOT / "annotations/train.jsonl").read_text(encoding="utf-8").splitlines()]
        names = {item["name"] for row in rows for item in row["target"]["items"]}
        for name in names:
            self.assertEqual(detokenize(tokenize(name)), normalize_name(name))

    def test_sep_off_by_one_and_sorting(self):
        vocab = vocabulary()
        truth = record([{"name": "brown rice", "weight_g": 2}, {"name": " BROWN RICE ", "weight_g": 3}])
        sequence, boundaries, weights = serialize_items(truth, vocab)
        self.assertEqual(sequence, [1, vocab.index["brown"], vocab.index["rice"], vocab.index["<SEP>"], 2])
        self.assertEqual(boundaries, [2])
        self.assertEqual(sequence[1:][boundaries[0]], vocab.index["<SEP>"])
        self.assertEqual(weights, [5])
        two = record([{"name": "brown rice", "weight_g": 2}, {"name": "apple", "weight_g": 1}])
        seq, _, _ = serialize_items(two, vocab)
        self.assertEqual(seq[1], vocab.index["apple"])

    def test_path_and_duplicate_group_isolation(self):
        with self.assertRaises(ValueError):
            safe_image_path(ROOT, "../private.png")
        with self.assertRaises(ValueError):
            validate_manifest_splits([{"sha256": "x", "duplicate_group": "g", "split": "train"},
                                      {"sha256": "y", "duplicate_group": "g", "split": "test"}])

    def test_training_statistics_reject_validation(self):
        with self.assertRaises(ValueError):
            TrainStatistics.fit([{"split": "val", "target": record()}])
        stats = TrainStatistics.fit([{"split": "train", "target": record()}])
        self.assertGreaterEqual(stats.weight_std, 1e-6 - 1e-12)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.vocab = vocabulary()
        self.images = torch.randn(2, 3, 32, 32)
        self.inputs = torch.tensor([[1, 4, 5, 0], [1, 4, 0, 0]])

    def test_resnet18_actual_layer4_shape_and_frozen_bn(self):
        encoder = VisualEncoder(ModelConfig(pretrained=False))
        encoder.train()
        bn = [m for m in encoder.features.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(bn and all(not m.training for m in bn))
        before = bn[0].running_mean.clone()
        with torch.no_grad():
            grid, v, z = encoder(torch.randn(1, 3, 224, 224))
        self.assertEqual(grid.shape, (1, 512, 7, 7))
        self.assertEqual(v.shape, (1, 512))
        self.assertEqual(z.shape, (1, 256))
        self.assertTrue(torch.equal(before, bn[0].running_mean))
        self.assertTrue(all(not p.requires_grad for p in encoder.features.parameters()))
        encoder.unfreeze_layer4()
        self.assertTrue(all(p.requires_grad for p in encoder.features[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in encoder.features[0].parameters()))

    def test_a_initial_condition_and_forward_backward(self):
        model = CaptionA(len(self.vocab), tiny())
        self.assertEqual(model.gru.input_size, 64)
        _, _, z = model.visual(self.images)
        logits, states, _ = model.decode(self.inputs, z)
        concat = torch.cat((model.embedding(self.inputs), z[:, None].expand(-1, 4, -1)), -1)
        expected, _ = model.gru(concat, z.unsqueeze(0))
        self.assertTrue(torch.equal(states, expected))
        token_loss(logits, torch.tensor([[4, 5, 2, 0], [4, 2, 0, 0]])).backward()
        self.assertGreater(float(model.gru.weight_ih_l0.grad.abs().sum()), 0)
        self.assertGreater(float(model.visual.projection.weight.grad.abs().sum()), 0)

    def test_padding_loss_is_token_normalized(self):
        logits = torch.randn(2, 4, 8)
        target = torch.tensor([[4, 5, 2, 0], [4, 2, 0, 0]])
        a = token_loss(logits, target)
        b = nn.functional.cross_entropy(logits[target.ne(0)], target[target.ne(0)])
        self.assertTrue(torch.allclose(a, b))

    def test_b_causal_padding_global_and_gradient(self):
        for global_only in (False, True):
            model = CaptionB(len(self.vocab), tiny(), global_only=global_only).eval()
            memory = model.memory(self.images)
            self.assertEqual(memory.shape, (2, 1 if global_only else 49, 32))
            tokens = torch.tensor([[1, 4, 5, 6], [1, 5, 4, 6]])
            changed = tokens.clone()
            changed[:, 2:] = 7
            first = model.decode(tokens, memory)
            second = model.decode(changed, memory)
            self.assertTrue(torch.allclose(first[:, :2], second[:, :2], atol=1e-6))
            short = model.decode(tokens[:, :2], memory)
            padded = model.decode(torch.cat((tokens[:, :2], torch.zeros(2, 2, dtype=torch.long)), 1), memory)
            self.assertTrue(torch.allclose(short, padded[:, :2], atol=1e-6))
            # Past PAD is visible to causality, so this isolates key padding masking.
            past_pad = torch.tensor([[1, 0, 5], [1, 0, 4]])
            before_pad_change = model.decode(past_pad, memory)[:, -1]
            saved_pad = model.embedding.weight[0].detach().clone()
            with torch.no_grad():
                model.embedding.weight[0].fill_(1000)
            after_pad_change = model.decode(past_pad, memory)[:, -1]
            self.assertTrue(torch.allclose(before_pad_change, after_pad_change, atol=1e-6))
            with torch.no_grad():
                model.embedding.weight[0].copy_(saved_pad)
            token_loss(model.decode(tokens, memory), torch.tensor([[4, 5, 6, 2], [5, 4, 6, 2]])).backward()
            self.assertGreater(float(model.grid_projection.weight.grad.abs().sum()), 0)
            self.assertEqual(predict_caption(model, self.images, max_tokens=3).shape[0], 2)
        mask = causal_mask(3)
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[2, 0])
        position = grid_positions(7, 7, 32, device="cpu", dtype=torch.float32)
        self.assertFalse(torch.equal(position[0], position[1]))
        self.assertFalse(torch.equal(position[0], position[7]))

    def test_hybrid_boundary_gradient_and_zero_items(self):
        model = Hybrid(len(self.vocab), tiny())
        samples = [{"id": str(i), "image": self.images[i], "target": record()} for i in range(2)]
        batch = collate_nutrition(samples, self.vocab)
        stats = TrainStatistics(10, 2, [10]*5, [2]*5)
        output = model(batch["images"], batch["inputs"], batch["boundaries"])
        selected = output["states"][:, 1]
        _, _, z = model.visual(batch["images"])
        expected = model.weight_head(torch.cat((selected, z), -1)).squeeze(-1)
        self.assertTrue(torch.allclose(output["weights"][:, 0], expected))
        self.assertTrue((output["weights"] >= 0).all() and (output["totals"] >= 0).all())
        hybrid_loss(output, batch, stats)["loss"].backward()
        for layer in (model.weight_head[0], model.total_head[0], model.gru):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))
        empty = collate_nutrition([{"id": "e", "image": self.images[0], "target": record([])}], self.vocab)
        self.assertEqual(float(hybrid_loss(model(empty["images"], empty["inputs"], empty["boundaries"]), empty, stats)["weight"].detach()), 0.0)

    def test_transfer_exact_rows_and_new_rows_unchanged(self):
        old = Vocabulary()
        old.extend_training(["rice"], split="train")
        new = Vocabulary(old.tokens)
        new.extend_training(["apple"], split="train")
        new.add_sep()
        source, target = CaptionA(len(old), tiny()), Hybrid(len(new), tiny())
        untouched = target.embedding.weight[len(old):].detach().clone()
        transfer_caption_a(source, old, target, new)
        self.assertTrue(torch.equal(source.embedding.weight, target.embedding.weight[:len(old)]))
        self.assertTrue(torch.equal(source.output.bias, target.output.bias[:len(old)]))
        self.assertTrue(torch.equal(untouched, target.embedding.weight[len(old):]))
        self.assertTrue(torch.equal(source.gru.weight_ih_l0, target.gru.weight_ih_l0))

    def test_checkpoint_optimizer_scheduler_rng(self):
        model = Hybrid(len(self.vocab), tiny())
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        optimizer.zero_grad()
        model(self.images, self.inputs, torch.tensor([[1], [1]]))["totals"].sum().backward()
        optimizer.step()
        scheduler.step()
        stats = TrainStatistics(1, 2, [1]*5, [2]*5)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            path = Path(temp) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, scheduler, vocab=self.vocab, statistics=stats, metadata={"test": True}, step=1)
            expected_python, expected_torch = random.random(), torch.rand(2)
            other = Hybrid(len(self.vocab), tiny())
            other_opt = torch.optim.AdamW(other.parameters())
            other_sched = torch.optim.lr_scheduler.StepLR(other_opt, 1)
            restored = load_checkpoint(path, other, other_opt, other_sched, restore_rng=True)
            self.assertEqual(expected_python, random.random())
            self.assertTrue(torch.equal(expected_torch, torch.rand(2)))
            self.assertEqual(restored["vocab"], self.vocab.tokens)
            self.assertEqual(other_opt.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
            self.assertTrue(other_opt.state)
            for a, b in zip(model.parameters(), other.parameters()):
                self.assertTrue(torch.equal(a, b))


class ScriptedModel:
    def __init__(self, vocab, sequence):
        self.vocab, self.sequence, self.step = vocab, sequence, 0

    def eval(self):
        return self

    def visual(self, images):
        return None, torch.zeros(len(images), 512), torch.zeros(len(images), 32)

    def total_head(self, condition):
        return torch.ones(len(condition), 5) * 99

    def weight_head(self, condition):
        return condition[:, :1]  # exposes precisely which state was gathered

    def decode(self, inputs, z, initial):
        logits = torch.full((1, 1, len(self.vocab)), -100.0)
        logits[0, 0, self.sequence[self.step]] = 100
        self.step += 1
        states = torch.full((1, 1, 32), float(self.step))
        return logits, states, states.transpose(0, 1)


class DecodeMetricTests(unittest.TestCase):
    def test_nonfinite_prediction_stays_json_readable_failure(self):
        vocab = vocabulary()
        tokens = [vocab.index["rice"], vocab.index["<SEP>"], 2]
        model = ScriptedModel(vocab, tokens)
        model.total_head = lambda condition: torch.tensor([[float("nan"), 1, 1, 1, 1]])
        prediction = predict_hybrid(model, torch.zeros(1, 3, 32, 32), vocab)[0]
        json.dumps(prediction, allow_nan=False)
        self.assertEqual(prediction["invalid_total_fields"], ["total_weight_g"])
        result = evaluate_records({"x": record()}, {"x": prediction})
        self.assertEqual(result["json_valid_rate"], 1)
        self.assertEqual(result["schema_valid_rate"], 0)
        self.assertEqual(result["totals"]["total_weight_g"]["mae"], 10)

    def test_sep_decode_alignment_and_unclosed_discard(self):
        vocab = vocabulary()
        tokens = [vocab.index["brown"], vocab.index["rice"], vocab.index["<SEP>"], 2]
        model = ScriptedModel(vocab, tokens)
        result = predict_hybrid(model, torch.zeros(1, 3, 32, 32), vocab)[0]
        self.assertEqual(result["record"]["items"], [{"name": "brown rice", "weight_g": 3.0}])
        self.assertEqual(result["record"]["total_weight_g"], 99)
        self.assertFalse(result["truncated"])
        unclosed = predict_hybrid(ScriptedModel(vocab, tokens), torch.zeros(1, 3, 32, 32), vocab, GenerationConfig(max_tokens=2))[0]
        self.assertEqual(unclosed["record"]["items"], [])
        self.assertTrue(unclosed["truncated"])
        self.assertEqual(len(unclosed["discarded_unclosed_tokens"]), 2)

    def test_input_isolation_and_qwen_boundary(self):
        params = inspect.signature(predict_hybrid).parameters
        self.assertEqual(list(params), ["model", "images", "vocab", "config"])
        with self.assertRaises(TypeError):
            predict_hybrid(None, None, None, target_text="answer")
        dataset = NutritionDataset(ROOT / "annotations/train.jsonl", ROOT, split="train", limit=1)
        self.assertEqual(set(dataset[0]), {"id", "image", "target"})
        self.assertEqual(dataset[0]["image"].shape, (3, 224, 224))
        self.assertEqual(len(qwen_messages("rgb")), 1)
        self.assertEqual(len(qwen_messages("rgb", training_target=record())), 2)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            manifest = Path(temp) / "rgb.jsonl"
            row = dataset.rows[0]
            manifest.write_text(json.dumps({"id": row["dish_id"], "image": row["image"]}), encoding="utf-8")
            self.assertEqual(set(RGBManifestDataset(manifest, ROOT)[0]), {"id", "image"})
            manifest.write_text(json.dumps({"id": "x", "image": "x.png", "metadata": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RGBManifestDataset(manifest, ROOT)

    def test_all_sample_failures_exact_metrics(self):
        truths = {"good": record(), "bad": record()}
        results = evaluate_records(truths, {"good": {"record": record(value=12)}, "bad": {"raw": "broken"}})
        self.assertEqual(results["samples"], 2)
        self.assertEqual(results["tp"], 1)
        self.assertEqual(results["fn"], 1)
        self.assertAlmostEqual(results["name_micro_f1"], 2/3)
        self.assertEqual(results["weight_union"]["mae"], 5)
        self.assertEqual(results["weight_union"]["mse"], 50)
        self.assertEqual(results["totals"]["protein_g"]["mae"], 6)
        self.assertEqual(results["totals"]["protein_g"]["mse"], 52)
        self.assertEqual(results["json_valid_rate"], 0.5)
        self.assertEqual(results["schema_valid_rate"], 0.5)

    def test_invalid_item_is_fp_and_weight_is_preserved(self):
        bad = record([{"name": " ", "weight_g": 3}, {"name": True, "weight_g": 4}])
        result = evaluate_records({"x": record()}, {"x": {"record": bad}})
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["weight_union"]["count"], 3)
        self.assertAlmostEqual(result["weight_union"]["mae"], 17/3)
        self.assertEqual(result["json_valid_rate"], 1)
        self.assertEqual(result["schema_valid_rate"], 0)
        self.assertIsNone(result["weight_matched"]["mae"])
        cleanup = evaluate_records({"x": record()}, {"x": {"record": record(), "failures": [{"weight_g": 5}]}})
        self.assertEqual(cleanup["fp"], 1)

    def test_missing_predictions_not_dropped_duplicate_merge(self):
        duplicate = record([{"name": " Rice", "weight_g": 3}, {"name": "ＲＩＣＥ", "weight_g": 7}])
        result = evaluate_records({"x": record(), "missing": record()}, {"x": {"record": duplicate}})
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["weight_matched"]["mae"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
