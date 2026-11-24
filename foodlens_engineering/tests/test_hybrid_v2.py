"""Hybrid V2 CPU contracts; real-data checks are separately optional."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from foodlens.ingredients import IngredientVocabulary, serialize_ingredients
from foodlens.data import TrainStatistics, collate_hybrid, read_jsonl, sha256, NutritionDataset
from foodlens.models import Hybrid, LegacyHybrid, CaptionA, ModelConfig, transfer_visual_backbone
from foodlens.losses import hybrid_loss
from foodlens.schema import TOTAL_FIELDS, canonical_items, validate_record
from foodlens.decode import predict_hybrid, GenerationConfig
from foodlens.metrics import evaluate_records
from foodlens.checkpoint import save_checkpoint, load_checkpoint
from foodlens.training import TrainConfig, run_training, restore_model, selection_value, is_better
from test_engineering import fixtures, ART

torch.set_num_threads(2)


def cfg():
    return ModelConfig(backbone="tiny", hidden=16, embedding=16, heads=4, layers=2, feedforward=32, dropout=0.)


def record(items=None):
    return {"items": items if items is not None else [{"name": "rice", "weight_g": 13}, {"name": "apple", "weight_g": 7}],
            **dict.fromkeys(TOTAL_FIELDS, 20.)}


def vocab():
    return IngredientVocabulary.fit([{"split": "train", "target": record()}])


def sample(target, i="x"):
    return {"id": i, "image": torch.randn(3, 16, 16), "target": target}


class Scripted:
    def eval(self):
        return self

    def encode(self, images):
        return torch.zeros(len(images), 49, 16), torch.zeros(len(images), 512), torch.zeros(len(images), 16)

    def total_head(self, x):
        return torch.ones(len(x), 5)

    def weight_head(self, x):
        return x[:, :1]

    def decode(self, prefix, memory):
        logits = torch.full((1, prefix.shape[1], 6), -100.)
        # Repeated/lower class scores dominate after consuming apple, but are forbidden.
        if prefix.shape[1] == 1:
            logits[0, -1, 4] = 100
        elif prefix.shape[1] == 2:
            logits[0, -1, 4] = 200
            logits[0, -1, 5] = 100
        else:
            logits[0, -1, 4] = 200
            logits[0, -1, 2] = 100
        # States expose the consumed class ID, not the predicted next ID.
        return logits, prefix.float().unsqueeze(-1).expand(-1, -1, 16)


class HybridV2Tests(unittest.TestCase):
    def test_production_dimensions_and_resnet_grid(self):
        m = Hybrid(len(vocab()), ModelConfig(pretrained=False)).eval()
        memory, v, z = m.encode(torch.randn(1, 3, 224, 224))
        self.assertEqual(memory.shape, (1, 49, 256))
        self.assertEqual(v.shape, (1, 512)); self.assertEqual(z.shape, (1, 256))
        self.assertEqual(len(m.decoder.layers), 2)
        self.assertEqual(m.decoder.layers[0].self_attn.num_heads, 4)
        self.assertEqual(m.decoder.layers[0].linear1.out_features, 1024)
        self.assertEqual(m.weight_head[0].in_features, 512)
        self.assertEqual(m.total_head[0].in_features, 768)
        self.assertEqual(m.decode(torch.tensor([[1, 4, 5]]), memory)[0].shape, (1, 3, len(vocab())))

    def test_mapping_stable_train_only_roundtrip_and_merge(self):
        v = vocab()
        self.assertEqual(v.tokens[4:], ["apple", "rice"])
        self.assertEqual(v.metadata(), IngredientVocabulary.restore(v.metadata()).metadata())
        r = record(list(reversed(record()["items"])))
        self.assertEqual(serialize_ingredients(r, v), serialize_ingredients(record(), v))
        duplicate = record([{"name": "ＲＩＣＥ", "weight_g": 3}, {"name": " rice ", "weight_g": 10}, {"name": "apple", "weight_g": 7}])
        self.assertEqual(serialize_ingredients(duplicate, v), serialize_ingredients(record(), v))
        duplicated = [{"name": "rice", "weight_g": w} for w in [0.1, 0.2, 3.0]]
        self.assertEqual(serialize_ingredients(record(duplicated), v), serialize_ingredients(record(list(reversed(duplicated))), v))
        self.assertEqual(IngredientVocabulary.fit([{"split": "train", "target": r}]).tokens, v.tokens)
        with self.assertRaises(ValueError):
            IngredientVocabulary.fit([{"split": "val", "target": r}])
        bad = v.metadata(); bad["normalization_version"] = "unknown"
        with self.assertRaises(ValueError):
            IngredientVocabulary.restore(bad)

    def test_shift_consumed_state_and_pad_loss(self):
        v = vocab()
        batch = collate_hybrid([sample(record()), sample(record([record()["items"][0]]), "y"), sample(record([]), "e")], v)
        self.assertEqual(batch["inputs"].tolist(), [[1, 4, 5], [1, 5, 0], [1, 0, 0]])
        self.assertEqual(batch["targets"].tolist(), [[4, 5, 2], [5, 2, 0], [2, 0, 0]])
        self.assertEqual(batch["boundaries"].tolist(), [[1, 2], [1, 0], [0, 0]])
        self.assertEqual(batch["weights"].tolist(), [[7, 13], [13, 0], [0, 0]])
        m = Hybrid(len(v), cfg()).eval()
        out = m(batch["images"], batch["inputs"], batch["boundaries"])
        _, _, z = m.encode(batch["images"])
        expected = m.weight_head(torch.cat((out["states"][0, 1:3], z[0].expand(2, -1)), -1)).flatten()
        torch.testing.assert_close(out["weights"][0], expected)
        stats = TrainStatistics(0, 1, [0]*5, [1]*5)
        out["states"].retain_grad(); out["weights"].retain_grad()
        hybrid_loss(out, batch, stats)["weight"].backward()
        self.assertTrue((out["weights"].grad[~batch["item_mask"]] == 0).all())
        self.assertTrue((out["weights"].grad[batch["item_mask"]] != 0).all())
        self.assertEqual(float(out["states"].grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(out["states"].grad[0, -1].abs().sum()), 0.)
        empty = collate_hybrid([sample(record([]))], v)
        self.assertEqual(float(hybrid_loss(m(empty["images"], empty["inputs"], empty["boundaries"]), empty, stats)["weight"].detach()), 0.)

    def test_causal_padding_grid_cross_attention_and_gradients(self):
        m = Hybrid(len(vocab()), cfg()).eval()
        images = torch.randn(2, 3, 16, 16)
        memory, _, _ = m.encode(images)
        self.assertEqual(memory.shape, (2, 49, 16))
        self.assertFalse(hasattr(m, "encoder"))
        x = torch.tensor([[1, 4, 5], [1, 4, 5]])
        y = x.clone(); y[:, -1] = 4
        torch.testing.assert_close(m.decode(x, memory)[0][:, :2], m.decode(y, memory)[0][:, :2])
        torch.testing.assert_close(m.decode(x[:, :2], memory)[0], m.decode(torch.cat((x[:, :2], torch.zeros(2, 1, dtype=torch.long)), 1), memory)[0][:, :2])
        for length in (1, 2, 3):
            torch.testing.assert_close(m.decode(x[:, :length], memory)[1][:, -1], m.decode(x, memory)[1][:, length-1], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(m.decode(x, memory)[0], m.decode(x, torch.zeros_like(memory))[0]))
        memory.retain_grad()
        m.decode(x, memory)[0].sum().backward()
        self.assertGreater(float(memory.grad.abs().sum()), 0)
        batch = collate_hybrid([sample(record()), sample(record(), "y")], vocab())
        m.zero_grad()
        hybrid_loss(m(batch["images"], batch["inputs"], batch["boundaries"]), batch, TrainStatistics(10, 2, [10]*5, [2]*5))["loss"].backward()
        for module in (m.decoder, m.weight_head, m.total_head, m.grid_projection):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_autoregressive_alignment_eos_limits_and_duplicates(self):
        images = torch.zeros(1, 3, 16, 16)
        p = predict_hybrid(Scripted(), images, vocab())[0]
        self.assertEqual(p["raw_token_ids"], [4, 5, 2])
        self.assertEqual(p["record"]["items"], [{"name": "apple", "weight_g": 4.}, {"name": "rice", "weight_g": 5.}])
        self.assertFalse(p["truncated"])
        self.assertEqual(p["schema_errors"], [])
        for config in (GenerationConfig(max_tokens=2), GenerationConfig(max_items=2)):
            p = predict_hybrid(Scripted(), images, vocab(), config)[0]
            self.assertEqual(len(p["record"]["items"]), 2)
            self.assertEqual(p["record"]["items"][-1]["weight_g"], 5.)
            self.assertTrue(p["truncated"])
        model = Scripted()
        model.decode = lambda x, m: (torch.tensor([[[0., 0., 100., 0., 0., 0.]]]), torch.zeros(1, 1, 16))
        p = predict_hybrid(model, images, vocab())[0]
        self.assertEqual(p["record"]["items"], [])
        self.assertFalse(p["truncated"])

    def test_nonfinite_outputs_and_failures(self):
        model = Scripted()
        model.weight_head = lambda x: torch.full((1, 1), float("nan"))
        model.total_head = lambda x: torch.tensor([[float("inf"), 1, 1, 1, 1]])
        p = predict_hybrid(model, torch.zeros(1, 3, 16, 16), vocab())[0]
        json.dumps(p, allow_nan=False)
        self.assertEqual(len(p["failures"]), 2)
        self.assertTrue(p["schema_errors"])
        model = Scripted()
        model.decode = lambda x, m: (torch.full((1, 1, 6), float("nan")), torch.zeros(1, 1, 16))
        p = predict_hybrid(model, torch.zeros(1, 3, 16, 16), vocab())[0]
        self.assertEqual(p["stop_reason"], "nonfinite_logits")
        self.assertTrue(p["truncated"])

    def test_oov_truth_is_independent_and_order_invariant(self):
        truth = record([{"name": "unknown one", "weight_g": 10}, {"name": "unknown two", "weight_g": 20}])
        batch = collate_hybrid([sample(truth)], vocab())
        self.assertEqual(batch["targets"].tolist(), [[0, 0, 2]])
        results = evaluate_records({"x": truth}, {"x": {"record": record([])}})
        self.assertEqual(results["fn"], 2)
        self.assertEqual(results["weight_union"]["count"], 2)
        self.assertEqual(results["weight_union"]["mae"], 15)
        r = record(); reversed_r = record(list(reversed(r["items"])))
        results = evaluate_records({"x": r}, {"x": {"record": reversed_r}})
        self.assertEqual(results["name_micro_precision"], 1)
        self.assertEqual(results["name_micro_recall"], 1)
        self.assertEqual(results["name_micro_f1"], 1)
        self.assertEqual(results["weight_union"]["mse"], 0)
        self.assertEqual(results["schema_valid_rate"], 1)

    def test_selection_weight_priority(self):
        c = TrainConfig(task="hybrid", family="hybrid", model=asdict(cfg()), engineering=True, metrics=[], monitor="hybrid")
        stats = TrainStatistics(10, 2, [10]*5, [2]*5)
        a = evaluate_records({"x": record()}, {"x": {"record": record()}})
        b = evaluate_records({"x": record()}, {"x": {"record": record([{"name": "rice", "weight_g": 15}, {"name": "apple", "weight_g": 9}])}})
        self.assertTrue(is_better(selection_value({"structured": a}, c, stats), selection_value({"structured": b}, c, stats), "hybrid", c.tolerance))

    def test_checkpoint_resume_and_legacy_loading(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            c = TrainConfig(task="hybrid", family="hybrid", model=asdict(cfg()), engineering=True, metrics=[], monitor="hybrid", epochs=2, batch_size=2, patience=10, max_tokens=5)
            common = dict(data_root=root, train_annotations=root/"train.jsonl", val_annotations=root/"val.jsonl")
            run_training(c, out=root/"full", **common)
            run_training(replace(c, epochs=1), out=root/"resumed", **common)
            run_training(c, out=root/"resumed", resume=root/"resumed/last.pt", **common)
            m, v, a = restore_model(root/"full/last.pt")
            _, _, b = restore_model(root/"resumed/last.pt")
            self.assertEqual(a["progress"], b["progress"])
            for k in a["model"]:
                self.assertTrue(torch.equal(a["model"][k], b["model"][k]), k)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            self.assertTrue(torch.equal(a["rng_torch"], b["rng_torch"]))
            self.assertEqual(a["ingredient_vocabulary"], v.metadata())
            from foodlens.text import Vocabulary
            old = Vocabulary(); old.extend_training(["rice"], split="train"); old.add_sep()
            legacy = LegacyHybrid(len(old), cfg())
            save_checkpoint(root/"legacy.pt", legacy, torch.optim.AdamW(legacy.parameters()), None, vocab=old, statistics=None, metadata={"model_family": "hybrid"}, step=0)
            lm, lv, _ = restore_model(root/"legacy.pt")
            self.assertIsInstance(lm, LegacyHybrid)
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                load_checkpoint(root/"legacy.pt", m)
            source = CaptionA(len(old), cfg())
            before = m.embedding.weight.detach().clone()
            report = transfer_visual_backbone(source, m)
            self.assertTrue(torch.equal(m.embedding.weight, before))
            self.assertEqual(report["modules"], ["visual.features"])
            self.assertTrue(all(k.startswith("visual.features.") for k in report["parameters"]))


class RealDataTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[2]/"data/annotations/train.jsonl").exists(), "real data absent")
    def test_all_train_roundtrip_permutations_and_split_oov(self):
        root = Path(__file__).resolve().parents[2]/"data"
        for entry in json.loads((root/"sources.json").read_text(encoding="utf-8"))["nutrition"]["splits"]:
            self.assertEqual(sha256(root/entry["path"]), entry["sha256"])
        rows = read_jsonl(root/"annotations/train.jsonl")
        v = IngredientVocabulary.fit(rows)
        self.assertEqual(len(rows), 2338); self.assertEqual(len(v)-4, 151)
        for r in rows:
            seq, positions, weights = serialize_ingredients(r["target"], v)
            recovered = [{"name": v.tokens[c], "weight_g": w} for c, w in zip(seq[1:-1], weights)]
            self.assertEqual(recovered, canonical_items(r["target"]["items"]))
            self.assertEqual(positions, list(range(1, len(weights)+1)))
            self.assertEqual(serialize_ingredients(record(list(reversed(r["target"]["items"]))), v), (seq, positions, weights))
        for split, count, oov in (("val", 414, 5), ("test", 506, 3)):
            other = read_jsonl(root/f"annotations/{split}.jsonl")
            self.assertEqual(len(other), count)
            unseen = {i["name"] for r in other for i in canonical_items(r["target"]["items"]) if i["name"] not in v.index}
            self.assertEqual(len(unseen), oov)
        # Decode every RGB through the existing preprocessing protocol; no training.
        for split in ("train", "val", "test"):
            ds = NutritionDataset(root/f"annotations/{split}.jsonl", root, split=split)
            for i in range(len(ds)):
                self.assertEqual(ds[i]["image"].shape, (3, 224, 224))


if __name__ == "__main__":
    unittest.main(verbosity=2)
