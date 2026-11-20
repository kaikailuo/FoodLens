"""Synthetic engineering fixtures only. No dataset, pretrained weights, API or CUDA."""
from dataclasses import asdict, replace
import csv
import importlib.util
import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from foodlens.caption_metrics import CiderDReward, build_df_cache, score_captions, record_text
from foodlens.checkpoint import seed_everything
from foodlens.data import collate_caption, read_jsonl, sha256, CaptionManifestDataset, safe_image_path
from foodlens.judge import (DIMENSIONS, JudgeConfig, export_human, import_human, lock_samples,
    parse_scores, prepare_candidates, run_judge, stable_id)
from foodlens.judge_stats import analyze_judge, cluster_bootstrap, spearman, weighted_kappa
from foodlens.losses import token_loss
from foodlens.models import CaptionA, Hybrid, ModelConfig, transfer_caption_a
from foodlens.qwen import QwenConfig, answer_labels, generate_qwen, inject_lora, lora_targets, QwenCollator
from foodlens.runtime import runtime, write_jsonl, reject_secrets
from foodlens.schema import TOTAL_FIELDS
from foodlens.scst import eos_mask, generate_policy, policy_loss, scst_loss
from foodlens.text import Vocabulary
from foodlens.training import TrainConfig, run_training, restore_model, is_better

torch.set_num_threads(2)
ART = Path(__file__).resolve().parents[1] / "artifacts/goal-20261120/test-temp"
ART.mkdir(parents=True, exist_ok=True)


def tiny():
    return ModelConfig(backbone="tiny", hidden=16, embedding=16, heads=4, layers=1, feedforward=32, dropout=0.1)


def record(value=10.):
    return {"items": [{"name": "brown rice", "weight_g": value}], **dict.fromkeys(TOTAL_FIELDS, value)}


def fixtures(root):
    """Make explicit synthetic RGB and labels in a disposable artifacts directory."""
    captions, nutrition = [], {"train": [], "val": [], "test": []}
    for i in range(8):
        split = "train" if i < 4 else "val" if i < 6 else "test"
        filename = f"synthetic_{i}.png"
        Image.new("RGB", (32, 36), (20*i, 100, 200-i)).save(root / filename)
        caption = f"a plate of {'brown rice' if i%2 else 'green apple'} number {i} ."
        captions.append({"id": str(i), "image": filename, "caption": caption,
            "official_filename": "SYNTHETIC/"+filename, "sha256": sha256(root/filename),
            "duplicate_group": str(i), "split": split})
        nutrition[split].append({"dish_id": str(i), "split": split, "image": filename, "target": record(10.+i)})
    write_jsonl(root / "caption.jsonl", captions)
    for split, rows in nutrition.items():
        write_jsonl(root / (split+".jsonl"), rows)
    return captions, nutrition


def tiny_qwen_components(root):
    """Real HF tokenizer/Processor/model with a synthetic vocabulary and random tiny weights."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor,
                              Qwen3VLVideoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration)
    special = ["<|endoftext|>", "<unk>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
               "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([json.dumps(record()), "user assistant Describe this RGB meal as one JSON object"],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=special))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, pad_token=special[0], unk_token=special[1],
                                 eos_token=special[3], additional_special_tokens=special[2:])
    # Deliberately minimal synthetic chat template; production uses the pinned official template.
    template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}{% for c in m['content'] %}{% if c['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    processor = Qwen3VLProcessor(tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=64),
        video_processor=Qwen3VLVideoProcessor(), chat_template=template)
    config = Qwen3VLConfig(text_config={"vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}, "pad_token_id": tokenizer.pad_token_id},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": [], "num_position_embeddings": 16},
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"), video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"), vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    return Qwen3VLForConditionalGeneration(config), processor


class PublicFoundationTests(unittest.TestCase):
    def test_config_device_and_secrets(self):
        with self.assertRaises(ValueError):
            safe_image_path(ART, str((ART/"absolute.png").resolve()))
        with self.assertRaises(ValueError):
            TrainConfig(engineering=True, monitor="val_loss", metrics=[], accumulation=0)
        with self.assertRaises(ValueError):
            TrainConfig(monitor="val_loss")
        with self.assertRaises(ValueError):
            TrainConfig(model=asdict(tiny()))
        with self.assertRaises(ValueError):
            runtime("cpu", "bf16")
        with self.assertRaises(ValueError):
            reject_secrets({"nested": {"api_key": "never-store"}})
        with self.assertRaises(ValueError):
            QwenConfig(revision="main")

    def test_caption_shift_truncation_and_pad(self):
        vocab = Vocabulary(); vocab.extend_training(["one two three four"], split="train")
        batch = collate_caption([{"id": "long", "image": torch.zeros(3, 4, 4), "caption": "one two three four"},
                                 {"id": "short", "image": torch.zeros(3, 4, 4), "caption": "one"}], vocab, 3)
        self.assertEqual(batch["truncated"], [True, False])
        self.assertEqual(batch["targets"].tolist(), [[vocab.index["one"], vocab.index["two"], 2], [vocab.index["one"], 2, 0]])
        self.assertEqual(batch["inputs"][0].tolist(), [1, vocab.index["one"], vocab.index["two"]])

    def test_epoch_resume_exact_with_accumulation(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            config = TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[],
                epochs=2, patience=10, batch_size=1, accumulation=3, max_tokens=10)
            full = root / "full"; resumed = root / "resumed"
            run_training(config, data_root=root, out=full, manifest=root/"caption.jsonl")
            run_training(replace(config, epochs=1), data_root=root, out=resumed, manifest=root/"caption.jsonl")
            run_training(config, data_root=root, out=resumed, manifest=root/"caption.jsonl", resume=resumed/"last.pt")
            a = torch.load(full/"last.pt", weights_only=False)
            b = torch.load(resumed/"last.pt", weights_only=False)
            self.assertEqual(a["progress"], b["progress"])

            self.assertEqual(a["scheduler"], b["scheduler"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertTrue(torch.equal(a["rng_torch"], b["rng_torch"]))
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            self.assertEqual(a["progress"]["step"], 4)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                run_training(replace(config, accumulation=2), data_root=root, out=resumed,
                             manifest=root/"caption.jsonl", resume=resumed/"last.pt")

    def test_hybrid_transfer_train_and_visual_ablation(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            c = TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[], epochs=1,
                            batch_size=2, max_tokens=12)
            run_training(c, data_root=root, out=root/"a", manifest=root/"caption.jsonl")
            h = replace(c, task="hybrid", family="hybrid")
            result = run_training(h, data_root=root, out=root/"h", train_annotations=root/"train.jsonl",
                                 val_annotations=root/"val.jsonl", initialize=root/"a/best.pt")
            self.assertEqual(result["step"], 2)
            source, old, _ = restore_model(root/"a/best.pt")
            model, new, ckpt = restore_model(root/"h/last.pt")
            self.assertEqual(new.tokens[:len(old)], old.tokens)
            self.assertEqual(ckpt["metadata"]["transfer"]["copied_tokens"], len(old))
            random_target = Hybrid(len(new), tiny())
            before = random_target.gru.weight_ih_l0.detach().clone()
            transfer_caption_a(source, old, random_target, new, mode="visual")
            self.assertTrue(torch.equal(before, random_target.gru.weight_ih_l0))
            self.assertTrue(torch.equal(source.visual.projection.weight, random_target.visual.projection.weight))
            self.assertTrue(is_better([.5, 1., .9], [.5, 2., 1.], "hybrid", 1e-6))

    def test_accumulation_uses_global_token_denominator(self):
        from foodlens.training import train_epoch
        from foodlens.runtime import runtime
        from foodlens.data import collate_caption
        seed_everything(9)
        vocab = Vocabulary(); vocab.extend_training(["rice apple pear"], split="train")
        config = TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[], epochs=1,
                            accumulation=2, gradient_clip=1e6)
        images = torch.randn(2, 3, 16, 16)
        samples = [{"id": str(i), "image": images[i], "caption": text} for i, text in enumerate(["rice", "apple pear rice"])]
        a = CaptionA(len(vocab), replace(tiny(), dropout=0.))
        b = CaptionA(len(vocab), replace(tiny(), dropout=0.)); b.load_state_dict(a.state_dict())
        opt_a = torch.optim.SGD(a.parameters(), lr=.01); opt_b = torch.optim.SGD(b.parameters(), lr=.01)
        device, scaler = runtime("cpu")
        train_epoch(a, [collate_caption([s], vocab) for s in samples], opt_a, scaler, device, config, None)
        batch = collate_caption(samples, vocab)
        token_loss(b(batch["images"], batch["inputs"]), batch["targets"]).backward(); opt_b.step()
        for x, y in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(x, y, atol=1e-7))


class SCSTTests(unittest.TestCase):
    def test_eos_pad_and_advantage_gradient_direction(self):
        tokens = torch.tensor([[4, 2, 0, 0], [4, 5, 2, 0]])
        self.assertEqual(eos_mask(tokens).tolist(), [[True, True, False, False], [True, True, True, False]])
        logp = torch.zeros(2, 4, requires_grad=True)
        rewards = torch.tensor([3., 0.], requires_grad=True)
        baseline = torch.tensor([1., 2.], requires_grad=True)
        policy_loss(logp, tokens, rewards, baseline).backward()
        self.assertEqual(logp.grad.tolist(), [[-1., -1., 0., 0.], [1., 1., 1., 0.]])
        self.assertIsNone(rewards.grad); self.assertIsNone(baseline.grad)
        self.assertFalse(eos_mask(torch.tensor([[2, 5, 2]])).tolist()[0][1])

    def test_fixed_df_actual_rewards_and_scst_backward(self):
        seed_everything(3)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); rows, _ = fixtures(root)
            train = rows[:4]
            cache = root/"df.json"
            build_df_cache(train, cache, manifest_sha256=sha256(root/"caption.jsonl"))
            refs = {r["id"]: [r["caption"]] for r in train}
            reward = CiderDReward(cache, refs, manifest_sha256=sha256(root/"caption.jsonl"))
            good, bad = reward(["0", "0"], [train[0]["caption"], "unrelated nonsense"])
            self.assertGreater(good, bad)
            self.assertEqual(reward(["0"], [train[0]["caption"]])[0], good)
            vocab = Vocabulary(); vocab.extend_training([r["caption"] for r in train], split="train")
            model = CaptionA(len(vocab), tiny()).eval()
            images = torch.randn(2, 3, 16, 16)
            with torch.no_grad():
                # Force a nonzero reward difference between greedy and stochastic outputs.
                model.output.bias[vocab.index["a"]] = .2
            loss, diagnostics = scst_loss(model, images, ["0", "1"], reward, vocab=vocab, max_tokens=6)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(all("advantage" in row for row in diagnostics))
            for bad_rows in ([{**train[0], "split": "val"}],):
                with self.assertRaises(ValueError):
                    build_df_cache(bad_rows, cache, manifest_sha256="x")

    def test_scst_training_and_ce_control_same_update_budget(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); rows, _ = fixtures(root)
            config = TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[],
                                 epochs=1, batch_size=2, max_tokens=6, updates_per_epoch=1)
            run_training(config, data_root=root, out=root/"ce", manifest=root/"caption.jsonl")
            build_df_cache(rows[:4], root/"df.json", manifest_sha256=sha256(root/"caption.jsonl"))
            for method in ("ce_continue", "scst"):
                result = run_training(replace(config, method=method), data_root=root, out=root/method,
                    manifest=root/"caption.jsonl", initialize=root/"ce/best.pt", df_cache=root/"df.json" if method == "scst" else None)
                self.assertEqual(result["step"], 1)


class MetricTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pycocoevalcap"), "optional standard metric dependency unavailable")
    def test_standard_metrics_and_vendored_cider_equivalence(self):
        refs = {"x": ["a green apple on plate"], "y": ["brown rice with sauce"]}
        preds = {"x": "a green apple on plate"}
        result = score_captions(refs, preds, metrics=["BLEU-4", "ROUGE-L", "CIDEr-D"])
        self.assertEqual(result["samples"], 2); self.assertEqual(result["missing"], 1)
        from pycocoevalcap.cider.cider import Cider
        reference_score, individual = Cider().compute_score(refs, {"x": [preds["x"]], "y": [""]})
        self.assertAlmostEqual(result["scores"]["CIDEr-D"], reference_score)
        self.assertEqual(record_text(record()), "brown rice")
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Java"):
                score_captions(refs, preds, metrics=["METEOR"])


class QwenProtocolTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("transformers"), "optional Qwen dependencies unavailable")
    def test_real_processor_collator_and_qwen_training_resume(self):
        from foodlens.qwen_training import run_qwen_training
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); fixtures(root)
            seed_everything(5)
            original, processor = tiny_qwen_components(root)
            base_state = {k: v.detach().clone() for k, v in original.state_dict().items()}
            samples = [{"id": "a", "image": Image.new("RGB", (16, 16)), "target": record()},
                       {"id": "b", "image": Image.new("RGB", (16, 20)), "target": record(12.)}]
            collator = QwenCollator(processor, max_length=1024)
            batch = collator(samples)
            labels = batch["labels"]
            image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self.assertTrue((labels[batch["input_ids"].eq(image_id)] == -100).all())
            self.assertTrue((labels[batch["attention_mask"].eq(0)] == -100).all())
            self.assertEqual(int(labels.eq(processor.tokenizer.eos_token_id).sum()), 2)
            self.assertGreater(batch["pixel_values"].numel(), 0)
            with self.assertRaisesRegex(ValueError, "never truncated"):
                QwenCollator(processor, max_length=10)(samples)
            original.save_pretrained(root/"tiny-base")
            processor.save_pretrained(root/"tiny-base")
            qconfig = QwenConfig(model_id=str(root/"tiny-base"), device="cpu", precision="fp32", gradient_checkpointing=False, dropout=0.,
                                 max_length=1024, max_new_tokens=4, min_pixels=16, max_pixels=64)
            c = TrainConfig(backend="qwen", task="hybrid", family="hybrid", model=asdict(tiny()),
                engineering=True, monitor="val_loss", metrics=[], batch_size=1, accumulation=2, epochs=2, patience=10)
            common = {"train_annotations": root/"train.jsonl", "val_annotations": root/"val.jsonl", "data_root": root}
            run_qwen_training(qconfig, c, out=root/"full", **common)
            run_qwen_training(qconfig, replace(c, epochs=1), out=root/"resumed", **common)
            run_qwen_training(qconfig, c, out=root/"resumed", resume=root/"resumed/last.json", **common)
            from safetensors.torch import load_file
            from foodlens.qwen_training import resolve_adapter
            full_path = resolve_adapter(root/"full/last.json")
            resumed_path = resolve_adapter(root/"resumed/last.json")
            full = load_file(full_path/"adapter_model.safetensors")
            resumed = load_file(resumed_path/"adapter_model.safetensors")
            self.assertEqual(full.keys(), resumed.keys())
            for key in full:
                self.assertTrue(torch.equal(full[key], resumed[key]), key)
            a = torch.load(full_path/"training_state.pt", weights_only=False)
            b = torch.load(resumed_path/"training_state.pt", weights_only=False)
            self.assertEqual(a["progress"], b["progress"])
            # Exercise paired inference and the frozen Judge using actual local tiny HF files.
            from types import SimpleNamespace
            from foodlens.commands import qwen_pair
            from foodlens.judge import LocalJudge
            config_path = root/"qwen.json"
            config_path.write_text(json.dumps(asdict(qconfig)), encoding="utf-8")
            write_jsonl(root/"rgb-only.jsonl", [{"id": "6", "image": "synthetic_6.png"}])
            qwen_pair(SimpleNamespace(qwen_config=config_path, adapter=root/"full/last.json",
                manifest=root/"rgb-only.jsonl", data_root=root, out=root/"pair"))
            for name in ("original", "finetuned"):
                prediction = read_jsonl(root/"pair"/(name+".jsonl"))
                self.assertEqual([row["id"] for row in prediction], ["6"])
                self.assertIn("raw", prediction[0])
            backend = LocalJudge(JudgeConfig(backend="local", qwen_config=str(config_path)))
            self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
            candidate = {"sample_id": stable_id("6", "anonymous"), "image_id": "6", "image": "synthetic_6.png",
                         "candidate_id": "anonymous", "caption": "a plate of rice", "status": "ready"}
            write_jsonl(root/"judge-candidate.jsonl", [candidate])
            summary = run_judge(JudgeConfig(backend="local", qwen_config=str(config_path)),
                root/"judge-candidate.jsonl", root, root/"tiny-judge.jsonl", backend=backend)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["successful"], 0)
            output = read_jsonl(root/"tiny-judge.jsonl")[0]
            self.assertIsNotNone(output["raw"])
            self.assertIsNone(output["scores"])
            with (full_path/"adapter_model.safetensors").open("ab") as stream:
                stream.write(b"synthetic_corruption_test")
            with self.assertRaisesRegex(ValueError, "incomplete or changed"):
                resolve_adapter(root/"full/last.json")

    def test_answer_mask_exact_prefix_padding_and_end(self):
        ids = torch.tensor([[0, 0, 10, 60, 11, 12, 40, 41, 2], [10, 60, 60, 11, 12, 40, 2, 0, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0]])
        labels = answer_labels(ids, mask, [[10, 60, 11, 12], [10, 60, 60, 11, 12]], end_id=2, forbidden_ids=[60])
        self.assertEqual(labels.tolist(), [[-100]*6+[40, 41, 2], [-100]*5+[40, 2, -100, -100]])
        with self.assertRaises(ValueError):
            answer_labels(ids, mask, [[10, 11], [10]], end_id=2)
        with self.assertRaises(ValueError):
            answer_labels(torch.tensor([[10, 60, 2]]), torch.ones(1, 3), [[10]], end_id=2, forbidden_ids=[60])
        self.assertEqual(list(inspect.signature(generate_qwen).parameters), ["model", "processor", "rgb_images", "config", "instruction"])

    @unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("transformers"), "optional Qwen dependencies unavailable")
    def test_real_tiny_qwen_language_lora_scope_gradient_and_roundtrip(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        config = Qwen3VLConfig(text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}, "pad_token_id": 0},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 32,
                "deepstack_visual_indexes": [], "num_position_embeddings": 16},
            image_token_id=60, video_token_id=63, vision_start_token_id=61, vision_end_token_id=62)
        seed_everything(3)
        base = Qwen3VLForConditionalGeneration(config)
        self.assertEqual(len(lora_targets(base)), 4)
        base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        qconfig = QwenConfig(device="cpu", precision="fp32", gradient_checkpointing=False, dropout=0.)
        model, scope = inject_lora(base, qconfig)
        self.assertTrue(all("language_model" in r["name"] and "visual" not in r["name"] for r in scope["trainable"]))
        ids = torch.tensor([[1, 61, 60, 60, 60, 60, 62, 5, 6, 2]])
        labels = torch.full_like(ids, -100); labels[:, -2:] = ids[:, -2:]
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
                  "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
        loss = model(**kwargs).loss
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        grads = {n: p.grad for n, p in model.named_parameters()}
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n))
        self.assertTrue(all(g is None for n, g in grads.items() if "visual" in n))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.step(); model.eval()
        with torch.no_grad():
            expected = model(**kwargs).logits
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            model.save_pretrained(tmp)
            fresh = Qwen3VLForConditionalGeneration(config); fresh.load_state_dict(base_state)
            loaded = PeftModel.from_pretrained(fresh, tmp).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, loaded(**kwargs).logits))


class JudgeTests(unittest.TestCase):
    def test_strict_score_parser(self):
        good = {dimension: {"score": 3, "evidence": "A plate is visible."} for dimension in DIMENSIONS}
        self.assertEqual(parse_scores(json.dumps(good)), good)
        for bad in ("", "```json\n{}\n```", "{}", json.dumps({**good, "model": "hidden"})):
            with self.assertRaises((ValueError, TypeError)):
                parse_scores(bad)
        for score in (True, 0, 6, 3.5, "3"):
            value = {**good, DIMENSIONS[0]: {"score": score, "evidence": "visible"}}
            with self.assertRaises(ValueError):
                parse_scores(json.dumps(value))

    def test_anonymity_human_roundtrip_and_backend_failure_denominator(self):
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            root = Path(tmp); rows, _ = fixtures(root)
            write_jsonl(root/"rgb.jsonl", [{"id": r["id"], "image": r["image"]} for r in rows])
            lock_samples(root/"rgb.jsonl", root/"locked.jsonl", count=3)
            for name in ("modelA", "modelB"):
                write_jsonl(root/(name+".jsonl"), [{"id": r["id"], "caption": "a plate of rice", "status": "ok"} for r in rows])
            candidates = prepare_candidates(root/"locked.jsonl", [root/"modelA.jsonl", root/"modelB.jsonl"], root/"blind")
            self.assertNotIn("modelA", (root/"blind/candidates.jsonl").read_text())
            class BoundaryBackend:
                version = {"backend": "synthetic_service_boundary_only"}
                calls = 0
                def score(self, image, caption):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("external service unavailable")
                    return (json.dumps({d: {"score": 3, "evidence": "Synthetic test image."} for d in DIMENSIONS})
                            if self.calls != 3 else "invalid"), {}
            summary = run_judge(JudgeConfig(backend="api", service_revision="synthetic-boundary-test"),
                root/"blind/candidates.jsonl", root, root/"judge.jsonl", backend=BoundaryBackend())
            self.assertEqual(summary["samples"], 6); self.assertEqual(summary["successful"], 4)
            self.assertEqual(len(read_jsonl(root/"judge.jsonl")), 6)
            for rater in ("R1", "R2"):
                export_human(root/"blind/candidates.jsonl", root/(rater+".csv"), rater=rater)
                with (root/(rater+".csv")).open(encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f); fields = reader.fieldnames; sheet = list(reader)
                self.assertNotIn("candidate_id", fields)
                for row in sheet:
                    row.update({d: "3" for d in DIMENSIONS})
                with (root/(rater+".csv")).open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(sheet)
                imported = import_human(root/(rater+".csv"), root/"blind/candidates.jsonl", root/(rater+".jsonl"))
                self.assertEqual(imported["ratings"], 6)
            humans = read_jsonl(root/"R1.jsonl")+read_jsonl(root/"R2.jsonl")
            result = analyze_judge(candidates, read_jsonl(root/"judge.jsonl"), humans, raters=["R1", "R2"], repeats=10)
            self.assertEqual(result["judge_successful"], 4)
            self.assertIsNone(result["dimensions"][DIMENSIONS[0]]["judge_human_spearman"]["estimate"])
            partial_humans = [r for r in humans if r["sample_id"] != candidates[0]["sample_id"]]
            traditional = [{"sample_id": r["sample_id"], "image_id": r["image_id"], "scores": {"BLEU-4": i/10}}
                           for i, r in enumerate(candidates)]
            partial = analyze_judge(candidates, read_jsonl(root/"judge.jsonl"), partial_humans,
                                    raters=["R1", "R2"], repeats=10, traditional=traditional)
            pairs = partial["dimensions"][DIMENSIONS[0]]["traditional"]["BLEU-4"]
            self.assertEqual(pairs["judge_spearman"]["pairs"], 4)
            self.assertEqual(pairs["human_spearman"]["pairs"], 5)

    def test_rank_kappa_and_image_cluster_bootstrap(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.)
        self.assertEqual(weighted_kappa([1, 2, 3], [1, 2, 3]), 1.)
        self.assertIsNone(weighted_kappa([3, 3], [3, 3]))
        self.assertAlmostEqual(weighted_kappa([1, 2, 3], [2, 2, 3]), 2/3)
        rows = [("i1", 1, 2), ("i1", 2, 1), ("i2", 3, 3), ("i2", 4, 4)]
        a = cluster_bootstrap(rows, spearman, repeats=30, seed=7)
        b = cluster_bootstrap(rows, spearman, repeats=30, seed=7)
        self.assertEqual(a, b); self.assertEqual(a["images"], 2)
        self.assertEqual(a["pairs"], 4)
        unequal = [("i1", 1, 1), ("i2", 2, 2), ("i2", 3, 3), ("i2", 4, 4)]
        clustered = cluster_bootstrap(unequal, lambda x, y: len(x), repeats=100, seed=7)
        self.assertEqual(clustered["ci95"], [2., 6.])

    def test_api_missing_configuration_never_returns_default_scores(self):
        from foodlens.judge import APIJudge
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "FOODLENS_JUDGE_API_KEY"):
                APIJudge(JudgeConfig(backend="api", service_revision="synthetic-interface-check"))


if __name__ == "__main__":
    unittest.main()
