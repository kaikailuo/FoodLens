"""Caption repair/cache acceptance on synthetic tensors and legacy-format artifacts."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import torch
from torch import nn
from foodlens.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from foodlens.config import load_config
from foodlens.decode import predict_caption
from foodlens.generation import cache_options, checkpoint_generation, effective_cache_options
from foodlens.models import CaptionA, CaptionB, Hybrid, ModelConfig, VisualEncoder, causal_mask, grid_positions, sinusoidal
from foodlens.training import TrainConfig, parameter_groups, resume_compatible_config, run_training
from foodlens.text import Vocabulary

torch.set_num_threads(2)
TEMP = Path(__file__).resolve().parents[1] / "artifacts/caption-cache-20261124/tests"
TEMP.mkdir(parents=True, exist_ok=True)


def tiny(dropout=.2):
    return ModelConfig(backbone="tiny", hidden=32, embedding=32, heads=4,
                       layers=2, feedforward=64, dropout=dropout)


def make_model(family, config=None):
    config = config or tiny()
    if family == "A":
        return CaptionA(16, config)
    if family == "hybrid":
        return Hybrid(16, config)
    return CaptionB(16, config, global_only=family == "B-global")


class LegacyA(nn.Module):
    """The pre-fix parameter registration/forward, independent of CaptionA.decode."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = VisualEncoder(config)
        self.embedding = nn.Embedding(16, config.embedding, padding_idx=0)
        self.gru = nn.GRU(config.embedding+config.hidden, config.hidden, batch_first=True)
        self.output = nn.Linear(config.hidden, 16)

    def forward(self, images, inputs):
        _, _, z = self.visual(images)
        x = torch.cat((self.embedding(inputs), z[:, None].expand(-1, inputs.shape[1], -1)), -1)
        states, _ = self.gru(x, z[None])
        return self.output(states)


class LegacyB(nn.Module):
    def __init__(self, config, global_only):
        super().__init__()
        self.config, self.global_only = config, global_only
        self.visual = VisualEncoder(config)
        self.grid_projection = nn.Linear(512, config.hidden)
        self.embedding = nn.Embedding(16, config.hidden, padding_idx=0)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(config.hidden, config.heads,
            config.feedforward, config.dropout, batch_first=True), config.layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(config.hidden, config.heads,
            config.feedforward, config.dropout, batch_first=True), config.layers)
        self.output = nn.Linear(config.hidden, 16)

    def forward(self, images, inputs):
        grid, v, _ = self.visual(images)
        tokens = self.grid_projection(v)[:, None] if self.global_only else self.grid_projection(grid.flatten(2).transpose(1, 2))
        if not self.global_only:
            tokens = tokens + grid_positions(*grid.shape[-2:], self.config.hidden, device=grid.device, dtype=grid.dtype)
        memory = self.encoder(tokens)
        text = self.embedding(inputs) * self.config.hidden**.5
        text = text + sinusoidal(inputs.shape[1], self.config.hidden, device=text.device, dtype=text.dtype)
        return self.output(self.decoder(text, memory, tgt_mask=causal_mask(inputs.shape[1], inputs.device),
                                        tgt_key_padding_mask=inputs.eq(0)))


class LegacyHybrid(LegacyA):
    def __init__(self, config):
        super().__init__(config)
        h = config.hidden
        self.weight_head = nn.Sequential(nn.Linear(h*2, h), nn.ReLU(), nn.Linear(h, 1), nn.Softplus())
        self.total_head = nn.Sequential(nn.Linear(512+h, h), nn.ReLU(), nn.Linear(h, 5), nn.Softplus())

    def forward(self, images, inputs, boundaries):
        _, v, z = self.visual(images)
        x = torch.cat((self.embedding(inputs), z[:, None].expand(-1, inputs.shape[1], -1)), -1)
        states, _ = self.gru(x, z[None])
        selected = states.gather(1, boundaries[:, :, None].expand(-1, -1, states.shape[-1]))
        return {"logits": self.output(states), "states": states,
                "weights": self.weight_head(torch.cat((selected, z[:, None]), -1)).squeeze(-1),
                "totals": self.total_head(torch.cat((v, z), -1))}


class CaptionRepairs(unittest.TestCase):
    def setUp(self):
        seed_everything(32)
        self.images = torch.randn(3, 3, 32, 32)
        self.inputs = torch.tensor([[1, 4, 5, 6], [1, 5, 0, 7], [1, 6, 7, 8]])

    def test_embedding_dropout_actual_training_eval_and_zero(self):
        for family in ("A", "hybrid"):
            model = make_model(family, tiny(.6)).train()
            _, _, z = model.visual(self.images)
            first = model.decode(self.inputs, z)[0]
            second = model.decode(self.inputs, z)[0]
            self.assertFalse(torch.equal(first, second))
            first.sum().backward()
            self.assertGreater(float(model.gru.weight_ih_l0.grad.abs().sum()), 0)
            self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)
            model.eval()
            self.assertTrue(torch.equal(model.decode(self.inputs, z)[0], model.decode(self.inputs, z)[0]))
            zero = make_model(family, tiny(0.)).train()
            self.assertTrue(torch.equal(zero.decode(self.inputs, z)[0], zero.decode(self.inputs, z)[0]))

    def test_legacy_checkpoint_state_and_optimizer_layout(self):
        for family in ("A", "hybrid", "B", "B-global"):
            legacy = (LegacyA(tiny()) if family == "A" else LegacyHybrid(tiny()) if family == "hybrid"
                      else LegacyB(tiny(), family == "B-global"))
            model = make_model(family)
            self.assertEqual(list(legacy.state_dict()), list(model.state_dict()))
            self.assertEqual([n for n, _ in legacy.named_parameters()], [n for n, _ in model.named_parameters()])
            legacy.eval(); model.eval()
            cfg = TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[])
            old_opt = torch.optim.AdamW(parameter_groups(legacy, cfg))
            old_sched = torch.optim.lr_scheduler.StepLR(old_opt, 1)
            old_opt.zero_grad()
            def forward(m):
                return m(self.images, self.inputs, torch.full((3, 1), 2, dtype=torch.long)) if family == "hybrid" else m(self.images, self.inputs)
            outputs = forward(legacy)
            (sum(t.sum() for t in outputs.values()) if isinstance(outputs, dict) else outputs.sum()).backward()
            old_opt.step(); old_sched.step()
            with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
                path = Path(tmp)/"legacy.pt"
                save_checkpoint(path, legacy, old_opt, old_sched, vocab=Vocabulary(["<PAD>", "<BOS>", "<EOS>", "<UNK>"]+[str(i) for i in range(12)]),
                    statistics=None, metadata={"model_family": family}, step=1)
                opt = torch.optim.AdamW(parameter_groups(model, cfg))
                sched = torch.optim.lr_scheduler.StepLR(opt, 1)
                loaded = load_checkpoint(path, model, opt, sched)
                self.assertEqual(loaded["step"], 1)
                self.assertEqual(old_opt.state_dict()["param_groups"], opt.state_dict()["param_groups"])
                for key, values in old_opt.state_dict()["state"].items():
                    for name, value in values.items():
                        loaded_value = opt.state_dict()["state"][key][name]
                        self.assertTrue(torch.equal(value, loaded_value) if isinstance(value, torch.Tensor) else value == loaded_value)
                with torch.inference_mode():
                    expected, actual = forward(legacy), forward(model)
                    if family == "hybrid":
                        for name in expected:
                            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                self.assertEqual(old_sched.state_dict(), sched.state_dict())

    def test_b_projection_not_executed_and_other_gradients_survive(self):
        for global_only in (False, True):
            model = CaptionB(16, tiny(), global_only=global_only).train()
            calls = []
            hook = model.visual.projection.register_forward_hook(lambda *args: calls.append(1))
            logits = model(self.images, self.inputs)
            logits.square().mean().backward(); hook.remove()
            self.assertEqual(calls, [])
            self.assertIsNone(model.visual.projection.weight.grad)
            for parameter in (model.grid_projection.weight, model.decoder.layers[0].self_attn.in_proj_weight,
                              model.decoder.layers[0].multihead_attn.in_proj_weight):
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        a = CaptionA(16, tiny())
        calls = []
        hook = a.visual.projection.register_forward_hook(lambda *args: calls.append(1))
        a(self.images, self.inputs); hook.remove()
        self.assertEqual(len(calls), 1)

    def test_default_visual_shapes_preserved(self):
        for global_only in (False, True):
            model = CaptionB(16, ModelConfig(pretrained=False), global_only=global_only).eval()
            with torch.inference_mode():
                memory = model.memory(torch.randn(1, 3, 224, 224))
            self.assertEqual(tuple(memory.shape), (1, 1 if global_only else 49, 256))


class CacheCorrectness(unittest.TestCase):
    def test_scst_cached_policy_preserves_actual_rewards_and_gradients(self):
        from foodlens.caption_metrics import CiderDReward, build_df_cache
        from foodlens.scst import scst_loss
        seed_everything(18)
        vocab = Vocabulary()
        vocab.extend_training(["apple banana beans bread carrot corn green plate rice salad soup with"], split="train")
        references = ["rice with green beans", "apple banana carrot corn", "bread soup salad plate"]
        rows = [{"id": str(i), "split": "train", "caption": text} for i, text in enumerate(references)]
        images = torch.randn(3, 3, 32, 32)
        full, cached = make_model("A", tiny(.4)), make_model("A", tiny(.4))
        cached.load_state_dict(full.state_dict())
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path = Path(tmp)/"df.json"
            build_df_cache(rows, path, manifest_sha256="synthetic-cache-test")
            reward = CiderDReward(path, {r["id"]: [r["caption"]] for r in rows})
            results = []
            for model, enabled in ((full, False), (cached, True)):
                seed_everything(55)
                loss, diagnostics = scst_loss(model, images, ["0", "1", "2"], reward, vocab=vocab,
                                               max_tokens=9, gru_cache_enabled=enabled)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertTrue(model.training)  # caller's mode restored
                results.append((loss.detach(), diagnostics))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=2e-6)
            self.assertEqual(results[0][1], results[1][1])
            self.assertGreater(float(cached.gru.weight_ih_l0.grad.abs().sum()), 0)
            for (name, left), (_, right) in zip(full.named_parameters(), cached.named_parameters()):
                if left.grad is None:
                    self.assertIsNone(right.grad)
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_gru_step_logits_and_true_single_token_work(self):
        seed_everything(12)
        model = make_model("A").eval()
        images = torch.randn(3, 3, 32, 32)
        inputs = torch.tensor([[1, 4, 2, 0], [1, 8, 9, 5], [1, 4, 5, 6]])
        with torch.inference_mode():
            _, _, z = model.visual(images)
            hidden = None
            for step in range(inputs.shape[1]):
                logits, _, hidden = model.decode(inputs[:, step:step+1], z, hidden)
                torch.testing.assert_close(logits[:, 0], model.decode(inputs[:, :step+1], z)[0][:, -1], rtol=1e-5, atol=2e-6)
            model.output.bias[2] = -100
        lengths = []
        hook = model.gru.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
        predict_caption(model, images, max_tokens=8); full = list(lengths); lengths.clear()
        predict_caption(model, images, max_tokens=8, gru_cache_enabled=True); hook.remove()
        self.assertEqual(full, list(range(1, 9)))
        self.assertEqual(lengths, [1]*8)

    def test_kv_step_logits_padding_positions_and_cache_contents(self):
        for seed in (3, 42, 81):
            for family in ("B", "B-global"):
                seed_everything(seed)
                model = make_model(family).eval()
                inputs = torch.randint(4, 16, (3, 17)); inputs[:, 0] = 1
                inputs[0, 3:6] = 0; inputs[1, 8] = 0
                with torch.inference_mode():
                    memory = model.memory(torch.randn(3, 3, 32, 32)); cache = None
                    for step in range(inputs.shape[1]):
                        logits, cache = model.decode_step(inputs[:, step:step+1], memory, cache)
                        reference = model.decode(inputs[:, :step+1], memory)[:, -1]
                        torch.testing.assert_close(logits[:, 0], reference, rtol=1e-5, atol=2e-6)
                        self.assertEqual(cache.position, step+1)
                        self.assertTrue(torch.equal(cache.valid_keys, inputs[:, :step+1].ne(0)))
                        for kv in cache.layers:
                            self.assertEqual(tuple(kv.self_key.shape), (3, 4, step+1, 8))
                            self.assertEqual(tuple(kv.memory_key.shape), (3, 4, 1 if family == "B-global" else 49, 8))

    def test_kv_uses_original_weights_without_full_decoder_calls(self):
        model = make_model("B").eval(); inputs = torch.tensor([[1, 4, 5, 6]])
        calls = []
        hook = model.decoder.register_forward_pre_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            memory = model.memory(torch.randn(1, 3, 32, 32)); cache = None
            for step in range(4):
                _, cache = model.decode_step(inputs[:, step:step+1], memory, cache)
                if step == 0:
                    pointers = [kv.memory_key.data_ptr() for kv in cache.layers]
                self.assertEqual(pointers, [kv.memory_key.data_ptr() for kv in cache.layers])
        hook.remove()
        self.assertEqual(calls, [])
        self.assertFalse(any("cache" in name for name in model.state_dict()))

    def test_norm_first_and_final_norm_semantics(self):
        model = make_model("B").eval()
        for layer in model.decoder.layers:
            layer.norm_first = True
            layer.activation = nn.functional.gelu
        model.decoder.norm = nn.LayerNorm(32)
        with torch.inference_mode():
            memory = model.memory(torch.randn(2, 3, 32, 32)); cache = None
            inputs = torch.tensor([[1, 4, 5, 6], [1, 8, 0, 7]])
            for step in range(4):
                logits, cache = model.decode_step(inputs[:, step:step+1], memory, cache)
                torch.testing.assert_close(logits[:, 0], model.decode(inputs[:, :step+1], memory)[:, -1], rtol=1e-5, atol=2e-6)

    def test_greedy_equivalence_independent_flags_batches_and_lengths(self):
        for seed in (2, 9, 32):
            seed_everything(seed)
            for family in ("A", "B", "B-global"):
                model = make_model(family).eval()
                for batch_size in (1, 3):
                    images = torch.randn(batch_size, 3, 32, 32)
                    for maximum in (1, 4, 13):
                        expected = predict_caption(model, images, max_tokens=maximum)
                        for gru, kv in ((True, False), (False, True), (True, True)):
                            actual = predict_caption(model, images, max_tokens=maximum,
                                gru_cache_enabled=gru, transformer_kv_cache_enabled=kv)
                            self.assertTrue(torch.equal(actual, expected), (family, seed, batch_size, maximum))

    def test_mixed_eos_and_limit_all_models(self):
        for family in ("A", "B", "B-global"):
            for ends in ([1, 3, None], [1, 1, 1], [None, None, None]):
                model = make_model(family).eval(); images = torch.randn(3, 3, 32, 32)
                def generate(cached):
                    count = 0
                    def control(module, args, output):
                        nonlocal count
                        count += 1
                        out = output.clone()
                        for i, end in enumerate(ends):
                            out[i, -1].fill_(-100)
                            out[i, -1, 2 if end is not None and count >= end else 4] = 100
                        return out
                    handle = model.output.register_forward_hook(control)
                    try:
                        return predict_caption(model, images, max_tokens=6, gru_cache_enabled=cached,
                                               transformer_kv_cache_enabled=cached)
                    finally:
                        handle.remove()
                old, new = generate(False), generate(True)
                self.assertTrue(torch.equal(old, new))
                for i, end in enumerate(ends):
                    if end is not None:
                        self.assertEqual(int(new[i, end]), 2)
                        self.assertTrue(new[i, end+1:].eq(0).all())
                    else:
                        self.assertEqual(new.shape[1], 7)

    def test_cache_is_request_local_and_training_path_stays_uncached(self):
        for family in ("A", "B", "B-global"):
            model = make_model(family).eval()
            x, y = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
            flags = {"gru_cache_enabled": True, "transformer_kv_cache_enabled": True}
            predict_caption(model, x, max_tokens=3, **flags)
            expected = predict_caption(model, y, max_tokens=7)
            self.assertTrue(torch.equal(expected, predict_caption(model, y, max_tokens=7, **flags)))
            if family != "A":
                with torch.inference_mode():
                    memory = model.memory(x)
                    _, cache = model.decode_step(torch.ones(2, 1, dtype=torch.long), memory)
                    with self.assertRaisesRegex(ValueError, "another model/image"):
                        model.decode_step(torch.ones(2, 1, dtype=torch.long), model.memory(y), cache)
                model.train()
                with self.assertRaisesRegex(ValueError, "inference-only"):
                    model.decode_step(torch.ones(2, 1, dtype=torch.long), memory)
                with patch.object(model, "decode_step", side_effect=AssertionError("teacher forcing used cache")):
                    model(x, torch.tensor([[1, 4, 5], [1, 4, 6]])).sum().backward()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: GPU cache/precision checks not executed")
    def test_cuda_logits_tokens_and_peak_memory(self):
        for family in ("A", "B", "B-global"):
            model = make_model(family).cuda().eval(); x = torch.randn(2, 3, 32, 32, device="cuda")
            for dtype in ([torch.float32, torch.bfloat16] if torch.cuda.is_bf16_supported() else [torch.float32]):
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    torch.cuda.reset_peak_memory_stats()
                    expected = predict_caption(model, x, max_tokens=13)
                    actual = predict_caption(model, x, max_tokens=13, gru_cache_enabled=True, transformer_kv_cache_enabled=True)
                self.assertTrue(torch.equal(actual, expected))
                self.assertGreater(torch.cuda.max_memory_allocated(), 0)


class CacheConfiguration(unittest.TestCase):
    def test_old_configs_and_invalid_switches(self):
        self.assertEqual(cache_options(), {"gru_cache_enabled": False, "transformer_kv_cache_enabled": False})
        for value in (0, 1, "false", None):
            with self.assertRaises(ValueError):
                cache_options({"gru_cache_enabled": value})
        with self.assertRaises(TypeError):
            cache_options({"typo": False})
        old = asdict(TrainConfig(model=asdict(tiny()), engineering=True, monitor="val_loss", metrics=[]))
        old.pop("generation")
        parsed = TrainConfig(**old)
        self.assertEqual(parsed.generation, cache_options())
        changed = {**old, "epochs": 100, "generation": {"gru_cache_enabled": True}}
        self.assertEqual(resume_compatible_config(old), resume_compatible_config(changed))
        self.assertNotEqual(resume_compatible_config(old), resume_compatible_config({**changed, "lr": .5}))
        baseline_path = Path(__file__).resolve().parents[1]/"configs/baseline.json"
        legacy = json.loads(baseline_path.read_text(encoding="utf-8"))
        for key in cache_options():
            legacy["caption_generation"].pop(key, None)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path = Path(tmp)/"old.json"; path.write_text(json.dumps(legacy), encoding="utf-8")
            self.assertFalse(load_config(path)["caption_generation"]["gru_cache_enabled"])

    def test_checkpoint_generation_migration_and_effective_flags(self):
        self.assertEqual(checkpoint_generation({"generation_config": {"max_tokens": 19, "max_items": 3}}),
                         {"max_tokens": 19, "strategy": "greedy", **cache_options()})
        both = {"gru_cache_enabled": True, "transformer_kv_cache_enabled": True}
        self.assertEqual(effective_cache_options("A", both), {**both, "transformer_kv_cache_enabled": False})
        self.assertEqual(effective_cache_options("B", both), {**both, "gru_cache_enabled": False})
        self.assertEqual(effective_cache_options("hybrid", both), cache_options())

    def test_training_validation_cli_and_legacy_resume_switching(self):
        from test_engineering import fixtures
        from foodlens.cli import main
        from foodlens.data import read_jsonl
        for family in ("A", "B", "B-global"):
            with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
                root = Path(tmp); fixtures(root)
                cfg = TrainConfig(family=family, model=asdict(tiny(0.)), engineering=True,
                    monitor="val_loss", metrics=[], epochs=1, batch_size=2, max_tokens=8)
                run = root/"run"
                run_training(cfg, manifest=root/"caption.jsonl", data_root=root, out=run)
                path = run/"last.pt"
                checkpoint = torch.load(path, weights_only=False)
                checkpoint["metadata"]["train_config"].pop("generation")
                checkpoint["metadata"].pop("caption_generation")
                checkpoint["metadata"].pop("caption_a_dropout_version")
                torch.save(checkpoint, path)
                cfg.epochs = 2
                options_path = root/"train.json"; options_path.write_text(json.dumps(asdict(cfg)), encoding="utf-8")
                argv = ["foodlens", "train", "--config", str(options_path), "--manifest", str(root/"caption.jsonl"),
                        "--data-root", str(root), "--out", str(run), "--resume", str(path),
                        "--gru-cache-enabled", "--transformer-kv-cache-enabled"]
                with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                        patch("foodlens.training.predict_caption", wraps=predict_caption) as generated:
                    main()
                self.assertTrue(generated.call_args.kwargs["gru_cache_enabled"])
                self.assertTrue(generated.call_args.kwargs["transformer_kv_cache_enabled"])
                current = torch.load(path, weights_only=False)
                self.assertEqual(current["progress"]["epoch"], 2)
                self.assertTrue(current["metadata"]["caption_generation"]["gru_cache_enabled"])
                self.assertTrue(current["metadata"]["caption_generation"]["transformer_kv_cache_enabled"])
                self.assertFalse(current["metadata"]["resume_generation_options"]["transformer_kv_cache_enabled"])
                manifest = root/"rgb.jsonl"
                manifest.write_text(json.dumps({"id": "6", "image": "synthetic_6.png"})+"\n", encoding="utf-8")
                results = []
                for enabled in (False, True):
                    output = root/f"pred-{enabled}.jsonl"
                    argv = ["foodlens", "predict", "--checkpoint", str(path), "--manifest", str(manifest), "--data-root", str(root),
                        "--out", str(output), "--gru-cache-enabled" if enabled else "--no-gru-cache-enabled",
                        "--transformer-kv-cache-enabled" if enabled else "--no-transformer-kv-cache-enabled"]
                    with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                        main()
                    results.append(read_jsonl(output))
                    protocol = json.loads(Path(str(output)+".protocol.json").read_text(encoding="utf-8"))
                    self.assertEqual(protocol["effective_caption_cache"], effective_cache_options(family,
                        {"gru_cache_enabled": enabled, "transformer_kv_cache_enabled": enabled}))
                self.assertEqual(results[0], results[1])

    def test_legacy_dropout_resume_records_changed_training_semantics(self):
        from test_engineering import fixtures
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            root = Path(tmp); fixtures(root)
            cfg = TrainConfig(model=asdict(tiny(.2)), engineering=True, monitor="val_loss", metrics=[],
                              epochs=1, batch_size=2, max_tokens=8)
            run_training(cfg, manifest=root/"caption.jsonl", data_root=root, out=root/"run")
            path = root/"run/last.pt"; state = torch.load(path, weights_only=False)
            state["metadata"].pop("caption_a_dropout_version"); torch.save(state, path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_training(replace(cfg, epochs=2), manifest=root/"caption.jsonl", data_root=root, out=root/"run", resume=path)
            self.assertTrue(any("historical training trajectory" in str(w.message) for w in caught))
            self.assertTrue(json.loads((root/"run/run.json").read_text(encoding="utf-8"))["legacy_dropout_resume"])


if __name__ == "__main__":
    unittest.main()
