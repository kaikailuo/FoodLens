"""Compare identical greedy outputs, latency and CUDA peaks with cache off/on.

Without --checkpoint this constructs a random tiny model and synthetic image
tensors. Its numbers are engineering microbenchmarks, never caption quality.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import statistics
import time
import torch
from foodlens.checkpoint import seed_everything
from foodlens.data import sha256
from foodlens.decode import predict_caption
from foodlens.models import ModelConfig
from foodlens.runtime import autocast, environment, runtime, write_json
from foodlens.training import build_model, restore_model


def _percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    lower = int(position); upper = min(lower+1, len(values)-1)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def benchmark(model, images, *, max_tokens, device, precision, warmup, repeats):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def generate(enabled):
        with autocast(device, precision):
            return predict_caption(model, images, max_tokens=max_tokens,
                gru_cache_enabled=enabled, transformer_kv_cache_enabled=enabled)
    for _ in range(warmup):
        generate(False); generate(True)
    expected = generate(False)
    if not torch.equal(expected, generate(True)):
        raise RuntimeError("cache changed greedy tokens; performance results rejected")
    results = {False: [], True: []}
    peaks = {False: [], True: []}
    deltas = {False: [], True: []}
    for repetition in range(repeats):
        # Alternate order to reduce systematic warmup/clock bias.
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            synchronize()
            if device.type == "cuda":
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output = generate(enabled)
            synchronize()
            results[enabled].append((time.perf_counter()-start)*1000)
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device)
                peaks[enabled].append(peak)
                deltas[enabled].append(peak-baseline)
            if not torch.equal(output, expected):
                raise RuntimeError("cache/reference token mismatch during measured repetition")
            del output
    report = {"tokens_equal": True, "batch": len(images), "max_tokens": max_tokens,
              "returned_sequence_length_including_bos": expected.shape[1], "warmup": warmup,
              "repeats": repeats, "device": str(device), "precision": precision,
              "cuda_peak_status": "measured" if device.type == "cuda" else "not_measured_no_cuda",
              "cache_modes": {}}
    for enabled in (False, True):
        report["cache_modes"]["enabled" if enabled else "disabled"] = {
            "latency_ms_median": statistics.median(results[enabled]),
            "latency_ms_p95": _percentile(results[enabled], .95), "latency_ms_samples": results[enabled],
            "cuda_peak_allocated_bytes": max(peaks[enabled]) if peaks[enabled] else None,
            "cuda_peak_delta_bytes": max(deltas[enabled]) if deltas[enabled] else None}
    report["median_speedup"] = statistics.median(results[False])/statistics.median(results[True])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["A", "B", "B-global"], default="A")
    parser.add_argument("--checkpoint", help="trusted local Caption checkpoint; family comes from its metadata")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if any(v < 1 for v in (args.batch_size, args.max_tokens, args.image_size, args.repeats, args.threads)) or args.warmup < 0:
        parser.error("positive sizes/repeats/threads and nonnegative warmup required")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device, _ = runtime(args.device, args.precision)
    if args.checkpoint:
        model, _, checkpoint = restore_model(args.checkpoint, device)
        family = checkpoint["metadata"].get("model_family")
        if family not in {"A", "B", "B-global"}:
            raise ValueError("benchmark requires a Caption A/B/B-global checkpoint")
    else:
        family = args.family
        config = ModelConfig(backbone="tiny", pretrained=False, hidden=32, embedding=32,
                             layers=2, feedforward=128, dropout=.1)
        model = build_model(family, 128, config).to(device)
        with torch.no_grad():
            model.output.bias[2] = -100  # explicit synthetic full-length benchmark, not model prediction quality
    model.eval()
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    report = benchmark(model, images, max_tokens=args.max_tokens, device=device, precision=args.precision,
                       warmup=args.warmup, repeats=args.repeats)
    report.update({"family": family, "model_config": asdict(model.config), "seed": args.seed,
        "threads": args.threads, "image_size": args.image_size, "environment": environment(),
        "purpose": "synthetic image tensor engineering benchmark; no model quality conclusion",
        "synthetic_random_model": not bool(args.checkpoint), "synthetic_eos_suppressed": not bool(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None})
    write_json(args.out, report)
    print(f"{family}: tokens_equal=True, median speedup={report['median_speedup']:.3f}, "
          f"CUDA peak status={report['cuda_peak_status']}")


if __name__ == "__main__":
    main()
