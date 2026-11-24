"""Exercise public CLI commands on explicit synthetic fixtures; never a quality experiment."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
from foodlens.data import sha256
from foodlens.models import ModelConfig
from foodlens.runtime import environment, write_json, write_jsonl
from foodlens.schema import TOTAL_FIELDS
from foodlens.training import TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="new independent artifacts subdirectory")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("choose a new acceptance output directory")
    out.mkdir(parents=True)
    (out/"SYNTHETIC_ONLY.txt").write_text("Random tiny models and synthetic RGB/labels. Engineering verification only. No real model quality results.\n", encoding="utf-8")
    rows, nutrition = [], {"train": [], "val": [], "test": []}
    for i in range(8):
        split = "train" if i < 4 else "val" if i < 6 else "test"
        image = f"synthetic_{i}.png"
        Image.new("RGB", (32, 40), (25*i, 50, 170)).save(out/image)
        rows.append({"id": str(i), "image": image, "caption": f"a plate of {'rice' if i%2 else 'apple'} number {i} .",
            "split": split, "official_filename": "SYNTHETIC/"+image, "sha256": sha256(out/image), "duplicate_group": str(i)})
        nutrition[split].append({"dish_id": str(i), "image": image, "split": split,
            "target": {"items": [{"name": "rice", "weight_g": 10.+i}], **dict.fromkeys(TOTAL_FIELDS, 20.+i)}})
    write_jsonl(out/"caption.jsonl", rows)
    write_jsonl(out/"rgb.jsonl", [{"id": r["id"], "image": r["image"]} for r in rows if r["split"] == "test"]
                + [{"id": "missing_rgb", "image": "intentionally_missing.png"}])
    for split, annotations in nutrition.items():
        write_jsonl(out/(split+".jsonl"), annotations)
    config = TrainConfig(engineering=True, monitor="val_loss", metrics=[], model=asdict(ModelConfig(
        backbone="tiny", hidden=16, embedding=16, layers=1, feedforward=32, dropout=.1)),
        epochs=1, batch_size=2, max_tokens=10, updates_per_epoch=1)
    commands = []
    def run(*arguments, expected=0):
        cmd = [sys.executable, "-m", "foodlens.cli", *map(str, arguments)]
        result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)
        entry = {"arguments": list(map(str, arguments)), "returncode": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr}
        commands.append(entry)
        write_json(out/"commands.json", commands)
        if result.returncode != expected:
            raise RuntimeError("CLI acceptance failed: "+str(arguments)+"\n"+result.stderr)
        print("OK", arguments[0], flush=True)
    run("audit", "--kind", "caption", "--inputs", out/"caption.jsonl", "--data-root", out, "--check-images", "--out", out/"audit.json")
    run("audit", "--kind", "nutrition", "--inputs", out/"train.jsonl", out/"val.jsonl", out/"test.jsonl", "--data-root", out,
        "--check-images", "--out", out/"nutrition_audit.json")
    run("build-df", "--manifest", out/"caption.jsonl", "--out", out/"df.json")
    for family in ("A", "B", "B-global"):
        path = out/(family+".json")
        write_json(path, asdict(replace(config, family=family)))
        run("train", "--config", path, "--manifest", out/"caption.jsonl", "--data-root", out, "--out", out/family)
        run("predict", "--checkpoint", out/family/"best.pt", "--manifest", out/"rgb.jsonl", "--data-root", out, "--out", out/(family+"-pred.jsonl"))
        run("evaluate-caption", "--manifest", out/"caption.jsonl", "--split", "test", "--predictions", out/(family+"-pred.jsonl"),
            "--metrics", "BLEU-4", "ROUGE-L", "CIDEr-D", "--out", out/(family+"-metrics.json"))
        assert len((out/(family+"-pred.jsonl")).read_text().splitlines()) == 3
    write_json(out/"A.json", asdict(replace(config, epochs=2)))
    run("train", "--config", out/"A.json", "--manifest", out/"caption.jsonl", "--data-root", out, "--out", out/"A", "--resume", out/"A/last.pt")
    for mode in ("visual",):
        name = "hybrid-"+mode
        write_json(out/(name+".json"), asdict(replace(config, task="hybrid", family="hybrid", transfer=mode, model={**config.model, "layers": 2})))
        run("train", "--config", out/(name+".json"), "--train-annotations", out/"train.jsonl", "--val-annotations", out/"val.jsonl",
            "--data-root", out, "--initialize", out/"A/best.pt", "--out", out/name)
        run("predict", "--checkpoint", out/name/"best.pt", "--manifest", out/"rgb.jsonl", "--data-root", out, "--out", out/(name+"-pred.jsonl"))
        run("evaluate", "--annotations", out/"test.jsonl", "--predictions", out/(name+"-pred.jsonl"),
            "--text-metrics", "BLEU-4", "ROUGE-L", "CIDEr-D", "--out", out/(name+"-metrics.json"))
    for method in ("ce_continue", "scst"):
        write_json(out/(method+".json"), asdict(replace(config, method=method)))
        parameters = ["train", "--config", out/(method+".json"), "--manifest", out/"caption.jsonl", "--data-root", out,
                      "--initialize", out/"A/best.pt", "--out", out/method]
        if method == "scst":
            parameters += ["--df-cache", out/"df.json"]
        run(*parameters)
    run("judge-lock", "--manifest", out/"rgb.jsonl", "--count", 2, "--out", out/"locked.jsonl")
    run("judge-prepare", "--manifest", out/"locked.jsonl", "--predictions", out/"A-pred.jsonl", out/"B-pred.jsonl", "--out", out/"blind")
    run("human-export", "--candidates", out/"blind/candidates.jsonl", "--rater", "SYNTHETIC_R1", "--out", out/"human.csv")
    write_json(out/"acceptance.json", {"purpose": "synthetic engineering acceptance only", "environment": environment(),
        "cli_commands_passed": len(commands), "caption_manifest_sha256": sha256(out/"caption.jsonl"),
        "not_executed": ["METEOR Java", "real pretrained Qwen/GPU", "external Judge service", "real course data"]})


if __name__ == "__main__":
    main()
