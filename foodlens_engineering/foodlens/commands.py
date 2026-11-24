"""CLI routing; optional model/service imports occur only in their own commands."""
import json
from pathlib import Path
from .data import read_jsonl, sha256
from .runtime import write_json


def train(args):
    from .training import load_train_config, run_training
    result = run_training(load_train_config(args.config), data_root=args.data_root, out=args.out,
        manifest=args.manifest, train_annotations=args.train_annotations, val_annotations=args.val_annotations,
        resume=args.resume, initialize=args.initialize, df_cache=args.df_cache)
    print(json.dumps(result))


def audit(args):
    from PIL import Image
    from .data import audit_caption_rows, safe_image_path
    from .schema import validate_record
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    if args.kind == "caption":
        counts = audit_caption_rows(rows)
    else:
        if not rows or any(row.get("split") not in {"train", "val", "test"} or validate_record(row["target"]) for row in rows):
            raise ValueError("invalid nutrition rows/Schema")
        if len({r["dish_id"] for r in rows}) != len(rows):
            raise ValueError("nutrition IDs repeated across original splits")
        counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}
    failures = []
    for row in rows:
        image = safe_image_path(args.data_root, row["image"])
        if args.check_images:
            try:
                if args.kind == "caption" and sha256(image) != row["sha256"]:
                    raise ValueError("image_hash_mismatch")
                with Image.open(image) as decoded:
                    decoded.verify()
            except (OSError, ValueError) as error:
                failures.append({"id": row.get("id", row.get("dish_id")), "error_type": type(error).__name__})
    result = {"kind": args.kind, "counts": counts, "input_hashes": {Path(p).name: sha256(p) for p in args.inputs},
        "images_checked": args.check_images, "image_failures": failures, "passed": not failures}
    write_json(args.out, result)
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


def build_df(args):
    from .caption_metrics import build_df_cache
    from .data import audit_caption_rows
    rows = read_jsonl(args.manifest)
    audit_caption_rows(rows)
    result = build_df_cache([r for r in rows if r["split"] == "train"], args.out, manifest_sha256=sha256(args.manifest))
    print(json.dumps({"documents": result["documents"], "ngrams": len(result["df"])}))


def evaluate_caption(args):
    from .caption_metrics import score_captions
    from .data import audit_caption_rows
    references = read_jsonl(args.manifest)
    audit_caption_rows(references)
    references = {r["id"]: [r["caption"]] for r in references if r["split"] == args.split}
    rows = read_jsonl(args.predictions)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate caption prediction IDs")
    predictions = {r["id"]: r.get("caption", "") if r.get("status", "ok") == "ok" else "" for r in rows}
    result = score_captions(references, predictions, metrics=args.metrics)
    result["failed"] = sum(r.get("status", "ok") != "ok" for r in rows if r["id"] in references)
    result["truncated"] = sum(bool(r.get("truncated")) for r in rows if r["id"] in references)
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}))


def qwen_train(args):
    from .qwen_training import load_qwen_config, load_qwen_train_config, run_qwen_training
    result = run_qwen_training(load_qwen_config(args.qwen_config), load_qwen_train_config(args.config),
        train_annotations=args.train_annotations, val_annotations=args.val_annotations,
        data_root=args.data_root, out=args.out, resume=args.resume)
    print(json.dumps(result))


def qwen_predict(args):
    from .qwen_training import load_qwen_config, qwen_predict_file
    qwen_predict_file(load_qwen_config(args.qwen_config), args.manifest, args.data_root, args.out, args.adapter)


def qwen_pair(args):
    from .qwen_training import load_qwen_config, qwen_predict_file
    config = load_qwen_config(args.qwen_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, adapter in (("original", None), ("finetuned", args.adapter)):
        qwen_predict_file(config, args.manifest, args.data_root, out / (name+".jsonl"), adapter)
    write_json(out / "pair.json", {"manifest_sha256": sha256(args.manifest),
        "qwen_config_sha256": sha256(args.qwen_config), "same_protocol": True})


def judge_lock(args):
    from .judge import lock_samples
    lock_samples(args.manifest, args.out, count=args.count, seed=args.seed)


def judge_prepare(args):
    from .judge import prepare_candidates
    prepare_candidates(args.manifest, args.predictions, args.out, seed=args.seed)


def judge_run(args):
    from .judge import JudgeConfig, run_judge
    config = JudgeConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    print(json.dumps(run_judge(config, args.candidates, args.data_root, args.out)))


def human_export(args):
    from .judge import export_human
    export_human(args.candidates, args.out, rater=args.rater, seed=args.seed)


def human_import(args):
    from .judge import import_human
    print(json.dumps(import_human(args.csv, args.candidates, args.out)))


def judge_analyze(args):
    from .judge_stats import analyze_judge
    result = analyze_judge(read_jsonl(args.candidates), read_jsonl(args.judge),
        [row for path in args.human for row in read_jsonl(path)], raters=args.raters, repeats=args.bootstrap,
        seed=args.seed, traditional=read_jsonl(args.traditional) if args.traditional else None)
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "dimensions"}))


def add_commands(commands):
    def command(name, function):
        parser = commands.add_parser(name)
        parser.set_defaults(function=function)
        return parser
    def required(parser, *names):
        for name in names:
            parser.add_argument("--"+name, required=True)
    def root(parser):
        parser.add_argument("--data-root", default="../data")
    run = command("train", train)
    required(run, "config", "out")
    for name in ("manifest", "train-annotations", "val-annotations", "resume", "initialize", "df-cache"):
        run.add_argument("--"+name)
    root(run)
    check = command("audit", audit)
    check.add_argument("--kind", choices=["caption", "nutrition"], required=True)
    check.add_argument("--inputs", nargs="+", required=True)
    check.add_argument("--check-images", action="store_true")
    required(check, "out"); root(check)
    cache = command("build-df", build_df)
    required(cache, "manifest", "out")
    metric = command("evaluate-caption", evaluate_caption)
    required(metric, "manifest", "predictions", "out")
    metric.add_argument("--split", choices=["train", "val", "test"], default="test")
    metric.add_argument("--metrics", nargs="+", choices=["BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"],
                        default=["BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"])
    qtrain = command("qwen-train", qwen_train)
    required(qtrain, "qwen-config", "config", "train-annotations", "val-annotations", "out")
    qtrain.add_argument("--resume"); root(qtrain)
    for name, function in (("qwen-predict", qwen_predict), ("qwen-pair", qwen_pair)):
        infer = command(name, function)
        required(infer, "qwen-config", "manifest", "out")
        infer.add_argument("--adapter", required=name == "qwen-pair")
        root(infer)
    lock = command("judge-lock", judge_lock)
    required(lock, "manifest", "out")
    lock.add_argument("--count", type=int, default=60)
    lock.add_argument("--seed", type=int, default=42)
    prepare = command("judge-prepare", judge_prepare)
    required(prepare, "manifest", "out")
    prepare.add_argument("--predictions", nargs="+", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    judge = commands.add_parser("judge-run", aliases=["judge"])
    judge.set_defaults(function=judge_run)
    required(judge, "config", "candidates", "out"); root(judge)
    export = command("human-export", human_export)
    required(export, "candidates", "out", "rater")
    export.add_argument("--seed", type=int, default=42)
    imp = command("human-import", human_import)
    required(imp, "csv", "candidates", "out")
    analysis = command("judge-analyze", judge_analyze)
    required(analysis, "candidates", "judge", "out")
    analysis.add_argument("--human", nargs="+", required=True)
    analysis.add_argument("--raters", nargs=2, required=True)
    analysis.add_argument("--traditional")
    analysis.add_argument("--bootstrap", type=int, default=2000)
    analysis.add_argument("--seed", type=int, default=42)
