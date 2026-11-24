"""Two independent raters; image-cluster bootstrap rather than caption bootstrap."""
from collections import defaultdict
import math
import numpy as np
from .data import read_jsonl
from .judge import DIMENSIONS
from .runtime import require


def spearman(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    require("scipy", "Judge Spearman statistics")
    from scipy.stats import spearmanr
    result = float(spearmanr(x, y).statistic)
    return result if math.isfinite(result) else None


def weighted_kappa(x, y):
    if len(x) != len(y) or not x:
        return None
    if any(type(v) is not int or not 1 <= v <= 5 for v in list(x)+list(y)):
        raise ValueError("kappa ratings must be integer 1..5")
    observed = np.zeros((5, 5), dtype=float)
    for a, b in zip(x, y, strict=True):
        observed[a-1, b-1] += 1
    weights = (np.arange(5)[:, None]-np.arange(5)[None, :])**2 / 16
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(x)
    denom = float((weights*expected).sum())
    return 1 - float((weights*observed).sum())/denom if denom else None


def cluster_bootstrap(rows, statistic, *, repeats=2000, seed=42):
    """Rows are (image_id, x, y); resample images with all candidate rows intact."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("positive bootstrap repeats required")
    groups = defaultdict(list)
    for image_id, x, y in rows:
        groups[image_id].append((x, y))
    point = statistic([r[1] for r in rows], [r[2] for r in rows])
    ids = sorted(groups)
    values = []
    rng = np.random.default_rng(seed)
    if ids:
        for _ in range(repeats):
            sampled = [pair for i in rng.integers(len(ids), size=len(ids)) for pair in groups[ids[i]]]
            value = statistic([p[0] for p in sampled], [p[1] for p in sampled])
            if value is not None and math.isfinite(value):
                values.append(value)
    return {"estimate": point, "ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
            "pairs": len(rows), "images": len(ids), "bootstrap_valid": len(values), "bootstrap_repeats": repeats,
            "undefined": point is None, "seed": seed, "resampling_unit": "image; all candidates retained"}


def analyze_judge(candidates, judge, human_rows, *, raters, repeats=2000, seed=42, traditional=None):
    from .judge import validate_candidates, parse_scores
    validate_candidates(candidates)
    if len(raters) != 2 or len(set(raters)) != 2:
        raise ValueError("choose exactly two independent original raters")
    expected = {r["sample_id"]: r for r in candidates}
    judges, humans = {}, {}
    for row in judge:
        sample_id = row["sample_id"]
        if sample_id not in expected or sample_id in judges or row["image_id"] != expected[sample_id]["image_id"]:
            raise ValueError("unknown/duplicate/misaligned Judge sample")
        if row["status"] == "ok":
            import json
            parse_scores(json.dumps(row["scores"]))
        judges[sample_id] = row
    for row in human_rows:
        if row["rater"] not in raters:
            continue  # Adjudicator is excluded from original inter-rater agreement.
        key = (row["sample_id"], row["rater"])
        if key in humans or row["sample_id"] not in expected or row["image_id"] != expected[row["sample_id"]]["image_id"]:
            raise ValueError("unknown/duplicate/misaligned human sample")
        if set(row["scores"]) != set(DIMENSIONS) or any(type(v) is not int or not 1 <= v <= 5 for v in row["scores"].values()):
            raise ValueError("invalid human dimension scores")
        humans[key] = row
    traditional_by_id = {}
    for row in traditional or []:
        sample_id = row["sample_id"]
        if sample_id not in expected or sample_id in traditional_by_id or row["image_id"] != expected[sample_id]["image_id"]:
            raise ValueError("unknown/duplicate traditional metric sample")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row["scores"].values()):
            raise ValueError("invalid traditional metric score")
        traditional_by_id[sample_id] = row["scores"]
    result = {"samples": len(expected), "judge_successful": sum(r["status"] == "ok" for r in judges.values()),
              "judge_missing": len(expected)-len(judges), "two_human_coverage": 0,
              "raters": list(raters), "human_mean": "arithmetic mean of two independent original ratings; no adjudication replacement",
              "kappa_weights": "quadratic, fixed 1..5 categories", "dimensions": {}}
    complete = {i for i in expected if all((i, r) in humans for r in raters)}
    result["two_human_coverage"] = len(complete)/len(expected)
    for dimension in DIMENSIONS:
        human_pairs, judge_pairs = [], []
        metric_pairs = defaultdict(lambda: {"judge": [], "human": []})
        for sample_id, candidate in expected.items():
            image_id = candidate["image_id"]
            mean = None
            if sample_id in complete:
                a, b = [humans[(sample_id, r)]["scores"][dimension] for r in raters]
                human_pairs.append((image_id, a, b))
                mean = (a+b)/2
            ok = sample_id in judges and judges[sample_id]["status"] == "ok"
            if ok:
                score = judges[sample_id]["scores"][dimension]["score"]
                if mean is not None:
                    judge_pairs.append((image_id, score, mean))
            for metric, value in traditional_by_id.get(sample_id, {}).items():
                if mean is not None:
                    metric_pairs[metric]["human"].append((image_id, value, mean))
                if ok:
                    metric_pairs[metric]["judge"].append((image_id, value, score))
        result["dimensions"][dimension] = {
            "judge_human_spearman": cluster_bootstrap(judge_pairs, spearman, repeats=repeats, seed=seed),
            "human_weighted_kappa": cluster_bootstrap(human_pairs, weighted_kappa, repeats=repeats, seed=seed),
            "traditional": {metric: {source+"_spearman": cluster_bootstrap(rows, spearman, repeats=repeats, seed=seed)
                for source, rows in values.items()} for metric, values in metric_pairs.items()}}
    return result
