"""COCO metric implementations with an explicit project word tokenizer protocol."""
from collections import defaultdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import numpy as np
from .text import tokenize
from .runtime import require, write_json
from .third_party.ciderD_scorer import CiderScorer, precook

TOKENIZER = "foodlens.word_v1: NFKC/lowercase/word-or-punctuation/whitespace-join"
CIDER_SOURCE = "ruotianluo/cider@dbb3960165d86202ed3c417b412a000fc8e717f3"


def check_metric_dependencies(metrics):
    if set(metrics) - {"BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"}:
        raise ValueError("unknown caption metric")
    if set(metrics) - {"CIDEr-D"}:
        require("pycocoevalcap", "standard caption metrics")
    if "METEOR" in metrics:
        if shutil.which("java") is None:
            raise RuntimeError("METEOR requires Java on PATH; no metric fallback")
        import pycocoevalcap.meteor.meteor as module
        folder = Path(module.__file__).parent
        if not (folder/"meteor-1.5.jar").is_file() or not (folder/"data/paraphrase-en.gz").is_file():
            raise RuntimeError("METEOR jar/English paraphrase resources missing")


def metric_text(text):
    if not isinstance(text, str):
        raise ValueError("caption must be a string")
    return " ".join(tokenize(text))


def normalize_references(references):
    if not references or any(not isinstance(refs, list) or not refs or
                             any(not isinstance(r, str) for r in refs)
                             for refs in references.values()):
        raise ValueError("nonempty reference corpus with nonempty caption lists required")
    return {sample_id: [metric_text(text) for text in refs] for sample_id, refs in references.items()}


def build_df_cache(rows, out, *, manifest_sha256):
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("CIDEr-D reward DF must use nonempty training rows only")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate reward reference IDs")
    references = normalize_references({r["id"]: [r["caption"]] for r in rows})
    frequencies = defaultdict(int)
    for refs in references.values():
        for gram in {g for ref in refs for g in precook(ref)}:
            frequencies[gram] += 1
    cache = {"format_version": 1, "split": "train", "tokenizer": TOKENIZER,
             "source": CIDER_SOURCE, "manifest_sha256": manifest_sha256,
             "documents": len(references), "sample_ids": sorted(references),
             "df": [[list(gram), count] for gram, count in sorted(frequencies.items())]}
    write_json(out, cache)
    return cache


class CiderDReward:
    def __init__(self, cache_path, references, *, manifest_sha256=None):
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        self.references = normalize_references(references)
        if cache.get("split") != "train" or cache.get("tokenizer") != TOKENIZER or cache.get("source") != CIDER_SOURCE:
            raise ValueError("incompatible reward cache protocol")
        if set(cache["sample_ids"]) != set(references) or cache["documents"] != len(references):
            raise ValueError("reward cache reference ID mismatch")
        if manifest_sha256 is not None and cache["manifest_sha256"] != manifest_sha256:
            raise ValueError("reward cache manifest hash mismatch")
        self.cache = cache
        self.frequency = defaultdict(float, {tuple(gram): count for gram, count in cache["df"]})

    def __call__(self, sample_ids, captions):
        if len(sample_ids) != len(captions):
            raise ValueError("reward ID/caption length mismatch")
        scorer = CiderScorer()
        # Set memory-backed fixed DF after construction; never use upstream pickle loader.
        scorer.df_mode = "fixed_training_cache"
        scorer.ref_len = math.log(self.cache["documents"])
        scorer.document_frequency = self.frequency.copy()
        for sample_id, caption in zip(sample_ids, captions, strict=True):
            scorer += (metric_text(caption), self.references[sample_id])
        scores = scorer.compute_cider()
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("nonfinite CIDEr-D reward")
        return [float(x) for x in scores]


def score_captions(references, predictions, *, metrics=("BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D")):
    check_metric_dependencies(metrics)
    refs = normalize_references(references)
    ids = list(refs)
    hyps = {sample_id: [metric_text(predictions.get(sample_id, ""))] for sample_id in ids}
    allowed = {"BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"}
    if not metrics or set(metrics) - allowed:
        raise ValueError("unknown/empty caption metric selection")
    result = {"samples": len(ids), "missing": len(set(ids)-set(predictions)),
              "prediction_extras": sorted(set(predictions)-set(ids)), "scores": {},
              "per_sample": {sample_id: {} for sample_id in ids},
              "protocol": {"tokenizer": TOKENIZER, "cider_source": CIDER_SOURCE,
                           "df": "evaluated reference corpus (offline); separate training-only cache for SCST",
                           "units": "BLEU-4/METEOR/ROUGE-L: raw 0..1; CIDEr-D: raw x10, no extra x100",
                           "cider_sigma": 6.0}}
    for name in metrics:
        if name == "CIDEr-D":
            scorer = CiderScorer()
            for sample_id in ids:
                scorer += (hyps[sample_id][0], refs[sample_id])
            # Empty hypotheses are valid and retain the full reference DF.
            if any(precook(ref) for values in refs.values() for ref in values):
                score, individual = scorer.compute_score()
            else:
                score, individual = 0.0, [0.0]*len(ids)
        else:
            require("pycocoevalcap", name)
            result["protocol"]["pycocoevalcap"] = version("pycocoevalcap")
            if name == "BLEU-4":
                from pycocoevalcap.bleu.bleu import Bleu
                score, individual = Bleu(4).compute_score(refs, hyps, verbose=0)
                score, individual = score[3], individual[3]
            elif name == "ROUGE-L":
                from pycocoevalcap.rouge.rouge import Rouge
                score, individual = Rouge().compute_score(refs, hyps)
            else:
                if shutil.which("java") is None:
                    raise RuntimeError("METEOR requires Java on PATH and pycocoevalcap bundled METEOR resources; no fallback")
                from .meteor import meteor_score
                score, individual = meteor_score(refs, hyps)
        result["scores"][name] = float(score)
        for sample_id, value in zip(ids, individual, strict=True):
            result["per_sample"][sample_id][name] = float(value)
    return result


def record_text(record):
    """Formal task2 text: complete canonical JSON, or empty for invalid output."""
    from .schema import canonical_record_text, validate_record
    if validate_record(record):
        return ""
    return canonical_record_text(record)


def score_record_texts(truth, predictions, *, metrics):
    """One formal JSON-text protocol for offline, Hybrid and both Qwen models."""
    from .schema import strict_loads
    def text(prediction):
        try:
            return record_text(strict_loads(prediction["raw"]) if "raw" in prediction else prediction.get("record"))
        except (ValueError, TypeError):
            return ""
    result = score_captions({i: [record_text(r)] for i, r in truth.items()},
        {i: text(p) for i, p in predictions.items()}, metrics=metrics)
    result["protocol"]["task2_serialization"] = "foodlens.complete_json.v1: NFKC/lower/merged/sorted names; items then TOTAL_FIELDS; compact JSON; one decimal"
    result["protocol"]["invalid_prediction"] = "empty hypothesis, full reference denominator and DF retained"
    return result
