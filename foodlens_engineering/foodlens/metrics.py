"""All-sample task2 metrics. Missing/unparseable predictions are never dropped."""
import json
from .schema import TOTAL_FIELDS, canonical_items, normalize_name, number, strict_loads, validate_record


def _mean(values):
    return sum(values) / len(values) if values else None


def _errors(values):
    return {"mae": _mean([abs(v) for v in values]), "mse": _mean([v*v for v in values]), "count": len(values)}


def evaluate_records(truth_by_id, prediction_by_id, *, _include_subset=True):
    tp = fp = fn = json_valid = schema_valid = truncated = 0
    matched, union, macro, per_sample, valid_ids = [], [], [], [], []
    total_errors = {key: [] for key in TOTAL_FIELDS}
    for sample_id, truth in truth_by_id.items():
        if validate_record(truth):
            raise ValueError(f"invalid ground truth: {sample_id}")
        prediction = prediction_by_id.get(sample_id, {"raw": ""})
        raw = prediction.get("raw", json.dumps(prediction.get("record")))
        status, errors = "parse_failed", []
        try:
            record = strict_loads(raw)
            json_valid += 1
            errors = validate_record(record)
            status = "schema_failed" if errors else "valid"
            if not errors:
                schema_valid += 1
                valid_ids.append(sample_id)
        except (ValueError, TypeError):
            record = {}
        if not isinstance(record, dict):
            record = {}
        true_map = {item["name"]: item["weight_g"] for item in canonical_items(truth["items"])}
        pred_map, invalid_weights = {}, []
        entries = record.get("items", [])
        if not isinstance(entries, list):
            entries = []
        for item in entries:
            name = item.get("name") if isinstance(item, dict) else None
            weight = item.get("weight_g") if isinstance(item, dict) else None
            if not isinstance(name, str) or not normalize_name(name) or "<unk>" in normalize_name(name):
                invalid_weights.append(float(weight) if number(weight) else 0.0)
                continue
            name = normalize_name(name)
            pred_map[name] = pred_map.get(name, 0.0) + (float(weight) if number(weight) else 0.0)
        # Cleanup failures must remain visible as FP, even if assembled JSON is valid.
        for failure in prediction.get("failures", []):
            invalid_weights.append(float(failure["weight_g"]) if number(failure.get("weight_g")) else 0.0)
        overlap = set(true_map) & set(pred_map)
        sample_tp = len(overlap)
        sample_fp = len(set(pred_map) - set(true_map)) + len(invalid_weights)
        sample_fn = len(set(true_map) - set(pred_map))
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        denom = 2*sample_tp + sample_fp + sample_fn
        macro.append(2*sample_tp/denom if denom else 1.0)
        matched.extend(pred_map[n] - true_map[n] for n in overlap)
        union.extend(pred_map.get(n, 0) - true_map.get(n, 0) for n in set(true_map) | set(pred_map))
        union.extend(invalid_weights)
        for key in TOTAL_FIELDS:
            value = record.get(key)
            total_errors[key].append((float(value) if number(value) else 0.0) - truth[key])
        truncated += bool(prediction.get("truncated", False))
        per_sample.append({"id": sample_id, "status": status, "schema_errors": errors,
                           "tp": sample_tp, "fp": sample_fp, "fn": sample_fn})
    n = len(truth_by_id)
    result = {"samples": n, "prediction_extras": sorted(set(prediction_by_id)-set(truth_by_id)),
              "name_micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
              "name_macro_f1": _mean(macro), "tp": tp, "fp": fp, "fn": fn,
              "weight_matched": _errors(matched), "weight_union": _errors(union),
              "matched_truth_coverage": tp / (tp+fn) if tp+fn else None,
              "totals": {key: _errors(errors) for key, errors in total_errors.items()},
              "json_valid_rate": json_valid/n if n else None,
              "schema_valid_rate": schema_valid/n if n else None,
              "truncation_rate": truncated/n if n else None,
              "valid_subset_coverage": len(valid_ids)/n if n else None,
              "per_sample": per_sample}
    if _include_subset:
        result["valid_subset"] = evaluate_records({i: truth_by_id[i] for i in valid_ids},
            {i: prediction_by_id[i] for i in valid_ids}, _include_subset=False)
    return result
