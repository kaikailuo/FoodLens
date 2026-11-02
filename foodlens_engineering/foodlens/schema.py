"""Strict delivery schema, normalization, and deterministic record assembly."""
import json
import math
import unicodedata

TOTAL_FIELDS = ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")


def normalize_name(name):
    return " ".join(unicodedata.normalize("NFKC", name).lower().split())


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_record(record):
    errors = []
    if not isinstance(record, dict):
        return ["root_not_object"]
    if set(record) != {"items", *TOTAL_FIELDS}:
        errors.append("root_fields")
    if not isinstance(record.get("items"), list):
        errors.append("items_not_list")
    else:
        for i, item in enumerate(record["items"]):
            if not isinstance(item, dict) or set(item) != {"name", "weight_g"}:
                errors.append(f"item_{i}_fields")
                continue
            if not isinstance(item["name"], str) or not normalize_name(item["name"]):
                errors.append(f"item_{i}_name")
            if not number(item["weight_g"]):
                errors.append(f"item_{i}_weight")
    for key in TOTAL_FIELDS:
        if not number(record.get(key)):
            errors.append(f"invalid_{key}")
    return errors


def strict_loads(raw):
    def reject(value):
        raise ValueError(f"non_JSON_constant:{value}")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_key:{key}")
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique_pairs)


def canonical_items(items):
    merged = {}
    for item in items:
        name = normalize_name(item["name"])
        merged[name] = merged.get(name, 0.0) + float(item["weight_g"])
    return [{"name": name, "weight_g": merged[name]} for name in sorted(merged)]


def assemble(items, totals):
    record = {"items": canonical_items(items)}
    record.update(zip(TOTAL_FIELDS, map(float, totals), strict=True))
    for item in record["items"]:
        item["weight_g"] = round(item["weight_g"], 1)
    for field in TOTAL_FIELDS:
        record[field] = round(record[field], 1)
    errors = validate_record(record)
    if errors:
        raise ValueError(errors)
    return record
