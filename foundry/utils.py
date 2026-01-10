import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def write_jsonl(path, records):
    path = Path(path)
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def stable_hash(text, length=8):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:length]


def slugify(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def map_evidence_type(tool_category, operation):
    mapping = {
        "Literature_Cross_Check": "Citation",
        "Quantitative_Analysis": "Statistical",
        "Figure_Analysis": "Figure",
        "Textual_Analysis": "Text",
        "Logical_Deduction": "Symbolic",
        "Argumentation_Validation": "Logical",
    }
    if tool_category in mapping:
        return mapping[tool_category]
    if operation:
        op = operation.lower()
        if "citation" in op or "prior" in op:
            return "Citation"
        if "table" in op or "figure" in op:
            return "Figure"
        if "test" in op or "effect" in op or "ci" in op:
            return "Statistical"
    return "Unknown"


def load_registry(path):
    path = Path(path)
    if not path.exists():
        return {"skills": []}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if isinstance(data, dict) and "skills" in data:
        return data
    if isinstance(data, dict) and not data:
        return {"skills": []}
    return {"skills": data if isinstance(data, list) else []}


def save_registry(path, registry):
    path = Path(path)
    ensure_dir(path)
    path.write_text(json.dumps(registry, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
