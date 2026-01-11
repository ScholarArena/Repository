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


def normalize_tool_calls(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (dict, str)):
                cleaned.append(item)
            else:
                cleaned.append(str(item))
        return cleaned
    return [str(value)]


INTENT_SPLIT_RE = re.compile(r"\s*(?:;|/|,|\||\band\b|&)\s*", re.IGNORECASE)


def split_intents(value):
    if value is None:
        return []
    items = []
    if isinstance(value, (list, tuple, set)):
        items.extend(value)
    else:
        items.append(value)
    tokens = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            item = item.get("intent") or item.get("strategic_intent") or str(item)
        text = str(item).strip()
        if not text:
            continue
        parts = INTENT_SPLIT_RE.split(text)
        tokens.extend([part.strip() for part in parts if part and part.strip()])
    return tokens


def map_evidence_type(tool_category, operation):
    mapping = {
        "Literature_Cross_Check": "Citation",
        "Quantitative_Analysis": "Statistical",
        "Figure_Analysis": "Figure",
        "Textual_Analysis": "Text",
        "Logical_Deduction": "Symbolic",
        "Argumentation_Validation": "Logical",
        "Contextual_Clarification": "Implementation",
        "Experimental_Addition": "Implementation",
        "Methodology_Comparison": "Implementation",
        "Performance_Analysis": "Statistical",
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


def map_ref_type_to_evidence(ref_type):
    mapping = {
        "figure": "Figure",
        "table": "Table",
        "appendix": "Document",
        "section": "Document",
        "abstract": "Document",
        "related_work": "Document",
        "substring": "Text",
    }
    return mapping.get(ref_type, "Unknown")


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
