import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import (
    map_evidence_type,
    map_ref_type_to_evidence,
    normalize_tool_calls,
    now_iso,
    read_json_or_jsonl,
    split_intents,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build issue/intent/evidence ontology summaries.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory")
    parser.add_argument("--labels", default="", help="Optional cluster labels JSON/JSONL")
    parser.add_argument("--intent-map", default="", help="Optional intent mapping JSON/JSONL")
    parser.add_argument("--evidence-map", default="", help="Optional evidence mapping JSON/JSONL")
    return parser.parse_args()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def load_labels(path):
    if not path:
        return {}
    records = read_json_or_jsonl(path)
    labels = {}
    for rec in records:
        cluster_id = rec.get("cluster_id") or rec.get("issue_cluster_id")
        if not cluster_id:
            continue
        labels[cluster_id] = rec
    return labels


def load_intent_map(path):
    if not path:
        return {}
    data = read_json_or_jsonl(path)
    if len(data) == 1 and isinstance(data[0], dict) and "intent_map" in data[0]:
        mapping = data[0].get("intent_map", {})
        return {str(k): str(v) for k, v in mapping.items() if k and v}
    if len(data) == 1 and isinstance(data[0], dict) and "mapping" in data[0]:
        mapping = data[0].get("mapping", {})
        return {str(k): str(v) for k, v in mapping.items() if k and v}
    if len(data) == 1 and isinstance(data[0], dict) and "items" in data[0]:
        mapping = {}
        for item in data[0].get("items", []):
            raw = item.get("raw_intent") or item.get("intent")
            mapped = item.get("mapped_intent") or item.get("canonical_intent")
            if raw and mapped:
                mapping[str(raw)] = str(mapped)
        return mapping
    mapping = {}
    for rec in data:
        if not isinstance(rec, dict):
            continue
        raw = rec.get("raw_intent") or rec.get("intent") or rec.get("name")
        mapped = rec.get("mapped_intent") or rec.get("canonical_intent")
        if raw and mapped:
            mapping[str(raw)] = str(mapped)
    if mapping:
        return mapping
    if len(data) == 1 and isinstance(data[0], dict):
        return {str(k): str(v) for k, v in data[0].items() if k and v}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if k and v}
    return {}


def load_evidence_map(path):
    if not path:
        return {}
    data = read_json_or_jsonl(path)
    if len(data) == 1 and isinstance(data[0], dict):
        record = data[0]
        if "mapping" in record:
            mapping = record.get("mapping", {})
            return {str(k): str(v) for k, v in mapping.items() if k and v}
        if "evidence_map" in record:
            mapping = record.get("evidence_map", {})
            return {str(k): str(v) for k, v in mapping.items() if k and v}
        if "items" in record:
            mapping = {}
            for item in record.get("items", []):
                cat = item.get("tool_category")
                evidence = item.get("evidence_type") or item.get("mapped_evidence")
                if cat and evidence:
                    mapping[str(cat)] = str(evidence)
            return mapping
        return {str(k): str(v) for k, v in record.items() if k and v}
    mapping = {}
    for rec in data:
        if not isinstance(rec, dict):
            continue
        cat = rec.get("tool_category")
        evidence = rec.get("evidence_type") or rec.get("mapped_evidence")
        if cat and evidence:
            mapping[str(cat)] = str(evidence)
    return mapping


def summarize(counter, total):
    items = [{"name": k, "count": v} for k, v in counter.most_common()]
    return {"generated_at": now_iso(), "total": total, "items": items}


def count_evidence(rec, evidence_map):
    counts = Counter()
    for call in normalize_tool_calls(rec.get("latent_tool_calls")):
        if not isinstance(call, dict):
            continue
        tool_category = call.get("tool_category")
        operation = call.get("operation")
        mapped = None
        if evidence_map:
            if tool_category and operation:
                mapped = evidence_map.get(f"{tool_category}::{operation}")
            if not mapped and tool_category:
                mapped = evidence_map.get(tool_category)
        counts[mapped or map_evidence_type(tool_category, operation)] += 1
    paper_span = rec.get("paper_span") or {}
    if paper_span.get("status") != "not_required":
        for ref_type in paper_span.get("ref_types") or []:
            counts[map_ref_type_to_evidence(ref_type)] += 1
    return counts


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    total = len(records)
    labels = load_labels(args.labels)
    intent_map = load_intent_map(args.intent_map)
    evidence_map = load_evidence_map(args.evidence_map)

    raw_intent_counts = Counter()
    intent_counts = Counter()
    evidence_counts = Counter()
    tool_category_counts = Counter()
    operation_counts = Counter()
    cluster_groups = defaultdict(list)

    for rec in records:
        cluster_id = rec.get("issue_cluster_id") or rec.get("cluster_id") or "unclustered"
        cluster_groups[cluster_id].append(rec)

        for raw_intent in split_intents(rec.get("strategic_intent")):
            raw_intent_counts[raw_intent] += 1
            mapped = intent_map.get(raw_intent, raw_intent)
            intent_counts[mapped] += 1

        for call in normalize_tool_calls(rec.get("latent_tool_calls")):
            if not isinstance(call, dict):
                continue
            tool_category = call.get("tool_category")
            operation = call.get("operation")
            if tool_category:
                tool_category_counts[tool_category] += 1
            if operation:
                operation_counts[operation] += 1

        evidence_counts.update(count_evidence(rec, evidence_map))

    issue_items = []
    for cluster_id, items in cluster_groups.items():
        cluster_intents = Counter()
        cluster_evidence = Counter()
        cluster_tools = Counter()
        for rec in items:
            for raw_intent in split_intents(rec.get("strategic_intent")):
                mapped = intent_map.get(raw_intent, raw_intent)
                cluster_intents[mapped] += 1
            for call in normalize_tool_calls(rec.get("latent_tool_calls")):
                if isinstance(call, dict) and call.get("tool_category"):
                    cluster_tools[call.get("tool_category")] += 1
            cluster_evidence.update(count_evidence(rec, evidence_map))

        label = labels.get(cluster_id, {}) if labels else {}
        issue_items.append(
            {
                "cluster_id": cluster_id,
                "count": len(items),
                "issue_type": label.get("issue_type"),
                "definition": label.get("definition"),
                "intent_patterns": label.get("intent_patterns"),
                "evidence_types": label.get("evidence_types"),
                "intent_counts": dict(cluster_intents),
                "evidence_hints": dict(cluster_evidence),
                "tool_categories": dict(cluster_tools),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "issue_ontology.json", {"generated_at": now_iso(), "items": issue_items})
    write_json(output_dir / "intent_ontology.json", summarize(intent_counts, total))
    write_json(output_dir / "intent_ontology_raw.json", summarize(raw_intent_counts, total))
    write_json(output_dir / "evidence_ontology.json", summarize(evidence_counts, total))
    write_json(output_dir / "tool_categories.json", summarize(tool_category_counts, total))
    write_json(output_dir / "operations.json", summarize(operation_counts, total))


if __name__ == "__main__":
    main()
