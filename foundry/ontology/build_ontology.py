import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import map_evidence_type, now_iso, read_json_or_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Build ontology summaries from issues.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory")
    return parser.parse_args()


def summarize(counter, total):
    items = [{"name": k, "count": v} for k, v in counter.most_common()]
    return {"generated_at": now_iso(), "total": total, "items": items}


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        __import__("json").dumps(data, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    total = len(records)

    intent_counter = Counter()
    issue_counter = Counter()
    tool_category_counter = Counter()
    operation_counter = Counter()
    evidence_counter = Counter()

    for rec in records:
        intent = rec.get("strategic_intent")
        if intent:
            intent_counter[intent] += 1
        issue_type = rec.get("issue_type")
        if issue_type:
            issue_counter[issue_type] += 1
        for call in rec.get("latent_tool_calls") or []:
            tool_category = call.get("tool_category")
            operation = call.get("operation")
            if tool_category:
                tool_category_counter[tool_category] += 1
            if operation:
                operation_counter[operation] += 1
            evidence_counter[map_evidence_type(tool_category, operation)] += 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "intent_types.json", summarize(intent_counter, total))
    write_json(out_dir / "issue_types.json", summarize(issue_counter, total))
    write_json(out_dir / "tool_categories.json", summarize(tool_category_counter, total))
    write_json(out_dir / "operations.json", summarize(operation_counter, total))
    write_json(out_dir / "evidence_types.json", summarize(evidence_counter, total))


if __name__ == "__main__":
    main()
