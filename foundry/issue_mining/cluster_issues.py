import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, stable_hash, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster issues by tool call signatures.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL")
    return parser.parse_args()


def issue_key(issue):
    calls = issue.get("latent_tool_calls") or []
    if calls:
        parts = []
        for call in calls:
            op = call.get("operation") or ""
            target = call.get("target_type") or ""
            outcome = call.get("outcome") or ""
            parts.append(f"{op}|{target}|{outcome}")
        return "||".join(parts)
    intent = issue.get("strategic_intent") or ""
    action = (issue.get("action") or "")[:120]
    return f"{intent}|{action}"


def derive_issue_type(issue):
    intent = issue.get("strategic_intent")
    if intent:
        return intent
    calls = issue.get("latent_tool_calls") or []
    if calls:
        return calls[0].get("operation") or "UNKNOWN"
    return "UNKNOWN"


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    out = []
    for rec in records:
        key = issue_key(rec)
        rec["cluster_id"] = stable_hash(key)
        rec["issue_type"] = derive_issue_type(rec)
        meta = rec.get("meta") or {}
        meta["cluster_key"] = key
        rec["meta"] = meta
        out.append(rec)
    write_jsonl(args.output_path, out)


if __name__ == "__main__":
    main()
