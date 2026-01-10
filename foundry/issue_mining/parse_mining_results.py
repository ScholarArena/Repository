import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Flatten mining_results into issue-level JSONL.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to mining_results.jsonl")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL path")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log interval (papers)")
    parser.add_argument("--quiet", action="store_true", help="Disable progress and summary logs")
    return parser.parse_args()


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    total_papers = len(records)
    total_issues = 0
    empty_mining = 0
    missing_forum = 0
    issues_with_tool_calls = 0
    issues_with_grounding_ref = 0
    role_counts = {}
    intent_counts = {}
    out = []
    for idx_paper, paper in enumerate(records, start=1):
        forum_id = paper.get("forum_id") or paper.get("forum")
        if not forum_id:
            missing_forum += 1
            forum_id = "UNKNOWN"
        title = paper.get("title") or ""
        timestamp = paper.get("timestamp")
        mining = (paper.get("analysis") or {}).get("mining_results") or []
        if not mining:
            empty_mining += 1
        for idx, item in enumerate(mining):
            issue_id = f"{forum_id}#{idx:04d}"
            total_issues += 1
            role = item.get("role")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
            intent = item.get("strategic_intent")
            if intent:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if item.get("latent_tool_calls"):
                issues_with_tool_calls += 1
            if item.get("grounding_ref"):
                issues_with_grounding_ref += 1
            out.append(
                {
                    "issue_id": issue_id,
                    "forum_id": forum_id,
                    "title": title,
                    "timestamp": timestamp,
                    "role": role,
                    "source_seg_ids": item.get("source_seg_ids") or [],
                    "grounding_ref": item.get("grounding_ref"),
                    "paper_span": None,
                    "strategic_intent": intent,
                    "action": item.get("action"),
                    "latent_tool_calls": item.get("latent_tool_calls") or [],
                    "issue_type": None,
                    "cluster_id": None,
                    "meta": {"cognitive_chain": item.get("cognitive_chain")},
                }
            )
        if not args.quiet and args.log_every > 0 and idx_paper % args.log_every == 0:
            print(
                f"[parse] {idx_paper}/{total_papers} papers | issues={total_issues}",
                file=sys.stderr,
            )
    write_jsonl(args.output_path, out)
    if not args.quiet:
        print(
            f"[summary] papers={total_papers} issues={total_issues} "
            f"empty_mining={empty_mining} missing_forum_id={missing_forum} "
            f"with_tool_calls={issues_with_tool_calls} with_grounding_ref={issues_with_grounding_ref}",
            file=sys.stderr,
        )
        if role_counts:
            role_summary = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
            print(f"[summary] roles: {role_summary}", file=sys.stderr)
        if intent_counts:
            top_intents = sorted(intent_counts.items(), key=lambda x: (-x[1], x[0]))[:12]
            intent_summary = ", ".join(f"{k}:{v}" for k, v in top_intents)
            print(f"[summary] intents(top12): {intent_summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
