import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Flatten mining_results into issue-level JSONL.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to mining_results.jsonl")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    out = []
    for paper in records:
        forum_id = paper.get("forum_id") or paper.get("forum") or "UNKNOWN"
        title = paper.get("title") or ""
        timestamp = paper.get("timestamp")
        mining = (paper.get("analysis") or {}).get("mining_results") or []
        for idx, item in enumerate(mining):
            issue_id = f"{forum_id}#{idx:04d}"
            out.append(
                {
                    "issue_id": issue_id,
                    "forum_id": forum_id,
                    "title": title,
                    "timestamp": timestamp,
                    "role": item.get("role"),
                    "source_seg_ids": item.get("source_seg_ids") or [],
                    "grounding_ref": item.get("grounding_ref"),
                    "paper_span": None,
                    "strategic_intent": item.get("strategic_intent"),
                    "action": item.get("action"),
                    "latent_tool_calls": item.get("latent_tool_calls") or [],
                    "issue_type": None,
                    "cluster_id": None,
                    "meta": {"cognitive_chain": item.get("cognitive_chain")},
                }
            )
    write_jsonl(args.output_path, out)


if __name__ == "__main__":
    main()
