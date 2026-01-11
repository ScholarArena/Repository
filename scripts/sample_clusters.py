import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample issues from each cluster for inspection.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSON file")
    parser.add_argument("--per-cluster", type=int, default=5, help="Samples per cluster")
    parser.add_argument("--max-clusters", type=int, default=50, help="Max clusters to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    random.seed(args.seed)

    records = read_jsonl(args.input_path)
    clusters = defaultdict(list)
    for rec in records:
        cluster_id = rec.get("cluster_id") or "unknown"
        clusters[cluster_id].append(rec)

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    selected = sorted_clusters[: args.max_clusters]

    output = []
    for cluster_id, items in selected:
        sample = items if len(items) <= args.per_cluster else random.sample(items, args.per_cluster)
        output.append(
            {
                "cluster_id": cluster_id,
                "size": len(items),
                "samples": [
                    {
                        "issue_id": rec.get("issue_id"),
                        "role": rec.get("role"),
                        "strategic_intent": rec.get("strategic_intent"),
                        "grounding_ref": rec.get("grounding_ref"),
                        "action": rec.get("action"),
                    }
                    for rec in sample
                ],
            }
        )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(
        json.dumps(output, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
