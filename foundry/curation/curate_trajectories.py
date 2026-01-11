import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Curate trajectories using observations.")
    parser.add_argument("--issues", required=True, help="Input issues JSONL")
    parser.add_argument("--obs", required=True, help="Input observations JSONL")
    parser.add_argument("--out", required=True, help="Output trajectories JSONL")
    parser.add_argument("--thread-index", default="", help="Optional act->thread index JSONL")
    return parser.parse_args()


def label_support(results):
    for res in results:
        if res.get("status") == "ok" and res.get("payload"):
            return "supported"
    return "inconclusive"


def main():
    args = parse_args()
    issues = read_json_or_jsonl(args.issues)
    observations = read_json_or_jsonl(args.obs)
    obs_by_issue = {obs.get("issue_id"): obs for obs in observations}
    thread_by_act = {}
    if args.thread_index:
        for rec in read_json_or_jsonl(args.thread_index):
            act_id = rec.get("act_id") or rec.get("issue_id")
            thread_id = rec.get("thread_id")
            if act_id and thread_id:
                thread_by_act[act_id] = thread_id

    out = []
    for issue in issues:
        issue_id = issue.get("issue_id")
        obs = obs_by_issue.get(issue_id) or {"results": [], "failure_codes": []}
        results = obs.get("results") or []
        tool_plan = []
        for res in results:
            tool_plan.append(
                {
                    "tool": res.get("tool"),
                    "args": res.get("args"),
                }
            )
        record = {
            "issue_id": issue_id,
            "issue_cluster_id": issue.get("issue_cluster_id") or issue.get("cluster_id"),
            "thread_id": issue.get("thread_id") or thread_by_act.get(issue.get("act_id") or issue_id),
            "state": {
                "role": issue.get("role"),
                "paper_span": issue.get("paper_span"),
                "issue_state": "Open",
            },
            "intent": issue.get("strategic_intent"),
            "tool_plan": tool_plan,
            "observation": results,
            "action_guidance": issue.get("action"),
            "labels": {"support": label_support(results)},
        }
        out.append(record)

    write_jsonl(args.out, out)


if __name__ == "__main__":
    main()
