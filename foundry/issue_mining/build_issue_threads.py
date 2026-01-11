import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from foundry.utils import normalize_tool_calls, read_json_or_jsonl, slugify, stable_hash, write_jsonl


SEC_RE = re.compile(r"\bsec(?:tion)?\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
APP_RE = re.compile(r"\bappendix\s*([a-z])\b", re.IGNORECASE)
FIG_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)\b", re.IGNORECASE)
TABLE_RE = re.compile(r"\btable\.?\s*(\d+)\b", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Build issue threads from semantic acts.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out-threads", required=True, help="Output threads JSONL")
    parser.add_argument("--out-index", required=True, help="Output act->thread index JSONL")
    parser.add_argument(
        "--labels",
        default="",
        help="Optional cluster labels JSONL to map issue_type (from label_clusters.py)",
    )
    parser.add_argument(
        "--thread-by",
        choices=["ontology", "cluster", "ontology_target", "cluster_target"],
        default="ontology",
        help="Thread grouping key",
    )
    parser.add_argument(
        "--out-issues",
        default="",
        help="Optional output issues JSONL with thread_id attached",
    )
    return parser.parse_args()


def normalize_target_from_ref(ref):
    if not ref:
        return "unknown"
    sec = SEC_RE.search(ref)
    if sec:
        return f"sec_{sec.group(1)}"
    app = APP_RE.search(ref)
    if app:
        return f"appendix_{app.group(1).lower()}"
    fig = FIG_RE.search(ref)
    if fig:
        return f"fig_{fig.group(1)}"
    table = TABLE_RE.search(ref)
    if table:
        return f"table_{table.group(1)}"
    return slugify(ref)[:40]


def derive_target_key(issue):
    for call in normalize_tool_calls(issue.get("latent_tool_calls")):
        if isinstance(call, dict):
            target = call.get("target_type")
            if target:
                return slugify(target)[:40]
        elif isinstance(call, str) and call.strip():
            return slugify(call)[:40]
    refs = (issue.get("paper_span") or {}).get("refs") or []
    if refs:
        return normalize_target_from_ref(refs[0])
    grounding_ref = issue.get("grounding_ref")
    return normalize_target_from_ref(grounding_ref)


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


def resolve_issue_ontology(issue, labels):
    cluster_id = issue.get("issue_cluster_id") or issue.get("cluster_id") or "unclustered"
    issue_type = issue.get("issue_type")
    if not issue_type and cluster_id in labels:
        issue_type = labels[cluster_id].get("issue_type") or labels[cluster_id].get("label")
    issue_ontology_id = issue_type or cluster_id
    ontology_key = slugify(issue_ontology_id)[:40]
    return cluster_id, issue_type, issue_ontology_id, ontology_key


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    labels = load_labels(args.labels)

    threads = defaultdict(list)
    thread_meta = {}

    for rec in records:
        forum_id = rec.get("forum_id") or "UNKNOWN"
        cluster_id, issue_type, issue_ontology_id, ontology_key = resolve_issue_ontology(rec, labels)
        target_key = derive_target_key(rec)
        if args.thread_by in {"ontology", "ontology_target"}:
            base_key = ontology_key
        else:
            base_key = cluster_id
        if args.thread_by in {"ontology_target", "cluster_target"}:
            thread_key = f"{forum_id}|{base_key}|{target_key}"
        else:
            thread_key = f"{forum_id}|{base_key}"
        thread_id = f"th_{stable_hash(thread_key, length=12)}"
        threads[thread_id].append(rec)
        thread_meta[thread_id] = {
            "thread_id": thread_id,
            "forum_id": forum_id,
            "issue_cluster_id": cluster_id,
            "issue_type": issue_type,
            "issue_ontology_id": issue_ontology_id,
            "target_key": target_key if "target" in args.thread_by else None,
        }

    thread_records = []
    index_records = []
    for thread_id, acts in threads.items():
        meta = thread_meta[thread_id]
        role_counts = Counter()
        intent_counts = Counter()
        for act in acts:
            if act.get("role"):
                role_counts[act.get("role")] += 1
            if act.get("strategic_intent"):
                intent_counts[act.get("strategic_intent")] += 1
            index_records.append(
                {
                    "act_id": act.get("act_id") or act.get("issue_id"),
                    "thread_id": thread_id,
                    "issue_cluster_id": meta["issue_cluster_id"],
                    "issue_ontology_id": meta.get("issue_ontology_id"),
                    "issue_type": meta.get("issue_type"),
                    "forum_id": meta["forum_id"],
                    "target_key": meta.get("target_key"),
                }
            )

        thread_records.append(
            {
                **meta,
                "act_ids": [act.get("act_id") or act.get("issue_id") for act in acts],
                "role_counts": dict(role_counts),
                "intent_counts": dict(intent_counts),
            }
        )

    write_jsonl(args.out_threads, thread_records)
    write_jsonl(args.out_index, index_records)

    if args.out_issues:
        thread_by_act = {item["act_id"]: item["thread_id"] for item in index_records if item.get("act_id")}
        updated = []
        for rec in records:
            act_id = rec.get("act_id") or rec.get("issue_id")
            cluster_id, issue_type, issue_ontology_id, _ = resolve_issue_ontology(rec, labels)
            if act_id and act_id in thread_by_act:
                rec = dict(rec)
                rec["thread_id"] = thread_by_act[act_id]
                if issue_type and not rec.get("issue_type"):
                    rec["issue_type"] = issue_type
                rec["issue_ontology_id"] = issue_ontology_id
            updated.append(rec)
        write_jsonl(args.out_issues, updated)


if __name__ == "__main__":
    main()
