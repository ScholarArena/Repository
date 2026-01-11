import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import split_intents


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def check_issues(path):
    total = 0
    with_tool = 0
    with_grounding = 0
    roles = Counter()
    intents = Counter()
    for rec in iter_jsonl(path):
        total += 1
        if rec.get("latent_tool_calls"):
            with_tool += 1
        if rec.get("grounding_ref"):
            with_grounding += 1
        role = rec.get("role") or "Unknown"
        roles[role] += 1
        for intent in split_intents(rec.get("strategic_intent")):
            intents[intent] += 1
    top_roles = ", ".join(f"{k}:{v}" for k, v in roles.most_common(5))
    top_intents = ", ".join(f"{k}:{v}" for k, v in intents.most_common(5))
    print(f"[check] issues total={total} with_tool_calls={with_tool} with_grounding_ref={with_grounding}")
    print(f"[check] roles_top={top_roles}")
    print(f"[check] intents_top={top_intents}")


def check_grounding(path):
    total = 0
    status_counts = Counter()
    for rec in iter_jsonl(path):
        total += 1
        status = (rec.get("paper_span") or {}).get("status") or "missing"
        status_counts[status] += 1
    resolved = status_counts.get("resolved", 0)
    not_required = status_counts.get("not_required", 0)
    effective = max(total - not_required, 1)
    rate = resolved / effective
    print(f"[check] issues={total} resolved={resolved} not_required={not_required} resolved_rate={rate:.3f}")
    print(f"[check] status_counts={dict(status_counts)}")


def check_clusters(path):
    cluster_counts = Counter()
    total = 0
    for rec in iter_jsonl(path):
        total += 1
        cluster_id = rec.get("issue_cluster_id") or rec.get("cluster_id") or "unclustered"
        cluster_counts[cluster_id] += 1
    sizes = list(cluster_counts.values())
    sizes_sorted = sorted(sizes)
    median = statistics.median(sizes_sorted) if sizes_sorted else 0
    print(f"[check] records={total} clusters={len(cluster_counts)} size_min={min(sizes_sorted) if sizes_sorted else 0} size_median={median} size_max={max(sizes_sorted) if sizes_sorted else 0}")
    top = ", ".join(f"{k}:{v}" for k, v in cluster_counts.most_common(5))
    print(f"[check] top_clusters={top}")


def check_label_clusters(path):
    total = 0
    parse_errors = 0
    issue_types = Counter()
    for rec in iter_jsonl(path):
        total += 1
        if rec.get("parse_error"):
            parse_errors += 1
        if rec.get("issue_type"):
            issue_types[rec["issue_type"]] += 1
    print(f"[check] clusters={total} parse_errors={parse_errors} labeled_types={len(issue_types)}")


def check_label_intents(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = data.get("canonical_intents") or []
    items = data.get("items") or []
    print(f"[check] canonical_intents={len(canonical)} mapped_items={len(items)} parse_error={data.get('parse_error')}")


def check_label_evidence(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") or []
    mapping = data.get("mapping") or {}
    pair = sum(1 for item in items if "::" in (item.get("key") or ""))
    cat = sum(1 for item in items if "::" not in (item.get("key") or ""))
    unknown = sum(1 for item in items if (item.get("evidence_type") or "") == "Unknown")
    print(f"[check] evidence_map items={len(items)} mapping={len(mapping)} pair={pair} category={cat} unknown={unknown}")


def check_ontology(dir_path):
    dir_path = Path(dir_path)
    issue = json.loads((dir_path / "issue_ontology.json").read_text(encoding="utf-8"))
    intent = json.loads((dir_path / "intent_ontology.json").read_text(encoding="utf-8"))
    evidence = json.loads((dir_path / "evidence_ontology.json").read_text(encoding="utf-8"))
    print(f"[check] issue_ontology items={len(issue.get('items', []))}")
    print(f"[check] intent_ontology items={len(intent.get('items', []))} total={intent.get('total')}")
    print(f"[check] evidence_ontology items={len(evidence.get('items', []))} total={evidence.get('total')}")


def check_threads(path):
    total = 0
    sizes = []
    for rec in iter_jsonl(path):
        total += 1
        act_ids = rec.get("act_ids") or []
        sizes.append(len(act_ids))
    median = statistics.median(sizes) if sizes else 0
    print(f"[check] threads={total} size_min={min(sizes) if sizes else 0} size_median={median} size_max={max(sizes) if sizes else 0}")


def check_primitives(registry_path, assignments_path=None, issues_path=None):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    prims = registry.get("primitives", [])
    unknown = sum(1 for p in prims if p.get("evidence_type") in (None, "", "Unknown"))
    print(f"[check] primitives={len(prims)} unknown_evidence={unknown}")
    if assignments_path:
        assigns = list(iter_jsonl(assignments_path))
        print(f"[check] assignments={len(assigns)}")
    if issues_path:
        prim_call_total = 0
        prim_call_with_id = 0
        for rec in iter_jsonl(issues_path):
            for call in rec.get("latent_tool_calls") or []:
                if isinstance(call, dict):
                    prim_call_total += 1
                    if call.get("primitive_id"):
                        prim_call_with_id += 1
        coverage = prim_call_with_id / prim_call_total if prim_call_total else 0
        print(f"[check] primitive_id_coverage={coverage:.3f}")


def check_skills(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    skills = data.get("skills", [])
    print(f"[check] skills={len(skills)}")


def check_observations(path):
    total = 0
    result_total = 0
    ok = 0
    for rec in iter_jsonl(path):
        total += 1
        results = rec.get("results") or []
        result_total += len(results)
        for res in results:
            if res.get("status") == "ok":
                ok += 1
    rate = ok / result_total if result_total else 0
    print(f"[check] observations={total} results={result_total} ok_rate={rate:.3f}")


def check_trajectories(path):
    total = sum(1 for _ in iter_jsonl(path))
    print(f"[check] trajectories={total}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline checks")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("issues")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("grounding")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("clusters")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("label_clusters")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("label_intents")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("label_evidence")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("ontology")
    s.add_argument("--dir", dest="dir_path", required=True)

    s = sub.add_parser("threads")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("primitives")
    s.add_argument("--registry", required=True)
    s.add_argument("--assignments", default="")
    s.add_argument("--issues", default="")

    s = sub.add_parser("skills")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("observations")
    s.add_argument("--in", dest="input_path", required=True)

    s = sub.add_parser("trajectories")
    s.add_argument("--in", dest="input_path", required=True)

    args = parser.parse_args()

    if args.command == "issues":
        check_issues(args.input_path)
    elif args.command == "grounding":
        check_grounding(args.input_path)
    elif args.command == "clusters":
        check_clusters(args.input_path)
    elif args.command == "label_clusters":
        check_label_clusters(args.input_path)
    elif args.command == "label_intents":
        check_label_intents(args.input_path)
    elif args.command == "label_evidence":
        check_label_evidence(args.input_path)
    elif args.command == "ontology":
        check_ontology(args.dir_path)
    elif args.command == "threads":
        check_threads(args.input_path)
    elif args.command == "primitives":
        check_primitives(args.registry, args.assignments or None, args.issues or None)
    elif args.command == "skills":
        check_skills(args.input_path)
    elif args.command == "observations":
        check_observations(args.input_path)
    elif args.command == "trajectories":
        check_trajectories(args.input_path)


if __name__ == "__main__":
    main()
