#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def read_json_records(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            if isinstance(data, list):
                return data
            return [data]
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                return [data]
    return records


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_role(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        roles = [normalize_role(item) for item in raw]
        roles = [role for role in roles if role]
        return roles[0] if roles else None
    value = str(raw).strip().lower()
    if not value:
        return None
    if "meta" in value or "area chair" in value or value == "ac":
        return "Meta"
    if "author" in value or "rebuttal" in value:
        return "Author"
    if "reviewer" in value or value == "review":
        return "Reviewer"
    if "chair" in value or "editor" in value:
        return "Meta"
    return None


def year_from_timestamp(ts):
    if ts is None:
        return None
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return None
    if value > 1e12:
        value = value / 1000.0
    try:
        return datetime.utcfromtimestamp(value).year
    except (OSError, OverflowError, ValueError):
        return None


def turn_id_from_segment(seg_id):
    if seg_id is None:
        return None
    token = str(seg_id)
    base, _, suffix = token.rpartition("_")
    if base and suffix.isdigit():
        return base
    return token


def load_acts(acts_path, mining_path):
    if acts_path and Path(acts_path).exists():
        acts = read_json_records(acts_path)
        forum_order = []
        forum_seen = set()
        for act in acts:
            forum_id = act.get("forum_id") or act.get("forum")
            if forum_id and forum_id not in forum_seen:
                forum_order.append(forum_id)
                forum_seen.add(forum_id)
        return acts, forum_order
    records = read_json_records(mining_path)
    acts = []
    forum_order = []
    forum_seen = set()
    for record in records:
        forum_id = record.get("forum_id") or record.get("forum")
        if forum_id and forum_id not in forum_seen:
            forum_order.append(forum_id)
            forum_seen.add(forum_id)
        title = record.get("title")
        timestamp = record.get("timestamp")
        mining = (record.get("analysis") or {}).get("mining_results") or []
        for idx, item in enumerate(mining):
            act_id = f"{forum_id}#{idx:04d}" if forum_id else None
            acts.append(
                {
                    "act_id": act_id,
                    "forum_id": forum_id,
                    "title": title,
                    "timestamp": timestamp,
                    "role": item.get("role"),
                    "strategic_intent": item.get("strategic_intent"),
                    "action": item.get("action"),
                    "source_seg_ids": item.get("source_seg_ids") or [],
                }
            )
    return acts, forum_order


def load_issue_assignments(path):
    if not path:
        return {}
    mapping = {}
    for record in iter_jsonl(path):
        act_id = record.get("act_id")
        issue_id = record.get("issue_id")
        if act_id and issue_id:
            mapping[act_id] = issue_id
    return mapping


def load_forum_meta(path):
    if not path:
        return {}
    meta = {}
    for record in iter_jsonl(path):
        forum_id = record.get("forum_id") or record.get("forum")
        if not forum_id:
            continue
        meta[forum_id] = record
    return meta


def load_context_segments(contexts_dir, forum_ids):
    segments = {}
    if not contexts_dir:
        return segments
    for forum_id in forum_ids:
        if not forum_id:
            continue
        json_path = os.path.join(contexts_dir, f"{forum_id}.json")
        jsonl_path = os.path.join(contexts_dir, f"{forum_id}.jsonl")
        path = json_path if os.path.exists(json_path) else jsonl_path
        if not os.path.exists(path):
            continue
        if path.endswith(".jsonl"):
            count = 0
            for record in iter_jsonl(path):
                if "id" in record and "text" in record:
                    count += 1
            segments[forum_id] = count
        else:
            data = read_json_records(path)
            if data and isinstance(data[0], dict) and "segments" in data[0]:
                segments[forum_id] = len(data[0].get("segments") or [])
            elif data and isinstance(data[0], dict) and "id" in data[0] and "text" in data[0]:
                segments[forum_id] = len(data)
    return segments


def load_pdf_pages(pdf_dir, forum_ids):
    pages = {}
    if not pdf_dir:
        return pages
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return pages
    for forum_id in forum_ids:
        if not forum_id:
            continue
        candidates = [
            os.path.join(pdf_dir, f"{forum_id}.pdf"),
            os.path.join(pdf_dir, forum_id, f"{forum_id}.pdf"),
            os.path.join(pdf_dir, forum_id, "paper.pdf"),
        ]
        pdf_path = next((p for p in candidates if os.path.exists(p)), None)
        if not pdf_path:
            continue
        try:
            reader = PdfReader(pdf_path)
            pages[forum_id] = len(reader.pages)
        except Exception:
            continue
    return pages


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def summarize_distribution(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "count": len(values),
    }


def parse_year_range(spec):
    if not spec:
        return []
    years = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                start = int(start)
                end = int(end)
                years.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                years.append(int(part))
            except ValueError:
                continue
    return years


def load_splits(path):
    if not path:
        return None
    data = read_json_records(path)
    if len(data) == 1 and isinstance(data[0], dict) and any(key in data[0] for key in ("train", "dev", "test")):
        out = {}
        for key in ("train", "dev", "test"):
            items = data[0].get(key) or []
            out[key] = [str(item) for item in items]
        return out
    splits = defaultdict(list)
    for record in data:
        if not isinstance(record, dict):
            continue
        forum_id = record.get("forum_id") or record.get("forum")
        split = (record.get("split") or record.get("set") or record.get("partition") or "").lower()
        if not forum_id or split not in {"train", "dev", "test"}:
            continue
        splits[split].append(str(forum_id))
    return splits if splits else None


def build_default_splits(forum_order, max_forums, train_count, dev_count, test_count):
    selected = forum_order[:max_forums] if max_forums else list(forum_order)
    train = selected[:train_count]
    dev = selected[train_count : train_count + dev_count]
    test = selected[train_count + dev_count : train_count + dev_count + test_count]
    return {"train": train, "dev": dev, "test": test}


def build_temporal_splits(forum_order, forum_years, max_forums, train_count, dev_count, test_count, train_years, test_years):
    selected = forum_order[:max_forums] if max_forums else list(forum_order)
    train_years = set(train_years or [])
    test_years = set(test_years or [])
    test = [fid for fid in selected if forum_years.get(fid) in test_years]
    remaining = [fid for fid in selected if fid not in test]
    train_dev_pool = [fid for fid in remaining if forum_years.get(fid) in train_years] or remaining
    train = train_dev_pool[:train_count]
    dev = train_dev_pool[train_count : train_count + dev_count]
    if test_count and len(test) > test_count:
        test = test[:test_count]
    return {"train": train, "dev": dev, "test": test}


def format_range(years):
    if not years:
        return None
    return f"{min(years)}--{max(years)}"


def format_mean_iqr(stats, digits=1):
    if not stats:
        return None
    mean = stats["mean"]
    p25 = stats["p25"]
    p75 = stats["p75"]
    if mean is None or p25 is None or p75 is None:
        return None
    mean_fmt = f"{mean:.{digits}f}" if digits is not None else str(mean)
    p25_fmt = f"{p25:.0f}" if digits is not None else str(p25)
    p75_fmt = f"{p75:.0f}" if digits is not None else str(p75)
    return f"{mean_fmt} [p25 {p25_fmt}, p75 {p75_fmt}]"


def load_foundry_status(events_paths):
    status_by_forum = defaultdict(Counter)
    for path in events_paths:
        if not path or not os.path.exists(path):
            continue
        for record in iter_jsonl(path):
            if record.get("type") != "move":
                continue
            observation = record.get("observation") or {}
            status = observation.get("status")
            if not status:
                continue
            thread_id = record.get("thread_id") or ""
            forum_id = record.get("forum_id")
            if not forum_id and "#" in thread_id:
                forum_id = thread_id.split("#", 1)[0]
            if not forum_id:
                continue
            status_by_forum[forum_id][status] += 1
    return status_by_forum


def main():
    parser = argparse.ArgumentParser(description="Compute data stats for tab:data_stats.")
    parser.add_argument("--mining-results", default="data/raw/mining_results.jsonl")
    parser.add_argument("--acts-in", default="")
    parser.add_argument("--issue-assignments", default="")
    parser.add_argument("--forum-meta", default="")
    parser.add_argument("--contexts-dir", default="")
    parser.add_argument("--pdf-dir", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument("--split-mode", choices=["default", "temporal"], default="default")
    parser.add_argument("--max-forums", type=int, default=2000)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--dev-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--train-years", default="2018-2023")
    parser.add_argument("--test-years", default="2024-2025")
    parser.add_argument("--thread-keys", default="issue_id,issue_type,issue_cluster_id")
    parser.add_argument("--events-jsonl", action="append", default=[])
    parser.add_argument("--events-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    acts, forum_order = load_acts(args.acts_in, args.mining_results)
    issue_map = load_issue_assignments(args.issue_assignments)
    forum_meta = load_forum_meta(args.forum_meta)

    thread_keys = [key.strip() for key in args.thread_keys.split(",") if key.strip()]

    forum_stats = {}
    for act in acts:
        forum_id = act.get("forum_id") or act.get("forum")
        if not forum_id:
            continue
        stats = forum_stats.setdefault(
            forum_id,
            {
                "act_count": 0,
                "roles": Counter(),
                "turn_ids": set(),
                "thread_ids": set(),
                "timestamp": None,
            },
        )
        stats["act_count"] += 1
        if stats["timestamp"] is None and act.get("timestamp") is not None:
            stats["timestamp"] = act.get("timestamp")
        role = normalize_role(act.get("role"))
        if role:
            stats["roles"][role] += 1
        source_seg_ids = act.get("source_seg_ids") or []
        for seg_id in source_seg_ids:
            turn_id = turn_id_from_segment(seg_id)
            if turn_id:
                stats["turn_ids"].add(turn_id)
        issue_id = act.get("issue_id") or issue_map.get(act.get("act_id"))
        if issue_id:
            stats["thread_ids"].add(issue_id)
        else:
            for key in thread_keys:
                val = act.get(key)
                if val:
                    stats["thread_ids"].add(f"{forum_id}::{val}")
                    break

    forum_years = {}
    for forum_id in forum_stats:
        meta = forum_meta.get(forum_id, {})
        year = meta.get("year") or meta.get("forum_year") or meta.get("submission_year")
        if year is None:
            year = year_from_timestamp(meta.get("timestamp"))
        if year is None:
            year = year_from_timestamp(forum_stats[forum_id].get("timestamp"))
        if year is not None:
            try:
                forum_years[forum_id] = int(year)
            except (TypeError, ValueError):
                continue

    context_counts = load_context_segments(args.contexts_dir, forum_stats.keys())
    pdf_pages = load_pdf_pages(args.pdf_dir, forum_stats.keys())

    score_changes = {}
    for forum_id, meta in forum_meta.items():
        for key in ("score_change", "max_score_change", "score_change_magnitude"):
            if key in meta:
                score_changes[forum_id] = meta[key]
                break

    splits = load_splits(args.splits)
    if splits is None:
        if args.split_mode == "temporal":
            splits = build_temporal_splits(
                forum_order or list(forum_stats.keys()),
                forum_years,
                args.max_forums,
                args.train_count,
                args.dev_count,
                args.test_count,
                parse_year_range(args.train_years),
                parse_year_range(args.test_years),
            )
        else:
            splits = build_default_splits(
                forum_order or list(forum_stats.keys()),
                args.max_forums,
                args.train_count,
                args.dev_count,
                args.test_count,
            )

    events_paths = list(args.events_jsonl or [])
    if args.events_dir and os.path.exists(args.events_dir):
        for root, _, files in os.walk(args.events_dir):
            for name in files:
                if name == "events.jsonl":
                    events_paths.append(os.path.join(root, name))
    foundry_status = load_foundry_status(events_paths)

    output = {"splits": {}, "notes": []}
    for split_name, forum_ids in splits.items():
        forum_ids = [fid for fid in forum_ids if fid in forum_stats]
        act_count = sum(forum_stats[fid]["act_count"] for fid in forum_ids)
        years = [forum_years.get(fid) for fid in forum_ids if forum_years.get(fid)]
        thread_counts = [len(forum_stats[fid]["thread_ids"]) for fid in forum_ids if forum_stats[fid]["thread_ids"]]
        turn_counts = [len(forum_stats[fid]["turn_ids"]) for fid in forum_ids if forum_stats[fid]["turn_ids"]]
        seg_counts = [context_counts[fid] for fid in forum_ids if fid in context_counts]
        page_counts = [pdf_pages[fid] for fid in forum_ids if fid in pdf_pages]
        score_vals = []
        for fid in forum_ids:
            value = score_changes.get(fid)
            if value is None:
                continue
            try:
                score_vals.append(float(value))
            except (TypeError, ValueError):
                continue
        role_counts = Counter()
        for fid in forum_ids:
            role_counts.update(forum_stats[fid]["roles"])
        total_roles = sum(role_counts.values()) or 1

        status_counts = Counter()
        for fid in forum_ids:
            status_counts.update(foundry_status.get(fid, {}))
        status_total = sum(status_counts.values())
        foundry_rates = None
        if status_total:
            foundry_rates = {
                "ok": status_counts.get("ok", 0) / status_total,
                "missing": status_counts.get("missing", 0) / status_total,
                "fail": status_counts.get("fail", 0) / status_total,
                "count": status_total,
            }

        output["splits"][split_name] = {
            "forums": len(forum_ids),
            "years_covered": format_range(years),
            "instances": act_count,
            "threads_per_forum": summarize_distribution(thread_counts),
            "turns_per_forum": summarize_distribution(turn_counts),
            "segments_per_context": summarize_distribution(seg_counts),
            "pdf_pages": summarize_distribution(page_counts),
            "score_change_magnitude": summarize_distribution(score_vals),
            "role_share": {
                "Reviewer": role_counts.get("Reviewer", 0) / total_roles,
                "Author": role_counts.get("Author", 0) / total_roles,
                "Meta": role_counts.get("Meta", 0) / total_roles,
            },
            "foundry_status": foundry_rates,
            "coverage": {
                "threads": len(thread_counts),
                "turns": len(turn_counts),
                "segments": len(seg_counts),
                "pages": len(page_counts),
                "score_change": len(score_vals),
            },
        }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    else:
        print(json.dumps(output, ensure_ascii=True, indent=2))

    if args.latex:
        splits_order = ["train", "dev", "test"]
        split_data = [output["splits"].get(name, {}) for name in splits_order]
        if not all(split_data):
            return
        rows = []
        rows.append(("Forums", [str(d.get("forums")) for d in split_data]))
        rows.append(("Years covered", [d.get("years_covered") or "-" for d in split_data]))
        rows.append(("Instances $\\mathcal{A}^{(0)}$", [str(d.get("instances")) for d in split_data]))
        rows.append((
            "Threads per forum",
            [format_mean_iqr(d.get("threads_per_forum"), digits=1) or "-" for d in split_data],
        ))
        rows.append((
            "Turns per $\\mathcal{D}^{raw}$",
            [format_mean_iqr(d.get("turns_per_forum"), digits=1) or "-" for d in split_data],
        ))
        rows.append((
            "Segments per $\\mathcal{C}$",
            [format_mean_iqr(d.get("segments_per_context"), digits=0) or "-" for d in split_data],
        ))
        rows.append((
            "PDF pages",
            [format_mean_iqr(d.get("pdf_pages"), digits=1) or "-" for d in split_data],
        ))
        rows.append((
            "Score-change magnitude",
            [format_mean_iqr(d.get("score_change_magnitude"), digits=2) or "-" for d in split_data],
        ))
        rows.append((
            "Reviewer share",
            [f"{(d.get('role_share') or {}).get('Reviewer', 0):.2f}" for d in split_data],
        ))
        rows.append((
            "Author share",
            [f"{(d.get('role_share') or {}).get('Author', 0):.2f}" for d in split_data],
        ))
        rows.append((
            "Meta share",
            [f"{(d.get('role_share') or {}).get('Meta', 0):.2f}" for d in split_data],
        ))
        rows.append((
            "OkRate after Foundry",
            [f"{(d.get('foundry_status') or {}).get('ok', 0):.3f}" if d.get("foundry_status") else "-" for d in split_data],
        ))
        rows.append((
            "MissingRate after Foundry",
            [f"{(d.get('foundry_status') or {}).get('missing', 0):.3f}" if d.get("foundry_status") else "-" for d in split_data],
        ))
        rows.append((
            "FailRate after Foundry",
            [f"{(d.get('foundry_status') or {}).get('fail', 0):.3f}" if d.get("foundry_status") else "-" for d in split_data],
        ))
        for label, values in rows:
            print(f"{label} & {values[0]} & {values[1]} & {values[2]} \\\\")


if __name__ == "__main__":
    main()
