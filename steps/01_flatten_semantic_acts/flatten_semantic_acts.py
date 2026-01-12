import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib import request as url_request

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing numpy. Install it to run clustering.") from exc


ROLE_SPLIT_RE = re.compile(r"\s*(?:,|/|;|\||\band\b|&)\s*", re.IGNORECASE)


def read_json_or_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def normalize_role_token(token):
    if token is None:
        return None
    cleaned = str(token).strip()
    if not cleaned:
        return None
    lowered = re.sub(r"[_-]+", " ", cleaned).strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    if lowered in {"ac", "area chair"} or "area chair" in lowered:
        return "Area Chair"
    if "meta reviewer" in lowered or "meta-reviewer" in lowered or "meta review" in lowered or lowered == "metareviewer":
        return "Meta-Reviewer"
    if "reviewer" in lowered or lowered == "review":
        return "Reviewer"
    if "author" in lowered or "rebuttal" in lowered:
        return "Author"
    if "editor" in lowered:
        return "Editor"
    if "chair" in lowered:
        return "Chair"
    return " ".join(part.capitalize() for part in lowered.split())


def normalize_roles(value):
    items = []
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                item = item.get("role") or item.get("type") or str(item)
            items.append(item)
    else:
        items.append(value)

    tokens = []
    for item in items:
        if item is None:
            continue
        token = item.get("role") if isinstance(item, dict) else str(item)
        token = token or ""
        parts = ROLE_SPLIT_RE.split(token)
        tokens.extend([part for part in parts if part])

    normalized = []
    seen = set()
    for token in tokens:
        norm = normalize_role_token(token)
        if not norm or norm in seen:
            continue
        normalized.append(norm)
        seen.add(norm)
    return normalized


def default_api_key():
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DMX_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    )


def call_embeddings(texts, model, api_key, base_url, max_retries, sleep_seconds):
    endpoint = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(1, max_retries + 1):
        try:
            req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
            with url_request.urlopen(req, timeout=360) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            embeddings = [item["embedding"] for item in data.get("data", [])]
            if len(embeddings) != len(texts):
                raise ValueError("Embedding count mismatch")
            return embeddings
        except Exception as exc:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_seconds)


def embed_texts(texts, model, api_key, base_url, batch_size, max_retries, sleep_seconds, log_prefix="embed"):
    embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = call_embeddings(
            batch,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        embeddings.extend(batch_embeddings)
        if log_prefix:
            print(f"[{log_prefix}] {min(start + batch_size, total)}/{total}", file=sys.stderr)
    return np.array(embeddings, dtype=float)


def normalize_vectors(matrix, in_place=False):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    if in_place:
        matrix /= norms
        return matrix
    return matrix / norms


def kmeans_plus_plus_init(data, k, rng):
    n = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=data.dtype)
    idx = rng.integers(0, n)
    centroids[0] = data[idx]
    distances = np.full(n, np.inf)
    for i in range(1, k):
        new_dist = np.linalg.norm(data - centroids[i - 1], axis=1) ** 2
        distances = np.minimum(distances, new_dist)
        probs = distances / distances.sum() if distances.sum() > 0 else np.full(n, 1 / n)
        idx = rng.choice(n, p=probs)
        centroids[i] = data[idx]
    return centroids


def kmeans(data, k, max_iter=50, seed=42):
    n = data.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return np.array([]), np.array([])
    if k > n:
        k = n
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(data, k, rng)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = data[labels == idx]
            if len(members) == 0:
                centroids[idx] = data[rng.integers(0, n)]
            else:
                centroids[idx] = members.mean(axis=0)

    return labels, centroids


def build_issue_text(act, mode):
    action = act.get("action") or ""
    grounding = act.get("grounding_ref") or ""
    tool_calls = act.get("latent_skill_calls") or []
    targets = []
    for call in tool_calls:
        if isinstance(call, dict):
            target = call.get("target_type")
            if target:
                targets.append(target)
    target_text = ", ".join(sorted(set(targets)))

    parts = []
    if "action" in mode:
        parts.append(action)
    if "grounding" in mode:
        parts.append(grounding)
    if "target" in mode and target_text:
        parts.append(target_text)
    return " | ".join([p for p in parts if p]) or action


def build_intent_text(act, mode):
    chain = (act.get("meta") or {}).get("cognitive_chain") or ""
    role_raw = (act.get("meta") or {}).get("role_raw") or ""
    action = act.get("action") or ""
    parts = []
    if "chain" in mode:
        parts.append(chain)
    if "role" in mode and role_raw:
        parts.append(f"role:{role_raw}")
    if "action" in mode:
        parts.append(action)
    return " | ".join([p for p in parts if p]) or action or chain


def parse_args():
    parser = argparse.ArgumentParser(description="Build semantic act instances with issue and intent labels.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to mining_results.jsonl")
    parser.add_argument("--out", dest="output_path", required=True, help="Output semantic acts JSONL")
    parser.add_argument("--issues-out", default="steps/01_flatten_semantic_acts/issues.jsonl")
    parser.add_argument("--issue-assignments-out", default="steps/01_flatten_semantic_acts/issue_assignments.jsonl")
    parser.add_argument("--intents-out", default="steps/01_flatten_semantic_acts/intents.jsonl")
    parser.add_argument("--intent-assignments-out", default="steps/01_flatten_semantic_acts/intent_assignments.jsonl")
    parser.add_argument("--issue-embeddings-out", default="steps/01_flatten_semantic_acts/issue_embeddings.npy")
    parser.add_argument("--intent-embeddings-out", default="steps/01_flatten_semantic_acts/intent_embeddings.npy")
    parser.add_argument("--limit-papers", type=int, default=0, help="Process only the first N papers")
    parser.add_argument("--sample-acts", type=int, default=0, help="Randomly sample N acts for a quick test run")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--issue-text-mode", default="action", choices=["action", "action+grounding", "action+target", "action+grounding+target"])
    parser.add_argument("--intent-text-mode", default="chain+role", choices=["chain", "chain+role", "chain+role+action", "action"])

    parser.add_argument("--issue-k", type=int, default=0, help="Number of clusters per forum (0 = auto)")
    parser.add_argument("--intent-k", type=int, default=0, help="Number of intent clusters (0 = auto)")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-cluster-scope", choices=["forum", "global"], default="global")

    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--embed-base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--embed-api-key", default="")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-sleep", type=float, default=0.3)
    parser.add_argument("--embed-retries", type=int, default=3)

    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-debug-dir", default="", help="Write raw LLM responses to this directory")
    parser.add_argument("--issue-sample-size", type=int, default=6)
    parser.add_argument("--intent-sample-size", type=int, default=6)
    parser.add_argument("--issue-memory-max", type=int, default=50)
    parser.add_argument("--intent-memory-max", type=int, default=50)
    parser.add_argument("--issue-memory-scope", choices=["forum", "global"], default="global")
    parser.add_argument("--skip-issue-labels", action="store_true")
    parser.add_argument("--skip-intent-labels", action="store_true")

    return parser.parse_args()


def call_chat(messages, model, api_key, base_url, max_retries, sleep_seconds):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(1, max_retries + 1):
        try:
            req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
            with url_request.urlopen(req, timeout=360) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_seconds)


def extract_json(text):
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    list_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if list_match:
        try:
            return json.loads(list_match.group(0))
        except json.JSONDecodeError:
            pass
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def write_llm_debug(debug_dir, kind, payload):
    if not debug_dir:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"{kind}_llm.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def normalize_label_payload(parsed):
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        reuse = None
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if item.get("action") == "reuse" and item.get("reuse_id"):
                reuse = item
            if "label" in item or "action" in item:
                return reuse or item
        return {}
    return {}


def label_with_llm(kind, samples, existing, model, api_key, base_url, debug_dir=None, debug_meta=None):
    kind = kind.lower().strip()
    if kind == "issue":
        definition = (
            "Issue label = a dispute/topic that multiple acts revolve around "
            "(e.g., missing baselines, unclear methodology, insufficient evidence, "
            "novelty claims)."
        )
    else:
        definition = (
            "Intent label = a communicative strategy behind an act "
            "(e.g., request clarification, defend method, concede minor point, "
            "challenge novelty)."
        )

    system = (
        "You are an expert scientific review analyst. "
        "Label clusters of semantic acts with concise, reusable labels. "
        "Return exactly one JSON object (not a list)."
    )
    user = {
        "task": f"Label {kind} cluster",
        "definition": definition,
        "existing_labels": existing or [],
        "samples": samples,
        "output_schema": {
            "action": "reuse|new",
            "label": "short label",
            "description": "one-sentence definition",
            "reuse_id": "if action=reuse, provide existing id",
        },
        "rules": [
            "Reuse an existing label if it matches the cluster meaning.",
            "Create a new label only when no existing label fits.",
            "Labels must be short, stable, and domain-agnostic (no paper-specific details).",
            "Description should explain the dispute/strategy in one sentence.",
            "Return a single JSON object, not an array.",
        ],
        "format": {
            "label_style": "Title_Case with underscores allowed",
            "description_style": "one sentence, no markdown",
        },
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
    ]
    response = call_chat(messages, model, api_key, base_url, max_retries=3, sleep_seconds=0.5)
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    parsed = extract_json(content) or {}
    normalized = normalize_label_payload(parsed)
    if debug_dir:
        write_llm_debug(
            debug_dir,
            kind,
            {
                "ts": time.time(),
                "kind": kind,
                "meta": debug_meta or {},
                "samples": samples,
                "existing_labels": existing or [],
                "messages": messages,
                "response": response,
                "parsed": parsed,
                "normalized": normalized,
            },
        )
    return normalized


def pick_samples(indices, acts, field, sample_size, rng, text_lookup=None):
    if not indices:
        return []
    selected = indices if len(indices) <= sample_size else rng.sample(indices, sample_size)
    samples = []
    for idx in selected:
        act = acts[idx]
        text = text_lookup[idx] if text_lookup else act.get(field)
        samples.append(
            {
                "act_id": act.get("act_id"),
                "role": act.get("role"),
                "text": text or "",
            }
        )
    return samples


def save_embeddings(path, matrix):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def auto_k(n):
    if n <= 1:
        return 1
    return max(2, int(round(math.sqrt(n))))


def main():
    args = parse_args()

    records = read_json_or_jsonl(args.input_path)
    if args.limit_papers > 0:
        records = records[: args.limit_papers]

    acts = []
    empty_mining = 0
    missing_forum = 0
    total_acts = 0

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
            act_id = f"{forum_id}#{idx:04d}"
            raw_role = item.get("role")
            roles = normalize_roles(raw_role)
            role = roles[0] if roles else None
            acts.append(
                {
                    "act_id": act_id,
                    "act_index": idx,
                    "issue_id": None,
                    "forum_id": forum_id,
                    "title": title,
                    "timestamp": timestamp,
                    "role": role,
                    "roles": roles,
                    "intent": None,
                    "intent_id": None,
                    "action": item.get("action"),
                    "act_text": item.get("action"),
                    "grounding_ref": item.get("grounding_ref"),
                    "source_seg_ids": item.get("source_seg_ids") or [],
                    "latent_skill_calls": item.get("latent_tool_calls") or [],
                    "meta": {"cognitive_chain": item.get("cognitive_chain"), "role_raw": raw_role},
                }
            )
            total_acts += 1

        if not args.quiet and args.log_every > 0 and idx_paper % args.log_every == 0:
            print(
                f"[flat] {idx_paper}/{len(records)} papers | acts={total_acts}",
                file=sys.stderr,
            )

    if not args.quiet:
        print(f"[summary] papers={len(records)} acts={total_acts} empty_mining={empty_mining} missing_forum_id={missing_forum}", file=sys.stderr)

    if total_acts == 0:
        write_jsonl(args.output_path, [])
        return

    if args.sample_acts and args.sample_acts > 0 and args.sample_acts < len(acts):
        rng = random.Random(args.sample_seed)
        sample_indices = sorted(rng.sample(range(len(acts)), args.sample_acts))
        acts = [acts[idx] for idx in sample_indices]
        if not args.quiet:
            print(f"[sample] acts={len(acts)} (seed={args.sample_seed})", file=sys.stderr)

    embed_key = args.embed_api_key or default_api_key()
    llm_key = args.llm_api_key or embed_key

    # Issue clustering
    issue_texts = [build_issue_text(act, args.issue_text_mode) for act in acts]
    if not embed_key:
        raise SystemExit("Missing API key for embeddings. Provide --embed-api-key or set OPENAI_API_KEY.")

    issue_embeddings = embed_texts(
        issue_texts,
        model=args.embed_model,
        api_key=embed_key,
        base_url=args.embed_base_url,
        batch_size=args.embed_batch_size,
        max_retries=args.embed_retries,
        sleep_seconds=args.embed_sleep,
        log_prefix="embed-issue",
    )
    save_embeddings(args.issue_embeddings_out, issue_embeddings)
    issue_embeddings = normalize_vectors(issue_embeddings, in_place=True)

    scope_to_indices = {}
    for idx, act in enumerate(acts):
        if args.issue_cluster_scope == "global":
            scope_key = "global"
        else:
            scope_key = act["forum_id"]
        scope_to_indices.setdefault(scope_key, []).append(idx)

    issue_catalog = []
    issue_assignments = []
    global_issue_memory = []
    rng = random.Random(args.seed)

    for scope_key, indices in scope_to_indices.items():
        data = issue_embeddings[indices]
        n = data.shape[0]
        k = args.issue_k if args.issue_k > 0 else auto_k(n)
        labels, _ = kmeans(data, k, max_iter=args.max_iter, seed=args.seed)

        cluster_to_indices = {}
        for local_idx, cluster_id in enumerate(labels):
            cluster_to_indices.setdefault(int(cluster_id), []).append(indices[local_idx])
        if not args.quiet:
            sizes = sorted(len(members) for members in cluster_to_indices.values())
            if sizes:
                size_min = sizes[0]
                size_max = sizes[-1]
                size_mean = sum(sizes) / len(sizes)
                size_median = float(np.median(sizes))
                print(
                    f"[issue] scope={scope_key} clusters={len(sizes)} "
                    f"size_min={size_min} size_median={size_median:.1f} "
                    f"size_mean={size_mean:.1f} size_max={size_max}",
                    file=sys.stderr,
                )

        forum_issue_memory = []
        for cluster_id, members in sorted(cluster_to_indices.items()):
            samples = pick_samples(members, acts, "action", args.issue_sample_size, rng, text_lookup=issue_texts)
            issue_payload = None
            if not args.skip_issue_labels and llm_key:
                memory = global_issue_memory if args.issue_memory_scope == "global" else forum_issue_memory
                issue_payload = label_with_llm(
                    "issue",
                    samples,
                    memory[-args.issue_memory_max :],
                    model=args.llm_model,
                    api_key=llm_key,
                    base_url=args.llm_base_url,
                    debug_dir=args.llm_debug_dir,
                    debug_meta={
                        "scope": scope_key,
                        "cluster_id": int(cluster_id),
                        "cluster_size": len(members),
                    },
                )
            if args.issue_cluster_scope == "global":
                issue_id = f"issue_{cluster_id:03d}"
            else:
                issue_id = f"{scope_key}#issue_{cluster_id:03d}"
            label = issue_payload.get("label") if issue_payload else f"Issue_{cluster_id:03d}"
            description = issue_payload.get("description") if issue_payload else ""
            reuse_id = issue_payload.get("reuse_id") if issue_payload else None
            action = issue_payload.get("action") if issue_payload else None

            if action == "reuse" and reuse_id:
                issue_id = reuse_id
            else:
                issue_catalog.append(
                    {
                        "issue_id": issue_id,
                        "forum_id": None if args.issue_cluster_scope == "global" else scope_key,
                        "label": label,
                        "description": description,
                        "examples": samples,
                    }
                )
                entry = {"issue_id": issue_id, "label": label, "description": description}
                forum_issue_memory.append(entry)
                global_issue_memory.append(entry)

            for act_idx in members:
                acts[act_idx]["issue_id"] = issue_id
                issue_assignments.append(
                    {
                        "act_id": acts[act_idx]["act_id"],
                        "issue_id": issue_id,
                        "forum_id": acts[act_idx]["forum_id"],
                        "cluster_id": f"{scope_key}#cluster_{cluster_id:03d}",
                    }
                )

    # Intent clustering (global)
    intent_texts = [build_intent_text(act, args.intent_text_mode) for act in acts]
    intent_embeddings = embed_texts(
        intent_texts,
        model=args.embed_model,
        api_key=embed_key,
        base_url=args.embed_base_url,
        batch_size=args.embed_batch_size,
        max_retries=args.embed_retries,
        sleep_seconds=args.embed_sleep,
        log_prefix="embed-intent",
    )
    save_embeddings(args.intent_embeddings_out, intent_embeddings)
    intent_embeddings = normalize_vectors(intent_embeddings, in_place=True)

    k_intent = args.intent_k if args.intent_k > 0 else auto_k(len(acts))
    intent_labels, _ = kmeans(intent_embeddings, k_intent, max_iter=args.max_iter, seed=args.seed)

    intent_catalog = []
    intent_assignments = []
    intent_memory = []

    cluster_to_indices = {}
    for idx, cluster_id in enumerate(intent_labels):
        cluster_to_indices.setdefault(int(cluster_id), []).append(idx)

    if not args.quiet:
        sizes = sorted(len(members) for members in cluster_to_indices.values())
        if sizes:
            size_min = sizes[0]
            size_max = sizes[-1]
            size_mean = sum(sizes) / len(sizes)
            size_median = float(np.median(sizes))
            top_clusters = sorted(
                ((cluster_id, len(members)) for cluster_id, members in cluster_to_indices.items()),
                key=lambda item: (-item[1], item[0]),
            )[:10]
            top_summary = ", ".join(f"{cid}:{count}" for cid, count in top_clusters)
            print(
                f"[intent] clusters={len(sizes)} size_min={size_min} "
                f"size_median={size_median:.1f} size_mean={size_mean:.1f} size_max={size_max}",
                file=sys.stderr,
            )
            print(f"[intent] top_clusters: {top_summary}", file=sys.stderr)

    for cluster_id, members in sorted(cluster_to_indices.items()):
        samples = pick_samples(members, acts, "act_text", args.intent_sample_size, rng, text_lookup=intent_texts)
        intent_payload = None
        if not args.skip_intent_labels and llm_key:
            intent_payload = label_with_llm(
                "intent",
                samples,
                intent_memory[-args.intent_memory_max :],
                model=args.llm_model,
                api_key=llm_key,
                base_url=args.llm_base_url,
                debug_dir=args.llm_debug_dir,
                debug_meta={
                    "cluster_id": int(cluster_id),
                    "cluster_size": len(members),
                },
            )
        intent_id = f"intent_{cluster_id:03d}"
        label = intent_payload.get("label") if intent_payload else f"Intent_{cluster_id:03d}"
        description = intent_payload.get("description") if intent_payload else ""
        reuse_id = intent_payload.get("reuse_id") if intent_payload else None
        action = intent_payload.get("action") if intent_payload else None

        if action == "reuse" and reuse_id:
            intent_id = reuse_id
        else:
            intent_catalog.append(
                {
                    "intent_id": intent_id,
                    "label": label,
                    "description": description,
                    "examples": samples,
                }
            )
            intent_memory.append({"intent_id": intent_id, "label": label, "description": description})

        for act_idx in members:
            acts[act_idx]["intent_id"] = intent_id
            acts[act_idx]["intent"] = label
            intent_assignments.append(
                {
                    "act_id": acts[act_idx]["act_id"],
                    "intent_id": intent_id,
                    "cluster_id": f"intent_cluster_{cluster_id:03d}",
                }
            )

    write_jsonl(args.output_path, acts)
    write_jsonl(args.issues_out, issue_catalog)
    write_jsonl(args.issue_assignments_out, issue_assignments)
    write_jsonl(args.intents_out, intent_catalog)
    write_jsonl(args.intent_assignments_out, intent_assignments)


if __name__ == "__main__":
    import os

    main()
