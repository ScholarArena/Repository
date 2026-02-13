import argparse
import json
import hashlib
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
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "were",
    "will",
    "with",
    "would",
    "you",
    "your",
}


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
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_seconds)

def read_progress(path):
    if not path or not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0


def write_progress(path, value):
    if not path:
        return
    path.write_text(str(int(value)), encoding="utf-8")


def embed_texts(
    texts,
    model,
    api_key,
    base_url,
    batch_size,
    max_retries,
    sleep_seconds,
    log_prefix="embed",
    save_path=None,
    resume=False,
):
    total = len(texts)
    embeddings = []
    memmap = None
    rng = np.random.default_rng(42)
    fallback_dim = None
    progress_path = Path(f"{save_path}.progress") if save_path else None
    start_at = 0

    if save_path and resume and Path(save_path).exists():
        if progress_path and progress_path.exists():
            start_at = read_progress(progress_path)
        memmap = np.lib.format.open_memmap(save_path, mode="r+")
        if memmap.shape[0] != total:
            memmap = None
            start_at = 0
        else:
            fallback_dim = int(memmap.shape[1])
        if log_prefix and start_at:
            print(f"[{log_prefix}] resume at {start_at}/{total}", file=sys.stderr)

    for start in range(start_at, total, batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_embeddings = call_embeddings(
                batch,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        except Exception:
            if fallback_dim is None:
                raise
            if log_prefix:
                print(
                    f"[{log_prefix}] warning: embedding request failed at {start}, using random fallback",
                    file=sys.stderr,
                )
            batch_embeddings = rng.standard_normal((len(batch), fallback_dim)).tolist()
        if save_path:
            if memmap is None:
                dim = len(batch_embeddings[0]) if batch_embeddings else 0
                memmap = np.lib.format.open_memmap(
                    save_path, mode="w+", dtype="float32", shape=(total, dim)
                )
                memmap[:] = np.nan
            end = start + len(batch_embeddings)
            memmap[start:end] = np.array(batch_embeddings, dtype="float32")
            memmap.flush()
            if progress_path:
                write_progress(progress_path, end)
        else:
            embeddings.extend(batch_embeddings)
        if fallback_dim is None and batch_embeddings:
            fallback_dim = len(batch_embeddings[0])
        if log_prefix:
            print(f"[{log_prefix}] {min(start + batch_size, total)}/{total}", file=sys.stderr)
    if save_path:
        if progress_path:
            write_progress(progress_path, total)
        return memmap
    return np.array(embeddings, dtype=float)


def normalize_vectors(matrix, in_place=False):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    if in_place:
        matrix /= norms
        return matrix
    return matrix / norms


def tokenize_for_hash(text):
    tokens = TOKEN_RE.findall((text or "").lower())
    return [token for token in tokens if token and token not in STOPWORDS]


def simhash64(tokens):
    if not tokens:
        return 0
    accum = [0] * 64
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bits = int.from_bytes(digest[:8], "big")
        for idx in range(64):
            if bits & (1 << idx):
                accum[idx] += 1
            else:
                accum[idx] -= 1
    fingerprint = 0
    for idx, value in enumerate(accum):
        if value >= 0:
            fingerprint |= 1 << idx
    return fingerprint


def heuristic_group_key(text, bits):
    tokens = tokenize_for_hash(text)
    if not tokens:
        return 0
    fingerprint = simhash64(tokens)
    if bits <= 0 or bits >= 64:
        return fingerprint
    return fingerprint >> (64 - bits)


def group_by_heuristic(texts, indices, bits):
    groups = {}
    for idx in indices:
        key = heuristic_group_key(texts[idx], bits)
        groups.setdefault(key, []).append(idx)
    return groups


def pick_representatives(indices, texts, reps, rng):
    if reps <= 1 or len(indices) <= reps:
        if not indices:
            return []
        longest = max(indices, key=lambda i: len((texts[i] or "")))
        if len(indices) == 1 or reps <= 1:
            return [longest]
        remaining = [idx for idx in indices if idx != longest]
        extra = rng.sample(remaining, reps - 1) if remaining else []
        return [longest] + extra
    lengths = sorted(indices, key=lambda i: len((texts[i] or "")), reverse=True)
    head = lengths[:1]
    remaining = lengths[1:]
    extra = rng.sample(remaining, reps - 1) if remaining else []
    return head + extra


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


def init_centroids(data, k, rng, method="kmeans++", sample_size=0):
    n = data.shape[0]
    if sample_size and sample_size < n:
        sample_size = max(sample_size, k)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        data_view = data[sample_idx]
    else:
        data_view = data
    if method == "random":
        if data_view.shape[0] < k:
            idx = rng.choice(data_view.shape[0], size=k, replace=True)
        else:
            idx = rng.choice(data_view.shape[0], size=k, replace=False)
        return data_view[idx].astype(np.float32, copy=False)
    return kmeans_plus_plus_init(data_view, k, rng).astype(np.float32, copy=False)


def assign_labels_chunked(data, centroids, chunk_size, log_prefix=None, log_every=0, iter_idx=None):
    n = data.shape[0]
    labels = np.empty(n, dtype=int)
    centroids = np.asarray(centroids, dtype=np.float32)
    c_norm = np.sum(centroids * centroids, axis=1)
    total_chunks = (n + chunk_size - 1) // chunk_size if chunk_size else 1
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(data[start:end], dtype=np.float32)
        x_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        dots = chunk @ centroids.T
        dist = x_norm + c_norm - 2.0 * dots
        labels[start:end] = dist.argmin(axis=1)
        if log_prefix and log_every:
            chunk_idx = (start // chunk_size) + 1
            if chunk_idx % log_every == 0 or chunk_idx == total_chunks:
                iter_note = f" iter={iter_idx}" if iter_idx is not None else ""
                print(
                    f"[{log_prefix}]{iter_note} chunks {chunk_idx}/{total_chunks}",
                    file=sys.stderr,
                )
    return labels


def kmeans(
    data,
    k,
    max_iter=50,
    seed=42,
    chunk_size=2048,
    log_prefix=None,
    log_every=0,
    method="full",
    init_method="kmeans++",
    init_sample=0,
    batch_size=2048,
):
    n = data.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return np.array([]), np.array([])
    if k > n:
        k = n
    if method == "auto":
        method = "minibatch" if n * k > 1_000_000 else "full"
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float32)
    centroids = init_centroids(data, k, rng, method=init_method, sample_size=init_sample)
    labels = np.zeros(n, dtype=int)

    if method == "minibatch":
        counts = np.zeros(k, dtype=np.int64)
        for iter_idx in range(1, max_iter + 1):
            batch_n = min(batch_size, n)
            batch_idx = rng.choice(n, size=batch_n, replace=False)
            batch = data[batch_idx]
            batch_labels = assign_labels_chunked(
                batch,
                centroids,
                chunk_size=min(chunk_size, batch_n),
                log_prefix=log_prefix,
                log_every=log_every,
                iter_idx=iter_idx,
            )
            for idx in range(k):
                members = batch[batch_labels == idx]
                if len(members) == 0:
                    continue
                counts[idx] += len(members)
                eta = len(members) / counts[idx]
                centroids[idx] = (1 - eta) * centroids[idx] + eta * members.mean(axis=0)
        labels = assign_labels_chunked(
            data,
            centroids,
            chunk_size=chunk_size,
            log_prefix=log_prefix,
            log_every=log_every,
            iter_idx="final",
        )
    else:
        for iter_idx in range(1, max_iter + 1):
            if log_prefix and log_every:
                print(f"[{log_prefix}] iter={iter_idx}/{max_iter} assign", file=sys.stderr)
            new_labels = assign_labels_chunked(
                data,
                centroids,
                chunk_size,
                log_prefix=log_prefix,
                log_every=log_every,
                iter_idx=iter_idx,
            )
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
    parser.add_argument("--kmeans-chunk-size", type=int, default=2048, help="Chunk size for k-means distance computation")
    parser.add_argument("--kmeans-log-every", type=int, default=0, help="Log every N chunks during k-means assignment (0 disables)")
    parser.add_argument("--kmeans-method", choices=["full", "minibatch", "auto"], default="auto")
    parser.add_argument("--kmeans-init", choices=["kmeans++", "random"], default="kmeans++")
    parser.add_argument("--kmeans-init-sample", type=int, default=0, help="Sample size for centroid init (0 = use full data)")
    parser.add_argument("--kmeans-batch-size", type=int, default=2048, help="Mini-batch size for k-means")

    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--embed-base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--embed-api-key", default="")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-sleep", type=float, default=0.3)
    parser.add_argument("--embed-retries", type=int, default=3)
    parser.add_argument("--embed-resume", action="store_true", help="Resume embeddings from saved .npy files")
    parser.add_argument("--issue-embed-strategy", choices=["full", "heuristic"], default="heuristic")
    parser.add_argument("--intent-embed-strategy", choices=["full", "heuristic"], default="heuristic")
    parser.add_argument("--issue-heuristic-bits", type=int, default=12)
    parser.add_argument("--intent-heuristic-bits", type=int, default=12)
    parser.add_argument("--issue-heuristic-reps", type=int, default=1)
    parser.add_argument("--intent-heuristic-reps", type=int, default=1)
    parser.add_argument("--issue-rep-embeddings-out", default="")
    parser.add_argument("--intent-rep-embeddings-out", default="")

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


def resolve_kmeans_method(method, n, k):
    if method == "auto":
        return "minibatch" if n * k > 1_000_000 else "full"
    return method


def describe_cluster_sizes(sizes):
    if not sizes:
        return None
    sizes = sorted(sizes)
    size_min = sizes[0]
    size_max = sizes[-1]
    size_mean = sum(sizes) / len(sizes)
    size_median = float(np.median(sizes))
    return size_min, size_median, size_mean, size_max


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


    issue_texts = [build_issue_text(act, args.issue_text_mode) for act in acts]
    if not embed_key:
        raise SystemExit("Missing API key for embeddings. Provide --embed-api-key or set OPENAI_API_KEY.")

    scope_to_indices = {}
    for idx, act in enumerate(acts):
        if args.issue_cluster_scope == "global":
            scope_key = "global"
        else:
            scope_key = act["forum_id"]
        scope_to_indices.setdefault(scope_key, []).append(idx)

    issue_embeddings = None
    if args.issue_embed_strategy == "full":
        if not args.quiet:
            print(f"[embed-issue] total_texts={len(issue_texts)}", file=sys.stderr)
        issue_embeddings = embed_texts(
            issue_texts,
            model=args.embed_model,
            api_key=embed_key,
            base_url=args.embed_base_url,
            batch_size=args.embed_batch_size,
            max_retries=args.embed_retries,
            sleep_seconds=args.embed_sleep,
            log_prefix="embed-issue",
            save_path=args.issue_embeddings_out,
            resume=args.embed_resume,
        )
        issue_embeddings = normalize_vectors(issue_embeddings, in_place=True)
        if not args.quiet and args.issue_embeddings_out:
            print(f"[embed-issue] saved embeddings to {args.issue_embeddings_out}", file=sys.stderr)

    issue_catalog = []
    issue_assignments = []
    issue_group_embeddings = []
    issue_group_counter = 0
    issue_group_for_act = {}
    global_issue_memory = []
    rng = random.Random(args.seed)

    for scope_key, indices in scope_to_indices.items():
        cluster_to_indices = {}
        group_id_by_local = {}
        if not args.quiet:
            print(f"[issue] scope={scope_key} acts={len(indices)}", file=sys.stderr)

        if args.issue_embed_strategy == "full":
            data = issue_embeddings[indices]
            n = data.shape[0]
            k = args.issue_k if args.issue_k > 0 else auto_k(n)
            method = resolve_kmeans_method(args.kmeans_method, n, k)
            issue_log_prefix = None
            if not args.quiet and args.kmeans_log_every:
                issue_log_prefix = f"kmeans-issue:{scope_key}"
            labels, _ = kmeans(
                data,
                k,
                max_iter=args.max_iter,
                seed=args.seed,
                chunk_size=args.kmeans_chunk_size,
                log_prefix=issue_log_prefix,
                log_every=args.kmeans_log_every,
                method=method,
                init_method=args.kmeans_init,
                init_sample=args.kmeans_init_sample,
                batch_size=args.kmeans_batch_size,
            )
            for local_idx, cluster_id in enumerate(labels):
                cluster_to_indices.setdefault(int(cluster_id), []).append(indices[local_idx])
        else:
            groups = group_by_heuristic(issue_texts, indices, args.issue_heuristic_bits)
            group_sizes = [len(members) for members in groups.values()]
            if not args.quiet:
                stats = describe_cluster_sizes(group_sizes)
                if stats:
                    size_min, size_median, size_mean, size_max = stats
                    print(
                        f"[heuristic-issue] scope={scope_key} groups={len(groups)} "
                        f"size_min={size_min} size_median={size_median:.1f} "
                        f"size_mean={size_mean:.1f} size_max={size_max}",
                        file=sys.stderr,
                    )

            group_keys = sorted(groups.keys())
            group_to_index = {key: idx for idx, key in enumerate(group_keys)}
            rep_texts = []
            rep_group_indices = []
            rep_act_indices = []
            for key in group_keys:
                local_idx = group_to_index[key]
                reps = pick_representatives(groups[key], issue_texts, args.issue_heuristic_reps, rng)
                for act_idx in reps:
                    rep_texts.append(issue_texts[act_idx])
                    rep_group_indices.append(local_idx)
                    rep_act_indices.append(act_idx)
            if not args.quiet:
                avg_reps = len(rep_texts) / max(1, len(group_keys))
                print(
                    f"[heuristic-issue] reps={len(rep_texts)} groups={len(group_keys)} "
                    f"avg_reps={avg_reps:.2f} reps_per_group={args.issue_heuristic_reps}",
                    file=sys.stderr,
                )

            rep_save_path = args.issue_rep_embeddings_out or None
            rep_embeddings = embed_texts(
                rep_texts,
                model=args.embed_model,
                api_key=embed_key,
                base_url=args.embed_base_url,
                batch_size=args.embed_batch_size,
                max_retries=args.embed_retries,
                sleep_seconds=args.embed_sleep,
                log_prefix="embed-issue-rep",
                save_path=rep_save_path,
                resume=args.embed_resume,
            )
            rep_embeddings = normalize_vectors(rep_embeddings, in_place=True)
            if not args.quiet:
                print(
                    f"[embed-issue-rep] embedded={len(rep_texts)} dim={rep_embeddings.shape[1]}",
                    file=sys.stderr,
                )

            group_embeddings = np.zeros((len(group_keys), rep_embeddings.shape[1]), dtype=np.float32)
            group_counts = np.zeros(len(group_keys), dtype=np.int32)
            for emb, group_idx in zip(rep_embeddings, rep_group_indices):
                group_embeddings[group_idx] += emb
                group_counts[group_idx] += 1
            for idx, count in enumerate(group_counts):
                if count > 0:
                    group_embeddings[idx] /= float(count)
            group_embeddings = normalize_vectors(group_embeddings, in_place=True)

            for local_idx, key in enumerate(group_keys):
                group_id = f"group_{issue_group_counter:06d}"
                issue_group_counter += 1
                group_id_by_local[local_idx] = group_id
            issue_group_embeddings.append(group_embeddings)

            n = group_embeddings.shape[0]
            k = args.issue_k if args.issue_k > 0 else auto_k(n)
            method = resolve_kmeans_method(args.kmeans_method, n, k)
            issue_log_prefix = None
            if not args.quiet and args.kmeans_log_every:
                issue_log_prefix = f"kmeans-issue:{scope_key}"
            labels, _ = kmeans(
                group_embeddings,
                k,
                max_iter=args.max_iter,
                seed=args.seed,
                chunk_size=args.kmeans_chunk_size,
                log_prefix=issue_log_prefix,
                log_every=args.kmeans_log_every,
                method=method,
                init_method=args.kmeans_init,
                init_sample=args.kmeans_init_sample,
                batch_size=args.kmeans_batch_size,
            )
            for key, members in groups.items():
                local_idx = group_to_index[key]
                cluster_id = int(labels[local_idx])
                cluster_to_indices.setdefault(cluster_id, []).extend(members)
                for act_idx in members:
                    issue_group_for_act[act_idx] = group_id_by_local[local_idx]

        if not args.quiet:
            sizes = sorted(len(members) for members in cluster_to_indices.values())
            stats = describe_cluster_sizes(sizes)
            if stats:
                size_min, size_median, size_mean, size_max = stats
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
                assignment = {
                    "act_id": acts[act_idx]["act_id"],
                    "issue_id": issue_id,
                    "forum_id": acts[act_idx]["forum_id"],
                    "cluster_id": f"{scope_key}#cluster_{cluster_id:03d}",
                }
                group_id = issue_group_for_act.get(act_idx)
                if group_id:
                    assignment["group_id"] = group_id
                issue_assignments.append(assignment)

    if issue_group_embeddings and args.issue_embeddings_out:
        merged = np.vstack(issue_group_embeddings)
        save_embeddings(args.issue_embeddings_out, merged)
        if not args.quiet:
            print(
                f"[embed-issue] saved group embeddings to {args.issue_embeddings_out} "
                f"shape={merged.shape}",
                file=sys.stderr,
            )


    intent_texts = [build_intent_text(act, args.intent_text_mode) for act in acts]
    intent_group_embeddings = []
    intent_group_for_act = {}
    intent_group_counter = 0
    intent_groups = None
    intent_group_to_index = None

    if args.intent_embed_strategy == "full":
        if not args.quiet:
            print(f"[embed-intent] total_texts={len(intent_texts)}", file=sys.stderr)
        intent_embeddings = embed_texts(
            intent_texts,
            model=args.embed_model,
            api_key=embed_key,
            base_url=args.embed_base_url,
            batch_size=args.embed_batch_size,
            max_retries=args.embed_retries,
            sleep_seconds=args.embed_sleep,
            log_prefix="embed-intent",
            save_path=args.intent_embeddings_out,
            resume=args.embed_resume,
        )
        intent_embeddings = normalize_vectors(intent_embeddings, in_place=True)
        if not args.quiet and args.intent_embeddings_out:
            print(f"[embed-intent] saved embeddings to {args.intent_embeddings_out}", file=sys.stderr)

        k_intent = args.intent_k if args.intent_k > 0 else auto_k(len(acts))
        intent_method = resolve_kmeans_method(args.kmeans_method, len(acts), k_intent)
        intent_log_prefix = None
        if not args.quiet and args.kmeans_log_every:
            intent_log_prefix = "kmeans-intent"
        intent_labels, _ = kmeans(
            intent_embeddings,
            k_intent,
            max_iter=args.max_iter,
            seed=args.seed,
            chunk_size=args.kmeans_chunk_size,
            log_prefix=intent_log_prefix,
            log_every=args.kmeans_log_every,
            method=intent_method,
            init_method=args.kmeans_init,
            init_sample=args.kmeans_init_sample,
            batch_size=args.kmeans_batch_size,
        )
    else:
        intent_groups = group_by_heuristic(intent_texts, list(range(len(acts))), args.intent_heuristic_bits)
        group_sizes = [len(members) for members in intent_groups.values()]
        if not args.quiet:
            stats = describe_cluster_sizes(group_sizes)
            if stats:
                size_min, size_median, size_mean, size_max = stats
                print(
                    f"[heuristic-intent] groups={len(intent_groups)} size_min={size_min} "
                    f"size_median={size_median:.1f} size_mean={size_mean:.1f} size_max={size_max}",
                    file=sys.stderr,
                )

        group_keys = sorted(intent_groups.keys())
        intent_group_to_index = {key: idx for idx, key in enumerate(group_keys)}
        rep_texts = []
        rep_group_indices = []
        for key in group_keys:
            local_idx = intent_group_to_index[key]
            reps = pick_representatives(intent_groups[key], intent_texts, args.intent_heuristic_reps, rng)
            for act_idx in reps:
                rep_texts.append(intent_texts[act_idx])
                rep_group_indices.append(local_idx)
        if not args.quiet:
            avg_reps = len(rep_texts) / max(1, len(group_keys))
            print(
                f"[heuristic-intent] reps={len(rep_texts)} groups={len(group_keys)} "
                f"avg_reps={avg_reps:.2f} reps_per_group={args.intent_heuristic_reps}",
                file=sys.stderr,
            )

        rep_save_path = args.intent_rep_embeddings_out or None
        rep_embeddings = embed_texts(
            rep_texts,
            model=args.embed_model,
            api_key=embed_key,
            base_url=args.embed_base_url,
            batch_size=args.embed_batch_size,
            max_retries=args.embed_retries,
            sleep_seconds=args.embed_sleep,
            log_prefix="embed-intent-rep",
            save_path=rep_save_path,
            resume=args.embed_resume,
        )
        rep_embeddings = normalize_vectors(rep_embeddings, in_place=True)
        if not args.quiet:
            print(
                f"[embed-intent-rep] embedded={len(rep_texts)} dim={rep_embeddings.shape[1]}",
                file=sys.stderr,
            )

        group_embeddings = np.zeros((len(group_keys), rep_embeddings.shape[1]), dtype=np.float32)
        group_counts = np.zeros(len(group_keys), dtype=np.int32)
        for emb, group_idx in zip(rep_embeddings, rep_group_indices):
            group_embeddings[group_idx] += emb
            group_counts[group_idx] += 1
        for idx, count in enumerate(group_counts):
            if count > 0:
                group_embeddings[idx] /= float(count)
        group_embeddings = normalize_vectors(group_embeddings, in_place=True)

        for local_idx, key in enumerate(group_keys):
            group_id = f"group_{intent_group_counter:06d}"
            intent_group_counter += 1
            for act_idx in intent_groups[key]:
                intent_group_for_act[act_idx] = group_id
        intent_group_embeddings.append(group_embeddings)

        k_intent = args.intent_k if args.intent_k > 0 else auto_k(len(group_keys))
        intent_method = resolve_kmeans_method(args.kmeans_method, len(group_keys), k_intent)
        intent_log_prefix = None
        if not args.quiet and args.kmeans_log_every:
            intent_log_prefix = "kmeans-intent"
        intent_labels, _ = kmeans(
            group_embeddings,
            k_intent,
            max_iter=args.max_iter,
            seed=args.seed,
            chunk_size=args.kmeans_chunk_size,
            log_prefix=intent_log_prefix,
            log_every=args.kmeans_log_every,
            method=intent_method,
            init_method=args.kmeans_init,
            init_sample=args.kmeans_init_sample,
            batch_size=args.kmeans_batch_size,
        )

    intent_catalog = []
    intent_assignments = []
    intent_memory = []

    cluster_to_indices = {}
    if args.intent_embed_strategy == "full":
        for idx, cluster_id in enumerate(intent_labels):
            cluster_to_indices.setdefault(int(cluster_id), []).append(idx)
    else:
        for key, members in intent_groups.items():
            local_idx = intent_group_to_index[key]
            cluster_id = int(intent_labels[local_idx])
            cluster_to_indices.setdefault(cluster_id, []).extend(members)

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
            assignment = {
                "act_id": acts[act_idx]["act_id"],
                "intent_id": intent_id,
                "cluster_id": f"intent_cluster_{cluster_id:03d}",
            }
            group_id = intent_group_for_act.get(act_idx)
            if group_id:
                assignment["group_id"] = group_id
            intent_assignments.append(assignment)

    if intent_group_embeddings and args.intent_embeddings_out:
        merged = np.vstack(intent_group_embeddings)
        save_embeddings(args.intent_embeddings_out, merged)
        if not args.quiet:
            print(
                f"[embed-intent] saved group embeddings to {args.intent_embeddings_out} "
                f"shape={merged.shape}",
                file=sys.stderr,
            )

    write_jsonl(args.output_path, acts)
    write_jsonl(args.issues_out, issue_catalog)
    write_jsonl(args.issue_assignments_out, issue_assignments)
    write_jsonl(args.intents_out, intent_catalog)
    write_jsonl(args.intent_assignments_out, intent_assignments)


if __name__ == "__main__":
    import os

    main()
