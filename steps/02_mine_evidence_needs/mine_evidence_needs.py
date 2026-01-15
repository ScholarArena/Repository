import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as url_request

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing numpy. Install it to run clustering.") from exc


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


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def log_line(log_path, message, also_stdout=True):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    if also_stdout:
        print(line, file=sys.stderr)
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_event(events_path, payload):
    if not events_path:
        return
    event = dict(payload or {})
    event.setdefault("ts", time.time())
    append_jsonl(events_path, event)


def count_prompt_chars(messages):
    total = 0
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else ""
        if content:
            total += len(content)
    return total


def extract_response_text(response):
    if not isinstance(response, dict):
        return ""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content") or ""


def slugify(text, max_len=48):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (text or ""))
    cleaned = cleaned.strip("_").lower()
    if not cleaned:
        cleaned = "artifact"
    return cleaned[:max_len]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def auto_k(n):
    if n <= 1:
        return 1
    return max(2, int(round(math.sqrt(n))))


def choose_k(n, target_size, explicit_k):
    if explicit_k and explicit_k > 0:
        return min(explicit_k, max(1, n))
    if target_size and target_size > 0:
        return max(1, int(math.ceil(n / target_size)))
    return auto_k(n)


def describe_cluster_sizes(sizes):
    if not sizes:
        return None
    sizes = sorted(sizes)
    size_min = sizes[0]
    size_max = sizes[-1]
    size_mean = sum(sizes) / len(sizes)
    size_median = float(np.median(sizes))
    return size_min, size_median, size_mean, size_max


def summarize_tool_calls(tool_calls):
    hints = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            for key in ("tool", "name", "type", "target_type", "target"):
                value = call.get(key)
                if value:
                    hints.append(str(value))
        else:
            hints.append(str(call))
    hints = sorted(set(hints))
    return ", ".join(hints)


def build_need_text(act, mode):
    intent = act.get("intent") or ""
    action = act.get("action") or act.get("act_text") or ""
    tool_calls = act.get("latent_skill_calls") or act.get("latent_tool_calls") or []
    tool_hint = summarize_tool_calls(tool_calls)
    grounding = act.get("grounding_ref") or ""
    source_ids = act.get("source_seg_ids") or []
    source_hint = ", ".join(str(item) for item in source_ids[:8]) if source_ids else ""

    parts = []
    if "intent" in mode and intent:
        parts.append(f"intent:{intent}")
    if "action" in mode and action:
        parts.append(f"action:{action}")
    if "tools" in mode and tool_hint:
        parts.append(f"tools:{tool_hint}")
    if "grounding" in mode and grounding:
        parts.append(f"grounding:{grounding}")
    if "source" in mode and source_hint:
        parts.append(f"source:{source_hint}")
    return " | ".join(parts) or action or intent


def pick_samples(indices, acts, need_texts, sample_size, rng):
    if not indices:
        return []
    selected = indices if len(indices) <= sample_size else rng.sample(indices, sample_size)
    samples = []
    for idx in selected:
        act = acts[idx]
        samples.append(
            {
                "act_id": act.get("act_id"),
                "intent": act.get("intent"),
                "action": act.get("action") or act.get("act_text"),
                "latent_skill_calls": act.get("latent_skill_calls") or act.get("latent_tool_calls") or [],
                "need_text": need_texts[idx],
            }
        )
    return samples


def call_chat(
    messages,
    model,
    api_key,
    base_url,
    max_retries,
    sleep_seconds,
    logger=None,
    usage_recorder=None,
    stage=None,
    meta=None,
):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
            with url_request.urlopen(req, timeout=360) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            duration = time.time() - start
            if usage_recorder:
                usage_recorder(
                    response=data,
                    messages=messages,
                    stage=stage,
                    model=model,
                    duration=duration,
                    meta=meta,
                )
            if logger:
                logger(
                    f"[llm] stage={stage or 'unknown'} attempt={attempt} status=ok "
                    f"duration={duration:.2f}s model={model}"
                )
            return data
        except Exception as exc:
            if logger:
                logger(
                    f"[llm] stage={stage or 'unknown'} attempt={attempt} status=error "
                    f"error={type(exc).__name__}"
                )
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


def extract_code(text):
    if not text:
        return ""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def write_llm_debug(debug_dir, kind, payload):
    if not debug_dir:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"{kind}_llm.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def normalize_spec_payload(payload, fallback_name, allow_primitive_specs=True):
    if not isinstance(payload, dict):
        payload = {}
    action = payload.get("action") or "accept"
    action = str(action).strip().lower()
    if action not in {"accept", "skip"}:
        action = "accept"
    skip_reason = payload.get("skip_reason") or ""
    kind = payload.get("kind") or payload.get("candidate_kind") or "primitive"
    kind = kind.strip().lower()
    if kind not in {"primitive", "skill"}:
        kind = "primitive"
    name = payload.get("name") or payload.get("label") or fallback_name
    summary = payload.get("need_summary") or payload.get("summary") or ""
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    tests = payload.get("tests") if isinstance(payload.get("tests"), list) else []
    dependencies = payload.get("dependencies") if isinstance(payload.get("dependencies"), list) else []
    if kind == "skill" and "dag" not in spec:
        spec["dag"] = []
    primitive_specs_raw = (
        payload.get("primitive_specs") if allow_primitive_specs and isinstance(payload.get("primitive_specs"), list) else []
    )
    primitive_specs = []
    for idx, prim in enumerate(primitive_specs_raw, start=1):
        prim_norm = normalize_spec_payload(prim, f"{name}_Primitive_{idx}", allow_primitive_specs=False)
        prim_norm["kind"] = "primitive"
        prim_norm["primitive_specs"] = []
        primitive_specs.append(prim_norm)
    normalized = {
        "action": action,
        "skip_reason": skip_reason,
        "kind": kind,
        "name": name,
        "need_summary": summary,
        "spec": spec,
        "tests": tests,
        "dependencies": [str(dep) for dep in dependencies if dep],
        "primitive_specs": primitive_specs,
    }
    return normalized


def normalize_template_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    template = {
        "spec_template": payload.get("spec_template") if isinstance(payload.get("spec_template"), dict) else {},
        "test_template": payload.get("test_template") if isinstance(payload.get("test_template"), dict) else {},
        "guidelines": payload.get("guidelines") if isinstance(payload.get("guidelines"), list) else [],
    }
    template["guidelines"] = [str(item) for item in template["guidelines"] if item]
    return template


SPEC_VAGUE_TERMS = {
    "robustness",
    "robust",
    "novelty",
    "significance",
    "important",
    "impact",
    "overall",
    "quality",
    "assess",
    "evaluate",
    "validate",
    "judge",
    "soundness",
}
SPEC_ACTION_TERMS = {
    "extract",
    "locate",
    "find",
    "match",
    "identify",
    "list",
    "count",
    "parse",
    "retrieve",
    "compare",
    "compute",
    "check",
    "verify",
    "detect",
}


def _text_has_any(text, terms):
    if not text:
        return False
    lowered = text.lower()
    return any(term in lowered for term in terms)


def extract_dag_primitives(dag):
    names = []
    for step in dag or []:
        if not isinstance(step, dict):
            continue
        if "primitive" in step and step.get("primitive"):
            names.append(str(step.get("primitive")))
            continue
        if step.get("type") == "primitive" and step.get("name"):
            names.append(str(step.get("name")))
            continue
    return sorted(set(names))


def validate_spec_payload(spec, existing_primitives=None):
    issues = []
    if not isinstance(spec, dict):
        return ["spec_not_dict"]
    if spec.get("action") == "skip":
        return issues
    kind = spec.get("kind")
    tests = spec.get("tests") or []
    spec_body = spec.get("spec") or {}
    external = bool(spec_body.get("external")) or bool(spec_body.get("source_type"))
    existing_primitives = set(existing_primitives or [])
    if not tests:
        issues.append("no_tests")
    statuses = set()
    for test in tests:
        expected = test.get("expected") or {}
        status = expected.get("status")
        if status:
            statuses.add(status)
        context = test.get("context") or {}
        segments = context.get("segments") or []
        if not segments and not external:
            issues.append("test_missing_segments")
            break
        for seg in segments:
            if "id" not in seg or "text" not in seg:
                issues.append("segment_missing_id_or_text")
                break
        if issues:
            break
    if "ok" not in statuses:
        issues.append("no_ok_test")
    if kind == "skill":
        dag = spec_body.get("dag") or []
        uses_ctrl = bool(spec_body.get("uses_controlled_llm"))
        if not dag and not uses_ctrl:
            issues.append("skill_without_dag_or_controlled_llm")
        dag_prims = extract_dag_primitives(dag)
        if dag_prims:
            missing_prims = [name for name in dag_prims if name not in existing_primitives]
            primitive_specs = spec.get("primitive_specs") or []
            prim_names = {prim.get("name") for prim in primitive_specs if prim.get("name")}
            if missing_prims and not primitive_specs:
                issues.append("missing_primitive_specs")
            else:
                missing_specs = [name for name in missing_prims if name not in prim_names]
                if missing_specs:
                    issues.append("missing_primitive_specs_for:" + ",".join(sorted(missing_specs)))
            for test in tests:
                stubs = test.get("primitive_stubs") or {}
                if not stubs:
                    issues.append("missing_primitive_stubs")
                    break
                if any(name not in stubs for name in dag_prims):
                    issues.append("incomplete_primitive_stubs")
                    break
    outputs = spec_body.get("outputs") or {}
    if "prov" not in outputs:
        issues.append("outputs_missing_prov")
    goal_text = " ".join(
        [
            str(spec.get("need_summary") or ""),
            str(spec_body.get("goal") or ""),
            str(spec.get("name") or ""),
        ]
    )
    if _text_has_any(goal_text, SPEC_VAGUE_TERMS) and not _text_has_any(goal_text, SPEC_ACTION_TERMS):
        issues.append("vague_goal_or_summary")
    primitive_specs = spec.get("primitive_specs") or []
    for prim in primitive_specs:
        prim_issues = validate_spec_payload(prim, existing_primitives=existing_primitives)
        if prim_issues:
            name = prim.get("name") or "unknown"
            issues.append(f"primitive_spec_invalid:{name}:{','.join(prim_issues)}")
            break
    return issues


def build_template_prompt(cluster_samples, max_tests):
    system = (
        "You design templates for ScholarArena spec/tests. "
        "Be explicit, operational, and deterministic. "
        "Output exactly one JSON object, no markdown."
    )
    user = {
        "task": "Propose a spec/tests template that improves feasibility and coverage.",
        "clusters": cluster_samples,
        "schema_notes": [
            "Observation = {type, payload, prov, status}. prov is a list of segment ids from context.segments.",
            "context.segments is a list of {id:int, text:str}.",
            "Tests must be runnable without file I/O.",
            "Network access is allowed but tests should remain stable and time-bounded.",
        ],
        "constraints": [
            "Template must stay within the existing schema keys.",
            "Use ASCII only.",
            "Tests must be deterministic and runnable without file I/O.",
            "Keep tests minimal but sufficient.",
            "Prefer at least one ok test and one missing/fail test when possible.",
            "Avoid external dependencies unless absolutely required.",
            "If external retrieval is needed, require explicit timeouts and simple checks.",
        ],
        "output_schema": {
            "spec_template": {
                "goal": "short description",
                "inputs": {"context": "dict", "params": "dict"},
                "outputs": {"type": "string", "payload": "dict", "prov": "list[int]"},
                "constraints": ["deterministic", "no network"],
                "dag": [],
                "uses_controlled_llm": False,
            },
            "test_template": {
                "name": "short test name",
                "context": {"segments": [{"id": 1, "text": "..."}]},
                "params": {"query": "value"},
                "primitive_stubs": {},
                "expected": {
                    "status": "ok|missing|fail",
                    "prov_contains": [1],
                    "payload_keys": ["key1", "key2"],
                },
            },
            "guidelines": ["short bullet strings"],
        },
        "max_tests": max_tests,
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
    ]
    return messages


def build_spec_prompt(samples, max_tests, existing_names, existing_primitives, template=None, issues=None):
    definition = (
        "Primitive: deterministic operator p(C, phi) -> Observation. "
        "Skill: deterministic DAG over primitives and controlled LLM subroutines "
        "that only normalize observed evidence (no new claims)."
    )
    system = (
        "You design evidence needs for ScholarArena. "
        "Output exactly one JSON object, no markdown. Keep it feasible and deterministic."
    )
    user = {
        "task": "Summarize the evidence need for this cluster and propose a spec/tests.",
        "definitions": definition,
        "samples": samples,
        "schema_notes": [
            "Observation = {type, payload, prov, status}. prov is a list of segment ids from context.segments.",
            "context.segments is a list of {id:int, text:str}.",
            "Tests must be runnable without file I/O.",
            "Network access is allowed but should be time-bounded and stable.",
            "Ignore subjective factors (novelty, significance, robustness, importance).",
        ],
        "constraints": [
            "Choose kind=primitive when a single deterministic operator is enough.",
            "Choose kind=skill only when a DAG of primitives is required.",
            "Tests must be small, deterministic, and runnable without file I/O.",
            "Limit tests to the provided max_tests.",
            "Prefer at least one ok test and one missing/fail test when possible.",
            "If an existing name fits, reuse it exactly.",
            "Avoid external dependencies unless absolutely required.",
            "Use ASCII only.",
            "Only propose tasks that can be deterministically derived from context segments.",
            "If the need is subjective or not directly computable, set action=skip with a reason.",
            "If external retrieval is needed, mark spec.external=true and include source_type.",
            "Do not create subjective scores or judgments; only objective retrieval or computation.",
            "Generate tests with explicit expected outputs that follow from the given context or retrieval.",
            "If kind=skill and DAG references primitives not in existing_primitives, include primitive_specs with tests.",
            "Skill tests must include primitive_stubs for all referenced primitives.",
        ],
        "existing_names": existing_names or [],
        "existing_primitives": existing_primitives or [],
        "quality_issues": issues or [],
        "output_schema": {
            "action": "accept|skip",
            "skip_reason": "if action=skip, provide a short reason",
            "kind": "primitive|skill",
            "name": "Title_Case identifier",
            "need_summary": "one sentence",
            "spec": {
                "goal": "short description",
                "inputs": {"context": "dict", "params": "dict"},
                "outputs": {"type": "string", "payload": "dict", "prov": "list[int]"},
                "constraints": ["deterministic"],
                "dag": "required for skill, list of steps (each step should reference a primitive name or controlled_llm)",
                "uses_controlled_llm": "bool (skill only)",
                "external": "bool (true if external retrieval is required)",
                "source_type": "list of strings, e.g., arxiv|semantic_scholar|doi|github|web",
            },
            "primitive_specs": [
                {
                    "kind": "primitive",
                    "name": "Title_Case identifier",
                    "need_summary": "one sentence",
                    "spec": {
                        "goal": "short description",
                        "inputs": {"context": "dict", "params": "dict"},
                        "outputs": {"type": "string", "payload": "dict", "prov": "list[int]"},
                        "constraints": ["deterministic"],
                    },
                    "tests": [
                        {
                            "name": "short test name",
                            "context": {"segments": [{"id": 1, "text": "..."}]},
                            "params": {"query": "value"},
                            "expected": {
                                "status": "ok|missing|fail",
                                "prov_contains": [1],
                                "payload_keys": ["key1", "key2"],
                            },
                        }
                    ],
                    "dependencies": ["pip_package_name"],
                }
            ],
            "tests": [
                {
                    "name": "short test name",
                    "context": {"segments": [{"id": 1, "text": "..."}, {"id": 2, "text": "..."}]},
                    "params": {"query": "value"},
                    "primitive_stubs": {"PrimitiveName": {"type": "text", "payload": {}, "prov": [1], "status": "ok"}},
                    "expected": {
                        "status": "ok|missing|fail",
                        "prov_contains": [1],
                        "payload_keys": ["key1", "key2"],
                    },
                }
            ],
            "dependencies": ["pip_package_name"],
        },
        "max_tests": max_tests,
        "format_rules": [
            "Return a single JSON object, not a list.",
            "Use ASCII only.",
            "If kind=primitive, omit dag or leave it empty.",
            "If tests do not need primitive_stubs, omit that field.",
        ],
    }
    if template:
        user["template"] = template
        user["template_rules"] = [
            "If template is provided, follow its structure and fill concrete values.",
            "Do not introduce new required keys beyond output_schema.",
        ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
    ]
    return messages


def build_codegen_prompt(spec_payload, error_text=None):
    system = (
        "You write deterministic Python code for ScholarArena primitives/skills. "
        "Output exactly one JSON object with a 'code' field, no markdown."
    )
    constraints = [
        "Use Python 3.10 standard library only.",
        "Network calls are allowed but must use short timeouts and simple GET/POST.",
        "No file I/O, no randomness, no time-based logic.",
        "Provide execute(context, params, primitives=None, controlled_llm=None).",
        "Return an Observation dict with keys: type, payload, prov, status.",
        "Use runtime.ok/missing/fail from runtime.py.",
        "If inputs are missing or evidence not found, return missing().",
        "Never raise; catch errors and return fail(error=...).",
        "Do not print or log.",
        "Primitive functions must be called as primitives[name](context, params).",
        "controlled_llm must be called as controlled_llm(prompt, evidence).",
    ]
    user = {
        "task": "Generate code that satisfies the spec and tests.",
        "spec": spec_payload,
        "behavior_notes": [
            "context is a dict; if present, context.segments is a list of {id:int, text:str}.",
            "For skills, use primitives[name] when available; if a required primitive is missing, return fail().",
            "Use controlled_llm only if spec.spec.uses_controlled_llm is true; otherwise ignore it.",
            "If spec.spec.external is true, include payload.sources with URLs or identifiers.",
            "If external retrieval fails or times out, return missing() or fail() with an error message.",
            "Keep code small and readable; no external dependencies.",
            "When calling primitives, pass context and params dicts (not arbitrary kwargs).",
        ],
        "constraints": constraints,
        "error_context": error_text or "",
        "output_schema": {"code": "python code only, no markdown"},
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
    ]
    return messages


def run_tests(artifact_dir, timeout_seconds, env):
    result = subprocess.run(
        [sys.executable, "tests.py"],
        cwd=artifact_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return result


def summarize_failure(result, max_lines=20):
    lines = []
    if result.stdout:
        lines.extend(result.stdout.strip().splitlines())
    if result.stderr:
        lines.extend(result.stderr.strip().splitlines())
    return "\n".join(lines[-max_lines:])


def write_tests_runner(test_runner_path):
    test_runner_path = Path(test_runner_path)
    code = f"""import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
runtime_root = None
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "runtime.py").exists():
        runtime_root = parent
        break
if runtime_root is None:
    raise RuntimeError("runtime.py not found")
sys.path.insert(0, str(runtime_root))

from runtime import normalize_observation, controlled_llm_stub

CODE_PATH = Path(__file__).parent / "code.py"
TESTS_PATH = Path(__file__).parent / "tests.json"

spec = importlib.util.spec_from_file_location("artifact", CODE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
execute = getattr(module, "execute")

def make_stub_primitives(stubs):
    primitives = {{}}
    if not stubs:
        return primitives
    for name, obs in stubs.items():
        obs_copy = json.loads(json.dumps(obs))
        def _fn(*args, **kwargs):
            return _obs
        primitives[name] = _fn
    return primitives

def assert_payload_contains(payload, expected):
    for key, value in expected.items():
        if key not in payload:
            raise AssertionError(f"payload missing key: {{key}}")
        actual = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value not in str(actual):
                raise AssertionError(f"payload key {{key}} does not contain '{{value}}'")
        else:
            if actual != value:
                raise AssertionError(f"payload key {{key}} expected {{value}} got {{actual}}")

def run():
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    for test in tests:
        context = test.get("context") or {{}}
        params = test.get("params") or {{}}
        stubs = test.get("primitive_stubs")
        primitives = make_stub_primitives(stubs)
        obs = execute(context, params, primitives=primitives, controlled_llm=controlled_llm_stub)
        obs = normalize_observation(obs)
        expected = test.get("expected") or {{}}
        if expected.get("status") and obs.get("status") != expected.get("status"):
            raise AssertionError(f"status expected {{expected.get('status')}} got {{obs.get('status')}}")
        if expected.get("type") and obs.get("type") != expected.get("type"):
            raise AssertionError(f"type expected {{expected.get('type')}} got {{obs.get('type')}}")
        if expected.get("prov_contains"):
            for item in expected.get("prov_contains"):
                if item not in obs.get("prov", []):
                    raise AssertionError(f"prov missing {{item}}")
        if expected.get("prov_len") is not None:
            if len(obs.get("prov", [])) != expected.get("prov_len"):
                raise AssertionError("prov length mismatch")
        if expected.get("payload_keys"):
            for key in expected.get("payload_keys"):
                if key not in obs.get("payload", {{}}):
                    raise AssertionError(f"payload missing key: {{key}}")
        if expected.get("payload_contains"):
            assert_payload_contains(obs.get("payload", {{}}), expected.get("payload_contains"))
    print("ok")

if __name__ == "__main__":
    run()
"""
    test_runner_path.write_text(code, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine evidence needs and compile primitives/skills.")
    parser.add_argument("--in", dest="input_path", default="data/interim/semantic_acts.jsonl")
    parser.add_argument("--out-dir", default="steps/02_mine_evidence_needs")
    parser.add_argument("--clusters-out", default="")
    parser.add_argument("--assignments-out", default="")
    parser.add_argument("--clusters-in", default="", help="Reuse clusters from this JSONL file")
    parser.add_argument("--reuse-clusters", action="store_true", help="Skip clustering and reuse existing clusters")
    parser.add_argument("--specs-out", default="")
    parser.add_argument("--library-index-out", default="")
    parser.add_argument("--requirements-out", default="")
    parser.add_argument("--stage", choices=["cluster", "spec", "codegen", "codegen-only"], default="codegen")
    parser.add_argument("--specs-in", default="")
    parser.add_argument("--reuse-specs", action="store_true", help="Skip spec generation and reuse existing specs")

    parser.add_argument("--limit-acts", type=int, default=0, help="Process only the first N acts")
    parser.add_argument("--sample-acts", type=int, default=0, help="Randomly sample N acts for a quick test run")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--need-text-mode",
        default="intent+action+tools",
        choices=["intent", "action", "tools", "intent+action", "intent+action+tools", "intent+action+grounding", "intent+action+tools+grounding", "intent+action+tools+grounding+source"],
    )

    parser.add_argument("--cluster-k", type=int, default=0, help="Number of clusters (0 = auto)")
    parser.add_argument("--target-cluster-size", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-chunk-size", type=int, default=2048)
    parser.add_argument("--kmeans-log-every", type=int, default=0)
    parser.add_argument("--kmeans-method", choices=["full", "minibatch", "auto"], default="auto")
    parser.add_argument("--kmeans-init", choices=["kmeans++", "random"], default="kmeans++")
    parser.add_argument("--kmeans-init-sample", type=int, default=0)
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)

    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--embed-base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--embed-api-key", default="")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-sleep", type=float, default=0.3)
    parser.add_argument("--embed-retries", type=int, default=3)
    parser.add_argument("--embed-resume", action="store_true")
    parser.add_argument("--embed-strategy", choices=["heuristic"], default="heuristic")
    parser.add_argument("--heuristic-bits", type=int, default=12)
    parser.add_argument("--heuristic-reps", type=int, default=1)
    parser.add_argument("--group-embeddings-out", default="")

    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-debug-dir", default="")
    parser.add_argument("--spec-sample-size", type=int, default=6)
    parser.add_argument("--spec-max-tests", type=int, default=3)
    parser.add_argument("--spec-retries", type=int, default=2)
    parser.add_argument(
        "--spec-template-mode",
        choices=["none", "static", "llm-global", "llm-per-cluster"],
        default="none",
    )
    parser.add_argument("--spec-template-out", default="")
    parser.add_argument("--spec-template-clusters", type=int, default=20)
    parser.add_argument("--max-clusters", type=int, default=0)

    parser.add_argument("--codegen-retries", type=int, default=3)
    parser.add_argument("--test-timeout", type=int, default=60)

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = out_dir / "progress.log"
    events_log = out_dir / "progress_events.jsonl"
    llm_usage_log = out_dir / "llm_usage.jsonl"
    llm_usage_summary = out_dir / "llm_usage_summary.json"

    def logger(message):
        log_line(progress_log, message, also_stdout=True)

    usage_totals = {
        "calls": 0,
        "tokens_reported_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_chars": 0,
        "completion_chars": 0,
    }
    stage_totals = {}

    def record_usage(response, messages, stage, model, duration, meta=None):
        usage = response.get("usage") if isinstance(response, dict) else {}
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        prompt_chars = count_prompt_chars(messages)
        completion_text = extract_response_text(response)
        completion_chars = len(completion_text)
        usage_totals["calls"] += 1
        usage_totals["prompt_chars"] += prompt_chars
        usage_totals["completion_chars"] += completion_chars
        if prompt_tokens is not None:
            usage_totals["tokens_reported_calls"] += 1
            usage_totals["prompt_tokens"] += int(prompt_tokens)
            usage_totals["completion_tokens"] += int(completion_tokens or 0)
            usage_totals["total_tokens"] += int(total_tokens or 0)

        stage_key = stage or "unknown"
        stage_entry = stage_totals.setdefault(
            stage_key,
            {
                "calls": 0,
                "tokens_reported_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
            },
        )
        stage_entry["calls"] += 1
        stage_entry["prompt_chars"] += prompt_chars
        stage_entry["completion_chars"] += completion_chars
        if prompt_tokens is not None:
            stage_entry["tokens_reported_calls"] += 1
            stage_entry["prompt_tokens"] += int(prompt_tokens)
            stage_entry["completion_tokens"] += int(completion_tokens or 0)
            stage_entry["total_tokens"] += int(total_tokens or 0)

        entry = {
            "ts": time.time(),
            "stage": stage_key,
            "model": model,
            "duration_sec": round(float(duration), 3),
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "usage": usage,
        }
        if meta:
            entry.update(meta)
        append_jsonl(llm_usage_log, entry)

    logger(f"[init] stage={args.stage} out_dir={out_dir}")
    log_event(events_log, {"event": "init", "stage": args.stage, "out_dir": str(out_dir)})
    clusters_out = Path(args.clusters_out) if args.clusters_out else out_dir / "need_clusters.jsonl"
    assignments_out = Path(args.assignments_out) if args.assignments_out else out_dir / "need_assignments.jsonl"
    specs_out = Path(args.specs_out) if args.specs_out else out_dir / "need_specs.jsonl"
    library_index_out = (
        Path(args.library_index_out) if args.library_index_out else out_dir / "library_index.jsonl"
    )
    requirements_out = Path(args.requirements_out) if args.requirements_out else out_dir / "requirements.txt"

    clusters = None
    specs = None

    if args.stage != "codegen-only":
        if args.reuse_clusters:
            clusters_path = Path(args.clusters_in) if args.clusters_in else clusters_out
            if not clusters_path.exists():
                raise SystemExit(f"Cluster file not found: {clusters_path}")
            clusters = read_json_or_jsonl(clusters_path)
            if not clusters:
                raise SystemExit(f"Cluster file is empty: {clusters_path}")
            logger(f"[cluster] reuse clusters from {clusters_path} clusters={len(clusters)}")
            log_event(
                events_log,
                {
                    "event": "cluster_reuse",
                    "clusters": len(clusters),
                    "clusters_path": str(clusters_path),
                },
            )
        else:
            cluster_start = time.time()
            logger(f"[stage] cluster start input={args.input_path}")
            log_event(events_log, {"event": "stage_start", "stage": "cluster"})
            acts = read_json_or_jsonl(args.input_path)
            if args.limit_acts > 0:
                acts = acts[: args.limit_acts]
            if args.sample_acts and args.sample_acts < len(acts):
                rng = random.Random(args.sample_seed)
                indices = sorted(rng.sample(range(len(acts)), args.sample_acts))
                acts = [acts[idx] for idx in indices]
                if not args.quiet:
                    print(f"[sample] acts={len(acts)}", file=sys.stderr)
                logger(f"[sample] acts={len(acts)} seed={args.sample_seed}")
            if not acts:
                write_jsonl(clusters_out, [])
                write_jsonl(assignments_out, [])
                logger("[stage] cluster empty input, exiting")
                log_event(events_log, {"event": "stage_end", "stage": "cluster", "status": "empty"})
                return

            need_texts = [build_need_text(act, args.need_text_mode) for act in acts]
            embed_key = args.embed_api_key or default_api_key()
            if not embed_key:
                raise SystemExit("Missing API key for embeddings. Provide --embed-api-key or set OPENAI_API_KEY.")
            logger(
                f"[cluster] acts={len(acts)} need_text_mode={args.need_text_mode} "
                f"embed_strategy=heuristic(required) target_cluster_size={args.target_cluster_size}"
            )

            cluster_to_indices = {}
            group_id_for_act = {}
            rng = random.Random(args.seed)
            indices = list(range(len(acts)))
            groups = group_by_heuristic(need_texts, indices, args.heuristic_bits)
            group_sizes = [len(members) for members in groups.values()]
            if not args.quiet:
                stats = describe_cluster_sizes(group_sizes)
                if stats:
                    size_min, size_median, size_mean, size_max = stats
                    print(
                        f"[heuristic] groups={len(groups)} size_min={size_min} "
                        f"size_median={size_median:.1f} size_mean={size_mean:.1f} size_max={size_max}",
                        file=sys.stderr,
                    )
            group_keys = sorted(groups.keys())
            group_to_index = {key: idx for idx, key in enumerate(group_keys)}
            rep_texts = []
            rep_group_indices = []
            for key in group_keys:
                local_idx = group_to_index[key]
                reps = pick_representatives(groups[key], need_texts, args.heuristic_reps, rng)
                for act_idx in reps:
                    rep_texts.append(need_texts[act_idx])
                    rep_group_indices.append(local_idx)
            rep_embeddings = embed_texts(
                rep_texts,
                model=args.embed_model,
                api_key=embed_key,
                base_url=args.embed_base_url,
                batch_size=args.embed_batch_size,
                max_retries=args.embed_retries,
                sleep_seconds=args.embed_sleep,
                log_prefix="embed-need-rep",
                save_path=None,
                resume=args.embed_resume,
            )
            rep_embeddings = normalize_vectors(rep_embeddings, in_place=True)

            group_embeddings = np.zeros((len(group_keys), rep_embeddings.shape[1]), dtype=np.float32)
            group_counts = np.zeros(len(group_keys), dtype=np.int32)
            for emb, group_idx in zip(rep_embeddings, rep_group_indices):
                group_embeddings[group_idx] += emb
                group_counts[group_idx] += 1
            for idx, count in enumerate(group_counts):
                if count > 0:
                    group_embeddings[idx] /= float(count)
            group_embeddings = normalize_vectors(group_embeddings, in_place=True)
            if args.group_embeddings_out:
                np.save(args.group_embeddings_out, group_embeddings)

            n = group_embeddings.shape[0]
            k = choose_k(n, args.target_cluster_size, args.cluster_k)
            labels, _ = kmeans(
                group_embeddings,
                k,
                max_iter=args.max_iter,
                seed=args.seed,
                chunk_size=args.kmeans_chunk_size,
                log_prefix="kmeans-needs" if args.kmeans_log_every else None,
                log_every=args.kmeans_log_every,
                method=args.kmeans_method,
                init_method=args.kmeans_init,
                init_sample=args.kmeans_init_sample,
                batch_size=args.kmeans_batch_size,
            )
            group_id_map = {}
            group_counter = 0
            for key in group_keys:
                group_id = f"group_{group_counter:06d}"
                group_id_map[key] = group_id
                group_counter += 1

            for key, members in groups.items():
                local_idx = group_to_index[key]
                cluster_id = int(labels[local_idx])
                cluster_to_indices.setdefault(cluster_id, []).extend(members)
                for act_idx in members:
                    group_id_for_act[act_idx] = group_id_map[key]

            clusters = []
            assignments = []
            for cluster_id, members in sorted(cluster_to_indices.items()):
                cluster_label = f"need_{cluster_id:04d}"
                samples = pick_samples(members, acts, need_texts, args.spec_sample_size, rng)
                clusters.append(
                    {
                        "cluster_id": cluster_label,
                        "size": len(members),
                        "samples": samples,
                    }
                )
                for act_idx in members:
                    assignment = {
                        "act_id": acts[act_idx].get("act_id"),
                        "cluster_id": cluster_label,
                    }
                    group_id = group_id_for_act.get(act_idx)
                    if group_id:
                        assignment["group_id"] = group_id
                    assignments.append(assignment)

            write_jsonl(clusters_out, clusters)
            write_jsonl(assignments_out, assignments)

            if not args.quiet:
                sizes = [len(members) for members in cluster_to_indices.values()]
                stats = describe_cluster_sizes(sizes)
                if stats:
                    size_min, size_median, size_mean, size_max = stats
                    print(
                        f"[cluster] clusters={len(sizes)} size_min={size_min} size_median={size_median:.1f} "
                        f"size_mean={size_mean:.1f} size_max={size_max}",
                        file=sys.stderr,
                    )
                    logger(
                        f"[cluster] clusters={len(sizes)} size_min={size_min} "
                        f"size_median={size_median:.1f} size_mean={size_mean:.1f} size_max={size_max}"
                    )

            cluster_duration = time.time() - cluster_start
            logger(f"[stage] cluster end duration={cluster_duration:.2f}s clusters={len(clusters)}")
            log_event(
                events_log,
                {
                    "event": "stage_end",
                    "stage": "cluster",
                    "duration_sec": round(cluster_duration, 3),
                    "clusters": len(clusters),
                },
            )

        if args.stage == "cluster":
            return

        if args.stage == "codegen" and (args.reuse_specs or args.specs_in):
            specs_path = Path(args.specs_in) if args.specs_in else specs_out
            if not specs_path.exists():
                raise SystemExit(f"Specs file not found: {specs_path}")
            specs = read_json_or_jsonl(specs_path)
            if not specs:
                raise SystemExit(f"Specs file is empty: {specs_path}")
            logger(f"[spec] reuse specs from {specs_path} specs={len(specs)}")
            log_event(
                events_log,
                {
                    "event": "spec_reuse",
                    "specs": len(specs),
                    "specs_path": str(specs_path),
                },
            )
        else:
            if not clusters:
                raise SystemExit("No clusters available. Run --stage cluster or use --reuse-clusters.")
            spec_start = time.time()
            logger(
                f"[stage] spec start clusters={len(clusters)} template_mode={args.spec_template_mode}"
            )
            log_event(
                events_log,
                {
                    "event": "stage_start",
                    "stage": "spec",
                    "clusters": len(clusters),
                    "template_mode": args.spec_template_mode,
                },
            )

            embed_key = args.embed_api_key or default_api_key()
            llm_key = args.llm_api_key or embed_key
            if not llm_key:
                raise SystemExit("Missing API key for LLM. Provide --llm-api-key or set OPENAI_API_KEY.")

            existing_names = []
            existing_primitives = []
            if library_index_out.exists():
                existing_entries = read_json_or_jsonl(library_index_out)
                existing_names = [item.get("name") for item in existing_entries if item.get("name")]
                existing_primitives = [
                    item.get("name")
                    for item in existing_entries
                    if item.get("name") and item.get("kind") == "primitive"
                ]

            spec_template = {}
            if args.spec_template_out:
                spec_template_path = Path(args.spec_template_out)
            else:
                default_name = (
                    "spec_test_template.jsonl"
                    if args.spec_template_mode == "llm-per-cluster"
                    else "spec_test_template.json"
                )
                spec_template_path = out_dir / default_name
            if args.spec_template_mode == "llm-global":
                template_clusters = clusters[: min(args.spec_template_clusters, len(clusters))]
                template_payload = None
                if template_clusters:
                    template_messages = build_template_prompt(template_clusters, args.spec_max_tests)
                    template_response = call_chat(
                        template_messages,
                        args.llm_model,
                        llm_key,
                        args.llm_base_url,
                        3,
                        0.5,
                        logger=logger,
                        usage_recorder=record_usage,
                        stage="template",
                        meta={"template_scope": "global", "clusters": len(template_clusters)},
                    )
                    template_content = extract_response_text(template_response)
                    template_parsed = extract_json(template_content) or {}
                    template_payload = normalize_template_payload(template_parsed)
                    if args.llm_debug_dir:
                        write_llm_debug(
                            args.llm_debug_dir,
                            "template",
                            {
                                "scope": "global",
                                "clusters": len(template_clusters),
                                "messages": template_messages,
                                "response": template_response,
                                "parsed": template_parsed,
                                "normalized": template_payload,
                            },
                        )
                spec_template = template_payload or {}
                spec_template_path.write_text(
                    json.dumps(spec_template, ensure_ascii=True, indent=2), encoding="utf-8"
                )
                logger(
                    f"[template] scope=global clusters={len(template_clusters)} "
                    f"saved={spec_template_path}"
                )
            elif args.spec_template_mode == "llm-per-cluster":
                spec_template_path.unlink(missing_ok=True)

            specs_out.unlink(missing_ok=True)
            requirements = set()
            skipped_specs = 0
            for idx, cluster in enumerate(clusters, start=1):
                if args.max_clusters and idx > args.max_clusters:
                    break
                cluster_id = cluster["cluster_id"]
                template_for_cluster = spec_template
                if args.spec_template_mode == "llm-per-cluster":
                    template_messages = build_template_prompt([cluster], args.spec_max_tests)
                    template_response = call_chat(
                        template_messages,
                        args.llm_model,
                        llm_key,
                        args.llm_base_url,
                        3,
                        0.5,
                        logger=logger,
                        usage_recorder=record_usage,
                        stage="template",
                        meta={"template_scope": "cluster", "cluster_id": cluster_id},
                    )
                    template_content = extract_response_text(template_response)
                    template_parsed = extract_json(template_content) or {}
                    template_for_cluster = normalize_template_payload(template_parsed)
                    append_jsonl(
                        spec_template_path,
                        {"cluster_id": cluster_id, "template": template_for_cluster},
                    )
                    if args.llm_debug_dir:
                        write_llm_debug(
                            args.llm_debug_dir,
                            "template",
                            {
                                "scope": "cluster",
                                "cluster_id": cluster_id,
                                "messages": template_messages,
                                "response": template_response,
                                "parsed": template_parsed,
                                "normalized": template_for_cluster,
                            },
                        )

                issues = []
                accepted = False
                for attempt in range(1, args.spec_retries + 1):
                    messages = build_spec_prompt(
                        cluster["samples"],
                        args.spec_max_tests,
                        existing_names,
                        existing_primitives,
                        template=template_for_cluster,
                        issues=issues,
                    )
                    response = call_chat(
                        messages,
                        args.llm_model,
                        llm_key,
                        args.llm_base_url,
                        3,
                        0.5,
                        logger=logger,
                        usage_recorder=record_usage,
                        stage="spec",
                        meta={
                            "cluster_id": cluster_id,
                            "cluster_size": cluster.get("size"),
                            "attempt": attempt,
                        },
                    )
                    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
                    parsed = extract_json(content) or {}
                    normalized = normalize_spec_payload(parsed, fallback_name=f"Need_{cluster_id}")
                    normalized["tests"] = (normalized.get("tests") or [])[: args.spec_max_tests]
                    normalized["cluster_id"] = cluster_id
                    issues = validate_spec_payload(normalized, existing_primitives=existing_primitives)
                    if normalized.get("action") == "skip":
                        skipped_specs += 1
                        logger(
                            f"[spec] cluster={cluster_id} action=skip reason={normalized.get('skip_reason')}"
                        )
                        log_event(
                            events_log,
                            {
                                "event": "spec_skip",
                                "cluster_id": cluster_id,
                                "reason": normalized.get("skip_reason"),
                            },
                        )
                        if args.llm_debug_dir:
                            write_llm_debug(
                                args.llm_debug_dir,
                                "spec",
                                {
                                    "cluster_id": cluster_id,
                                    "samples": cluster["samples"],
                                    "messages": messages,
                                    "response": response,
                                    "parsed": parsed,
                                    "normalized": normalized,
                                    "issues": issues,
                                },
                            )
                        break
                    if not issues:
                        append_jsonl(specs_out, normalized)
                        for dep in normalized.get("dependencies", []):
                            if dep:
                                requirements.add(dep)
                        if args.llm_debug_dir:
                            write_llm_debug(
                                args.llm_debug_dir,
                                "spec",
                                {
                                    "cluster_id": cluster_id,
                                    "samples": cluster["samples"],
                                    "messages": messages,
                                    "response": response,
                                    "parsed": parsed,
                                    "normalized": normalized,
                                    "issues": issues,
                                },
                            )
                        accepted = True
                        break
                    if args.llm_debug_dir:
                        write_llm_debug(
                            args.llm_debug_dir,
                            "spec",
                            {
                                "cluster_id": cluster_id,
                                "samples": cluster["samples"],
                                "messages": messages,
                                "response": response,
                                "parsed": parsed,
                                "normalized": normalized,
                                "issues": issues,
                            },
                        )
                if not accepted and issues:
                    skipped_specs += 1
                    logger(f"[spec] cluster={cluster_id} rejected issues={issues}")
                    log_event(
                        events_log,
                        {
                            "event": "spec_reject",
                            "cluster_id": cluster_id,
                            "issues": issues,
                        },
                    )
                if not args.quiet and args.log_every and idx % args.log_every == 0:
                    print(f"[spec] {idx}/{len(clusters)} clusters", file=sys.stderr)
                    logger(f"[spec] {idx}/{len(clusters)} clusters")

            if requirements:
                requirements_out.write_text("\n".join(sorted(requirements)) + "\n", encoding="utf-8")

            if args.stage == "spec":
                spec_duration = time.time() - spec_start
                logger(f"[stage] spec end duration={spec_duration:.2f}s skipped={skipped_specs}")
                log_event(
                    events_log,
                    {
                        "event": "stage_end",
                        "stage": "spec",
                        "duration_sec": round(spec_duration, 3),
                        "skipped": skipped_specs,
                    },
                )
                summary_payload = {
                    "totals": usage_totals,
                    "by_stage": stage_totals,
                }
                llm_usage_summary.write_text(
                    json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8"
                )
                logger(
                    f"[llm] calls={usage_totals['calls']} "
                    f"tokens_reported_calls={usage_totals['tokens_reported_calls']} "
                    f"total_tokens={usage_totals['total_tokens']} prompt_chars={usage_totals['prompt_chars']} "
                    f"completion_chars={usage_totals['completion_chars']}"
                )
                return

            specs = read_json_or_jsonl(specs_out)
    else:
        if not args.specs_in:
            raise SystemExit("--specs-in is required for stage=codegen-only")
        specs = read_json_or_jsonl(args.specs_in)

    llm_key = args.llm_api_key or default_api_key()
    if not llm_key:
        raise SystemExit("Missing API key for LLM. Provide --llm-api-key or set OPENAI_API_KEY.")

    if "spec_start" in locals():
        spec_duration = time.time() - spec_start
        logger(f"[stage] spec end duration={spec_duration:.2f}s")
        log_event(
            events_log,
            {
                "event": "stage_end",
                "stage": "spec",
                "duration_sec": round(spec_duration, 3),
            },
        )

    existing_primitives = set()
    if library_index_out.exists():
        existing_entries = read_json_or_jsonl(library_index_out)
        for entry in existing_entries:
            if entry.get("kind") == "primitive" and entry.get("name"):
                existing_primitives.add(entry.get("name"))

    spec_queue = []
    seen_primitives = set(existing_primitives)
    for spec in specs:
        prim_specs = spec.get("primitive_specs") or []
        for idx, prim in enumerate(prim_specs, start=1):
            prim_name = prim.get("name")
            if prim_name and prim_name in seen_primitives:
                continue
            prim_copy = dict(prim)
            prim_copy["kind"] = "primitive"
            prim_copy["cluster_id"] = f"{spec.get('cluster_id', 'cluster')}__prim_{idx:02d}"
            prim_copy["parent_cluster_id"] = spec.get("cluster_id")
            spec_queue.append(prim_copy)
            if prim_name:
                seen_primitives.add(prim_name)
        spec_queue.append(spec)

    codegen_start = time.time()
    logger(f"[stage] codegen start artifacts={len(spec_queue)}")
    log_event(
        events_log,
        {"event": "stage_start", "stage": "codegen", "artifacts": len(spec_queue)},
    )

    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    library_dir = out_dir / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(out_dir) + os.pathsep + env.get("PYTHONPATH", "")

    for idx, spec in enumerate(spec_queue, start=1):
        if args.max_clusters and idx > args.max_clusters:
            break
        if spec.get("action") == "skip":
            logger(f"[codegen] artifact skipped name={spec.get('name')}")
            continue
        kind = spec.get("kind", "primitive")
        name = spec.get("name") or f"Need_{idx:04d}"
        cluster_id = spec.get("cluster_id") or f"need_{idx:04d}"
        artifact_id = f"{kind}_{slugify(name)}_{cluster_id}"
        spec_for_codegen = dict(spec)
        spec_for_codegen.pop("primitive_specs", None)
        tests = spec.get("tests") or []
        if not tests:
            append_jsonl(
                out_dir / "artifact_attempts.jsonl",
                {
                    "artifact_id": artifact_id,
                    "cluster_id": cluster_id,
                    "attempt": 0,
                    "status": "no_tests",
                    "error": "spec has no tests",
                },
            )
            logger(f"[codegen] artifact={artifact_id} status=no_tests")
            continue
        last_error = ""
        for attempt in range(1, args.codegen_retries + 1):
            attempt_dir = attempts_dir / f"{artifact_id}_attempt_{attempt}"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            spec_path = attempt_dir / "spec.json"
            tests_path = attempt_dir / "tests.json"
            code_path = attempt_dir / "code.py"
            test_runner_path = attempt_dir / "tests.py"
            spec_path.write_text(json.dumps(spec, ensure_ascii=True, indent=2), encoding="utf-8")
            tests_path.write_text(json.dumps(tests, ensure_ascii=True, indent=2), encoding="utf-8")
            write_tests_runner(test_runner_path)

            messages = build_codegen_prompt(spec_for_codegen, error_text=last_error)
            response = call_chat(
                messages,
                args.llm_model,
                llm_key,
                args.llm_base_url,
                3,
                0.5,
                logger=logger,
                usage_recorder=record_usage,
                stage="codegen",
                meta={"artifact_id": artifact_id, "attempt": attempt, "cluster_id": cluster_id},
            )
            content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
            parsed = extract_json(content) or {}
            code = parsed.get("code") if isinstance(parsed, dict) else None
            if not code:
                code = extract_code(content)
            code_path.write_text(code, encoding="utf-8")

            test_start = time.time()
            result = run_tests(attempt_dir, args.test_timeout, env)
            test_duration = time.time() - test_start
            status = "pass" if result.returncode == 0 else "fail"
            error_text = summarize_failure(result)

            append_jsonl(
                out_dir / "artifact_attempts.jsonl",
                {
                    "artifact_id": artifact_id,
                    "cluster_id": cluster_id,
                    "attempt": attempt,
                    "status": status,
                    "error": error_text,
                },
            )
            logger(
                f"[codegen] artifact={artifact_id} attempt={attempt} "
                f"status={status} test_duration={test_duration:.2f}s"
            )
            log_event(
                events_log,
                {
                    "event": "codegen_attempt",
                    "artifact_id": artifact_id,
                    "cluster_id": cluster_id,
                    "attempt": attempt,
                    "status": status,
                    "duration_sec": round(test_duration, 3),
                },
            )

            if args.llm_debug_dir:
                write_llm_debug(
                    args.llm_debug_dir,
                    "codegen",
                    {
                        "artifact_id": artifact_id,
                        "attempt": attempt,
                        "messages": messages,
                        "response": response,
                        "status": status,
                        "error": error_text,
                    },
                )

            if status == "pass":
                target_dir = library_dir / kind / artifact_id
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(attempt_dir, target_dir)
                append_jsonl(
                    library_index_out,
                    {
                        "artifact_id": artifact_id,
                        "kind": kind,
                        "name": name,
                        "cluster_id": cluster_id,
                        "spec_path": str(target_dir / "spec.json"),
                        "code_path": str(target_dir / "code.py"),
                        "tests_path": str(target_dir / "tests.json"),
                        "code_sha256": sha256_file(target_dir / "code.py"),
                    },
                )
                logger(f"[codegen] artifact={artifact_id} status=pass saved={target_dir}")
                break

            last_error = error_text

        if not args.quiet and args.log_every and idx % args.log_every == 0:
            print(f"[codegen] {idx}/{len(spec_queue)} artifacts", file=sys.stderr)
            logger(f"[codegen] {idx}/{len(spec_queue)} artifacts")

    codegen_duration = time.time() - codegen_start
    logger(f"[stage] codegen end duration={codegen_duration:.2f}s")
    log_event(
        events_log,
        {
            "event": "stage_end",
            "stage": "codegen",
            "duration_sec": round(codegen_duration, 3),
        },
    )

    summary_payload = {
        "totals": usage_totals,
        "by_stage": stage_totals,
    }
    llm_usage_summary.write_text(
        json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    logger(
        f"[llm] calls={usage_totals['calls']} tokens_reported_calls={usage_totals['tokens_reported_calls']} "
        f"total_tokens={usage_totals['total_tokens']} prompt_chars={usage_totals['prompt_chars']} "
        f"completion_chars={usage_totals['completion_chars']}"
    )


if __name__ == "__main__":
    main()
