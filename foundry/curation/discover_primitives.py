import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import (
    map_evidence_type,
    normalize_tool_calls,
    now_iso,
    read_json_or_jsonl,
    slugify,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Discover deterministic primitives from tool calls.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output primitives registry JSON")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum occurrences to keep a primitive")
    parser.add_argument("--write-stubs", action="store_true", help="Write primitive stub files and tests")
    parser.add_argument("--primitives-dir", default="skills/primitives", help="Output primitives directory")
    parser.add_argument("--method", choices=["auto", "minibatch", "kmeans", "exact"], default="auto")
    parser.add_argument("--num-clusters", type=int, default=0, help="Number of clusters (0 = heuristic)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Assignment chunk size")
    parser.add_argument("--log-every", type=int, default=5000, help="Progress log interval")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    parser.set_defaults(normalize=True)
    parser.add_argument("--include-outcome", action="store_true", help="Include outcome text in embeddings")
    parser.add_argument("--include-strings", action="store_true", help="Include non-dict tool calls as operations")
    parser.add_argument("--embeddings", default="skills/primitives/embeddings.npy", help="Embeddings .npy path")
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings via API")
    parser.add_argument("--embed-model", default="text-embedding-3-large", help="Embedding model name")
    parser.add_argument(
        "--embed-base-url",
        default=os.environ.get("DMX_API_BASE_URL", "https://www.dmxapi.cn/v1/"),
        help="Embedding API base URL",
    )
    parser.add_argument(
        "--embed-api-key",
        default=os.environ.get("DMX_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        help="Embedding API key (DMX_API_KEY or OPENAI_API_KEY)",
    )
    parser.add_argument("--embed-batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--embed-workers", type=int, default=4, help="Concurrent embedding requests")
    parser.add_argument("--embed-timeout", type=int, default=360, help="Embedding API timeout (seconds)")
    parser.add_argument("--embed-max-retries", type=int, default=3, help="Embedding API retries")
    parser.add_argument("--embed-retry-sleep", type=float, default=2.0, help="Retry backoff base (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume embedding generation")
    parser.add_argument("--overwrite-embeddings", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--progress-path", default="", help="Optional progress JSON path")
    parser.add_argument("--out-issues", default="", help="Optional output issues JSONL with primitive_id")
    parser.add_argument("--out-assignments", default="", help="Optional tool-call assignments JSONL")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    return parser.parse_args()


def default_signature(tool_category, operation):
    if operation and "Extract" in operation:
        return {"paper_span": "SpanRef", "locator": "string"}
    if tool_category == "Literature_Cross_Check":
        return {"paper_span": "SpanRef", "query": "string"}
    if tool_category == "Quantitative_Analysis":
        return {"table": "TableRef", "metric": "string"}
    return {"paper_span": "SpanRef"}


def default_schema(tool_category, operation):
    if tool_category == "Literature_Cross_Check":
        return {"matches": "list", "missing_refs": "list"}
    if tool_category == "Quantitative_Analysis":
        return {"value": "number"}
    if operation and "Extract" in operation:
        return {"text": "string"}
    return {"note": "string"}


def build_call_text(call, include_outcome):
    tool_category = call.get("tool_category") or ""
    operation = call.get("operation") or ""
    target_type = call.get("target_type") or ""
    outcome = call.get("outcome") or ""
    parts = [tool_category, operation, target_type]
    if include_outcome and outcome:
        parts.append(f"outcome={outcome}")
    text = " | ".join([part for part in parts if part])
    return text or "unknown"


def load_embeddings(path, use_mmap=True):
    mmap_mode = "r" if use_mmap else None
    embeddings = np.load(path, mmap_mode=mmap_mode)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    return embeddings


def normalize_embeddings(embeddings):
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def heuristic_k(n):
    return int(max(10, min(2000, round(math.sqrt(n)))))


def request_embeddings(texts, model, api_key, base_url, timeout):
    if not api_key:
        raise ValueError("Missing API key. Set DMX_API_KEY or OPENAI_API_KEY, or pass --embed-api-key.")
    endpoint = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
    with url_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(body)
    if "data" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    ordered = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in ordered]


def embed_with_retries(texts, args):
    last_exc = None
    for attempt in range(args.embed_max_retries):
        try:
            return request_embeddings(
                texts,
                args.embed_model,
                args.embed_api_key,
                args.embed_base_url,
                args.embed_timeout,
            )
        except (url_error.HTTPError, url_error.URLError, RuntimeError, ValueError) as exc:
            last_exc = exc
            if attempt == args.embed_max_retries - 1:
                break
            sleep_for = args.embed_retry_sleep * (2 ** attempt)
            if not args.quiet:
                print(f"[embed] retry {attempt + 1}/{args.embed_max_retries} after {sleep_for:.1f}s: {exc}", file=sys.stderr)
            time.sleep(sleep_for)
    raise last_exc


def generate_embeddings(texts, args):
    total = len(texts)
    if not args.quiet:
        print(f"[embed] total={total} model={args.embed_model}", file=sys.stderr)

    out_path = Path(args.embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_path) if args.progress_path else out_path.with_suffix(out_path.suffix + ".progress.json")

    if args.overwrite_embeddings and out_path.exists():
        out_path.unlink()
    if args.overwrite_embeddings and progress_path.exists():
        progress_path.unlink()

    memmap = None
    dim = None
    completed_ranges = set()
    completed_rows = 0

    if out_path.exists():
        if not args.resume and not args.overwrite_embeddings and not args.quiet:
            print("[embed] embeddings file exists; resuming by default", file=sys.stderr)
        memmap = np.load(out_path, mmap_mode="r+")
        if memmap.ndim != 2 or memmap.shape[0] != total:
            raise ValueError(
                f"Existing embeddings shape {memmap.shape} does not match records {total}. "
                "Use --overwrite-embeddings to regenerate."
            )
        dim = memmap.shape[1]

    if progress_path.exists() and (args.resume or out_path.exists()):
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            for start, end in progress.get("completed", []):
                completed_ranges.add((int(start), int(end)))
            completed_rows = sum(end - start for start, end in completed_ranges)
        except Exception:
            completed_ranges = set()
            completed_rows = 0

    batches = [(start, min(total, start + args.embed_batch_size)) for start in range(0, total, args.embed_batch_size)]
    pending = [batch for batch in batches if batch not in completed_ranges]
    if not args.quiet and completed_rows:
        print(f"[embed] resume completed_rows={completed_rows}", file=sys.stderr)

    if not pending:
        if memmap is None:
            raise RuntimeError("No embeddings to resume and embeddings file missing.")
        return memmap

    def fetch_batch(start, end):
        batch = texts[start:end]
        embeddings = embed_with_retries(batch, args)
        if len(embeddings) != len(batch):
            raise RuntimeError(f"Embedding batch mismatch: got {len(embeddings)} expected {len(batch)}")
        return start, end, embeddings

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    with ThreadPoolExecutor(max_workers=max(1, args.embed_workers)) as executor:
        batch_iter = iter(pending)
        futures = {}
        for _ in range(min(args.embed_workers, len(pending))):
            start, end = next(batch_iter)
            futures[executor.submit(fetch_batch, start, end)] = (start, end)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                batch_start, batch_end, embeddings = future.result()
                if memmap is None:
                    dim = len(embeddings[0])
                    memmap = np.lib.format.open_memmap(out_path, mode="w+", dtype="float32", shape=(total, dim))
                memmap[batch_start:batch_end] = np.asarray(embeddings, dtype=np.float32)
                completed_ranges.add((batch_start, batch_end))
                completed_rows += batch_end - batch_start
                progress_path.write_text(
                    json.dumps(
                        {
                            "total": total,
                            "batch_size": args.embed_batch_size,
                            "completed": sorted(list(completed_ranges)),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if not args.quiet and args.log_every > 0 and completed_rows % args.log_every == 0:
                    print(f"[embed] {completed_rows}/{total} texts", file=sys.stderr)

                try:
                    start, end = next(batch_iter)
                    futures[executor.submit(fetch_batch, start, end)] = (start, end)
                except StopIteration:
                    pass

    if memmap is None:
        raise RuntimeError("No embeddings generated.")
    memmap.flush()
    return memmap


def compute_labels(chunk, centers, normalized):
    if normalized:
        scores = chunk @ centers.T
        labels = np.argmax(scores, axis=1)
        dist = 1.0 - scores[np.arange(len(labels)), labels]
    else:
        x_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        c_norm = np.sum(centers * centers, axis=1)
        distances = x_norm + c_norm - 2.0 * (chunk @ centers.T)
        labels = np.argmin(distances, axis=1)
        dist = distances[np.arange(len(labels)), labels]
    return labels, dist


def assign_all(embeddings, centers, normalized, chunk_size, log_every, quiet):
    n = embeddings.shape[0]
    labels = np.empty(n, dtype=np.int32)
    distances = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = embeddings[start:end]
        labels_chunk, dist_chunk = compute_labels(chunk, centers, normalized)
        labels[start:end] = labels_chunk
        distances[start:end] = dist_chunk
        if not quiet and log_every > 0 and end % log_every == 0:
            print(f"[assign] {end}/{n} calls", file=sys.stderr)
    return labels, distances


def kmeans_numpy(embeddings, num_clusters, batch_size, max_iter, seed, normalized, log_every, quiet):
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if num_clusters > n:
        raise ValueError("num_clusters cannot exceed number of samples")
    init_idx = rng.choice(n, size=num_clusters, replace=False)
    centers = embeddings[init_idx].astype(np.float32, copy=True)
    counts = np.zeros(num_clusters, dtype=np.int64)

    for iteration in range(1, max_iter + 1):
        batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
        batch = embeddings[batch_idx]
        labels, _ = compute_labels(batch, centers, normalized)
        for i, label in enumerate(labels):
            counts[label] += 1
            eta = 1.0 / counts[label]
            centers[label] = (1.0 - eta) * centers[label] + eta * batch[i]
        if not quiet and log_every > 0 and iteration % log_every == 0:
            print(f"[cluster] iter {iteration}/{max_iter}", file=sys.stderr)
    return centers


def kmeans_sklearn(embeddings, num_clusters, batch_size, max_iter, seed, method, quiet):
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for this method") from exc

    if method == "kmeans":
        if not quiet:
            print("[cluster] sklearn KMeans fit start", file=sys.stderr)
        model = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=seed, n_init=10)
        labels = model.fit_predict(embeddings)
        if not quiet:
            print("[cluster] sklearn KMeans fit done", file=sys.stderr)
        return model.cluster_centers_, labels

    if not quiet:
        print("[cluster] sklearn MiniBatchKMeans fit start", file=sys.stderr)
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_init=10,
    )
    labels = model.fit_predict(embeddings)
    if not quiet:
        print("[cluster] sklearn MiniBatchKMeans fit done", file=sys.stderr)
    return model.cluster_centers_, labels


def main():
    args = parse_args()
    issues = read_json_or_jsonl(args.input_path)

    call_entries = []
    for rec in issues:
        issue_id = rec.get("issue_id")
        act_id = rec.get("act_id") or issue_id
        calls = normalize_tool_calls(rec.get("latent_tool_calls"))
        for idx, call in enumerate(calls):
            if isinstance(call, dict):
                entry = {
                    "issue_id": issue_id,
                    "act_id": act_id,
                    "call_index": idx,
                    "tool_category": call.get("tool_category"),
                    "operation": call.get("operation"),
                    "target_type": call.get("target_type"),
                    "outcome": call.get("outcome"),
                }
                call_entries.append(entry)
            elif args.include_strings and isinstance(call, str) and call.strip():
                entry = {
                    "issue_id": issue_id,
                    "act_id": act_id,
                    "call_index": idx,
                    "tool_category": "unknown",
                    "operation": call.strip(),
                    "target_type": "unknown",
                    "outcome": "",
                }
                call_entries.append(entry)

    if not call_entries:
        raise RuntimeError("No tool calls found to discover primitives.")

    if not args.quiet:
        print(f"[load] issues={len(issues)} calls={len(call_entries)}", file=sys.stderr)

    labels = None
    if args.method != "exact":
        texts = [build_call_text(call, args.include_outcome) for call in call_entries]
        emb_path = Path(args.embeddings)
        if args.generate_embeddings:
            embeddings = generate_embeddings(texts, args)
            if not args.quiet:
                print(f"[embed] saved embeddings to {emb_path}", file=sys.stderr)
        else:
            if not emb_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
            embeddings = load_embeddings(args.embeddings)
            if embeddings.shape[0] != len(texts):
                raise ValueError(
                    f"Embedding count {embeddings.shape[0]} does not match tool calls {len(texts)}."
                )

        if args.normalize:
            embeddings = normalize_embeddings(embeddings)
            if not args.quiet:
                print("[load] embeddings normalized (L2)", file=sys.stderr)

        k = args.num_clusters if args.num_clusters > 0 else heuristic_k(len(call_entries))
        if not args.quiet:
            print(f"[cluster] method={args.method} k={k}", file=sys.stderr)

        if args.method in {"kmeans", "minibatch"}:
            _, labels = kmeans_sklearn(
                embeddings,
                k,
                args.batch_size,
                args.max_iter,
                args.seed,
                args.method,
                args.quiet,
            )
        else:
            try:
                _, labels = kmeans_sklearn(
                    embeddings,
                    k,
                    args.batch_size,
                    args.max_iter,
                    args.seed,
                    "minibatch",
                    args.quiet,
                )
            except Exception:
                centers = kmeans_numpy(
                    embeddings,
                    k,
                    args.batch_size,
                    args.max_iter,
                    args.seed,
                    args.normalize,
                    max(1, args.log_every // 100),
                    args.quiet,
                )
                labels, _ = assign_all(
                    embeddings,
                    centers,
                    args.normalize,
                    args.chunk_size,
                    args.log_every,
                    args.quiet,
                )
    else:
        labels = np.zeros(len(call_entries), dtype=np.int32)
        key_groups = defaultdict(list)
        for idx, call in enumerate(call_entries):
            key = (call.get("tool_category"), call.get("operation"), call.get("target_type"))
            key_groups[key].append(idx)
        sorted_keys = sorted(key_groups.items(), key=lambda x: (-len(x[1]), str(x[0])))
        label_map = {}
        for label, (key, idxs) in enumerate(sorted_keys):
            for idx in idxs:
                labels[idx] = label
            label_map[label] = key
        if not args.quiet:
            print(f"[cluster] method=exact clusters={len(label_map)}", file=sys.stderr)

    cluster_groups = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_groups[int(label)].append(call_entries[idx])

    primitives = []
    assignments = []
    for label, items in cluster_groups.items():
        if len(items) < args.min_count:
            continue
        tool_counts = Counter(call.get("tool_category") or "unknown" for call in items)
        op_counts = Counter(call.get("operation") or "unknown" for call in items)
        target_counts = Counter(call.get("target_type") or "unknown" for call in items)
        tool_category = tool_counts.most_common(1)[0][0]
        operation = op_counts.most_common(1)[0][0]
        target_type = target_counts.most_common(1)[0][0]
        evidence_type = map_evidence_type(tool_category, operation)

        name = operation or "unknown"
        base_id = f"prim_{label:04d}_{slugify(name)[:24]}"
        primitive_id = base_id

        definition = (
            f"Deterministic primitive to {operation or 'process evidence'} on "
            f"{target_type or 'paper artifacts'} (category: {tool_category})."
        )

        sample_calls = [
            {
                "tool_category": call.get("tool_category"),
                "operation": call.get("operation"),
                "target_type": call.get("target_type"),
                "outcome": call.get("outcome"),
            }
            for call in items[:3]
        ]

        primitives.append(
            {
                "primitive_id": primitive_id,
                "cluster_id": f"prim_cluster_{label:04d}",
                "name": name,
                "definition": definition,
                "tool_category": tool_category,
                "operation": operation,
                "target_type": target_type,
                "count": len(items),
                "evidence_type": evidence_type,
                "signature": default_signature(tool_category, operation),
                "evidence_schema": default_schema(tool_category, operation),
                "failure_codes": ["NOT_IMPLEMENTED"],
                "source_counts": {
                    "tool_category": dict(tool_counts),
                    "operation": dict(op_counts),
                    "target_type": dict(target_counts),
                },
                "sample_calls": sample_calls,
            }
        )

        for call in items:
            assignments.append(
                {
                    "issue_id": call.get("issue_id"),
                    "act_id": call.get("act_id"),
                    "call_index": call.get("call_index"),
                    "primitive_id": primitive_id,
                    "cluster_id": f"prim_cluster_{label:04d}",
                }
            )

    primitives.sort(key=lambda x: (-x.get("count", 0), x.get("primitive_id")))

    registry = {
        "generated_at": now_iso(),
        "method": args.method,
        "num_clusters": len(cluster_groups),
        "total_calls": len(call_entries),
        "primitives": primitives,
    }

    primitives_dir = Path(args.primitives_dir)
    primitives_dir.mkdir(parents=True, exist_ok=True)

    if args.write_stubs:
        for primitive in primitives:
            prim_dir = primitives_dir / primitive["primitive_id"]
            prim_dir.mkdir(parents=True, exist_ok=True)
            (prim_dir / "primitive.json").write_text(
                json.dumps(primitive, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            tests_dir = prim_dir / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            (tests_dir / "test_manifest.py").write_text(
                "import json\n"
                "from pathlib import Path\n\n"
                "def test_manifest_has_required_fields():\n"
                "    data = json.loads((Path(__file__).resolve().parents[1] / 'primitive.json').read_text())\n"
                "    assert 'primitive_id' in data\n"
                "    assert 'signature' in data\n"
                "    assert 'evidence_schema' in data\n",
                encoding="utf-8",
            )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(
        json.dumps(registry, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    if args.out_assignments:
        write_jsonl(args.out_assignments, assignments)

    if args.out_issues:
        prim_by_key = {
            (item.get("issue_id"), item.get("call_index")): item.get("primitive_id") for item in assignments
        }
        updated = []
        for rec in issues:
            issue_id = rec.get("issue_id")
            calls = rec.get("latent_tool_calls") or []
            new_calls = []
            for idx, call in enumerate(calls):
                if isinstance(call, dict):
                    prim_id = prim_by_key.get((issue_id, idx))
                    if prim_id:
                        call = dict(call)
                        call["primitive_id"] = prim_id
                    new_calls.append(call)
                else:
                    new_calls.append(call)
            rec = dict(rec)
            rec["latent_tool_calls"] = new_calls
            updated.append(rec)
        write_jsonl(args.out_issues, updated)


if __name__ == "__main__":
    main()
