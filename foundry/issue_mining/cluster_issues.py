import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster issues using embedding vectors.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume embedding generation if embeddings/progress files exist",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Overwrite existing embeddings and progress files",
    )
    parser.add_argument(
        "--progress-path",
        default="",
        help="Optional progress JSON path (default: embeddings.npy.progress.json)",
    )
    parser.add_argument(
        "--embed-input-mode",
        choices=["tool_calls", "intent_action"],
        default="tool_calls",
        help="How to build embedding input text",
    )
    parser.add_argument(
        "--no-embed-outcome",
        dest="embed_outcome",
        action="store_false",
        help="Disable including outcome in tool_calls embedding text",
    )
    parser.set_defaults(embed_outcome=True)
    parser.add_argument(
        "--no-embed-grounding-ref",
        dest="embed_grounding_ref",
        action="store_false",
        help="Disable including grounding_ref in embedding text",
    )
    parser.set_defaults(embed_grounding_ref=True)
    parser.add_argument("--num-clusters", type=int, default=0, help="Number of clusters (0 = heuristic)")
    parser.add_argument(
        "--method",
        choices=["auto", "minibatch", "kmeans"],
        default="auto",
        help="Clustering method",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Assignment chunk size")
    parser.add_argument("--log-every", type=int, default=5000, help="Progress log interval (issues)")
    parser.add_argument("--log-every-iter", type=int, default=10, help="Progress log interval (iterations)")
    parser.add_argument("--stats-out", default="", help="Optional JSON path for cluster stats")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    parser.set_defaults(normalize=True)
    parser.add_argument("--no-mmap", dest="mmap", action="store_false", help="Disable mmap for embeddings")
    parser.set_defaults(mmap=True)
    return parser.parse_args()


def derive_issue_type(issue):
    intent = issue.get("strategic_intent")
    if intent:
        return intent
    calls = issue.get("latent_tool_calls") or []
    if isinstance(calls, dict):
        return calls.get("operation") or "UNKNOWN"
    if isinstance(calls, str):
        return calls.strip()[:80] or "UNKNOWN"
    if isinstance(calls, list) and calls:
        first = calls[0]
        if isinstance(first, dict):
            return first.get("operation") or "UNKNOWN"
        if isinstance(first, str):
            return first.strip()[:80] or "UNKNOWN"
    return "UNKNOWN"


def load_embeddings(path, use_mmap):
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


def normalize_tool_calls(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (dict, str)):
                cleaned.append(item)
            else:
                cleaned.append(str(item))
        return cleaned
    return [str(value)]


def build_embedding_text(issue, mode, include_outcome, include_grounding_ref):
    if mode == "intent_action":
        intent = issue.get("strategic_intent") or ""
        action = issue.get("action") or ""
        text = " | ".join([part for part in [intent, action] if part])
        return text if text else "unknown"

    calls = normalize_tool_calls(issue.get("latent_tool_calls"))
    parts = []
    for call in calls:
        if isinstance(call, str):
            segment = call.strip()
            if segment:
                parts.append(segment)
            continue
        tool_category = call.get("tool_category") or ""
        operation = call.get("operation") or ""
        target_type = call.get("target_type") or ""
        outcome = call.get("outcome") or ""
        segment_parts = [tool_category, operation, target_type]
        if include_outcome and outcome:
            segment_parts.append(f"outcome={outcome}")
        segment = " | ".join([part for part in segment_parts if part])
        if segment:
            parts.append(segment)
    if parts:
        text = " || ".join(parts)
        if include_grounding_ref:
            span = issue.get("paper_span") or {}
            if span.get("status") != "not_required":
                grounding_ref = issue.get("grounding_ref") or ""
                if grounding_ref:
                    text = f"{text} || grounding_ref: {grounding_ref}"
        return text
    return build_embedding_text(issue, "intent_action", include_outcome, include_grounding_ref)


def request_embeddings(texts, model, api_key, base_url, timeout):
    if not api_key:
        raise ValueError("Missing API key. Set DMX_API_KEY or OPENAI_API_KEY, or pass --embed-api-key.")
    endpoint = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
    with url_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(body)
    if "data" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    ordered = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in ordered]


def embed_with_retries(texts, model, api_key, base_url, timeout, max_retries, retry_sleep, quiet):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return request_embeddings(texts, model, api_key, base_url, timeout)
        except (url_error.HTTPError, url_error.URLError, RuntimeError, ValueError) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            sleep_for = retry_sleep * (2 ** attempt)
            if not quiet:
                print(f"[embed] retry {attempt + 1}/{max_retries} after {sleep_for:.1f}s: {exc}", file=sys.stderr)
            time.sleep(sleep_for)
    raise last_exc


def generate_embeddings(records, args):
    texts = [
        build_embedding_text(rec, args.embed_input_mode, args.embed_outcome, args.embed_grounding_ref)
        for rec in records
    ]
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
        embeddings = embed_with_retries(
            batch,
            args.embed_model,
            args.embed_api_key,
            args.embed_base_url,
            args.embed_timeout,
            args.embed_max_retries,
            args.embed_retry_sleep,
            args.quiet,
        )
        if len(embeddings) != len(batch):
            raise RuntimeError(f"Embedding batch mismatch: got {len(embeddings)} expected {len(batch)}")
        return start, end, embeddings

    with ThreadPoolExecutor(max_workers=max(1, args.embed_workers)) as executor:
        batch_iter = iter(pending)
        futures = {}
        for _ in range(min(args.embed_workers, len(pending))):
            start, end = next(batch_iter)
            futures[executor.submit(fetch_batch, start, end)] = (start, end)

        while futures:
            future = next(as_completed(futures))
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
            print(f"[assign] {end}/{n} issues", file=sys.stderr)
    return labels, distances


def kmeans_numpy(
    embeddings,
    num_clusters,
    batch_size,
    max_iter,
    seed,
    normalized,
    log_every_iter,
    quiet,
):
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
        if not quiet and log_every_iter > 0 and iteration % log_every_iter == 0:
            print(f"[cluster] iter {iteration}/{max_iter}", file=sys.stderr)
    return centers


def kmeans_sklearn(
    embeddings,
    num_clusters,
    batch_size,
    max_iter,
    seed,
    method,
    quiet,
):
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
        return model.cluster_centers_, labels, getattr(model, "inertia_", None)

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
    return model.cluster_centers_, labels, getattr(model, "inertia_", None)


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    total = len(records)
    if not args.quiet:
        print(f"[load] records={total}", file=sys.stderr)

    emb_path = Path(args.embeddings)
    if args.generate_embeddings:
        embeddings = generate_embeddings(records, args)
        if not args.quiet:
            print(f"[embed] saved embeddings to {emb_path}", file=sys.stderr)
    else:
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        embeddings = load_embeddings(args.embeddings, args.mmap)
        if embeddings.shape[0] != total:
            raise ValueError(
                f"Embedding count {embeddings.shape[0]} does not match input records {total}. "
                "Ensure the input JSONL aligns with the embeddings order, or use --generate-embeddings."
            )
        if not args.quiet:
            print(f"[load] embeddings shape={embeddings.shape}", file=sys.stderr)

    if args.normalize:
        embeddings = normalize_embeddings(embeddings)
        if not args.quiet:
            print("[load] embeddings normalized (L2)", file=sys.stderr)

    k = args.num_clusters if args.num_clusters > 0 else heuristic_k(total)
    if not args.quiet:
        print(f"[cluster] method={args.method} k={k}", file=sys.stderr)

    centers = None
    labels = None
    inertia = None
    if args.method in {"kmeans", "minibatch"}:
        centers, labels, inertia = kmeans_sklearn(
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
            centers, labels, inertia = kmeans_sklearn(
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
                args.log_every_iter,
                args.quiet,
            )
            labels, distances = assign_all(
                embeddings,
                centers,
                args.normalize,
                args.chunk_size,
                args.log_every,
                args.quiet,
            )
            inertia = float(np.sum(distances))

    if labels is None:
        labels, distances = assign_all(
            embeddings,
            centers,
            args.normalize,
            args.chunk_size,
            args.log_every,
            args.quiet,
        )
        inertia = float(np.sum(distances))

    cluster_counts = Counter(labels.tolist())
    sizes = list(cluster_counts.values())
    size_min = min(sizes) if sizes else 0
    size_max = max(sizes) if sizes else 0
    size_mean = sum(sizes) / len(sizes) if sizes else 0
    size_median = float(np.median(sizes)) if sizes else 0
    if not args.quiet:
        print(
            f"[summary] clusters={len(cluster_counts)} size_min={size_min} "
            f"size_median={size_median:.1f} size_mean={size_mean:.1f} size_max={size_max}",
            file=sys.stderr,
        )
        top_clusters = cluster_counts.most_common(10)
        top_summary = ", ".join(f"{label}:{count}" for label, count in top_clusters)
        print(f"[summary] top_clusters: {top_summary}", file=sys.stderr)
        if inertia is not None:
            print(f"[summary] inertia={inertia:.3f}", file=sys.stderr)

    out = []
    for idx, rec in enumerate(records):
        label = int(labels[idx])
        rec["cluster_id"] = f"emb_{label:04d}"
        rec["issue_type"] = derive_issue_type(rec)
        meta = rec.get("meta") or {}
        meta["cluster_method"] = "embedding"
        meta["cluster_label"] = label
        rec["meta"] = meta
        out.append(rec)
    write_jsonl(args.output_path, out)

    if args.stats_out:
        stats = {
            "num_records": total,
            "num_clusters": len(cluster_counts),
            "cluster_counts": dict(cluster_counts),
            "size_min": size_min,
            "size_median": size_median,
            "size_mean": size_mean,
            "size_max": size_max,
            "inertia": inertia,
            "method": args.method,
            "num_clusters_requested": args.num_clusters,
            "seed": args.seed,
            "normalized": args.normalize,
        }
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(
            json.dumps(stats, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
