import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster issues using embedding vectors.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
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
    if calls:
        return calls[0].get("operation") or "UNKNOWN"
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
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    embeddings = load_embeddings(args.embeddings, args.mmap)
    if embeddings.shape[0] != total:
        raise ValueError(
            f"Embedding count {embeddings.shape[0]} does not match input records {total}. "
            "Ensure the input JSONL aligns with the embeddings order."
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
