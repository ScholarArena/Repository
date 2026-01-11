import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run, require_api_key


def main():
    grounded_in = os.environ.get("CLUSTER_INPUT", "data/interim/grounded_issues.jsonl")
    issues_out = os.environ.get("CLUSTER_OUT", "data/processed/issues.jsonl")
    embeddings = os.environ.get("CLUSTER_EMBEDDINGS", "data/raw/embeddings.npy")

    embed_input_mode = os.environ.get("EMBED_INPUT_MODE", "semantic_act")
    generate = os.environ.get("GENERATE_EMBEDDINGS", "1")
    embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
    embed_workers = os.environ.get("EMBED_WORKERS", "8")
    embed_batch = os.environ.get("EMBED_BATCH_SIZE", "64")
    resume = os.environ.get("EMBED_RESUME", "1")

    method = os.environ.get("CLUSTER_METHOD", "auto")
    num_clusters = os.environ.get("CLUSTER_K", "0")
    max_iter = os.environ.get("CLUSTER_MAX_ITER", "200")

    sample_size = os.environ.get("SAMPLE_SIZE", "")
    sample_mode = os.environ.get("SAMPLE_MODE", "head")

    cmd = [
        PYTHON,
        "foundry/issue_mining/cluster_issues.py",
        "--in",
        grounded_in,
        "--embeddings",
        embeddings,
        "--out",
        issues_out,
        "--embed-input-mode",
        embed_input_mode,
        "--method",
        method,
        "--num-clusters",
        num_clusters,
        "--max-iter",
        max_iter,
    ]

    if generate == "1":
        require_api_key()
        cmd.extend(
            [
                "--generate-embeddings",
                "--embed-model",
                embed_model,
                "--embed-workers",
                embed_workers,
                "--embed-batch-size",
                embed_batch,
            ]
        )
        if resume == "1":
            cmd.append("--resume")

    if sample_size:
        cmd.extend(["--sample-size", sample_size, "--sample-mode", sample_mode])

    run(cmd)
    run([PYTHON, "scripts/pipeline/checks.py", "clusters", "--in", issues_out])


if __name__ == "__main__":
    main()
