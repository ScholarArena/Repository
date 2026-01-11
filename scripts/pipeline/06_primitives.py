import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run, require_api_key


def main():
    issues_in = os.environ.get("PRIMITIVES_INPUT", "data/processed/issues.jsonl")
    registry_out = os.environ.get("PRIMITIVES_REGISTRY", "skills/primitives/registry.json")
    embeddings = os.environ.get("PRIMITIVES_EMBEDDINGS", "skills/primitives/embeddings.npy")
    evidence_map = os.environ.get("EVIDENCE_MAP", "foundry/ontology/evidence_map.json")

    assignments_out = os.environ.get("PRIMITIVES_ASSIGNMENTS", "data/interim/primitive_assignments.jsonl")
    issues_out = os.environ.get("PRIMITIVES_ISSUES_OUT", "data/processed/issues_with_primitives.jsonl")

    method = os.environ.get("PRIMITIVES_METHOD", "auto")
    min_count = os.environ.get("PRIMITIVES_MIN_COUNT", "5")
    embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
    embed_workers = os.environ.get("EMBED_WORKERS", "8")
    embed_batch = os.environ.get("EMBED_BATCH_SIZE", "64")
    generate = os.environ.get("GENERATE_EMBEDDINGS", "1")
    resume = os.environ.get("EMBED_RESUME", "1")

    cmd = [
        PYTHON,
        "foundry/curation/discover_primitives.py",
        "--in",
        issues_in,
        "--out",
        registry_out,
        "--method",
        method,
        "--min-count",
        min_count,
        "--embeddings",
        embeddings,
        "--evidence-map",
        evidence_map,
        "--out-assignments",
        assignments_out,
        "--out-issues",
        issues_out,
        "--write-stubs",
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

    run(cmd)
    run(
        [
            PYTHON,
            "scripts/pipeline/checks.py",
            "primitives",
            "--registry",
            registry_out,
            "--assignments",
            assignments_out,
            "--issues",
            issues_out,
        ]
    )


if __name__ == "__main__":
    main()
