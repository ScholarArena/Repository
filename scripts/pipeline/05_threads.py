import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run


def main():
    issues_in = os.environ.get("THREADS_INPUT", "data/processed/issues.jsonl")
    labels = os.environ.get("CLUSTER_LABELS", "foundry/ontology/cluster_labels.jsonl")
    threads_out = os.environ.get("THREADS_OUT", "data/processed/threads.jsonl")
    index_out = os.environ.get("THREAD_INDEX", "data/processed/thread_index.jsonl")
    issues_out = os.environ.get("THREADS_ISSUES_OUT", "data/processed/issues_with_threads.jsonl")

    cmd = [
        PYTHON,
        "foundry/issue_mining/build_issue_threads.py",
        "--in",
        issues_in,
        "--labels",
        labels,
        "--out-threads",
        threads_out,
        "--out-index",
        index_out,
    ]
    if issues_out:
        cmd.extend(["--out-issues", issues_out])

    run(cmd)
    run([PYTHON, "scripts/pipeline/checks.py", "threads", "--in", threads_out])


if __name__ == "__main__":
    main()
