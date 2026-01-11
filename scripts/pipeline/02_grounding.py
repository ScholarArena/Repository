import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run


def main():
    mining_flat = os.environ.get("MINING_FLAT", "data/interim/mining_flat.jsonl")
    papers_dir = os.environ.get("PAPERS_DIR", "data/raw/papers_md")
    grounded_out = os.environ.get("GROUNDED_ISSUES", "data/interim/grounded_issues.jsonl")
    log_every = os.environ.get("LOG_EVERY_ISSUES", "5000")
    no_snippet = os.environ.get("NO_SNIPPET", "1")

    cmd = [
        PYTHON,
        "foundry/issue_mining/link_grounding_ref.py",
        "--in",
        mining_flat,
        "--papers",
        papers_dir,
        "--out",
        grounded_out,
        "--log-every",
        log_every,
    ]
    if no_snippet == "1":
        cmd.append("--no-snippet")

    run(cmd)
    run([PYTHON, "scripts/pipeline/checks.py", "grounding", "--in", grounded_out])


if __name__ == "__main__":
    main()
