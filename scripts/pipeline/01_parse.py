import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, ROOT, run


def main():
    mining_raw = os.environ.get("MINING_RAW", "data/raw/mining_results.jsonl")
    mining_flat = os.environ.get("MINING_FLAT", "data/interim/mining_flat.jsonl")
    log_every = os.environ.get("LOG_EVERY_PAPERS", "100")

    run(
        [
            PYTHON,
            "foundry/issue_mining/parse_mining_results.py",
            "--in",
            mining_raw,
            "--out",
            mining_flat,
            "--log-every",
            log_every,
        ]
    )
    run([PYTHON, "scripts/pipeline/checks.py", "issues", "--in", mining_flat])


if __name__ == "__main__":
    main()
