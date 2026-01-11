import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run


def main():
    primitives_registry = os.environ.get("PRIMITIVES_REGISTRY", "skills/primitives/registry.json")
    skills_dir = os.environ.get("SKILLS_DIR", "skills/library")
    skills_registry = os.environ.get("SKILLS_REGISTRY", "skills/registry.json")

    issues_in = os.environ.get("PRIMITIVES_ISSUES_OUT", "data/processed/issues_with_primitives.jsonl")
    observations_out = os.environ.get("OBSERVATIONS_OUT", "data/interim/observations.jsonl")
    thread_index = os.environ.get("THREAD_INDEX", "data/processed/thread_index.jsonl")
    trajectories_out = os.environ.get("TRAJECTORIES_OUT", "data/processed/trajectories.jsonl")

    run(
        [
            PYTHON,
            "foundry/curation/compile_skills.py",
            "--primitives",
            primitives_registry,
            "--skills-dir",
            skills_dir,
            "--registry",
            skills_registry,
        ]
    )
    run([PYTHON, "scripts/pipeline/checks.py", "skills", "--in", skills_registry])

    run(
        [
            PYTHON,
            "foundry/curation/run_skills.py",
            "--issues",
            issues_in,
            "--registry",
            skills_registry,
            "--out",
            observations_out,
        ]
    )
    run([PYTHON, "scripts/pipeline/checks.py", "observations", "--in", observations_out])

    run(
        [
            PYTHON,
            "foundry/curation/curate_trajectories.py",
            "--issues",
            issues_in,
            "--obs",
            observations_out,
            "--thread-index",
            thread_index,
            "--out",
            trajectories_out,
        ]
    )
    run([PYTHON, "scripts/pipeline/checks.py", "trajectories", "--in", trajectories_out])


if __name__ == "__main__":
    main()
