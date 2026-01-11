import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run


def main():
    steps = [
        "scripts/pipeline/01_parse.py",
        "scripts/pipeline/02_grounding.py",
        "scripts/pipeline/03_cluster.py",
        "scripts/pipeline/04_ontology.py",
        "scripts/pipeline/05_threads.py",
        "scripts/pipeline/06_primitives.py",
        "scripts/pipeline/07_skills_and_trajectories.py",
    ]
    for step in steps:
        run([PYTHON, step])


if __name__ == "__main__":
    main()
