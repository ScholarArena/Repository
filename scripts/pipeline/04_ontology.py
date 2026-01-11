import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import PYTHON, run, require_api_key


def main():
    issues_in = os.environ.get("ONTOLOGY_INPUT", "data/processed/issues.jsonl")
    labels_out = os.environ.get("CLUSTER_LABELS", "foundry/ontology/cluster_labels.jsonl")
    intent_map = os.environ.get("INTENT_MAP", "foundry/ontology/intent_map.json")
    evidence_map = os.environ.get("EVIDENCE_MAP", "foundry/ontology/evidence_map.json")
    ontology_dir = os.environ.get("ONTOLOGY_DIR", "foundry/ontology")

    run_label_clusters = os.environ.get("RUN_LABEL_CLUSTERS", "1") == "1"
    run_label_intents = os.environ.get("RUN_LABEL_INTENTS", "1") == "1"
    run_label_evidence = os.environ.get("RUN_LABEL_EVIDENCE", "1") == "1"

    if run_label_clusters:
        require_api_key()
        run(
            [
                PYTHON,
                "foundry/ontology/label_clusters.py",
                "--in",
                issues_in,
                "--out",
                labels_out,
            ]
        )
        run([PYTHON, "scripts/pipeline/checks.py", "label_clusters", "--in", labels_out])

    if run_label_intents:
        require_api_key()
        run(
            [
                PYTHON,
                "foundry/ontology/label_intents.py",
                "--in",
                issues_in,
                "--out",
                intent_map,
            ]
        )
        run([PYTHON, "scripts/pipeline/checks.py", "label_intents", "--in", intent_map])

    if run_label_evidence:
        require_api_key()
        run(
            [
                PYTHON,
                "foundry/ontology/label_evidence_types.py",
                "--in",
                issues_in,
                "--out",
                evidence_map,
                "--category-fallback",
            ]
        )
        run([PYTHON, "scripts/pipeline/checks.py", "label_evidence", "--in", evidence_map])

    run(
        [
            PYTHON,
            "foundry/ontology/build_ontology.py",
            "--in",
            issues_in,
            "--out",
            ontology_dir,
            "--labels",
            labels_out,
            "--intent-map",
            intent_map,
            "--evidence-map",
            evidence_map,
        ]
    )
    run([PYTHON, "scripts/pipeline/checks.py", "ontology", "--dir", ontology_dir])


if __name__ == "__main__":
    main()
