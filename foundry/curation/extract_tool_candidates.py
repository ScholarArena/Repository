import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import map_evidence_type, read_json_or_jsonl, write_jsonl


DEFAULT_EXECUTABLE_OPS = {
    "Extract_Span",
    "Extract_Equation",
    "Extract_Table",
    "Figure_Crop",
    "Resolve_Citation",
    "Local_Scholar_Search",
    "T_Test",
    "Bootstrap_CI",
    "Effect_Size",
    "Sympy_Simplify",
    "Dimensional_Check",
    "Sandbox_Run",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract tool candidates from issues.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output tool candidates JSONL")
    parser.add_argument(
        "--executable-operations",
        default="",
        help="Comma-separated operation names to treat as executable.",
    )
    parser.add_argument(
        "--executable-categories",
        default="",
        help="Comma-separated tool categories to treat as executable.",
    )
    return parser.parse_args()


def parse_set(value):
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def is_executable(operation, tool_category, allow_ops, allow_cats):
    if operation in allow_ops or tool_category in allow_cats:
        return True
    if operation in DEFAULT_EXECUTABLE_OPS:
        return True
    return False


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    allow_ops = parse_set(args.executable_operations)
    allow_cats = parse_set(args.executable_categories)

    out = []
    for rec in records:
        issue_id = rec.get("issue_id")
        calls = rec.get("latent_tool_calls") or []
        for idx, call in enumerate(calls):
            tool_category = call.get("tool_category")
            operation = call.get("operation")
            target_type = call.get("target_type")
            outcome = call.get("outcome")
            executability = "yes" if is_executable(operation, tool_category, allow_ops, allow_cats) else "no"
            out.append(
                {
                    "candidate_id": f"{issue_id}#{idx}",
                    "issue_id": issue_id,
                    "tool_category": tool_category,
                    "operation": operation,
                    "target_type": target_type,
                    "outcome": outcome,
                    "executability": executability,
                    "evidence_type": map_evidence_type(tool_category, operation),
                }
            )

    write_jsonl(args.output_path, out)


if __name__ == "__main__":
    main()
