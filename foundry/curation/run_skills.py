import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import load_registry, read_json_or_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run skills for issues and produce observations.")
    parser.add_argument("--issues", required=True, help="Input issues JSONL")
    parser.add_argument("--registry", required=True, help="Skills registry JSON")
    parser.add_argument("--out", required=True, help="Output observations JSONL")
    return parser.parse_args()


def load_entrypoint(entrypoint):
    path_str, func_name = entrypoint.split(":", 1)
    path = Path(path_str)
    module_name = f"skill_{path.stem}_{abs(hash(path_str))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def build_args(issue, call):
    args = {}
    if issue.get("paper_span"):
        args["paper_span"] = issue.get("paper_span")
    if call.get("target_type"):
        args["query"] = call.get("target_type")
    if call.get("outcome"):
        args["hint"] = call.get("outcome")
    return args


def index_skills_by_operation(skills):
    index = {}
    for skill in skills:
        source = skill.get("source") or {}
        operation = source.get("operation")
        if not operation:
            continue
        index.setdefault(operation, []).append(skill)
    return index


def index_skills_by_primitive(skills):
    index = {}
    for skill in skills:
        source = skill.get("source") or {}
        primitive_id = source.get("primitive_id")
        if primitive_id:
            index[primitive_id] = skill
    return index


def main():
    args = parse_args()
    issues = read_json_or_jsonl(args.issues)
    registry = load_registry(args.registry)
    skills = registry.get("skills", [])
    skills_by_op = index_skills_by_operation(skills)
    skills_by_primitive = index_skills_by_primitive(skills)

    out_records = []
    for issue in issues:
        issue_id = issue.get("issue_id")
        results = []
        failure_codes = []
        calls = issue.get("latent_tool_calls") or []
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            primitive_id = call.get("primitive_id")
            skill = skills_by_primitive.get(primitive_id) if primitive_id else None
            if not skill:
                operation = call.get("operation")
                candidates = skills_by_op.get(operation) or []
                skill = candidates[0] if candidates else None
            if not skill:
                continue
            entrypoint = skill.get("entrypoint")
            args_dict = build_args(issue, call)
            try:
                runner = load_entrypoint(entrypoint)
                response = runner(**args_dict)
                status = response.get("status") if isinstance(response, dict) else "ok"
                payload = response.get("payload") if isinstance(response, dict) else response
                results.append(
                    {
                        "call_id": f"c{idx}",
                        "tool": skill.get("name"),
                        "skill_id": skill.get("skill_id"),
                        "status": status or "ok",
                        "payload": payload,
                        "args": args_dict,
                    }
                )
            except Exception as exc:
                failure_codes.append("EXEC_ERROR")
                results.append(
                    {
                        "call_id": f"c{idx}",
                        "tool": skill.get("name"),
                        "skill_id": skill.get("skill_id"),
                        "status": "error",
                        "payload": None,
                        "args": args_dict,
                        "error": str(exc),
                    }
                )

        out_records.append(
            {
                "issue_id": issue_id,
                "results": results,
                "failure_codes": failure_codes,
            }
        )

    write_jsonl(args.out, out_records)


if __name__ == "__main__":
    main()
