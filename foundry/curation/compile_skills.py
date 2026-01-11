import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import load_registry, save_registry, slugify, read_json_or_jsonl, now_iso


def parse_args():
    parser = argparse.ArgumentParser(description="Compile primitives or tool candidates into skills.")
    parser.add_argument("--in", dest="input_path", default="", help="Tool candidates JSONL (deprecated)")
    parser.add_argument("--primitives", dest="primitives_path", default="", help="Primitives registry JSON")
    parser.add_argument("--skills-dir", dest="skills_dir", required=True, help="Skills library directory")
    parser.add_argument("--registry", dest="registry_path", required=True, help="Skills registry JSON")
    return parser.parse_args()


def default_signature(tool_category, operation):
    if operation and "Extract" in operation:
        return {"paper_span": "SpanRef", "locator": "string"}
    if tool_category == "Literature_Cross_Check":
        return {"paper_span": "SpanRef", "query": "string"}
    if tool_category == "Quantitative_Analysis":
        return {"table": "TableRef", "metric": "string"}
    return {"paper_span": "SpanRef"}


def default_evidence_schema(tool_category, operation):
    if tool_category == "Literature_Cross_Check":
        return {"matches": "list", "missing_refs": "list"}
    if tool_category == "Quantitative_Analysis":
        return {"value": "number"}
    if operation and "Extract" in operation:
        return {"text": "string"}
    return {"note": "string"}


def ensure_skill_files(skill_dir, manifest):
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "tests").mkdir(parents=True, exist_ok=True)
    skill_py = skill_dir / "skill.py"
    if not skill_py.exists():
        skill_py.write_text(
            "def run(**kwargs):\n"
            "    return {\"status\": \"not_implemented\", \"payload\": {\"args\": kwargs}}\n",
            encoding="utf-8",
        )
    (skill_dir / "skill.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (skill_dir / "tests" / "README.md").write_text(
        "Stub tests placeholder for skill.\n",
        encoding="utf-8",
    )


def load_primitives(path):
    data = read_json_or_jsonl(path)
    if len(data) == 1 and isinstance(data[0], dict) and "primitives" in data[0]:
        return data[0].get("primitives", [])
    return [item for item in data if isinstance(item, dict)]


def main():
    args = parse_args()
    registry = load_registry(args.registry_path)
    skills_dir = Path(args.skills_dir)

    existing_ids = {s.get("skill_id") for s in registry.get("skills", [])}
    new_skills = []

    if args.primitives_path:
        primitives = load_primitives(args.primitives_path)
        for prim in primitives:
            operation = prim.get("operation") or "unknown"
            tool_category = prim.get("tool_category") or "unknown"
            primitive_id = prim.get("primitive_id") or slugify(f"{operation}_{tool_category}")
            name = prim.get("name") or slugify(operation)
            base_id = slugify(f"skill_{primitive_id}")
            skill_id = base_id
            suffix = 1
            while skill_id in existing_ids:
                suffix += 1
                skill_id = f"{base_id}_{suffix}"

            manifest = {
                "skill_id": skill_id,
                "name": name,
                "definition": prim.get("definition"),
                "signature": prim.get("signature") or default_signature(tool_category, operation),
                "evidence_schema": prim.get("evidence_schema") or default_evidence_schema(tool_category, operation),
                "evidence_type": prim.get("evidence_type"),
                "failure_codes": prim.get("failure_codes") or ["NOT_IMPLEMENTED"],
                "entrypoint": f"{skills_dir / skill_id / 'skill.py'}:run",
                "source": {
                    "primitive_id": primitive_id,
                    "operation": operation,
                    "target_type": prim.get("target_type"),
                    "tool_category": tool_category,
                },
                "generated_at": now_iso(),
            }

            ensure_skill_files(skills_dir / skill_id, manifest)
            new_skills.append(manifest)
            existing_ids.add(skill_id)
    elif args.input_path:
        candidates = read_json_or_jsonl(args.input_path)
        for cand in candidates:
            if cand.get("executability") != "yes":
                continue
            operation = cand.get("operation") or "unknown"
            tool_category = cand.get("tool_category") or "unknown"
            base_id = slugify(f"{operation}_{tool_category}")
            skill_id = base_id
            suffix = 1
            while skill_id in existing_ids:
                suffix += 1
                skill_id = f"{base_id}_{suffix}"

            manifest = {
                "skill_id": skill_id,
                "name": slugify(operation),
                "signature": default_signature(tool_category, operation),
                "evidence_schema": default_evidence_schema(tool_category, operation),
                "failure_codes": ["NOT_IMPLEMENTED"],
                "entrypoint": f"{skills_dir / skill_id / 'skill.py'}:run",
                "source": {
                    "operation": operation,
                    "target_type": cand.get("target_type"),
                    "tool_category": tool_category,
                },
                "generated_at": now_iso(),
            }

            ensure_skill_files(skills_dir / skill_id, manifest)
            new_skills.append(manifest)
            existing_ids.add(skill_id)
    else:
        raise ValueError("Provide --primitives or --in for skill compilation.")

    registry.setdefault("skills", [])
    registry["skills"].extend(new_skills)
    registry["generated_at"] = now_iso()
    save_registry(args.registry_path, registry)


if __name__ == "__main__":
    main()
