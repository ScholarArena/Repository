#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_tests(root, kind):
    roots = []
    if kind in {"all", "primitive"}:
        roots.append(root / "library" / "primitive")
    if kind in {"all", "skill"}:
        roots.append(root / "library" / "skill")
    tests = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("tests.py"):
            if path.name != "tests.py":
                continue
            if "__pycache__" in path.parts:
                continue
            tests.append(path)
    return sorted(tests)


def run_test(path, timeout):
    path = path.resolve()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    status = "pass" if result.returncode == 0 else "fail"
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return {
        "path": str(path),
        "artifact": path.parent.name,
        "status": status,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all primitive/skill tests in the library.")
    parser.add_argument("--root", default="steps/02_mine_evidence_needs")
    parser.add_argument("--kind", choices=["all", "primitive", "skill"], default="all")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tests = find_tests(root, args.kind)
    if not tests:
        print("no tests found", file=sys.stderr)
        return 1

    results = []
    for path in tests:
        try:
            result = run_test(path, args.timeout)
        except subprocess.TimeoutExpired:
            result = {
                "path": str(path),
                "artifact": path.parent.name,
                "status": "timeout",
                "returncode": None,
                "stdout": "",
                "stderr": f"timeout>{args.timeout}s",
            }
        results.append(result)
        print(f"{result['status']}: {result['artifact']}")
        if result["status"] != "pass" and result["stderr"]:
            print(result["stderr"])

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timed = sum(1 for r in results if r["status"] == "timeout")
    print(f"summary: pass={passed} fail={failed} timeout={timed} total={len(results)}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0 if failed == 0 and timed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
