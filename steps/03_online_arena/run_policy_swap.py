import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from run_online_arena import run_arena
from utils import read_json, read_jsonl, write_json, write_jsonl


def load_policies(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    data = read_json(path)
    if isinstance(data, dict) and "policies" in data:
        return data["policies"]
    if isinstance(data, list):
        return data
    raise ValueError("Invalid policy config format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy swap experiment for ScholarArena.")
    parser.add_argument("--policies", required=True, help="JSON or JSONL file of policy configs")
    parser.add_argument("--threads-in", default="")
    parser.add_argument("--acts-in", default="")
    parser.add_argument("--contexts-dir", default="")
    parser.add_argument("--context-path", default="")
    parser.add_argument("--intents-in", default="")
    parser.add_argument("--library-index", default="steps/02_mine_evidence_needs/library_index.jsonl")
    parser.add_argument("--out-dir", default="steps/03_online_arena/outputs")
    parser.add_argument("--base-run-id", default="policy_swap")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--threads-per-round", type=int, default=0)
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument("--policy-mode", choices=["llm", "stub"], default="llm")
    parser.add_argument("--log-raw-llm", action="store_true")
    args = parser.parse_args()

    policies = load_policies(args.policies)
    if not policies:
        raise ValueError("No policies to run")

    results = []
    for policy in policies:
        policy_name = policy.get("name") or policy.get("model") or "policy"
        run_id = f"{args.base_run_id}_{policy_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        arena_args = copy.copy(args)
        arena_args.run_id = run_id
        arena_args.policy_model = policy.get("model") or policy.get("policy_model") or "gpt-4o-mini"
        arena_args.policy_base_url = policy.get("base_url") or "https://api.openai.com/v1"
        arena_args.policy_api_key = policy.get("api_key") or os.environ.get(
            policy.get("api_key_env", "OPENAI_API_KEY"), ""
        )
        arena_args.threads_in = args.threads_in
        arena_args.acts_in = args.acts_in
        arena_args.contexts_dir = args.contexts_dir
        arena_args.context_path = args.context_path
        arena_args.intents_in = args.intents_in
        arena_args.library_index = args.library_index
        arena_args.out_dir = args.out_dir
        arena_args.max_rounds = args.max_rounds
        arena_args.threads_per_round = args.threads_per_round
        arena_args.disable_gating = args.disable_gating
        arena_args.policy_mode = args.policy_mode
        arena_args.log_raw_llm = args.log_raw_llm
        if not arena_args.policy_api_key and arena_args.policy_mode == "llm":
            raise ValueError(f"Missing API key for policy: {policy_name}")
        result = run_arena(arena_args)
        results.append({
            "policy": policy_name,
            "model": arena_args.policy_model,
            "run_id": result["run_id"],
            "metrics": result["metrics"],
            "out_dir": result["out_dir"],
        })

    summary_path = os.path.join(args.out_dir, f"{args.base_run_id}_summary.json")
    write_json(summary_path, {"results": results})
    summary_jsonl = os.path.join(args.out_dir, f"{args.base_run_id}_summary.jsonl")
    write_jsonl(summary_jsonl, results)
    print(json.dumps({"summary": summary_path, "count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
