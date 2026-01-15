import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from executor import Library
from ledger import allowed_obs_ids, init_thread_state, next_role, summarize_state, update_state
from metrics import compute_metrics
from policy import PolicyClient
from utils import ensure_dir, iter_jsonl, read_json, read_jsonl, write_json, write_jsonl


def default_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DMX_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    )


def build_threads_from_acts(acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_issue: Dict[str, List[Dict[str, Any]]] = {}
    for act in acts:
        issue_id = act.get("issue_id")
        if not issue_id:
            continue
        by_issue.setdefault(issue_id, []).append(act)
    threads = []
    for issue_id, items in by_issue.items():
        items.sort(key=lambda x: x.get("timestamp") or 0)
        first = items[0]
        issue_text = first.get("action") or first.get("grounding_ref") or ""
        request_seed = None
        if (first.get("role") or "").lower().startswith("review"):
            request_seed = {"id": "R1", "text": issue_text}
        thread = {
            "issue_id": issue_id,
            "forum_id": first.get("forum_id") or first.get("forum"),
            "issue_tag": first.get("issue_type") or first.get("issue_cluster_id") or first.get("cluster_id"),
            "issue_text": issue_text,
            "severity": first.get("severity") or "medium",
            "budget": 6,
            "phase": "Open",
            "requests": [request_seed] if request_seed else [],
            "context_id": first.get("forum_id") or first.get("forum"),
        }
        threads.append(thread)
    return threads


def load_context_for_thread(
    thread: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    contexts_dir: Optional[str],
    default_context_path: Optional[str],
    papers_md_dir: Optional[str],
    md_max_chars: int,
) -> Dict[str, Any]:
    if default_context_path:
        if default_context_path in cache:
            return cache[default_context_path]
        context = load_context_file(default_context_path)
        cache[default_context_path] = context
        return context
    context_path = thread.get("context_path")
    if context_path:
        if context_path in cache:
            return cache[context_path]
        context = load_context_file(context_path)
        cache[context_path] = context
        return context
    context_id = thread.get("context_id")
    if contexts_dir and context_id:
        candidate_json = os.path.join(contexts_dir, f"{context_id}.json")
        candidate_jsonl = os.path.join(contexts_dir, f"{context_id}.jsonl")
        path = candidate_json if os.path.exists(candidate_json) else candidate_jsonl
        if os.path.exists(path):
            if path in cache:
                return cache[path]
            context = load_context_file(path)
            cache[path] = context
            return context
    if papers_md_dir and context_id:
        md_path = os.path.join(papers_md_dir, context_id, "auto", f"{context_id}.md")
        if os.path.exists(md_path):
            if md_path in cache:
                return cache[md_path]
            context = load_markdown_context(md_path, md_max_chars)
            cache[md_path] = context
            return context
    return {"segments": []}


def load_context_file(path: str) -> Dict[str, Any]:
    if path.endswith(".jsonl"):
        segments = []
        for record in iter_jsonl(path):
            if "id" in record and "text" in record:
                segments.append({"id": record["id"], "text": record["text"]})
        return {"segments": segments}
    return read_json(path)


def load_markdown_context(path: str, max_chars: int) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    segments = parse_markdown_segments(text, max_chars)
    return {"segments": segments}


def parse_markdown_segments(text: str, max_chars: int) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    segments: List[Dict[str, Any]] = []
    buffer: List[str] = []
    current_section = None

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        chunk = "\n".join(buffer).strip()
        buffer = []
        if not chunk:
            return
        for part in split_long_text(chunk, max_chars):
            segments.append({"id": len(segments) + 1, "text": part, "section": current_section})

    for line in lines:
        heading = parse_heading(line)
        if heading:
            flush()
            current_section = heading
            segments.append({"id": len(segments) + 1, "text": heading, "section": current_section})
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return segments


def parse_heading(line: str) -> Optional[str]:
    match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    if not match:
        return None
    return match.group(2).strip()


def split_long_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        parts.append(text[start:end])
        start = end
    return parts


def enforce_gating(action: Dict[str, Any], observation: Dict[str, Any], allowed_ids: List[str]) -> Dict[str, Any]:
    status = observation.get("status")
    claims = action.get("claims") or []
    filtered_claims = []
    if status == "ok":
        for claim in claims:
            cites = claim.get("cites") or []
            cites = [c for c in cites if c in allowed_ids]
            if cites:
                filtered_claims.append({"text": claim.get("text", ""), "cites": cites})
    action["claims"] = filtered_claims
    if status != "ok":
        action_type = action.get("action_type")
        if action_type not in {"clarification", "evidence_plan", "conditional_commitment", "request"}:
            action["action_type"] = "clarification"
    return action


def load_intents(path: Optional[str]) -> List[str]:
    if not path:
        return ["Request_Evidence", "Clarify", "Respond", "Close_Issue"]
    intents = []
    for record in read_jsonl(path):
        label = record.get("label") or record.get("intent")
        if label and label not in intents:
            intents.append(label)
    return intents or ["Request_Evidence", "Clarify", "Respond", "Close_Issue"]


def load_threads(path: Optional[str], acts_path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        return read_jsonl(path)
    if acts_path:
        acts = read_jsonl(acts_path)
        return build_threads_from_acts(acts)
    raise ValueError("Provide --threads-in or --acts-in")


def run_arena(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    ensure_dir(out_dir)

    threads_seed = load_threads(args.threads_in, args.acts_in)
    threads = [init_thread_state(seed) for seed in threads_seed]

    library = Library(args.library_index)
    skills = list(library.available_skills().keys()) + list(library.available_primitives().keys())
    intents = load_intents(args.intents_in)

    policy_mode = args.policy_mode
    policy = PolicyClient(
        model=args.policy_model,
        api_key=args.policy_api_key,
        base_url=args.policy_base_url,
        mode=policy_mode,
        log_raw=args.log_raw_llm,
    )

    events = []
    context_cache: Dict[str, Dict[str, Any]] = {}

    for round_id in range(args.max_rounds):
        active = [t for t in threads if t.get("phase") != "Closed" and t.get("budget", 0) > 0]
        if not active:
            break
        summaries = [summarize_state(t) for t in active]
        order = policy.schedule_threads(summaries)
        selected = [t for t in active if t.get("issue_id") in order]
        if args.threads_per_round and args.threads_per_round > 0:
            selected = selected[: args.threads_per_round]
        events.append({
            "type": "meta_schedule",
            "round": round_id,
            "selected": [t.get("issue_id") for t in selected],
        })

        for thread in selected:
            context = load_context_for_thread(
                thread,
                context_cache,
                args.contexts_dir,
                args.context_path,
                args.papers_md_dir,
                args.md_max_chars,
            )
            role = next_role(thread)
            plan = policy.plan_move(role, summarize_state(thread), intents, skills)
            observation = library.execute(plan.get("skill_call"), context)
            action = policy.compose_action(
                role,
                summarize_state(thread),
                observation,
                plan.get("intent"),
                allowed_obs_ids(thread, observation),
            )
            if not args.disable_gating:
                action = enforce_gating(action, observation, allowed_obs_ids(thread, observation))
            update_state(thread, role, plan.get("intent"), observation, action)
            gold_plan = None
            oracle_moves = thread.get("oracle_moves") or []
            oracle_index = int(thread.get("oracle_index") or 0)
            if oracle_index < len(oracle_moves):
                gold_plan = oracle_moves[oracle_index]
                thread["oracle_index"] = oracle_index + 1
            events.append({
                "type": "move",
                "round": round_id,
                "thread_id": thread.get("issue_id"),
                "role": role,
                "plan": plan,
                "gold_plan": gold_plan,
                "observation": observation,
                "action": action,
                "state": summarize_state(thread),
            })

    metrics = compute_metrics(events, threads)

    write_jsonl(os.path.join(out_dir, "events.jsonl"), events)
    write_jsonl(os.path.join(out_dir, "threads_final.jsonl"), threads)
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    if args.log_raw_llm:
        write_jsonl(os.path.join(out_dir, "raw_llm.jsonl"), policy.raw_logs)

    return {"run_id": run_id, "out_dir": out_dir, "metrics": metrics}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ScholarArena Online Arena.")
    parser.add_argument("--threads-in", default="")
    parser.add_argument("--acts-in", default="")
    parser.add_argument("--contexts-dir", default="")
    parser.add_argument("--context-path", default="")
    parser.add_argument("--papers-md-dir", default="data/raw/papers_md")
    parser.add_argument("--md-max-chars", type=int, default=900)
    parser.add_argument("--intents-in", default="")
    parser.add_argument(
        "--library-index",
        default="steps/02_mine_evidence_needs/library_index.jsonl",
    )
    parser.add_argument("--out-dir", default="steps/03_online_arena/outputs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--policy-model", default="gpt-4o-mini")
    parser.add_argument("--policy-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--policy-api-key", default=default_api_key())
    parser.add_argument("--policy-mode", choices=["llm", "stub"], default="llm")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--threads-per-round", type=int, default=0)
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument("--log-raw-llm", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.threads_in and not args.acts_in:
        parser.error("Provide --threads-in or --acts-in")
    if not args.policy_api_key and args.policy_mode == "llm":
        parser.error("Missing --policy-api-key or OPENAI_API_KEY")
    result = run_arena(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
