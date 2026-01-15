import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODULE_DIR = os.path.dirname(__file__)
RUNTIME_DIR = os.path.join(MODULE_DIR, "..", "02_mine_evidence_needs")
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

try:
    import runtime
except Exception as exc:
    raise RuntimeError(f"Failed to import runtime from {RUNTIME_DIR}: {exc}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


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
            segments.append({
                "id": len(segments) + 1,
                "text": part,
                "section": current_section,
            })

    for line in lines:
        heading = parse_heading(line)
        if heading:
            flush()
            current_section = heading
            segments.append({
                "id": len(segments) + 1,
                "text": heading,
                "section": current_section,
            })
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


def make_observation_id(obs: Dict[str, Any]) -> str:
    payload = {"type": obs.get("type", "evidence"), "prov": obs.get("prov") or []}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"O{digest[:12]}"


class Library:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.primitives: Dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for record in read_jsonl(self.index_path):
            name = record.get("name")
            if not name:
                continue
            self.artifacts[name] = {
                "kind": record.get("kind"),
                "code_path": record.get("code_path"),
            }
        self._loaded = True

    def _load_execute(self, name: str):
        entry = self.artifacts.get(name)
        if not entry:
            return None
        code_path = entry.get("code_path")
        if not code_path:
            return None
        spec = importlib.util.spec_from_file_location(f"reground_{name}", code_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "execute", None)

    def _build_primitives(self) -> Dict[str, Any]:
        if self.primitives:
            return self.primitives
        self.load()
        prims = {}
        for name, entry in self.artifacts.items():
            if entry.get("kind") != "primitive":
                continue
            execute = self._load_execute(name)
            if execute:
                def _wrap(ctx, params, _exec=execute):
                    obs = _exec(ctx, params, primitives=prims, controlled_llm=runtime.controlled_llm_stub)
                    return runtime.normalize_observation(obs)
                prims[name] = _wrap
        self.primitives = prims
        return prims

    def execute(self, name: str, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        self.load()
        entry = self.artifacts.get(name)
        if not entry:
            obs = runtime.fail(error=f"unknown_artifact:{name}")
            obs = runtime.normalize_observation(obs)
            obs["id"] = make_observation_id(obs)
            return obs
        execute = self._load_execute(name)
        if not execute:
            obs = runtime.fail(error=f"missing_execute:{name}")
            obs = runtime.normalize_observation(obs)
            obs["id"] = make_observation_id(obs)
            return obs
        prims = self._build_primitives()
        obs = execute(context, params, primitives=prims, controlled_llm=runtime.controlled_llm_stub)
        obs = runtime.normalize_observation(obs)
        obs["id"] = make_observation_id(obs)
        obs["skill_name"] = name
        return obs


def normalize_ref(value: str) -> str:
    if not value:
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in {"n/a", "na", "none", "not_required"}:
        return ""
    if lowered.startswith("n/a") or lowered.startswith("na "):
        return ""
    if "not_required" in lowered or "not required" in lowered:
        return ""
    return cleaned


def build_query(act: Dict[str, Any]) -> str:
    ref = normalize_ref(act.get("grounding_ref") or "")
    if ref:
        return ref
    action = normalize_ref(act.get("action") or "")
    if action:
        return action
    intent = normalize_ref(act.get("intent") or act.get("strategic_intent") or "")
    if intent:
        return intent
    calls = act.get("latent_skill_calls") or act.get("latent_tool_calls") or []
    if calls:
        first = calls[0]
        operation = normalize_ref(first.get("operation") or first.get("tool") or first.get("name") or "")
        target = normalize_ref(first.get("target_type") or "")
        query = " ".join([p for p in [operation, target] if p])
        return query
    return "evidence"


def extract_id(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def build_skill_call(query: str, available: List[str]) -> Tuple[str, Dict[str, Any]]:
    lower = query.lower()
    if "Extract_Table" in available:
        if "table" in lower or "tab" in lower:
            table_id = extract_id(query, r"(?:table|tab)\s*([0-9]+)")
            if table_id:
                return "Extract_Table", {"table_id": table_id}
    if "Extract_Figure_Caption" in available:
        if "figure" in lower or "fig" in lower:
            fig_id = extract_id(query, r"(?:figure|fig\.?)[^0-9]*([0-9]+)")
            if fig_id:
                return "Extract_Figure_Caption", {"figure_id": fig_id}
    if "Extract_Equation" in available:
        if "equation" in lower or "eq" in lower:
            eq_id = extract_id(query, r"(?:eq\.?|equation)[^0-9]*([0-9]+)")
            if eq_id:
                return "Extract_Equation", {"eq_id": eq_id}
    if "Extract_Algorithm" in available:
        if "algorithm" in lower or "alg" in lower:
            algo_id = extract_id(query, r"(?:alg\.?|algorithm)[^0-9]*([0-9]+)")
            if algo_id:
                return "Extract_Algorithm", {"algo_id": algo_id}
    if "Extract_Section" in available:
        if "section" in lower or "sec" in lower or "appendix" in lower:
            title = query
            if "appendix" in lower:
                title = "Appendix"
            return "Extract_Section", {"title": title}
    if "Extract_Span" in available:
        return "Extract_Span", {"query": query}
    return available[0], {}


def load_context(forum_id: str, papers_md_dir: str, max_chars: int, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if forum_id in cache:
        return cache[forum_id]
    md_path = os.path.join(papers_md_dir, forum_id, "auto", f"{forum_id}.md")
    if not os.path.exists(md_path):
        cache[forum_id] = {"segments": []}
        return cache[forum_id]
    text = Path(md_path).read_text(encoding="utf-8", errors="ignore")
    segments = parse_markdown_segments(text, max_chars)
    cache[forum_id] = {"segments": segments}
    return cache[forum_id]


def write_context(out_dir: str, forum_id: str, context: Dict[str, Any]) -> None:
    if not out_dir:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{forum_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for seg in context.get("segments", []):
            f.write(json.dumps({"id": seg.get("id"), "text": seg.get("text", "")}, ensure_ascii=True) + "\n")


def build_thread_seed(act: Dict[str, Any]) -> Dict[str, Any]:
    issue_id = act.get("issue_id") or act.get("act_id")
    return {
        "issue_id": issue_id,
        "forum_id": act.get("forum_id") or act.get("forum"),
        "issue_tag": act.get("issue_type") or act.get("issue_cluster_id") or act.get("cluster_id"),
        "issue_text": act.get("action") or "",
        "severity": act.get("severity") or "medium",
        "budget": 6,
        "phase": "Open",
        "requests": [],
        "context_id": act.get("forum_id") or act.get("forum"),
        "oracle_moves": [],
        "oracle_index": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.5: re-ground semantic acts with executable evidence.")
    parser.add_argument("--acts-in", required=True, help="Semantic acts JSONL")
    parser.add_argument("--library-index", default="steps/02_mine_evidence_needs/library_index.jsonl")
    parser.add_argument("--papers-md-dir", default="data/raw/papers_md")
    parser.add_argument("--contexts-out", default="steps/02_5_reground_observations/contexts")
    parser.add_argument("--out-acts", default="steps/02_5_reground_observations/semantic_acts_regrounded.jsonl")
    parser.add_argument("--out-observations", default="steps/02_5_reground_observations/observations.jsonl")
    parser.add_argument("--out-threads", default="steps/02_5_reground_observations/thread_oracle_seeds.jsonl")
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--max-acts", type=int, default=0)
    args = parser.parse_args()

    acts = read_jsonl(args.acts_in)
    if args.max_acts and args.max_acts > 0:
        acts = acts[: args.max_acts]

    library = Library(args.library_index)
    library.load()
    available = list(library.artifacts.keys())

    context_cache: Dict[str, Dict[str, Any]] = {}
    threads: Dict[str, Dict[str, Any]] = {}

    Path(args.out_observations).unlink(missing_ok=True)
    regrounded = []

    for act in acts:
        forum_id = act.get("forum_id") or act.get("forum") or ""
        if not forum_id:
            forum_id = "unknown"
        context = load_context(forum_id, args.papers_md_dir, args.max_chars, context_cache)
        write_context(args.contexts_out, forum_id, context)

        query = build_query(act)
        skill_name, params = build_skill_call(query, available)
        obs = library.execute(skill_name, context, params)
        rho_star = obs.get("prov") if obs.get("status") == "ok" else []

        act_id = act.get("act_id") or act.get("issue_id") or f"{forum_id}#{len(regrounded):06d}"
        enriched = dict(act)
        enriched["act_id"] = act_id
        enriched["skill_call"] = {"name": skill_name, "arguments": params}
        enriched["observation"] = obs
        enriched["rho_0"] = act.get("grounding_ref")
        enriched["rho_star"] = rho_star
        regrounded.append(enriched)

        append_jsonl(args.out_observations, {
            "act_id": act_id,
            "forum_id": forum_id,
            "skill_call": {"name": skill_name, "arguments": params},
            "observation": obs,
        })

        issue_id = act.get("issue_id") or act_id
        if issue_id not in threads:
            threads[issue_id] = build_thread_seed(act)
        threads[issue_id]["oracle_moves"].append({
            "intent": act.get("intent") or act.get("strategic_intent"),
            "skill_call": {"name": skill_name, "arguments": params},
        })

    write_jsonl(args.out_acts, regrounded)
    write_jsonl(args.out_threads, list(threads.values()))

    print(json.dumps({
        "acts": len(regrounded),
        "threads": len(threads),
        "out_acts": args.out_acts,
        "out_observations": args.out_observations,
        "out_threads": args.out_threads,
    }, indent=2))


if __name__ == "__main__":
    main()
