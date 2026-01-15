# Step 03 - Online Arena (Multi-Agent)

Goal
- Implement the Online Arena from `ScholarArena.tex`: ledger-based thread execution with Meta scheduling, Move planning/execution/action, and hard evidence gating.
- Produce replayable logs and metrics to support policy swap experiments (tab:policy_swap).

Input
- Thread seeds (JSONL) **or** acts (JSONL) to build threads.
- Paper context `C` as segments (JSON or JSONL) for tool execution.
- Capability library from Step 02: `steps/02_mine_evidence_needs/library_index.jsonl`.
- Intent inventory (optional): `steps/01_flatten_semantic_acts/intents.jsonl`.

Thread seed JSONL format (one per issue):
```json
{
  "issue_id": "forum#0001",
  "forum_id": "forum",
  "issue_tag": "issue_0003",
  "issue_text": "Request evidence about Table 2",
  "severity": "medium",
  "budget": 6,
  "phase": "Open",
  "requests": [{"id": "R1", "text": "Request evidence about Table 2"}],
  "context_id": "forum"
}
```

Context formats
- JSON with `{"segments": [{"id": 1, "text": "..."}, ...]}`
- JSONL with one segment per line (`{"id": 1, "text": "..."}`)

Process
1. Meta schedules active threads from ledger summaries only.
2. Per thread: Plan `(intent, skill_call)` → Execute (skill or primitive) → Act with evidence gating.
3. Ledger/FSM update and deterministic logging.

Outputs (per run)
- `steps/03_online_arena/outputs/<run_id>/events.jsonl`
- `steps/03_online_arena/outputs/<run_id>/threads_final.jsonl`
- `steps/03_online_arena/outputs/<run_id>/metrics.json`
- `steps/03_online_arena/outputs/<run_id>/raw_llm.jsonl` (optional)

Command (single policy)
```bash
python steps/03_online_arena/run_online_arena.py \
  --acts-in data/processed/issues.sample.jsonl \
  --contexts-dir data/processed/contexts \
  --intents-in steps/01_flatten_semantic_acts/intents.jsonl \
  --policy-model gpt-4o-mini \
  --policy-base-url https://api.openai.com/v1
```

Stub-only (no API key)
```bash
python steps/03_online_arena/run_online_arena.py \
  --acts-in data/processed/issues.sample.jsonl \
  --contexts-dir data/processed/contexts \
  --policy-mode stub
```

Command (policy swap)
```bash
python steps/03_online_arena/run_policy_swap.py \
  --policies steps/03_online_arena/policy_swap.json \
  --acts-in data/processed/issues.sample.jsonl \
  --contexts-dir data/processed/contexts
```

Policy swap config example (`steps/03_online_arena/policy_swap.json`)
```json
{
  "policies": [
    {"name": "gpt4o", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
    {"name": "claude35", "model": "claude-3-5-sonnet", "api_key_env": "ANTHROPIC_API_KEY"}
  ]
}
```

Notes / Missing prerequisites
- Paper contexts (`C`) are required for tool execution. If you do not have parsed PDF segments, tools will return `missing` and gating will force clarification-only actions.
- PlanAcc requires re-grounded supervision with explicit `(intent, skill_call)` targets (i.e., `A^{(*)}`); the current pipeline does not generate these yet.
- Policy swap across GPT/Claude/Gemini/etc. assumes OpenAI-compatible endpoints or provider-specific adapters.
