# Step 02 - Mine Evidence Needs and Compile Capabilities

Goal
- Implement the Offline Foundry stage from `ScholarArena.tex`:
  - Mine evidence needs `u = Psi(intent, x, rho^(0))`.
  - Cluster needs, summarize each cluster with an LLM into `spec` + `tests`.
  - Generate executable `code` (Primitive or Skill), run tests, and gate into a library.

Input
- `data/interim/semantic_acts.jsonl` (from Step 01)
  - Required fields per act: `intent`, `action`, `latent_skill_calls` (or `latent_tool_calls`).

Process
1. Build a need text from `(intent, action, latent_skill_calls)` for each act.
2. Heuristic merge (simhash) to reduce embedding volume, embed group reps, then cluster.
3. (Optional) Use LLM to generate a spec/tests template (global or per-cluster).
4. For each cluster, sample acts and call the LLM to produce:
   - `spec` (Primitive or Skill) + `tests` (feasible, deterministic).
5. Call the LLM to generate `code` for each spec and run tests.
6. Only passing artifacts are added to the capability library.

Outputs (minimal)
- `steps/02_mine_evidence_needs/need_clusters.jsonl`
- `steps/02_mine_evidence_needs/need_assignments.jsonl`
- `steps/02_mine_evidence_needs/need_specs.jsonl`
- `steps/02_mine_evidence_needs/library_index.jsonl`
- `steps/02_mine_evidence_needs/library/` (passed Primitives and Skills)
- `steps/02_mine_evidence_needs/spec_test_template.json` (if template mode is global)
- `steps/02_mine_evidence_needs/spec_test_template.jsonl` (if template mode is per-cluster)
- `steps/02_mine_evidence_needs/llm_usage.jsonl` (per-call usage)
- `steps/02_mine_evidence_needs/llm_usage_summary.json` (aggregated usage)
- `steps/02_mine_evidence_needs/progress.log` / `progress_events.jsonl` (detailed progress logs)

Command (full pipeline)
```bash
python steps/02_mine_evidence_needs/mine_evidence_needs.py \
  --in data/interim/semantic_acts.jsonl \
  --embed-model text-embedding-3-large \
  --llm-model gpt-4o-mini
```

Reuse pipeline (recommended)
```bash
# 1) cluster only
python steps/02_mine_evidence_needs/mine_evidence_needs.py --stage cluster

# 2) reuse clusters to generate spec/tests
python steps/02_mine_evidence_needs/mine_evidence_needs.py --stage spec --reuse-clusters

# 3) reuse specs to run codegen + test gating
python steps/02_mine_evidence_needs/mine_evidence_needs.py --stage codegen --reuse-specs
```

Failure prevention plan (recommended)
1. **Cluster only** to verify sizes and sampling:
   ```bash
   python steps/02_mine_evidence_needs/mine_evidence_needs.py --stage cluster
   ```
2. **Spec-only** to inspect feasibility and dependency list before codegen:
   ```bash
   python steps/02_mine_evidence_needs/mine_evidence_needs.py --stage spec
   ```
   This writes `requirements.txt` so you can pre-install packages.
3. **Codegen on a small subset first** using `--max-clusters 10`.
4. **Full run** once the environment is stable.

Notes
- Heuristic pre-merge is required (no full-embedding mode).
- Default `--spec-template-mode none` disables templates so LLM can explore freely per cluster.
- Spec generation retries invalid outputs (see `--spec-retries`) and skips clusters that remain invalid.
- Network access is allowed during spec/code generation; keep tests time-bounded and stable.
- Use `--target-cluster-size` to keep clusters small.
- Use `--codegen-retries` to control repair attempts with test errors.
- Use `--llm-debug-dir` to save raw LLM responses.
- Use `--spec-template-mode llm-global|llm-per-cluster` to let LLM generate spec/tests templates.
- `llm_usage.jsonl` logs token usage if the API returns `usage`; otherwise it logs prompt/completion character counts.
- Use `--reuse-clusters` with `--clusters-in` (optional) to skip re-clustering.
- Use `--reuse-specs` with `--specs-in` (optional) to skip spec generation.
