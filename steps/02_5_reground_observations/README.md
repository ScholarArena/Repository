# Step 02.5 - Evidence Re-grounding

Goal
- Execute the compiled library on paper contexts to rewrite approximate evidence pointers `rho^(0)` into executable provenance `rho^(*)`.
- Produce `A^(*)` style supervision with observation IDs + provenance.
- Emit thread seeds with `oracle_moves` for replay and PlanAcc.

Inputs
- Semantic acts JSONL (from Step 01), e.g. `data/interim/semantic_acts.jsonl`.
- Library index from Step 02: `steps/02_mine_evidence_needs/library_index.jsonl`.
- Parsed paper markdown under `data/raw/papers_md/<forum_id>/auto/<forum_id>.md`.

Outputs
- `steps/02_5_reground_observations/semantic_acts_regrounded.jsonl`
- `steps/02_5_reground_observations/observations.jsonl`
- `steps/02_5_reground_observations/thread_oracle_seeds.jsonl`
- `steps/02_5_reground_observations/contexts/<forum_id>.jsonl`

Command
```bash
python steps/02_5_reground_observations/reground_observations.py \
  --acts-in data/interim/semantic_acts.jsonl \
  --papers-md-dir data/raw/papers_md
```

Notes
- The script derives a deterministic skill call per act, defaulting to `Extract_Span` when no specific cue is found.
- If a paper markdown file is missing, the context is empty and the observation will be `missing`.
