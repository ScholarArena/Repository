# Step 01 - Flatten mining_results into semantic act instances

Goal
- Convert `data/raw/mining_results.jsonl` into semantic act instances defined in `论文新方案.md`.

Input
- `data/raw/mining_results.jsonl`
- Required fields per paper:
  - `forum_id` (or `forum`), `title`, `timestamp`
  - `analysis.mining_results[]` items with:
    - `role`, `strategic_intent`, `action`
    - `grounding_ref`, `source_seg_ids`, `latent_tool_calls`
    - optional `cognitive_chain`

Process
- For each paper, iterate `analysis.mining_results` and emit one act per item.
- Normalize role strings into canonical roles (Reviewer/Author/Meta-Reviewer/etc.).
- Map to semantic act instance fields:
  - r -> `role`
  - i -> `issue_id` (forum_id + index)
  - intent -> `intent`/`strategic_intent`
  - x -> `action` / `act_text`
  - rho -> `grounding_ref`, `source_seg_ids`, `latent_tool_calls`

Output
- Default target: `data/interim/semantic_acts.jsonl`
- Each line is a JSON object with core fields:
  - `act_id`, `act_index`, `issue_id`, `forum_id`, `title`, `timestamp`
  - `role`, `roles`, `intent`, `action`
  - `grounding_ref`, `source_seg_ids`, `latent_tool_calls`
  - `meta.cognitive_chain`, `meta.role_raw`

Command
```bash
python steps/01_flatten_semantic_acts/flatten_semantic_acts.py \
  --in data/raw/mining_results.jsonl \
  --out data/interim/semantic_acts.jsonl \
  --log-every 100
```

Notes
- If `grounding_ref` or tool calls are missing, the act is still emitted; it is just evidence-incomplete.
- Summary stats are printed unless `--quiet` is provided.
