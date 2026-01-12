# Step 01 - Build Semantic Act Instances

Goal
- Convert `data/raw/mining_results.jsonl` into Semantic Act Instances as defined in `论文新方案.md`.
- Derive issue/thread identifiers by clustering actions (per forum).
- Derive intent labels by clustering cognitive chains (global) and labeling via LLM.

Input
- `data/raw/mining_results.jsonl`
- Required fields per paper:
  - `forum_id` (or `forum`), `title`, `timestamp`
  - `analysis.mining_results[]` items with:
    - `role`, `strategic_intent`, `action`
    - `grounding_ref`, `source_seg_ids`, `latent_tool_calls`
    - optional `cognitive_chain`

Process
1. **Flatten** mining results into act-level records.
2. **Issue clustering** per forum:
   - Encode `action` (default) and cluster to produce thread-level issue IDs.
   - For each cluster, sample acts and use LLM to name/describe the issue.
3. **Intent clustering** (global):
   - Encode `meta.cognitive_chain + role_raw` and cluster to obtain intent labels.
   - For each cluster, sample acts and use LLM to label the intent.
4. Update Semantic Act Instances with `issue_id` and `intent` (label), and rename
   `latent_tool_calls` -> `latent_skill_calls`.

Outputs (minimal)
- `data/interim/semantic_acts.jsonl` (final Semantic Act Instances)
- `steps/01_flatten_semantic_acts/issues.jsonl` (issue list with labels/descriptions)
- `steps/01_flatten_semantic_acts/issue_assignments.jsonl` (act_id -> issue_id mapping)
- `steps/01_flatten_semantic_acts/intents.jsonl` (intent list with labels/descriptions)
- `steps/01_flatten_semantic_acts/intent_assignments.jsonl` (act_id -> intent_id mapping)
- `steps/01_flatten_semantic_acts/issue_embeddings.npy` (issue embeddings, same order as acts)
- `steps/01_flatten_semantic_acts/intent_embeddings.npy` (intent embeddings, same order as acts)

Command (full pipeline)
```bash
python steps/01_flatten_semantic_acts/flatten_semantic_acts.py \
  --in data/raw/mining_results.jsonl \
  --out data/interim/semantic_acts.jsonl \
  --embed-model text-embedding-3-large \
  --llm-model gpt-4o-mini
```

Notes
- Issue text source is `action` by default. If needed, use `--issue-text-mode action+grounding+target`.
- Intent text source defaults to `meta.cognitive_chain + role_raw`. Override via `--intent-text-mode`.
- LLM labeling reuses existing issue/intent labels to avoid duplicates.
- To skip LLM labeling, add `--skip-issue-labels` and/or `--skip-intent-labels`.
- Quick test run on a small sample: add `--sample-acts 200` (acts are randomly selected).
