# Foundry 4.5 Implementation Guide

This document describes the concrete scripts and I/O for the Foundry pipeline (section 4.5 in `技术方案.md`). The scripts are minimal, deterministic, and dependency-free by default.

## Input Assumptions

- `mining_results.jsonl` contains one JSON object per paper, matching the sample structure.
- Each record has `analysis.mining_results`, a list of issue-level items with `latent_tool_calls`.

## Scripts and Responsibilities

### 1) Parse mining_results into issue-level JSONL

- Script: `foundry/issue_mining/parse_mining_results.py`
- Input: `data/raw/mining_results.jsonl`
- Output: `data/interim/mining_flat.jsonl`

```
python foundry/issue_mining/parse_mining_results.py \
  --in data/raw/mining_results.jsonl \
  --out data/interim/mining_flat.jsonl
```

### 2) Resolve grounding_ref to paper_span

- Script: `foundry/issue_mining/link_grounding_ref.py`
- Input: `data/interim/mining_flat.jsonl` + `data/raw/papers_md/`
- Output: `data/interim/grounded_issues.jsonl`

```
python foundry/issue_mining/link_grounding_ref.py \
  --in data/interim/mining_flat.jsonl \
  --papers data/raw/papers_md \
  --out data/interim/grounded_issues.jsonl
```

MinerU layout is supported by default: `data/raw/papers_md/<forum_id>/auto/<forum_id>.md`, with `images/` alongside the markdown. The resolver stores `doc_path`, `images_dir`, and simple matches (sections, figures, tables) with character offsets; snippets can be disabled via `--no-snippet`.
Each match now includes `ref_start/ref_end` for the ref anchor and `section_start/section_end` for the enclosing section span, plus `section_title` when available.

Optional flags:
- `--log-every 5000` progress by issue count
- `--sample-failures 20` unresolved sample prints
- `--no-snippet` avoid storing snippets to reduce output size
- `paper_span.status` is set to `resolved`, `unresolved`, or `not_required`, with `paper_span.reason` describing why grounding is skipped.

### 3) Cluster semantic acts into issue clusters

- Script: `foundry/issue_mining/cluster_issues.py`
- Input: `data/interim/grounded_issues.jsonl` + `data/raw/embeddings.npy`
- Output: `data/processed/issues.jsonl`

```
python foundry/issue_mining/cluster_issues.py \
  --in data/interim/grounded_issues.jsonl \
  --embeddings data/raw/embeddings.npy \
  --out data/processed/issues.jsonl
```

Clustering uses embedding vectors aligned with `data/interim/mining_flat.jsonl`. The input JSONL must preserve the same order and length (e.g., `grounded_issues.jsonl` produced by the resolver without filtering).

Common flags:
- `--num-clusters 0` (heuristic) or a fixed K
- `--method auto|minibatch|kmeans`
- `--log-every 5000` and `--log-every-iter 10`
- `--stats-out data/processed/cluster_stats.json`
- Quick run: `--sample-size 2000 --sample-mode random` (write to a separate embeddings/output file)
- Partial embedding: `--embed-limit 2000 --stop-after-embed` (then rerun with `--resume` to complete)

Optional embedding generation (DMXAPI):
```
export DMX_API_KEY=sk-...
python foundry/issue_mining/cluster_issues.py \
  --in data/interim/grounded_issues.jsonl \
  --embeddings data/raw/embeddings.npy \
  --generate-embeddings \
  --embed-model text-embedding-3-large \
  --out data/processed/issues.jsonl
```

Embedding text (default `tool_calls`): `tool_category | operation | target_type | outcome=...` with optional `grounding_ref` appended unless `paper_span.status=not_required`. For semantic-ontology clustering, use `--embed-input-mode semantic_act` to include `strategic_intent`, `action`, and tool calls together. Use `--no-embed-outcome` or `--no-embed-grounding-ref` to disable.

Concurrency and resume:
- `--embed-workers 4` concurrent requests
- `--resume` resume from existing `embeddings.npy` and `embeddings.npy.progress.json`
- `--overwrite-embeddings` regenerate from scratch

### 4) Build issue threads (T3 ledger inputs)

- Script: `foundry/issue_mining/build_issue_threads.py`
- Input: `data/processed/issues.jsonl`
- Output: `data/processed/threads.jsonl` + `data/processed/thread_index.jsonl`

```
python foundry/issue_mining/build_issue_threads.py \
  --in data/processed/issues.jsonl \
  --out-threads data/processed/threads.jsonl \
  --out-index data/processed/thread_index.jsonl
```

Threads group semantic acts by `(forum_id, issue_cluster_id, target_key)` to approximate multi-round issue conversations.
Use `--out-issues` to attach `thread_id` back into the issue records.

### 5) Optional: Label clusters with a teacher LLM

- Script: `foundry/ontology/label_clusters.py`
- Input: `data/processed/issues.jsonl`
- Output: `foundry/ontology/cluster_labels.jsonl`

```
export DMX_API_KEY=sk-...
python foundry/ontology/label_clusters.py \
  --in data/processed/issues.jsonl \
  --out foundry/ontology/cluster_labels.jsonl
```

### 6) Optional: Map raw intents to a canonical intent ontology

- Script: `foundry/ontology/label_intents.py`
- Input: `data/processed/issues.jsonl`
- Output: `foundry/ontology/intent_map.json`

```
export DMX_API_KEY=sk-...
python foundry/ontology/label_intents.py \
  --in data/processed/issues.jsonl \
  --out foundry/ontology/intent_map.json
```

### 7) Build ontology summaries (issue / intent / evidence)

- Script: `foundry/ontology/build_ontology.py`
- Input: `data/processed/issues.jsonl`
- Output: `foundry/ontology/*.json`

```
python foundry/ontology/build_ontology.py \
  --in data/processed/issues.jsonl \
  --out foundry/ontology \
  --labels foundry/ontology/cluster_labels.jsonl \
  --intent-map foundry/ontology/intent_map.json
```

Outputs include `issue_ontology.json`, `intent_ontology.json`, `intent_ontology_raw.json`, `evidence_ontology.json`, and tool/operation summaries.

### 8) Discover deterministic primitives (clustered tool calls)

- Script: `foundry/curation/discover_primitives.py`
- Input: `data/processed/issues.jsonl`
- Output: `skills/primitives/registry.json`

```
python foundry/curation/discover_primitives.py \
  --in data/processed/issues.jsonl \
  --out skills/primitives/registry.json \
  --method auto \
  --write-stubs
```

Use `--out-issues data/processed/issues_with_primitives.jsonl` to attach `primitive_id` back into issues. Use `--out-assignments` to save the call→primitive index.
If `skills/primitives/embeddings.npy` is missing, add `--generate-embeddings` and the usual DMX embedding flags.

### 9) Compile primitives into skills

- Script: `foundry/curation/compile_skills.py`
- Input: `skills/primitives/registry.json`
- Output: `skills/library/*` + `skills/registry.json`

```
python foundry/curation/compile_skills.py \
  --primitives skills/primitives/registry.json \
  --skills-dir skills/library \
  --registry skills/registry.json
```

This creates stub `skill.py` files and manifests that reference discovered primitives.

### 10) Run skills and collect observations

- Script: `foundry/curation/run_skills.py`
- Input: `data/processed/issues.jsonl` + `skills/registry.json`
- Output: `data/interim/observations.jsonl`

```
python foundry/curation/run_skills.py \
  --issues data/processed/issues.jsonl \
  --registry skills/registry.json \
  --out data/interim/observations.jsonl
```

### 11) Curate trajectories

- Script: `foundry/curation/curate_trajectories.py`
- Input: `data/processed/issues.jsonl` + `data/interim/observations.jsonl`
- Output: `data/processed/trajectories.jsonl`

```
python foundry/curation/curate_trajectories.py \
  --issues data/processed/issues.jsonl \
  --obs data/interim/observations.jsonl \
  --thread-index data/processed/thread_index.jsonl \
  --out data/processed/trajectories.jsonl
```

## Notes

- The current skill implementations are stubs and return `status=not_implemented` by default.
- All outputs are JSONL with ASCII-safe serialization for reproducibility.
- The pipeline is deterministic and can be extended with real executables later.

## Run Log (parse_mining_results)

Command:
```
python foundry/issue_mining/parse_mining_results.py \
  --in data/raw/mining_results.jsonl \
  --out data/interim/mining_flat.jsonl \
  --log-every 100
```

Output:
```
[parse] 100/2011 papers | issues=2723
[parse] 200/2011 papers | issues=5070
[parse] 300/2011 papers | issues=6389
[parse] 400/2011 papers | issues=8422
[parse] 500/2011 papers | issues=11007
[parse] 600/2011 papers | issues=13336
[parse] 700/2011 papers | issues=15562
[parse] 800/2011 papers | issues=18383
[parse] 900/2011 papers | issues=20995
[parse] 1000/2011 papers | issues=23846
[parse] 1100/2011 papers | issues=26369
[parse] 1200/2011 papers | issues=28752
[parse] 1300/2011 papers | issues=31428
[parse] 1400/2011 papers | issues=34220
[parse] 1500/2011 papers | issues=36714
[parse] 1600/2011 papers | issues=39789
[parse] 1700/2011 papers | issues=42741
[parse] 1800/2011 papers | issues=45305
[parse] 1900/2011 papers | issues=47891
[parse] 2000/2011 papers | issues=50647
[summary] papers=2011 issues=50991 empty_mining=2 missing_forum_id=0 with_tool_calls=50901 with_grounding_ref=50459
[summary] multi_role_issues=20 unknown_role_issues=0
[summary] roles: Area Chair:14, Author:26083, Reviewer:24912
[summary] intents(top12): Clarify_Misunderstanding:1906, Expose_Methodological_Weakness:1688, Defend_Novelty_Against_Prior_Work:1068, Concede_Minor_Point_To_Win_Major:649, Establish_Theoretical_Grounding:608, Shift_Burden_Of_Proof:442, Defend_Methodological_Choice:340, Request_Clarification:183, Improve_Presentation:152, Challenge_Novelty:145, Demonstrate_Responsiveness:145, Trap_With_Definition_Request:140
```

## Run Log (link_grounding_ref)

Command:
```
python foundry/issue_mining/link_grounding_ref.py \
  --in data/interim/mining_flat.jsonl \
  --papers data/raw/papers_md \
  --out data/interim/grounded_issues.jsonl \
  --no-snippet
```

Output:
```
[link] 5000/50991 issues | resolved=4267
[link] 10000/50991 issues | resolved=8652
[link] 15000/50991 issues | resolved=12803
[link] 20000/50991 issues | resolved=17156
[link] 25000/50991 issues | resolved=21385
[link] 30000/50991 issues | resolved=25585
[link] 35000/50991 issues | resolved=29788
[link] 40000/50991 issues | resolved=34185
[link] 45000/50991 issues | resolved=38536
[link] 50000/50991 issues | resolved=42743
[summary] issues=50991 resolved=43617 resolved_rate=0.889 not_required=1942 required_unresolved=5432 missing_docs=0
[summary] refs.abstract: total=2394 matched=2394 rate=1.000
[summary] refs.appendix: total=7463 matched=2837 rate=0.380
[summary] refs.figure: total=7537 matched=7270 rate=0.965 image_matched=3174
[summary] refs.other: total=22687 matched=8758 rate=0.386
[summary] refs.related_work: total=854 matched=698 rate=0.817
[summary] refs.section: total=46938 matched=44635 rate=0.951
[summary] refs.table: total=9303 matched=8650 rate=0.930
[summary] unresolved_samples:
  - {'issue_id': 'viNQSOadLg#0027', 'forum_id': 'viNQSOadLg', 'grounding_ref': "Author's response f1bXYRwmqc_com_6, f1bXYRwmqc_com_7", 'doc_path': 'data\\raw\\papers_md\\viNQSOadLg\\auto\\viNQSOadLg.md'}
  - {'issue_id': 'TTrzgEZt9s#0010', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix D (pages referenced)', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0015', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix E', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0028', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix D', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0031', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix E', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0032', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix length', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0034', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix B, Appendix C', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0036', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Appendix E', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'TTrzgEZt9s#0037', 'forum_id': 'TTrzgEZt9s', 'grounding_ref': 'Page 33 (Appendix D.5.2)', 'doc_path': 'data\\raw\\papers_md\\TTrzgEZt9s\\auto\\TTrzgEZt9s.md'}
  - {'issue_id': 'jUNSBetmAo#0013', 'forum_id': 'jUNSBetmAo', 'grounding_ref': 'Appendix C', 'doc_path': 'data\\raw\\papers_md\\jUNSBetmAo\\auto\\jUNSBetmAo.md'}
  - {'issue_id': '9ceadCJY4B#0015', 'forum_id': '9ceadCJY4B', 'grounding_ref': 'Sec 4 (Mitigation Methods)', 'doc_path': 'data\\raw\\papers_md\\9ceadCJY4B\\auto\\9ceadCJY4B.md'}
  - {'issue_id': 'jFiFmHrIfD#0030', 'forum_id': 'jFiFmHrIfD', 'grounding_ref': 'Appendix B', 'doc_path': 'data\\raw\\papers_md\\jFiFmHrIfD\\auto\\jFiFmHrIfD.md'}
  - {'issue_id': 'jFiFmHrIfD#0031', 'forum_id': 'jFiFmHrIfD', 'grounding_ref': 'Appendix D', 'doc_path': 'data\\raw\\papers_md\\jFiFmHrIfD\\auto\\jFiFmHrIfD.md'}
  - {'issue_id': 'qBL04XXex6#0005', 'forum_id': 'qBL04XXex6', 'grounding_ref': 'Appendix Sec E (Influence of Bad Feedback)', 'doc_path': 'data\\raw\\papers_md\\qBL04XXex6\\auto\\qBL04XXex6.md'}
  - {'issue_id': 'B0wJ5oCPdB#0018', 'forum_id': 'B0wJ5oCPdB', 'grounding_ref': 'Reviewer_Comments_tveC_c5jB_LjHX', 'doc_path': 'data\\raw\\papers_md\\B0wJ5oCPdB\\auto\\B0wJ5oCPdB.md'}
  - {'issue_id': 'B0wJ5oCPdB#0021', 'forum_id': 'B0wJ5oCPdB', 'grounding_ref': 'Reviewer_Comments_Doc8_tveC', 'doc_path': 'data\\raw\\papers_md\\B0wJ5oCPdB\\auto\\B0wJ5oCPdB.md'}
  - {'issue_id': 'B0wJ5oCPdB#0026', 'forum_id': 'B0wJ5oCPdB', 'grounding_ref': 'BBH_Benchmark', 'doc_path': 'data\\raw\\papers_md\\B0wJ5oCPdB\\auto\\B0wJ5oCPdB.md'}
  - {'issue_id': '6ARlSgun7J#0005', 'forum_id': '6ARlSgun7J', 'grounding_ref': 'Eq. (9)', 'doc_path': 'data\\raw\\papers_md\\6ARlSgun7J\\auto\\6ARlSgun7J.md'}
  - {'issue_id': 'M11LONBkx1#0008', 'forum_id': 'M11LONBkx1', 'grounding_ref': 'Appendix C.4 (to be added)', 'doc_path': 'data\\raw\\papers_md\\M11LONBkx1\\auto\\M11LONBkx1.md'}
  - {'issue_id': 'M11LONBkx1#0012', 'forum_id': 'M11LONBkx1', 'grounding_ref': 'Appendix C.3, Appendix C.4', 'doc_path': 'data\\raw\\papers_md\\M11LONBkx1\\auto\\M11LONBkx1.md'}
```

## Run Log (cluster_issues: KMeans)

Notes:
- Selected KMeans over MiniBatchKMeans; `issues_kmeans.jsonl` renamed to `issues.jsonl`.

Output:
```
[load] records=50991
[cluster] method=kmeans k=226
[cluster] sklearn KMeans fit start
[cluster] sklearn KMeans fit done
[summary] clusters=226 size_min=23 size_median=220.0 size_mean=225.6 size_max=496
[summary] top_clusters: 199:496, 100:436, 149:419, 47:403, 106:388, 141:371, 191:369, 181:367, 19:367, 156:366
[summary] inertia=22165.010
```
