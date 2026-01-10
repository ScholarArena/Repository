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

Optional flags:
- `--log-every 5000` progress by issue count
- `--sample-failures 20` unresolved sample prints
- `--no-snippet` avoid storing snippets to reduce output size

### 3) Cluster issues and assign issue_type

- Script: `foundry/issue_mining/cluster_issues.py`
- Input: `data/interim/grounded_issues.jsonl`
- Output: `data/processed/issues.jsonl`

```
python foundry/issue_mining/cluster_issues.py \
  --in data/interim/grounded_issues.jsonl \
  --out data/processed/issues.jsonl
```

Clustering uses a stable hash over `(operation, target_type, outcome)` (or intent/action fallback).

### 4) Build ontology summaries

- Script: `foundry/ontology/build_ontology.py`
- Input: `data/processed/issues.jsonl`
- Output: `foundry/ontology/*.json`

```
python foundry/ontology/build_ontology.py \
  --in data/processed/issues.jsonl \
  --out foundry/ontology
```

Outputs include intent types, issue types, tool categories, operations, and evidence type counts.

### 5) Extract tool candidates

- Script: `foundry/curation/extract_tool_candidates.py`
- Input: `data/processed/issues.jsonl`
- Output: `data/interim/tool_candidates.jsonl`

```
python foundry/curation/extract_tool_candidates.py \
  --in data/processed/issues.jsonl \
  --out data/interim/tool_candidates.jsonl
```

By default, only known deterministic operations are marked executable. You can override:

```
python foundry/curation/extract_tool_candidates.py \
  --in data/processed/issues.jsonl \
  --out data/interim/tool_candidates.jsonl \
  --executable-operations "Resolve_Citation,Extract_Table" \
  --executable-categories "Literature_Cross_Check"
```

### 6) Compile executable candidates into skills

- Script: `foundry/curation/compile_skills.py`
- Input: `data/interim/tool_candidates.jsonl`
- Output: `skills/library/*` + `skills/registry.json`

```
python foundry/curation/compile_skills.py \
  --in data/interim/tool_candidates.jsonl \
  --skills-dir skills/library \
  --registry skills/registry.json
```

This creates stub `skill.py` files and a minimal manifest for each executable candidate.

### 7) Run skills and collect observations

- Script: `foundry/curation/run_skills.py`
- Input: `data/processed/issues.jsonl` + `skills/registry.json`
- Output: `data/interim/observations.jsonl`

```
python foundry/curation/run_skills.py \
  --issues data/processed/issues.jsonl \
  --registry skills/registry.json \
  --out data/interim/observations.jsonl
```

### 8) Curate trajectories

- Script: `foundry/curation/curate_trajectories.py`
- Input: `data/processed/issues.jsonl` + `data/interim/observations.jsonl`
- Output: `data/processed/trajectories.jsonl`

```
python foundry/curation/curate_trajectories.py \
  --issues data/processed/issues.jsonl \
  --obs data/interim/observations.jsonl \
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
