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

MinerU layout is supported by default: `data/raw/papers_md/<forum_id>/auto/<forum_id>.md`, with `images/` alongside the markdown. The resolver stores `doc_path`, `images_dir`, and simple matches/snippets for `grounding_ref` items (sections, figures, tables).

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
