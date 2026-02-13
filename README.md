# ScholarArena

**Evidence-Gated, Replayable Multi-Party Scientific Discourse**

ScholarArena is a verifiable framework for scientific review discussions.
It turns evidence availability into a hard constraint: factual fields must cite traceable Observations.
If evidence is missing or execution fails, the system emits auditable next-step actions instead of unsupported claims.

[Read the paper](manuscript.pdf)

## Core Design

- **Offline Foundry**: Mines evidence needs from multi-round dialogue, compiles and tests Primitives/Skills, then re-grounds outputs into citable Observations.
- **Online Arena**: Runs issue-threaded Reviewer/Author/Meta interaction with ledger + FSM control for replayable execution.
- **Hard Evidence Gate**: Every factual claim must cite an Observation. Under `missing/fail`, actions are restricted to auditable follow-ups.

## Key Results (Temporal Split)

| System | Support | HallucMissing | CloseRate |
| --- | ---: | ---: | ---: |
| ScholarArena | **0.90** | **0.04** | **0.66** |
| w/o Gating | 0.79 | 0.26 | 0.57 |

Hard evidence gating sharply reduces unsupported claims while keeping thread progress under uncertainty.

## Quick Start

```bash
python steps/03_online_arena/run_online_arena.py \
  --acts-in data/processed/issues.sample.jsonl \
  --policy-mode stub \
  --max-rounds 1
```

For the full pipeline (Foundry / Re-grounding / Arena), see:

- `steps/01_flatten_semantic_acts/README.md`
- `steps/02_mine_evidence_needs/README.md`
- `steps/02_5_reground_observations/README.md`
- `steps/03_online_arena/README.md`

## Repository Layout

```text
steps/01_flatten_semantic_acts/
steps/02_mine_evidence_needs/
steps/02_5_reground_observations/
steps/03_online_arena/
data/
scripts/
```
