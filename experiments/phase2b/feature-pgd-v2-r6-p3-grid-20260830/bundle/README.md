# Phase 2B P3 GPU Migration Bundle r6

This bundle runs one preregistered Phase 2B P3 validation-only grid on CUDA.
It is a controlled development experiment, not a promotion package.

## Scope

- Fixed seed: `20260805`.
- Three chronological folds, 50 epochs per run.
- Baseline plus seven pure Feature-PGD candidates.
- PGD steps fixed at 3; only epsilon and beta vary.
- `structured_missing` is excluded.
- The bundle never executes short OOS and does not contain a short-OOS entrypoint.
- The baseline is trained inside this bundle; external checkpoint reuse is rejected.

## GPU entrypoint

Run from the extracted bundle root:

```bash
bash run_validation_grid_cuda.sh
```

The entrypoint verifies the source contract, records CUDA preflight information,
runs focused tests, trains the grid, validates artifacts, writes the deterministic
P3 selection, and creates a results archive.

## Acceptance behavior

Every candidate must pass all clean guards before it may be ranked for P4:

- return MAE: no more than 1% relative degradation;
- direction Brier: no more than 1% relative degradation;
- volatility MAE: no more than 2% relative degradation;
- NDCG@20: no more than 0.005 absolute degradation;
- RankIC: no more than 0.005 absolute degradation.

Stress evaluation uses fold-equal circular moving-block bootstrap. Every stress
set must have a strictly negative point estimate in all three folds and a 95%
one-sided upper confidence bound strictly below zero for relative stress error.
No candidate passing only some guards is eligible for P4.

`P3_GRID_SELECTION.json` may nominate at most two clean-guard-pass candidates
for a new-seed P4 replication. It never marks Phase 2B as exited or promoted.

## Integrity

The expected dataset SHA-256 is
`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`.
`MANIFEST.sha256` covers all source files in this bundle. Results are separately
validated before they can be archived.
