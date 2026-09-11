# r6 Audit Record

## Source package

- Source bundle: r6 Phase 2B Feature-PGD P3 validation grid.
- Dataset SHA-256: `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`.
- CUDA entrypoint: `run_validation_grid_cuda.sh`.
- Frozen configuration: `configs/ml/phase2b_feature_pgd_v2_validation_grid_cuda.json`.

## Controls added in r6

- Replaced legacy validation and short-OOS configuration with one validation-only grid.
- Removed short-OOS execution entrypoint and configuration from the migration bundle.
- Excluded `structured_missing`; all challengers are pure `feature_pgd` with no missing-data perturbation during training.
- Registered custom candidate IDs and exact epsilon/beta values before training.
- Added strict clean metric guards, fold-equal block bootstrap, and strict relative-stress gates.
- Added timestamp-level NDCG@20 and RankIC measurements for paired inference.
- Added deterministic Top-2 selection that first requires all clean guards.
- Added result validation for contract hashes, checkpoint-selection hashes, manifests, and strict gate shape.
- Prohibited external baseline checkpoint reuse so all reported checkpoints are contained in the run output.

## Required execution evidence

Before accepting a GPU result archive, retain:

1. `runtime_preflight_validation_grid.json`.
2. `preregistered_contract.json` and `phase2b_oos_report.json` with valid self-hashes.
3. `P3_GRID_SELECTION.json` and `RESULT_VALIDATION.json` reporting PASS.
4. The results archive SHA-256 sidecar.

No result archive is a promotion certificate. A P3 Top-2 is only permission to
preregister a new-seed P4 replication.
