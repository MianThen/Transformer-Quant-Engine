# Phase 2B Decision Record

## Current decision

Phase 2B does not advance, does not exit research, and does not enter production.
One P3 validation-only optimization round is authorized because the earlier pure
Feature-PGD result improved return MAE and ranking measures but failed the clean
volatility guard and cross-window stress stability requirements.

## Frozen P3 protocol

P3 uses the fixed validation folds and seed `20260805`. It changes only pure
Feature-PGD epsilon and beta. The baseline and every challenger are trained in
the bundle. This makes the paired comparison reproducible and excludes imported
checkpoints as a source of untracked variation.

The former short OOS ending on 2026-07-13 has already been observed. It is not
untouched OOS, must not be rerun for tuning, and is not used by this bundle.

## Decision rule

Candidates must pass every preregistered clean guard. Candidates that fail even
one guard are stopped and cannot be combined with other Phase 2B techniques.
Only up to two candidates passing every clean guard may be registered for P4
multi-seed replication using seeds `20260805`, `20260806`, and `20260807`.

P3 selection is not a Phase 2B promotion event. Phase exit and promotion fields
remain false even when a Top-2 is selected.

## Phase 5 boundary

The available Phase 5 InfoTS artifact is a GPU-trained pretraining reference
pending OOS. It has no supervised six-output prediction evidence and no OOS or
cost gate. It cannot offset a Phase 2B failure or support promotion.
