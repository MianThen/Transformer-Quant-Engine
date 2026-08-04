from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from python.qbt_ml.research.hypothesis_registry import Hypothesis, HypothesisRegistry
from python.qbt_ml.research.multiple_testing_oracle import (
    adjusted_p_values,
    paired_block_bootstrap_mean,
)


def hypothesis(**overrides) -> Hypothesis:
    values = {
        "hypothesis_id": "risk.lw_lin_cc.v1",
        "family_id": "risk_estimator_v1",
        "trial_id": "trial.lw_lin_cc.topk",
        "method_version": "LW-LIN-CC/v1",
        "portfolio_policy_id": "TOPK_EQUAL_WEIGHT/v1",
        "policy_config_hash": 101,
        "expected_direction": "POSITIVE",
        "primary_metric": "realized_to_predicted_variance_ratio",
        "validation_windows": ("wf-001", "wf-002", "wf-003"),
        "p_value_method": "paired_circular_block_bootstrap_mean/v1",
        "created_before_test_at": "2026-07-30T01:00:00Z",
    }
    values.update(overrides)
    return Hypothesis(**values)


def test_registry_is_sealed_hashed_and_schema_valid(tmp_path: Path):
    registry = HypothesisRegistry(
        hypotheses=(hypothesis(),), sealed_at="2026-07-30T02:00:00Z"
    )
    registry.validate(test_data_available_at="2026-07-30T03:00:00Z")
    output = tmp_path / "hypothesis_registry.json"
    registry.write(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["registry_sha256"] == registry.sha256()
    schema = json.loads(
        Path("schemas/hypothesis_registry_v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)


def test_registry_rejects_post_test_and_duplicate_trials():
    with pytest.raises(ValueError, match="before test"):
        hypothesis(created_before_test_at="2026-07-30T04:00:00Z").validate(
            "2026-07-30T03:00:00Z"
        )
    duplicate = replace(hypothesis(), hypothesis_id="risk.other.v1")
    with pytest.raises(ValueError, match="duplicate trial"):
        HypothesisRegistry(
            hypotheses=(hypothesis(), duplicate), sealed_at="2026-07-30T02:00:00Z"
        ).validate()


def test_registry_groups_only_predeclared_families():
    second = hypothesis(
        hypothesis_id="tail.empirical_es.v1",
        family_id="tail_risk_v1",
        trial_id="trial.tail_empirical_es",
        method_version="TAIL-EMPIRICAL-ES/v1",
        primary_metric="fz0_joint_score",
    )
    registry = HypothesisRegistry(
        hypotheses=(second, hypothesis()), sealed_at="2026-07-30T02:00:00Z"
    )
    families = registry.families()
    assert tuple(families) == ("risk_estimator_v1", "tail_risk_v1")
    assert families["tail_risk_v1"][0].trial_id == "trial.tail_empirical_es"


def test_multiple_testing_oracle_golden_and_seed_replay():
    p_values = np.asarray([0.01, 0.04, 0.03, 0.002, 0.80])
    np.testing.assert_allclose(
        adjusted_p_values(p_values), [0.025, 0.05, 0.05, 0.01, 0.80]
    )
    differences = np.asarray([0.02, 0.01, -0.01, 0.03, 0.02, 0.00, 0.01, 0.04])
    first = paired_block_bootstrap_mean(
        differences, block_length=3, resamples=999, seed=1234567
    )
    second = paired_block_bootstrap_mean(
        differences, block_length=3, resamples=999, seed=1234567
    )
    assert first == second
    assert first == (0.001, 0)
