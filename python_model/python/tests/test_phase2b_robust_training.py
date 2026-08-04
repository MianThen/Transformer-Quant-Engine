from __future__ import annotations

import numpy as np
import pytest

from python.qbt_ml.data.schemas import BAR_V1
from python.qbt_ml.models.temporal_transformer import (
    TemporalTransformerConfig,
    TemporalTransformerV1,
)
from python.qbt_ml.training.robust_training import (
    apl_direction_loss,
    build_stress_sets,
    direction_noise_audit,
    feature_pgd_perturbation,
    latent_fgm_loss,
    validate_robust_training_config,
)


def test_noise_audit_is_deterministic_and_non_mutating():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.float32)
    original = labels.copy()
    first = direction_noise_audit(labels, flip_rates=(0.0, 0.5), seed=9)
    second = direction_noise_audit(labels, flip_rates=(0.0, 0.5), seed=9)
    assert first == second
    np.testing.assert_array_equal(labels, original)
    assert first["records"][0]["flipped_count"] == 0


def test_apl_direction_loss_has_finite_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    probability = torch.sigmoid(logits)
    target = torch.tensor([0.0, 1.0, 1.0])
    loss = apl_direction_loss(probability, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_feature_pgd_preserves_padding_and_state_features():
    torch = pytest.importorskip("torch")
    features = torch.zeros((1, 3, len(BAR_V1.feature_names)))
    features[:, :, 0] = 1.0
    features[:, :, BAR_V1.feature_names.index("is_st")] = 1.0
    valid_mask = torch.tensor([[0, 1, 1]], dtype=torch.uint8)

    def objective(value):
        return value[..., 0].square().mean()

    perturbed = feature_pgd_perturbation(
        features, valid_mask, objective, epsilon=0.1, steps=2,
        feature_names=BAR_V1.feature_names,
    )
    assert torch.equal(perturbed[:, 0], features[:, 0])
    assert torch.equal(
        perturbed[:, :, BAR_V1.feature_names.index("is_st")],
        features[:, :, BAR_V1.feature_names.index("is_st")],
    )
    assert torch.all((perturbed - features).abs() <= 0.100001)


def test_latent_fgm_returns_clean_and_adversarial_diagnostics():
    torch = pytest.importorskip("torch")
    model = TemporalTransformerV1(TemporalTransformerConfig(
        feature_count=4, lookback=3, d_model=4, nhead=2,
        num_layers=1, dim_feedforward=8, dropout=0.0,
    ))
    features = torch.randn(2, 3, 4)
    valid_mask = torch.ones(2, 3, dtype=torch.uint8)
    target = {
        "expected_return": torch.zeros(2),
        "direction": torch.ones(2),
        "realized_volatility": torch.ones(2),
        "rank_utility": torch.zeros(2),
        "rank_relevance": torch.zeros(2),
        "timestamp": torch.tensor([1, 1]),
        "symbol_tie_breaker": torch.tensor([0, 1]),
    }

    def clean_loss(prediction, _target):
        return prediction["expected_return"].square().mean()

    loss, diagnostics = latent_fgm_loss(
        model, features, valid_mask, target, clean_loss,
        epsilon=0.05, beta=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["latent_perturbation_norm"] >= 0.0


def test_stress_sets_keep_shape_and_boolean_states():
    features = np.zeros((2, 4, len(BAR_V1.feature_names)), dtype=np.float32)
    state_indices = [BAR_V1.feature_names.index(name) for name in (
        "is_suspended", "is_listed", "is_st", "is_tradable",
    )]
    features[:, :, state_indices] = 1.0
    valid_mask = np.ones((2, 4), dtype=np.uint8)
    stress = build_stress_sets(
        features, valid_mask, feature_names=BAR_V1.feature_names, seed=3,
    )
    assert set(stress) == {"price", "volume", "missing", "extreme_volatility"}
    for value in stress.values():
        assert value["features"].shape == features.shape
        assert value["valid_mask"].shape == valid_mask.shape
        np.testing.assert_array_equal(value["features"][..., state_indices], 1.0)


def test_robust_config_is_opt_in_and_feature_pgd_is_registered():
    baseline = validate_robust_training_config(None, feature_names=BAR_V1.feature_names)
    assert baseline.mode == "none"
    candidate = validate_robust_training_config({
        "mode": "feature_pgd", "hypothesis_id": "phase2b-pgd-v1",
        "epsilon": 0.03, "pgd_steps": 2,
    }, feature_names=BAR_V1.feature_names)
    assert candidate.adversarial_enabled
    with pytest.raises(ValueError, match="eval/ONNX"):
        validate_robust_training_config({
            "mode": "latent_fgm", "hypothesis_id": "bad", "production_eval": True,
        }, feature_names=BAR_V1.feature_names)


def test_train_phase2b_candidates_write_research_only_contracts(tmp_path):
    torch = pytest.importorskip("torch")
    from python.qbt_ml.cli import _train
    from python.tests.test_phase1e_gradients import _phase1e_training_fixture

    dataset, base_config = _phase1e_training_fixture(tmp_path)
    for mode in ("direction_apl", "latent_fgm", "feature_pgd"):
        config = dict(base_config)
        config["training"] = dict(base_config["training"])
        config["training"]["output"] = str(tmp_path / mode)
        config["robust_training"] = {
            "mode": mode, "hypothesis_id": f"phase2b-{mode}-smoke",
            "epsilon": 0.01, "beta": 0.25, "pgd_steps": 1,
        }
        _train(config, str(dataset), None)
        checkpoint = torch.load(
            tmp_path / mode / "checkpoint.pt", map_location="cpu", weights_only=True,
        )
        metrics = (tmp_path / mode / "metrics.json").read_text(encoding="utf-8")
        assert checkpoint["robust_training_mode"] == mode
        assert f'"robust_training_mode": "{mode}"' in metrics
