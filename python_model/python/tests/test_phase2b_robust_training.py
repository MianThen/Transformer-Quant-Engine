from __future__ import annotations

import json
import numpy as np
import pytest

from python.qbt_ml.data.schemas import BAR_V1
from python.qbt_ml.models.temporal_transformer import (
    TemporalTransformerConfig,
    TemporalTransformerV1,
)
from python.qbt_ml.features.scaling import FeatureStandardizerV1
from python.qbt_ml.training.robust_training import (
    apl_direction_loss,
    build_stress_sets,
    direction_noise_audit,
    feature_pgd_perturbation,
    latent_fgm_loss,
    structured_missing_augmentation,
    validate_robust_training_config,
)
from python.qbt_ml.training.phase2b import _paired_block_bootstrap


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


def test_train_fold_standardizer_excludes_padding_and_preserves_states(tmp_path):
    values = np.asarray([
        [[10.0, 1.0, 1.0], [14.0, 0.0, 1.0]],
        [[999.0, 9.0, 0.0], [18.0, 3.0, 0.0]],
    ], dtype=np.float32)
    mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    scaler = FeatureStandardizerV1.fit(
        values, mask, ("continuous", "is_st", "other"),
    )
    assert scaler.mean[0] == pytest.approx(12.0)
    assert scaler.mean[1] == 0.0
    assert scaler.scale[1] == 1.0
    transformed = scaler.transform(values, mask)
    assert np.all(transformed[1] == 0.0)
    np.testing.assert_allclose(
        scaler.inverse_transform(transformed[:1]), values[:1], atol=1e-5,
    )
    path = tmp_path / "scaler.json"
    scaler.write(path)
    restored = FeatureStandardizerV1.read(path)
    assert restored.sha256 == scaler.sha256


def test_model_raw_and_standardized_forward_are_equivalent():
    torch = pytest.importorskip("torch")
    scaler = FeatureStandardizerV1.fit(
        np.asarray([[[2.0, 1.0], [4.0, 0.0]]], dtype=np.float32),
        np.ones((1, 2), dtype=np.uint8),
        ("continuous", "is_st"),
    )
    model = TemporalTransformerV1(TemporalTransformerConfig(
        feature_count=2, lookback=2, d_model=4, nhead=2,
        num_layers=1, dim_feedforward=8, dropout=0.0,
        input_mean=scaler.mean, input_scale=scaler.scale,
        input_protected=scaler.protected,
    )).eval()
    features = torch.tensor([[[2.0, 1.0], [4.0, 0.0]]])
    mask = torch.ones((1, 2), dtype=torch.uint8)
    with torch.no_grad():
        raw = model(features, mask)
        standardized = model.forward_standardized(
            model.normalize_features(features, mask), mask,
        )
    for name in ("expected_return", "expected_volatility", "direction_probability"):
        torch.testing.assert_close(raw[name], standardized[name])


def test_structured_missing_only_updates_valid_continuous_features():
    torch = pytest.importorskip("torch")
    features = torch.ones((1, 3, 3))
    mask = torch.tensor([[0, 1, 1]], dtype=torch.uint8)
    augmented, update = structured_missing_augmentation(
        features, mask,
        feature_names=("continuous", "is_st", "other"),
        center=torch.tensor([5.0, 7.0, 9.0]), rate=1.0,
    )
    assert not update[0, 0].any()
    assert augmented[0, 1, 0] == 5.0
    assert augmented[0, 1, 2] == 9.0
    assert augmented[0, 1, 1] == 1.0


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


def test_center_impute_stress_keeps_valid_mask_and_uses_center():
    names = BAR_V1.feature_names
    values = np.ones((2, 4, len(names)), dtype=np.float32)
    masks = np.ones((2, 4), dtype=np.uint8)
    center = np.arange(len(names), dtype=np.float32)
    stress = build_stress_sets(
        values, masks, feature_names=names, seed=3,
        missing_mode="continuous_center_impute", missing_center=center,
    )
    missing = stress["missing"]
    assert missing["metadata"]["mode"] == "continuous_center_impute"
    np.testing.assert_array_equal(missing["valid_mask"], masks)
    continuous = np.asarray([
        name not in {"is_suspended", "is_listed", "is_st", "is_tradable"}
        for name in names
    ])
    changed = missing["features"] != values
    assert np.all(missing["features"][changed] == np.broadcast_to(
        center, values.shape,
    )[changed])
    assert not changed[..., ~continuous].any()


def test_paired_block_bootstrap_is_reproducible_and_zero_for_equal_series():
    first = _paired_block_bootstrap(
        [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],
        block_length=2, draws=50, seed=7,
    )
    second = _paired_block_bootstrap(
        [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],
        block_length=2, draws=50, seed=7,
    )
    assert first == second
    assert first["estimate"] == 0.0
    assert first["ci_low"] == 0.0
    assert first["ci_high"] == 0.0


def test_robust_config_is_opt_in_and_feature_pgd_is_registered():
    baseline = validate_robust_training_config(None, feature_names=BAR_V1.feature_names)
    assert baseline.mode == "none"
    candidate = validate_robust_training_config({
        "mode": "feature_pgd", "hypothesis_id": "phase2b-pgd-v1",
        "epsilon": 0.03, "pgd_steps": 2,
    }, feature_names=BAR_V1.feature_names)
    assert candidate.adversarial_enabled
    missing = validate_robust_training_config({
        "mode": "structured_missing", "hypothesis_id": "phase2b-missing-v2",
        "missing_mode": "continuous_center_impute", "missing_rate": 0.1,
    }, feature_names=BAR_V1.feature_names)
    assert missing.mode == "structured_missing"
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


def test_feature_pgd_v2_smoke_persists_pure_pgd_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from python.qbt_ml.cli import _train
    from python.tests.test_phase1e_gradients import _phase1e_training_fixture

    dataset, base_config = _phase1e_training_fixture(tmp_path)
    config = dict(base_config)
    config["training"] = dict(base_config["training"])
    config["training"]["output"] = str(tmp_path / "feature-pgd-v2")
    config["preprocessing"] = {
        "mode": "train_fold_standardizer_v1", "scale_floor": 1e-6,
    }
    config["robust_training"] = {
        "mode": "feature_pgd", "hypothesis_id": "phase2b-feature-pgd-v2-smoke",
        "epsilon": 0.01, "beta": 0.25, "pgd_steps": 1,
    }
    _train(config, str(dataset), None)
    checkpoint = torch.load(
        tmp_path / "feature-pgd-v2" / "checkpoint.pt",
        map_location="cpu", weights_only=True,
    )
    assert checkpoint["preprocessing_mode"] == "train_fold_standardizer_v1"
    assert checkpoint["preprocessing_spec_sha256"]
    assert checkpoint["model_config"]["input_mean"]
    assert checkpoint["robust_training_spec"]["missing_mode"] == "none"
    assert (tmp_path / "feature-pgd-v2" / "checkpoint_best.pt").is_file()
    assert (tmp_path / "feature-pgd-v2" / "checkpoint_last.pt").is_file()


def test_feature_pgd_rejects_implicit_structured_missing():
    with pytest.raises(ValueError, match="纯 PGD"):
        validate_robust_training_config({
            "mode": "feature_pgd",
            "hypothesis_id": "phase2b-feature-pgd-invalid-hybrid",
            "missing_mode": "continuous_center_impute",
            "missing_rate": 0.1,
        }, feature_names=BAR_V1.feature_names)


def test_validation_only_training_never_evaluates_or_writes_test(tmp_path):
    torch = pytest.importorskip("torch")
    from python.qbt_ml.cli import _train
    from python.tests.test_phase1e_gradients import _phase1e_training_fixture

    dataset, base_config = _phase1e_training_fixture(tmp_path)
    config = dict(base_config)
    config["training"] = dict(base_config["training"])
    config["training"]["output"] = str(tmp_path / "validation-blind")
    config["training"]["evaluation_split"] = "validation"
    _train(config, str(dataset), None)
    output = tmp_path / "validation-blind"
    assert (output / "validation_predictions.npz").is_file()
    assert (output / "validation_embeddings.npz").is_file()
    assert not (output / "test_predictions.npz").exists()
    assert not (output / "test_embeddings.npz").exists()
    assert not (output / "embedding_manifest.json").exists()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_split"] == "validation"
    assert not any(key.startswith("test") for key in metrics)
    checkpoint = torch.load(
        output / "checkpoint.pt", map_location="cpu", weights_only=True,
    )
    assert checkpoint["evaluation_split"] == "validation"
