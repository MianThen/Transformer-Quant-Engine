from __future__ import annotations

import numpy as np
import pytest

from python.qbt_ml.research.infots import (
    InfoTSArtifactValidationError,
    InfoTSAugmentationSpecV1,
    apply_causal_augmentations,
    build_infots_pretraining_artifact,
    info_nce_loss,
    select_information_aware_augmentation,
    validate_infots_pretraining_artifact,
)
from python.qbt_ml.research.infots_training import (
    InfoTSTrainingSpecV1,
    InfoTSTrainingUnavailable,
    PRODUCTION_OUTPUT_KEYS,
    train_infots_pretraining,
    validate_six_output_contract,
)


def _inputs():
    features = np.asarray([
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0], [4.0, 40.0, 400.0], [0.0, 0.0, 0.0]],
        [[5.0, 50.0, 500.0], [6.0, 60.0, 600.0], [7.0, 70.0, 700.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    valid_mask = np.asarray([[True, True, True, True, False], [True, True, True, False, False]])
    timestamps = np.asarray([10, 20, 30, 40, 50])
    return features, valid_mask, timestamps


def test_causal_augmentation_is_deterministic_and_guards_state_padding_future():
    features, valid_mask, timestamps = _inputs()
    spec = InfoTSAugmentationSpecV1(
        seed=37,
        crop_min_length=2,
        crop_max_length=3,
        noise_std=0.1,
        time_mask_probability=0.4,
        continuous_feature_indices=(0, 1),
        state_feature_indices=(2,),
    )
    first = apply_causal_augmentations(features, valid_mask, timestamps, spec)
    second = apply_causal_augmentations(features, valid_mask, timestamps, spec)
    for key in ("global_view", "local_view", "global_valid_mask", "local_valid_mask", "global_time_mask", "local_time_mask"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["metadata"] == second["metadata"]
    assert first["metadata"]["future_guard_passed"] is True
    assert first["metadata"]["padding_guard_passed"] is True
    np.testing.assert_array_equal(first["global_view"][:, :, 2][valid_mask], features[:, :, 2][valid_mask])
    np.testing.assert_array_equal(first["local_view"][:, :, 2][first["local_valid_mask"]], features[:, :, 2][first["local_valid_mask"]])
    assert np.all(first["global_view"][~valid_mask] == 0.0)
    assert np.all(first["local_view"][~first["local_valid_mask"]] == 0.0)
    mutated = features.copy()
    mutated[:, -1, :] = 1e9
    mutated_result = apply_causal_augmentations(mutated, valid_mask, timestamps, spec)
    np.testing.assert_array_equal(first["global_view"][:, :-1], mutated_result["global_view"][:, :-1])
    np.testing.assert_array_equal(first["local_view"][:, :-1], mutated_result["local_view"][:, :-1])


def test_infotype_nce_and_selector_are_deterministic():
    global_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    local_embeddings = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.7]])
    first = info_nce_loss(global_embeddings, local_embeddings, temperature=0.2)
    second = info_nce_loss(global_embeddings, local_embeddings, temperature=0.2)
    assert first == second
    assert first["loss"] > 0.0
    candidates = [
        {"augmentation_id": "baseline", "info_nce_loss": first["loss"], "downstream_probe_score": 0.6, "future_guard_passed": True, "state_guard_passed": True, "padding_guard_passed": True},
        {"augmentation_id": "candidate", "info_nce_loss": first["loss"] - 0.1, "downstream_probe_score": 0.65, "future_guard_passed": True, "state_guard_passed": True, "padding_guard_passed": True},
        {"augmentation_id": "unsafe", "info_nce_loss": 0.01, "downstream_probe_score": 1.0, "future_guard_passed": False, "state_guard_passed": True, "padding_guard_passed": True},
    ]
    selection = select_information_aware_augmentation(candidates, baseline_id="baseline", minimum_improvement=0.01)
    assert selection["selected_id"] == "candidate"
    assert selection["evidence_level"] == "REFERENCE_ONLY"
    assert selection["selector_sha256"]


def test_train_fold_only_artifact_hash_and_future_guard():
    features, valid_mask, timestamps = _inputs()
    spec = InfoTSAugmentationSpecV1(seed=3, continuous_feature_indices=(0, 1), state_feature_indices=(2,))
    augmented = apply_causal_augmentations(features, valid_mask, timestamps, spec)
    candidates = [{"augmentation_id": "baseline", "info_nce_loss": 0.4, "downstream_probe_score": 0.5, "future_guard_passed": True, "state_guard_passed": True, "padding_guard_passed": True}]
    selector = select_information_aware_augmentation(candidates, baseline_id="baseline")
    artifact = build_infots_pretraining_artifact(
        dataset_sha256="a" * 64,
        train_fold_end=30,
        train_timestamps=[10, 20, 30],
        augmentation_metadata=augmented["metadata"],
        selector_artifact=selector,
        supervised_budget=100,
        seed=3,
    )
    validate_infots_pretraining_artifact(artifact)
    changed = dict(artifact, train_timestamp_count=4)
    with pytest.raises(InfoTSArtifactValidationError):
        validate_infots_pretraining_artifact(changed)
    with pytest.raises(InfoTSArtifactValidationError):
        build_infots_pretraining_artifact(
            dataset_sha256="a" * 64,
            train_fold_end=30,
            train_timestamps=[10, 20, 31],
            augmentation_metadata=augmented["metadata"],
            selector_artifact=selector,
            supervised_budget=100,
            seed=3,
        )


def test_training_entry_requires_torch_or_cuda_and_six_output_contract():
    prediction = {key: np.asarray([0.1]) for key in PRODUCTION_OUTPUT_KEYS}
    validate_six_output_contract(prediction)
    with pytest.raises(InfoTSArtifactValidationError):
        validate_six_output_contract({"expected_return": np.asarray([0.1])})
    features, valid_mask, timestamps = _inputs()
    augmentation_spec = InfoTSAugmentationSpecV1(
        seed=5, continuous_feature_indices=(0, 1), state_feature_indices=(2,), crop_min_length=2,
    )
    training_spec = InfoTSTrainingSpecV1(
        seed=5, epochs=1, batch_size=2, train_fold_end=40, device="cuda",
    )
    with pytest.raises(InfoTSTrainingUnavailable):
        train_infots_pretraining(features, valid_mask, timestamps, augmentation_spec, training_spec)
