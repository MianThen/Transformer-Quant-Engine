from __future__ import annotations

import copy
import json
from collections import OrderedDict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from python.qbt_ml.training import (
    GRADIENT_TASKS,
    GradientConflictRecorder,
    GradientDiagnosticSpecV1,
    GradNormController,
    GradNormSpecV1,
    PCGradSpecV1,
    backward_pcgrad,
    pcgrad_project,
    shared_named_parameters,
    shared_parameter_set_sha256,
)
from python.qbt_ml.cli import _train


TASK_WEIGHTS = {
    "return": 1.0,
    "direction": 0.25,
    "volatility": 0.25,
    "quantile": 0.25,
    "rank": 0.1,
}


class ToyMultitaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        self.return_head = torch.nn.Parameter(torch.tensor(1.0))
        self.direction_head = torch.nn.Parameter(torch.tensor(1.0))
        self.volatility_head = torch.nn.Parameter(torch.tensor(1.0))
        self.quantile_head = torch.nn.Parameter(torch.tensor(1.0))
        self.rank_head = torch.nn.Parameter(torch.tensor(1.0))

    def components(self):
        return OrderedDict((
            ("return", self.return_head * (self.shared[0] + 2.0)),
            ("direction", self.direction_head * (-self.shared[0] + self.shared[1] + 2.0)),
            ("volatility", self.volatility_head * (self.shared[1] + 2.0)),
            ("quantile", self.quantile_head * (self.shared.sum() + 2.0)),
            ("rank", self.rank_head * (-self.shared[0] - self.shared[1] + 2.0)),
        ))


def _weighted(components):
    return OrderedDict(
        (task, TASK_WEIGHTS[task] * components[task]) for task in GRADIENT_TASKS
    )


def _pcgrad_spec() -> PCGradSpecV1:
    return PCGradSpecV1(hypothesis_id="phase1e-pcgrad-test")


def _gradnorm_spec(**overrides) -> GradNormSpecV1:
    values = {"hypothesis_id": "phase1e-gradnorm-test", **overrides}
    return GradNormSpecV1(**values)


def test_gradient_diagnostics_do_not_mutate_gradients_or_update() -> None:
    baseline = ToyMultitaskModel()
    instrumented = copy.deepcopy(baseline)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    instrumented_optimizer = torch.optim.SGD(instrumented.parameters(), lr=0.05)

    baseline_components = baseline.components()
    sum(_weighted(baseline_components).values()).backward()
    baseline_optimizer.step()

    components = instrumented.components()
    recorder = GradientConflictRecorder(GradientDiagnosticSpecV1())
    recorder.sample(
        components,
        shared_named_parameters(instrumented),
        TASK_WEIGHTS,
        step=0,
        epoch=0,
        fold=1,
    )
    assert all(parameter.grad is None for parameter in instrumented.parameters())
    sum(_weighted(components).values()).backward()
    instrumented_optimizer.step()

    for baseline_parameter, instrumented_parameter in zip(
        baseline.parameters(), instrumented.parameters()
    ):
        torch.testing.assert_close(
            baseline_parameter, instrumented_parameter, rtol=0.0, atol=0.0
        )


def test_gradient_diagnostic_oracle_and_artifact_replay() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    components = OrderedDict((
        ("return", parameter[0] + 10.0),
        ("direction", -parameter[0] + 10.0),
        ("volatility", 2.0 * parameter[1] + 10.0),
        ("quantile", 0.5 * parameter.sum() + 10.0),
        ("rank", -parameter[1] + 10.0),
    ))
    weights = {task: 1.0 for task in GRADIENT_TASKS}
    recorder = GradientConflictRecorder(GradientDiagnosticSpecV1())
    sample = recorder.sample(
        components, (("shared", parameter),), weights,
        step=0, epoch=2, fold=3, regime="HIGH_VOL",
    )

    assert parameter.grad is None
    assert sample["gradient_norms"]["return"] == pytest.approx(1.0)
    assert sample["dot_products"]["return|direction"] == pytest.approx(-1.0)
    assert sample["cosine_similarity"]["return|direction"] == pytest.approx(-1.0)
    assert sample["negative_cosine_conflict_rate"] > 0.0
    assert set(sample["relative_training_rate"]) == set(GRADIENT_TASKS)
    assert all(value == pytest.approx(1.0) for value in sample["normalized_losses"].values())

    artifact = recorder.artifact(
        (("shared", parameter),), model_contract_sha256="a" * 64,
        dataset_sha256="b" * 64,
    )
    replay = recorder.artifact(
        (("shared", parameter),), model_contract_sha256="a" * 64,
        dataset_sha256="b" * 64,
    )
    assert artifact["report_sha256"] == replay["report_sha256"]
    assert artifact["aggregates"][0]["sample_count"] == 1
    assert "return|direction" in artifact["aggregates"][0]["conflict_matrix"]


def test_shared_parameter_hash_captures_name_and_shape() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    baseline = shared_parameter_set_sha256((("shared.weight", parameter),))
    renamed = shared_parameter_set_sha256((("encoder.weight", parameter),))
    reshaped = shared_parameter_set_sha256(
        (("shared.weight", torch.nn.Parameter(torch.ones(1, 2))),)
    )
    assert len(baseline) == 64
    assert baseline != renamed
    assert baseline != reshaped


def test_pcgrad_negative_projection_zero_guard_and_replay() -> None:
    gradients = OrderedDict((
        ("return", torch.tensor([1.0, 0.0])),
        ("direction", torch.tensor([-1.0, 1.0])),
        ("volatility", torch.zeros(2)),
        ("quantile", torch.zeros(2)),
        ("rank", torch.zeros(2)),
    ))
    first, first_aggregate = pcgrad_project(gradients, _pcgrad_spec(), step=7)
    second, second_aggregate = pcgrad_project(gradients, _pcgrad_spec(), step=7)

    torch.testing.assert_close(first["return"], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(first["direction"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(first_aggregate, second_aggregate)
    for task in GRADIENT_TASKS:
        torch.testing.assert_close(first[task], second[task])
        assert torch.isfinite(first[task]).all()


def test_pcgrad_only_replaces_shared_gradient_and_preserves_accumulation() -> None:
    baseline = ToyMultitaskModel()
    challenger = copy.deepcopy(baseline)
    accumulated = copy.deepcopy(baseline)

    baseline_components = baseline.components()
    sum(_weighted(baseline_components).values()).backward()
    baseline_head_gradients = {
        name: parameter.grad.clone()
        for name, parameter in baseline.named_parameters()
        if name.endswith("_head")
    }
    baseline_shared_gradient = baseline.shared.grad.clone()

    challenger_components = challenger.components()
    weighted = _weighted(challenger_components)
    metrics = backward_pcgrad(
        weighted, sum(weighted.values()), (challenger.shared,), _pcgrad_spec(), step=4
    )
    assert not torch.equal(challenger.shared.grad, baseline_shared_gradient)
    for name, parameter in challenger.named_parameters():
        if name.endswith("_head"):
            torch.testing.assert_close(parameter.grad, baseline_head_gradients[name])
    assert metrics["spec_sha256"] == _pcgrad_spec().sha256
    assert 0.0 <= metrics["projection_frequency"] <= 1.0

    previous = torch.tensor([0.3, -0.2])
    accumulated.shared.grad = previous.clone()
    accumulated_components = accumulated.components()
    accumulated_weighted = _weighted(accumulated_components)
    backward_pcgrad(
        accumulated_weighted,
        sum(accumulated_weighted.values()),
        (accumulated.shared,),
        _pcgrad_spec(),
        step=4,
    )
    torch.testing.assert_close(accumulated.shared.grad, challenger.shared.grad + previous)


def test_pcgrad_rejects_incomplete_or_mismatched_tasks() -> None:
    gradients = OrderedDict((task, torch.ones(2)) for task in GRADIENT_TASKS)
    gradients.pop("rank")
    with pytest.raises(ValueError):
        pcgrad_project(gradients, _pcgrad_spec())

    mismatched = OrderedDict((task, torch.ones(2)) for task in GRADIENT_TASKS)
    mismatched["rank"] = torch.ones(3)
    with pytest.raises(ValueError):
        pcgrad_project(mismatched, _pcgrad_spec())


def test_gradnorm_weights_positive_normalized_and_rank_fixed() -> None:
    controller = GradNormController(_gradnorm_spec(rank_weight=0.1))
    weights = controller.normalized_weights()
    assert torch.all(weights > 0.0)
    assert float(weights.sum().detach()) == pytest.approx(4.0)

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    components = OrderedDict((
        ("return", (parameter[0] - 2.0).square()),
        ("direction", (parameter[1] + 2.0).square()),
        ("volatility", parameter.square().mean()),
        ("quantile", (parameter.sum() - 1.0).square()),
        ("rank", (parameter[0] + 3.0).square()),
    ))
    adaptive_total = sum(controller.weighted_task_losses(components).values())
    assert float(controller.model_loss(components).detach()) == pytest.approx(
        float(adaptive_total.detach() + 0.1 * components["rank"].detach())
    )


def test_gradnorm_oracle_initial_loss_freeze_and_finite_weight_update() -> None:
    controller = GradNormController(_gradnorm_spec(alpha=0.0))
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))

    def components(scale: float):
        return OrderedDict((
            ("return", scale * (parameter[0] - 2.0).square()),
            ("direction", scale * (parameter[1] + 2.0).square()),
            ("volatility", scale * parameter.square().mean()),
            ("quantile", scale * (parameter.sum() - 1.0).square()),
            ("rank", scale * (parameter[0] + 3.0).square()),
        ))

    objective, diagnostics = controller.gradnorm_objective(components(1.0), (parameter,))
    initial = controller.initial_losses.clone()
    targets = list(diagnostics["target_gradient_norms"].values())
    assert targets == pytest.approx([sum(targets) / len(targets)] * len(targets))
    objective.backward()
    assert torch.isfinite(controller.raw_weights.grad).all()

    optimizer = torch.optim.SGD(controller.parameters(), lr=0.05)
    optimizer.step()
    assert torch.all(controller.normalized_weights() > 0.0)
    assert float(controller.normalized_weights().sum().detach()) == pytest.approx(4.0)

    controller.gradnorm_objective(components(3.0), (parameter,))
    torch.testing.assert_close(controller.initial_losses, initial)


def test_gradnorm_constant_zero_loss_and_checkpoint_restore() -> None:
    controller = GradNormController(_gradnorm_spec())
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    components = OrderedDict((
        ("return", parameter.sum() * 0.0),
        ("direction", parameter.sum() * 0.0 + 2.0),
        ("volatility", parameter.square().sum() * 0.0),
        ("quantile", parameter[0] * 0.0 + 4.0),
        ("rank", parameter.sum() * 0.0 + 1.0),
    ))
    objective, diagnostics = controller.gradnorm_objective(components, (parameter,))
    assert torch.isfinite(objective)
    assert all(
        torch.isfinite(torch.tensor(value))
        for values in diagnostics.values() if isinstance(values, dict)
        for value in values.values()
    )

    restored = GradNormController(_gradnorm_spec())
    restored.load_state_dict(controller.state_dict())
    torch.testing.assert_close(restored.raw_weights, controller.raw_weights)
    torch.testing.assert_close(restored.initial_losses, controller.initial_losses)
    torch.testing.assert_close(restored.normalized_weights(), controller.normalized_weights())


def test_gradnorm_spec_rejects_adaptive_rank() -> None:
    with pytest.raises(ValueError):
        GradNormSpecV1(
            hypothesis_id="invalid",
            adaptive_tasks=GRADIENT_TASKS,
        )
    with pytest.raises(ValueError):
        GradNormSpecV1(hypothesis_id="invalid-rank-weight", rank_weight=0.2)


def _phase1e_training_fixture(tmp_path):
    timestamps = np.repeat(np.arange(1, 15, dtype=np.int64), 3)
    symbol_index = np.tile(np.arange(3, dtype=np.float32), 14)
    generator = np.random.default_rng(20260801)
    features = generator.normal(size=(timestamps.size, 4, 3)).astype(np.float32)
    expected = (0.001 * timestamps + 0.002 * symbol_index).astype(np.float32)
    direction = (expected > np.median(expected)).astype(np.float32)
    volatility = (0.01 + 0.001 * symbol_index).astype(np.float32)
    dataset = tmp_path / "phase1e-dataset.npz"
    np.savez_compressed(
        dataset,
        features=features,
        valid_mask=np.ones((timestamps.size, 4), dtype=np.uint8),
        timestamps=timestamps,
        symbols=np.asarray(["A", "B", "C"] * 14),
        expected_return=expected,
        direction=direction,
        realized_volatility=volatility,
        rank_utility=expected,
        rank_relevance=np.tile(np.asarray([0.0, 0.5, 1.0], dtype=np.float32), 14),
        label_spec_sha256=np.asarray("a" * 64),
        ranking_score_spec_sha256=np.asarray("b" * 64),
        feature_schema_sha256=np.asarray("c" * 64),
    )
    return dataset, {
        "enabled": True,
        "seed": 20260801,
        "label_v2": {"horizon_bars": 1, "execution_lag_bars": 1},
        "ranking_score": {
            "mode": "raw_return",
            "production_top_k": 2,
            "lambda_rank": 0.1,
        },
        "ranking": {
            "loss_variant": "legacy",
            "cross_sections_per_batch": 1,
            "return_weight": 1.0,
            "direction_weight": 0.25,
            "volatility_weight": 0.25,
            "quantile_weight": 0.25,
        },
        "split": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "purge_timestamps": 2,
            "embargo_timestamps": 1,
        },
        "model": {
            "d_model": 4,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 8,
            "dropout": 0.0,
        },
        "training": {
            "dataset": str(dataset),
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "cpu_threads": 1,
            "device": "cpu",
        },
    }


def test_phase1e_training_modes_are_auditable_and_diagnostics_are_zero_impact(
    tmp_path, capsys,
) -> None:
    dataset, base_config = _phase1e_training_fixture(tmp_path)
    runs = {}
    modes = {
        "none": {"mode": "none"},
        "diagnostics": {"mode": "diagnostics", "cadence_steps": 2},
        "pcgrad": {"mode": "pcgrad", "hypothesis_id": "phase1e-pcgrad-smoke"},
        "gradnorm": {
            "mode": "gradnorm",
            "hypothesis_id": "phase1e-gradnorm-smoke",
            "alpha": 0.0,
            "weight_learning_rate": 1e-3,
        },
    }
    for mode, gradient_config in modes.items():
        run = tmp_path / mode
        config = copy.deepcopy(base_config)
        config["training"]["output"] = str(run)
        config["gradient_optimization"] = gradient_config
        _train(config, str(dataset), None)
        progress = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert progress["event"] == "training_epoch_completed"
        assert progress["epoch"] == progress["epochs"] == 1
        assert progress["gradient_mode"] == mode
        assert progress["train_loss"] > 0.0
        checkpoint = torch.load(
            run / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        assert checkpoint["gradient_optimization_mode"] == mode
        assert metrics["gradient_optimization_mode"] == mode
        runs[mode] = (run, checkpoint, metrics)

    baseline_state = runs["none"][1]["model_state_dict"]
    diagnostic_state = runs["diagnostics"][1]["model_state_dict"]
    assert baseline_state.keys() == diagnostic_state.keys()
    for name in baseline_state:
        torch.testing.assert_close(
            baseline_state[name], diagnostic_state[name], rtol=0.0, atol=0.0
        )
    assert runs["none"][1]["history"] == runs["diagnostics"][1]["history"]

    artifact = json.loads((
        runs["diagnostics"][0] / "gradient_conflict_artifact.json"
    ).read_text(encoding="utf-8"))
    assert artifact["samples"]
    assert artifact["aggregates"]
    assert artifact["report_sha256"] == runs["diagnostics"][2][
        "gradient_conflict_report_sha256"
    ]

    pcgrad_diagnostics = runs["pcgrad"][1]["gradient_mechanism_diagnostics"]
    assert pcgrad_diagnostics
    assert all(0.0 <= item["projection_frequency"] <= 1.0
               for item in pcgrad_diagnostics)

    gradnorm_checkpoint = runs["gradnorm"][1]
    assert gradnorm_checkpoint["gradient_optimization_state_dict"] is not None
    assert gradnorm_checkpoint["gradient_mechanism_diagnostics"]
    assert gradnorm_checkpoint["gradient_optimization_spec"]["rank_weight"] == 0.1
    assert torch.isfinite(
        gradnorm_checkpoint["gradient_optimization_state_dict"]["initial_losses"]
    ).all()


def test_phase1e_training_rejects_non_frozen_or_unregistered_combinations(
    tmp_path,
) -> None:
    dataset, base_config = _phase1e_training_fixture(tmp_path)

    missing_hypothesis = copy.deepcopy(base_config)
    missing_hypothesis["gradient_optimization"] = {"mode": "pcgrad"}
    with pytest.raises(ValueError, match="hypothesis_id"):
        _train(missing_hypothesis, str(dataset), str(tmp_path / "missing"))

    changed_rank = copy.deepcopy(base_config)
    changed_rank["ranking_score"]["lambda_rank"] = 0.2
    changed_rank["gradient_optimization"] = {"mode": "diagnostics"}
    with pytest.raises(ValueError, match="scalar weights"):
        _train(changed_rank, str(dataset), str(tmp_path / "changed-rank"))

    changed_loss = copy.deepcopy(base_config)
    changed_loss["ranking"]["loss_variant"] = "lambda"
    changed_loss["gradient_optimization"] = {"mode": "diagnostics"}
    with pytest.raises(ValueError, match="legacy rank"):
        _train(changed_loss, str(dataset), str(tmp_path / "changed-loss"))

    combined_kendall = copy.deepcopy(base_config)
    combined_kendall["multitask_weighting"] = {
        "mode": "kendall",
        "frozen_ranking_loss": "legacy",
    }
    combined_kendall["gradient_optimization"] = {"mode": "gradnorm", "hypothesis_id": "x"}
    with pytest.raises(ValueError, match="fixed weights"):
        _train(combined_kendall, str(dataset), str(tmp_path / "combined"))
