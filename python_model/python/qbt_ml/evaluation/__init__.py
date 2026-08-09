"""Evaluation entry points, loaded lazily to keep optional stacks isolated."""

from importlib import import_module


_EXPORTS = {
    "prediction_metrics": (".metrics", "prediction_metrics"),
    "run_feature_ablation": (".ablation", "run_feature_ablation"),
    "run_model_benchmark": (".benchmark", "run_model_benchmark"),
    "run_cpp_portfolio_ablation": (".portfolio_benchmark", "run_cpp_portfolio_ablation"),
    "run_cpp_portfolio_benchmark": (".portfolio_benchmark", "run_cpp_portfolio_benchmark"),
    "run_deep_baseline_suite": (".deep_walk_forward", "run_deep_baseline_suite"),
    "run_promotion_review": (".promotion", "run_promotion_review"),
    "run_transformer_feature_ablation": (".ablation", "run_transformer_feature_ablation"),
    "run_transformer_walk_forward": (".deep_walk_forward", "run_transformer_walk_forward"),
    "run_walk_forward_baseline": (".walk_forward", "run_walk_forward_baseline"),
    "Phase3CCalibrationError": (".phase3c_calibration", "Phase3CCalibrationError"),
    "run_phase3c_calibration": (".phase3c_calibration", "run_phase3c_calibration"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
