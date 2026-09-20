from pathlib import Path

import numpy as np

from regression_benchmark import build_models, load_dataset, run_benchmark, save_results


def test_dataset_has_expected_shape_and_no_missing_values() -> None:
    features, target = load_dataset()

    assert features.shape == (442, 10)
    assert target.shape == (442,)
    assert not features.isna().any().any()
    assert not target.isna().any()


def test_model_registry_contains_baseline_and_five_candidates() -> None:
    models = build_models()

    assert list(models) == [
        "Mean baseline",
        "Linear regression",
        "Polynomial ridge",
        "RBF SVR",
        "Decision tree",
        "Random forest",
    ]


def test_benchmark_returns_finite_metrics_and_predictions() -> None:
    features, target = load_dataset()
    result = run_benchmark(features, target, cv_splits=3)

    assert result.metrics.shape == (6, 7)
    assert result.predictions.shape == (89, 7)
    assert np.isfinite(result.metrics.select_dtypes(include="number").to_numpy()).all()
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_save_results_writes_expected_artifacts(tmp_path: Path) -> None:
    features, target = load_dataset()
    result = run_benchmark(features, target, cv_splits=2)
    written = save_results(result, tmp_path)

    assert set(written) == {
        "metrics",
        "predictions",
        "comparison_plot",
        "predictions_plot",
        "residuals_plot",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())

