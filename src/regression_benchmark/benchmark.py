"""Reproducible training, evaluation, and visualization for regression models."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "regression-model-benchmark-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluation outputs produced by a benchmark run."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, RegressorMixin]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the built-in diabetes regression dataset as pandas objects."""

    dataset = load_diabetes(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename("disease_progression").copy()
    return features, target


def build_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Create the baseline and candidate regression models."""

    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Polynomial ridge": Pipeline(
            steps=[
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RBF SVR": TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "Decision tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def run_benchmark(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.20,
    cv_splits: int = 5,
    random_state: int = 42,
) -> BenchmarkResult:
    """Evaluate all models with cross-validation and a held-out test set."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    cross_validation = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    metric_rows: list[dict[str, float | str]] = []
    prediction_columns: dict[str, np.ndarray] = {
        "Actual": y_test.to_numpy(),
    }
    fitted_models: dict[str, RegressorMixin] = {}

    for model_name, model in build_models(random_state).items():
        cv_scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=None,
        )
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        metric_rows.append(
            {
                "Model": model_name,
                "CV MAE": float(-cv_scores["test_mae"].mean()),
                "CV RMSE": float(-cv_scores["test_rmse"].mean()),
                "CV R2": float(cv_scores["test_r2"].mean()),
                "Test MAE": float(mean_absolute_error(y_test, predicted)),
                "Test RMSE": float(root_mean_squared_error(y_test, predicted)),
                "Test R2": float(r2_score(y_test, predicted)),
            }
        )
        prediction_columns[model_name] = np.asarray(predicted)
        fitted_models[model_name] = model

    metrics = pd.DataFrame(metric_rows).sort_values("Test RMSE").reset_index(drop=True)
    predictions = pd.DataFrame(prediction_columns, index=y_test.index).sort_index()
    predictions.index.name = "sample_id"

    return BenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        fitted_models=fitted_models,
    )


def save_results(result: BenchmarkResult, output_dir: str | Path) -> dict[str, Path]:
    """Write metrics, predictions, and diagnostic plots to disk."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.csv"
    predictions_path = destination / "predictions.csv"
    comparison_path = destination / "model_comparison.png"
    predictions_plot_path = destination / "predicted_vs_actual.png"
    residuals_path = destination / "residuals.png"

    result.metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    result.predictions.to_csv(predictions_path, float_format="%.4f")

    colors = ["#1877F2", "#00A67E", "#E4572E", "#6C5CE7", "#D4A017", "#44546A"]

    ordered = result.metrics.sort_values("Test RMSE", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.barh(ordered["Model"], ordered["Test RMSE"], color=colors[: len(ordered)])
    axis.bar_label(bars, fmt="%.1f", padding=4)
    axis.set_title("Held-out Test RMSE by Model")
    axis.set_xlabel("RMSE (lower is better)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    model_names = [column for column in result.predictions.columns if column != "Actual"]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    actual = result.predictions["Actual"]
    lower = min(actual.min(), result.predictions[model_names].min().min())
    upper = max(actual.max(), result.predictions[model_names].max().max())
    for axis, model_name, color in zip(axes.flat, model_names, colors, strict=True):
        predicted = result.predictions[model_name]
        axis.scatter(actual, predicted, alpha=0.72, color=color, edgecolor="white", linewidth=0.4)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#333333", linewidth=1)
        axis.set_title(model_name)
        axis.grid(alpha=0.2)
    figure.supxlabel("Actual target")
    figure.supylabel("Predicted target")
    figure.suptitle("Predicted vs. Actual Values", fontsize=15)
    figure.tight_layout()
    figure.savefig(predictions_plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    residual_values = [actual - result.predictions[name] for name in model_names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    boxplot = axis.boxplot(
        residual_values,
        tick_labels=model_names,
        orientation="horizontal",
        patch_artist=True,
    )
    for patch, color in zip(boxplot["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.axvline(0, linestyle="--", color="#333333", linewidth=1)
    axis.set_title("Residual Distribution by Model")
    axis.set_xlabel("Actual - predicted")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(residuals_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "comparison_plot": comparison_path,
        "predictions_plot": predictions_plot_path,
        "residuals_plot": residuals_path,
    }
