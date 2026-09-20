"""Command-line interface for the regression benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import load_dataset, run_benchmark, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare classical regression models with reproducible evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics, predictions, and plots (default: results).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features, target = load_dataset()
    result = run_benchmark(
        features,
        target,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )
    written_files = save_results(result, args.output_dir)

    display_metrics = result.metrics.copy()
    numeric_columns = display_metrics.select_dtypes(include="number").columns
    display_metrics[numeric_columns] = display_metrics[numeric_columns].round(3)
    print(display_metrics.to_string(index=False))
    print("\nSaved artifacts:")
    for label, path in written_files.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()

