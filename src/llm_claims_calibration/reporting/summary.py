from __future__ import annotations

import platform
from pathlib import Path
from importlib import metadata

import pandas as pd


def _table_text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False)


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def write_summary_report(
    output_path: str,
    config_title: str,
    dataset_size: int,
    selected_method: str,
    calibration_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    threshold_results: pd.DataFrame,
    posterior_summary: pd.DataFrame,
    split_sizes: dict[str, int],
    random_seed: int,
    bayesian_prior: dict[str, float],
    reproduction_command: str,
) -> None:
    best_row = calibration_results.sort_values("ece").iloc[0]
    lines = [
        f"# {config_title}",
        "",
        "## Summary",
        f"- Synthetic claims generated: {dataset_size}",
        f"- Best calibration method by ECE: `{selected_method}`",
        f"- Best observed test-set ECE: {best_row['ece']:.4f}",
        f"- Best observed test-set Brier score: {best_row['brier_score']:.4f}",
        "",
        "## Data Split",
        f"- Development/train size: {split_sizes['train']}",
        f"- Calibration size: {split_sizes['calibration']}",
        f"- Evaluation/test size: {split_sizes['test']}",
        "",
        "## Reproducibility",
        f"- Random seed: {random_seed}",
        f"- Python version: {platform.python_version()}",
        f"- `numpy`: {_package_version('numpy')}",
        f"- `pandas`: {_package_version('pandas')}",
        f"- `scipy`: {_package_version('scipy')}",
        f"- `scikit-learn`: {_package_version('scikit-learn')}",
        f"- `matplotlib`: {_package_version('matplotlib')}",
        f"- `pydantic`: {_package_version('pydantic')}",
        f"- `PyYAML`: {_package_version('PyYAML')}",
        f"- Reproduction command: `{reproduction_command}`",
        "",
        "## Calibration Comparison",
        "```text",
        _table_text(calibration_results),
        "```",
        "",
        "## Bootstrap Confidence Intervals",
        "```text",
        _table_text(bootstrap_results),
        "```",
        "",
        "## Expected-Loss Thresholds",
        "```text",
        _table_text(threshold_results),
        "```",
        "",
        "## Bayesian Prior",
        f"- alpha0: {bayesian_prior['alpha0']}",
        f"- beta0: {bayesian_prior['beta0']}",
        "",
        "## Bayesian Posterior by Risk Tier",
        "```text",
        _table_text(posterior_summary),
        "```",
        "",
        "## Reproduction",
        f"- `{reproduction_command}`",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
