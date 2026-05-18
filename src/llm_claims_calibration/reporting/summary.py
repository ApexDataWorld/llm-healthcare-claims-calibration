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


def write_cost_sensitivity_section(
    sensitivity_results: pd.DataFrame,
    output_path: Path,
) -> None:
    from llm_claims_calibration.cost.sensitivity import get_cost_scenarios

    scenarios = get_cost_scenarios()
    threshold_table = (
        sensitivity_results.pivot(index="scenario_name", columns="risk_tier", values="threshold")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    threshold_table.insert(1, "cost_ratio", threshold_table["scenario_name"].map(lambda name: scenarios[str(name)].cost_ratio))
    threshold_table = threshold_table.rename(
        columns={
            "scenario_name": "Scenario",
            "cost_ratio": "Cost Ratio",
            "low": "Low Tier",
            "medium": "Medium Tier",
            "high": "High Tier",
            "mandatory_review": "Mandatory Review",
        }
    )

    medium_table = sensitivity_results[sensitivity_results["risk_tier"] == "medium"].copy()
    medium_table.insert(1, "cost_ratio", medium_table["scenario_name"].map(lambda name: scenarios[str(name)].cost_ratio))
    medium_table = medium_table.rename(
        columns={
            "scenario_name": "Scenario",
            "cost_ratio": "Cost Ratio",
            "threshold": "Threshold",
            "coverage": "Coverage",
            "accepted_accuracy": "Accepted Accuracy",
            "expected_loss": "Expected Loss",
        }
    )[
        ["Scenario", "Cost Ratio", "Threshold", "Coverage", "Accepted Accuracy", "Expected Loss"]
    ]

    section_lines = [
        "",
        "## 8.7 Sensitivity Analysis: Cost Ratio Variation",
        "",
        "### 8.7.1 Cost Scenarios",
        "- `baseline_4x`: wrong decisions are modeled as four times the cost of human review.",
        "- `high_decision_cost_10x`: wrong decisions are modeled as ten times the cost of human review.",
        "- `high_review_cost_2x`: human review is modeled as twice the cost of a wrong decision.",
        "- `symmetric_1x`: wrong decisions and human review have equal cost.",
        "",
        "### 8.7.2 Results",
        "Table 8.7.1: Optimal Thresholds Across Cost Scenarios",
        "```text",
        _table_text(threshold_table),
        "```",
        "",
        "Table 8.7.2: Coverage and Accepted Accuracy by Scenario (Medium Tier)",
        "```text",
        _table_text(medium_table),
        "```",
        "",
        "### 8.7.3 Implications for Organizations",
        "- The framework recomputes thresholds directly from the selected cost ratio rather than assuming one universal operating point.",
        "- Organizations that treat wrong automated decisions as more expensive will generally prefer lower-coverage operating points.",
        "- Organizations that treat manual review as more expensive will generally prefer higher-coverage operating points.",
        "",
        "### 8.7.4 Uncertainty in Cost Assumptions",
        "- These scenarios are illustrative sensitivity checks, not claims that one cost ratio is universally correct.",
        "- In practice, organizations should define local costs with stakeholders and rerun the same analysis using their own assumptions.",
    ]

    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    output_path.write_text(existing.rstrip() + "\n" + "\n".join(section_lines) + "\n", encoding="utf-8")


def write_confidence_variant_section(
    variant_results: pd.DataFrame,
    output_path: Path,
) -> None:
    threshold_stability = variant_results.attrs.get("threshold_stability")
    best_by_variant = (
        variant_results.sort_values(["variant", "ece", "brier_score", "nll"])
        .groupby("variant", as_index=False)
        .first()[["variant", "method", "ece"]]
    )
    best_summary = ", ".join(
        f"{row['variant']} -> {row['method']} (ECE={row['ece']:.4f})" for _, row in best_by_variant.iterrows()
    )
    results_table = variant_results.rename(
        columns={
            "variant": "Confidence Source",
            "method": "Method",
            "accuracy": "Accuracy",
            "ece": "ECE",
            "brier_score": "Brier",
            "nll": "NLL",
        }
    )[
        ["Confidence Source", "Method", "Accuracy", "ECE", "Brier", "NLL"]
    ]
    lines = [
        "",
        "## 8.6 Sensitivity Analysis: Confidence Source Variation",
        "",
        "### 8.6.1 Confidence Source Definitions",
        "- `simulated`: the existing synthetic confidence generated in the baseline pipeline.",
        "- `consistency`: agreement across three deterministic perturbed logit samples.",
        "- `logit_margin`: sigmoid-normalized margin between the top two logits.",
        "",
        "### 8.6.2 Experiment Procedure",
        "- The same train/calibration/test split was reused for all confidence sources.",
        "- Each non-raw calibration method was re-fit on the variant-specific calibration confidence before test evaluation.",
        "- This robustness check uses the synthetic logits already exposed by `SyntheticArtifacts`; no external model calls were introduced.",
        "",
        "### 8.6.3 Results",
        "Table 8.6.1: Calibration Results Across Confidence Sources",
        "```text",
        _table_text(results_table),
        "```",
        "",
        "Key Findings",
        f"- Best ECE by confidence source in this synthetic run: {best_summary}.",
        "- Accuracy remained stable because the underlying class predictions were held fixed while only the confidence source changed.",
        "- Consistency and logit-margin confidence provide plausible alternative uncertainty proxies without changing the core adjudication simulation.",
    ]
    if threshold_stability is not None and not threshold_stability.empty:
        stability_table = threshold_stability.rename(
            columns={
                "variant": "Confidence Source",
                "threshold": "Threshold",
                "coverage": "Coverage",
                "accepted_accuracy": "Accepted Accuracy",
                "expected_loss": "Expected Loss",
            }
        )[
            ["Confidence Source", "Threshold", "Coverage", "Accepted Accuracy", "Expected Loss"]
        ]
        lines.extend(
            [
                "",
                "### 8.6.4 Threshold Stability",
                "Table 8.6.2: Threshold Stability Across Confidence Sources",
                "```text",
                _table_text(stability_table),
                "```",
            ]
        )

    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    output_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


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
    variant_results: pd.DataFrame | None = None,
    sensitivity_results: pd.DataFrame | None = None,
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
    if variant_results is not None and not variant_results.empty:
        write_confidence_variant_section(variant_results, Path(output_path))
    if sensitivity_results is not None and not sensitivity_results.empty:
        write_cost_sensitivity_section(sensitivity_results, Path(output_path))
