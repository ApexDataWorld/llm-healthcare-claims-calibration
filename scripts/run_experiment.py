from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_claims_calibration.abstention.risk_coverage import apply_thresholds_by_risk_tier, risk_coverage_curve
from llm_claims_calibration.bayesian.beta_binomial import fit_beta_binomial
from llm_claims_calibration.bayesian.expected_loss import optimize_thresholds
from llm_claims_calibration.calibration.methods import (
    CalibrationOutputs,
    apply_correctness_calibrator,
    apply_temperature_scaling,
)
from llm_claims_calibration.config import load_config
from llm_claims_calibration.data.generate_synthetic_claims import LABELS, generate_synthetic_claims, split_dataset
from llm_claims_calibration.evaluation.bootstrap import bootstrap_calibration_metrics
from llm_claims_calibration.evaluation.metrics import binary_forecast_metrics
from llm_claims_calibration.reporting.plots import save_reliability_diagram, save_risk_coverage_curve
from llm_claims_calibration.reporting.summary import write_summary_report

BOOTSTRAP_RESAMPLES = 1000
BAYESIAN_ALPHA0 = 1.0
BAYESIAN_BETA0 = 1.0


def build_method_frame(split_frame: pd.DataFrame, outputs: CalibrationOutputs, label_to_index: dict[str, int]) -> pd.DataFrame:
    frame = split_frame.copy()
    frame["predicted_label"] = outputs.predicted_label
    frame["raw_confidence"] = outputs.raw_confidence
    frame["calibrated_confidence"] = outputs.calibrated_confidence
    true_index = frame["true_label"].map(label_to_index).to_numpy()
    predicted_index = pd.Series(outputs.predicted_label).map(label_to_index).to_numpy()
    frame["correct"] = (predicted_index == true_index).astype(int)
    return frame


def choose_best_method(results: pd.DataFrame) -> str:
    ordered = results.sort_values(["ece", "brier_score", "nll"], ascending=True)
    return str(ordered.iloc[0]["method"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic calibration-aware claims experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = generate_synthetic_claims(n_claims=config.data.n_claims, seed=config.random_seed)
    dataset_path = Path(config.outputs.dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.claims.to_csv(dataset_path, index=False)

    splits = split_dataset(
        artifacts.claims,
        train_fraction=config.data.train_fraction,
        calibration_fraction=config.data.calibration_fraction,
    )
    n_train = len(splits["train"])
    n_calibration = len(splits["calibration"])
    n_test = len(splits["test"])
    calibration_logits = artifacts.logits[n_train : n_train + n_calibration]
    test_logits = artifacts.logits[n_train + n_calibration :]

    label_to_index = artifacts.label_to_index
    calibration_true_index = splits["calibration"]["true_label"].map(label_to_index).to_numpy()
    calibration_correct = splits["calibration"]["raw_correct"].to_numpy()
    calibration_confidence = splits["calibration"]["raw_confidence"].to_numpy()
    test_confidence = splits["test"]["raw_confidence"].to_numpy()
    raw_predicted_labels = splits["test"]["predicted_label_raw"].to_numpy()

    outputs_by_method: dict[str, CalibrationOutputs] = {
        "raw": CalibrationOutputs(
            predicted_label=raw_predicted_labels,
            raw_confidence=test_confidence,
            calibrated_confidence=test_confidence,
        ),
        "temperature_scaling": apply_temperature_scaling(calibration_logits, calibration_true_index, test_logits, LABELS),
        "platt_scaling": apply_correctness_calibrator(
            "platt_scaling",
            calibration_confidence,
            calibration_correct,
            test_confidence,
            raw_predicted_labels,
        ),
        "isotonic_regression": apply_correctness_calibrator(
            "isotonic_regression",
            calibration_confidence,
            calibration_correct,
            test_confidence,
            raw_predicted_labels,
        ),
    }

    calibration_rows = []
    method_frames: dict[str, pd.DataFrame] = {}
    for method in config.calibration.methods:
        frame = build_method_frame(splits["test"], outputs_by_method[method], label_to_index)
        method_frames[method] = frame
        metrics = binary_forecast_metrics(
            frame["calibrated_confidence"].to_numpy(),
            frame["correct"].to_numpy(),
            n_bins=config.calibration.bins,
        )
        calibration_rows.append({"method": method, **metrics})

    calibration_results = pd.DataFrame(calibration_rows).sort_values("ece").reset_index(drop=True)
    best_method = choose_best_method(calibration_results)
    best_frame = method_frames[best_method]
    bootstrap_frames = []
    for method in config.calibration.methods:
        method_bootstrap = bootstrap_calibration_metrics(
            confidence=method_frames[method]["calibrated_confidence"].to_numpy(),
            correct=method_frames[method]["correct"].to_numpy(),
            n_bins=config.calibration.bins,
            n_bootstrap=BOOTSTRAP_RESAMPLES,
            random_seed=config.random_seed,
        )
        method_bootstrap.insert(0, "method", method)
        bootstrap_frames.append(method_bootstrap)
    bootstrap_results = pd.concat(bootstrap_frames, ignore_index=True)

    curve = risk_coverage_curve(best_frame, thresholds=config.abstention.thresholds)

    costs = {
        "false_approval_cost": config.expected_loss.false_approval_cost,
        "false_denial_cost": config.expected_loss.false_denial_cost,
        "false_request_info_cost": config.expected_loss.false_request_info_cost,
        "human_review_cost": config.expected_loss.human_review_cost,
        "high_risk_delay_cost": config.expected_loss.high_risk_delay_cost,
    }

    threshold_results = optimize_thresholds(
        best_frame,
        candidate_thresholds=config.abstention.thresholds,
        costs=costs,
        risk_tiers=list(config.abstention.risk_tiers.keys()),
    )

    tuned_thresholds = dict(zip(threshold_results["risk_tier"], threshold_results["threshold"]))
    for tier, default_threshold in config.abstention.risk_tiers.items():
        tuned_thresholds.setdefault(tier, default_threshold)

    thresholded = apply_thresholds_by_risk_tier(best_frame, tuned_thresholds)
    posterior_rows = []
    for risk_tier, group in thresholded.groupby("risk_tier"):
        posterior = fit_beta_binomial(
            int(group["correct"].sum()),
            int(len(group)),
            alpha_prior=BAYESIAN_ALPHA0,
            beta_prior=BAYESIAN_BETA0,
        )
        lower, upper = posterior.credible_interval()
        posterior_rows.append(
            {
                "risk_tier": risk_tier,
                "cases": int(len(group)),
                "posterior_mean": posterior.mean,
                "credible_interval_lower": lower,
                "credible_interval_upper": upper,
                "probability_correct_gt_0_80": posterior.probability_above(0.80),
            }
        )
    posterior_summary = pd.DataFrame(posterior_rows).sort_values("risk_tier").reset_index(drop=True)

    tables_dir = Path(config.outputs.tables_dir)
    figures_dir = Path(config.outputs.figures_dir)
    results_dir = REPO_ROOT / "results"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    calibration_results.to_csv(tables_dir / "calibration_results.csv", index=False)
    threshold_results.to_csv(tables_dir / "threshold_results.csv", index=False)
    bootstrap_results.to_csv(results_dir / "bootstrap_results.csv", index=False)
    save_reliability_diagram(best_frame, str(figures_dir / "reliability_diagram.png"), config.calibration.bins)
    save_risk_coverage_curve(curve, str(figures_dir / "risk_coverage_curve.png"))
    reproduction_command = f"python3 scripts/run_experiment.py --config {args.config}"
    write_summary_report(
        output_path=config.outputs.report_path,
        config_title=config.title,
        dataset_size=len(artifacts.claims),
        selected_method=best_method,
        calibration_results=calibration_results,
        bootstrap_results=bootstrap_results,
        threshold_results=threshold_results,
        posterior_summary=posterior_summary,
        split_sizes={"train": n_train, "calibration": n_calibration, "test": n_test},
        random_seed=config.random_seed,
        bayesian_prior={"alpha0": BAYESIAN_ALPHA0, "beta0": BAYESIAN_BETA0},
        reproduction_command=reproduction_command,
    )


if __name__ == "__main__":
    main()
