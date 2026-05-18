from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
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
    sigmoid,
)
from llm_claims_calibration.confidence.variants import (
    generate_consistency_confidence,
    generate_logit_margin_confidence,
    get_simulated_confidence,
)
from llm_claims_calibration.config import load_config
from llm_claims_calibration.cost.sensitivity import get_cost_scenarios, run_threshold_sensitivity_analysis
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


def _logit_transform(confidence: np.ndarray) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=float)
    clipped = np.clip(confidence, 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _scalar_temperature_scale(
    calibration_confidence: np.ndarray,
    calibration_correct: np.ndarray,
    test_confidence: np.ndarray,
) -> np.ndarray:
    calibration_logit = _logit_transform(calibration_confidence)
    candidates = np.linspace(0.05, 10.0, 400)
    best_temperature = 1.0
    best_loss = None
    for candidate in candidates:
        calibrated = sigmoid(calibration_logit / candidate)
        calibrated = np.clip(calibrated, 1e-12, 1.0 - 1e-12)
        loss = float(
            -np.mean(
                calibration_correct * np.log(calibrated)
                + (1.0 - calibration_correct) * np.log(1.0 - calibrated)
            )
        )
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_temperature = float(candidate)
    return np.clip(sigmoid(_logit_transform(test_confidence) / best_temperature), 1e-6, 0.999999)


def run_confidence_variant_analysis(
    splits: dict[str, pd.DataFrame],
    artifacts,
    config,
    best_method: str,
    method_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    _ = best_method
    n_train = len(splits["train"])
    n_calibration = len(splits["calibration"])
    calibration_logits = artifacts.logits[n_train : n_train + n_calibration]
    test_logits = artifacts.logits[n_train + n_calibration :]
    calibration_correct = splits["calibration"]["raw_correct"].to_numpy(dtype=float)
    test_correct = splits["test"]["raw_correct"].to_numpy(dtype=float)
    risk_tiers = list(config.abstention.risk_tiers.keys())
    label_to_index = artifacts.label_to_index
    predicted_test_labels = splits["test"]["predicted_label_raw"].to_numpy()
    true_test_index = splits["test"]["true_label"].map(label_to_index).to_numpy()

    simulated = get_simulated_confidence(artifacts, splits)
    confidence_sources: dict[str, dict[str, np.ndarray]] = {
        "simulated": simulated,
        "consistency": {
            "calibration": generate_consistency_confidence(
                splits["calibration"],
                calibration_logits,
                k_samples=config.confidence_variants.consistency_k_samples,
                seed=config.random_seed,
                perturbation_scale=config.confidence_variants.consistency_perturbation_scale,
            ),
            "test": generate_consistency_confidence(
                splits["test"],
                test_logits,
                k_samples=config.confidence_variants.consistency_k_samples,
                seed=config.random_seed,
                perturbation_scale=config.confidence_variants.consistency_perturbation_scale,
            ),
        },
        "logit_margin": {
            "calibration": generate_logit_margin_confidence(calibration_logits),
            "test": generate_logit_margin_confidence(test_logits),
        },
    }

    selected_variants = []
    for variant in config.confidence_variants.variants:
        if variant == "logit_margin" and not config.confidence_variants.logit_margin_enabled:
            continue
        selected_variants.append(variant)

    rows = []
    threshold_rows = []
    costs = {
        "false_approval_cost": config.expected_loss.false_approval_cost,
        "false_denial_cost": config.expected_loss.false_denial_cost,
        "false_request_info_cost": config.expected_loss.false_request_info_cost,
        "human_review_cost": config.expected_loss.human_review_cost,
        "high_risk_delay_cost": config.expected_loss.high_risk_delay_cost,
    }

    for variant in selected_variants:
        calibration_confidence = confidence_sources[variant]["calibration"]
        test_confidence = confidence_sources[variant]["test"]
        variant_frames: dict[str, pd.DataFrame] = {}
        for method in config.calibration.methods:
            if method == "raw":
                calibrated_confidence = np.clip(test_confidence, 1e-6, 0.999999)
            elif method == "temperature_scaling":
                calibrated_confidence = _scalar_temperature_scale(
                    calibration_confidence=calibration_confidence,
                    calibration_correct=calibration_correct,
                    test_confidence=test_confidence,
                )
            else:
                calibrated_confidence = apply_correctness_calibrator(
                    method=method,
                    calibration_confidence=calibration_confidence,
                    calibration_correct=calibration_correct,
                    test_confidence=test_confidence,
                    predicted_labels=predicted_test_labels,
                ).calibrated_confidence

            frame = splits["test"].copy()
            frame["predicted_label"] = predicted_test_labels
            frame["raw_confidence"] = test_confidence
            frame["calibrated_confidence"] = calibrated_confidence
            frame["correct"] = (pd.Series(predicted_test_labels).map(label_to_index).to_numpy() == true_test_index).astype(int)
            variant_frames[method] = frame
            metrics = binary_forecast_metrics(
                frame["calibrated_confidence"].to_numpy(),
                test_correct,
                n_bins=config.calibration.bins,
            )
            rows.append({"method": method, "variant": variant, **metrics})

        variant_results = pd.DataFrame([row for row in rows if row["variant"] == variant]).sort_values(["ece", "brier_score", "nll"])
        selected_method = str(variant_results.iloc[0]["method"])
        thresholds = optimize_thresholds(
            variant_frames[selected_method],
            candidate_thresholds=config.abstention.thresholds,
            costs=costs,
            risk_tiers=risk_tiers,
        )
        medium = thresholds[thresholds["risk_tier"] == "medium"]
        if not medium.empty:
            medium_row = medium.iloc[0]
            threshold_rows.append(
                {
                    "variant": variant,
                    "method": selected_method,
                    "threshold": float(medium_row["threshold"]),
                    "coverage": float(medium_row["coverage"]),
                    "accepted_accuracy": float(medium_row["accepted_accuracy"]),
                    "expected_loss": float(medium_row["expected_loss"]),
                }
            )

    results = pd.DataFrame(rows).sort_values(["variant", "ece", "brier_score", "nll"]).reset_index(drop=True)
    results.attrs["threshold_stability"] = pd.DataFrame(threshold_rows)
    return results


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
    variant_results = None
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
    sensitivity_results = None

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
    results_dir = Path(config.outputs.results_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if config.confidence_variants and config.confidence_variants.enabled:
        variant_results = run_confidence_variant_analysis(
            splits=splits,
            artifacts=artifacts,
            config=config,
            best_method=best_method,
            method_frames=method_frames,
        )
        variant_results.to_csv(results_dir / "confidence_variant_results.csv", index=False)
    if config.cost_sensitivity and config.cost_sensitivity.enabled:
        cost_scenarios = get_cost_scenarios()
        if config.cost_sensitivity.scenarios:
            cost_scenarios = {name: cost_scenarios[name] for name in config.cost_sensitivity.scenarios}
        sensitivity_results = run_threshold_sensitivity_analysis(
            best_frame=best_frame,
            candidate_thresholds=config.abstention.thresholds,
            cost_scenarios=cost_scenarios,
            risk_tiers=list(config.abstention.risk_tiers.keys()),
        )
        sensitivity_results.to_csv(results_dir / "cost_sensitivity_results.csv", index=False)
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
        variant_results=variant_results,
        sensitivity_results=sensitivity_results,
    )


if __name__ == "__main__":
    main()
