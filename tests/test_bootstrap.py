import numpy as np

from llm_claims_calibration.evaluation.bootstrap import bootstrap_calibration_metrics


def test_bootstrap_metrics_returns_expected_rows() -> None:
    confidence = np.array([0.95, 0.80, 0.70, 0.40, 0.20, 0.10])
    correct = np.array([1, 1, 1, 0, 0, 0])
    results = bootstrap_calibration_metrics(
        confidence=confidence,
        correct=correct,
        n_bins=3,
        n_bootstrap=100,
        random_seed=42,
    )
    assert set(results["metric"]) == {"ece", "brier_score", "nll"}
    assert (results["ci_lower"] <= results["point_estimate"]).all()
    assert (results["point_estimate"] <= results["ci_upper"]).all()
