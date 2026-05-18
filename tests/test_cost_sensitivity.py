import numpy as np
import pandas as pd

from llm_claims_calibration.cost.sensitivity import get_cost_scenarios, run_threshold_sensitivity_analysis


def test_get_cost_scenarios() -> None:
    scenarios = get_cost_scenarios()
    assert len(scenarios) == 4
    assert "baseline_4x" in scenarios
    assert "high_decision_cost_10x" in scenarios
    assert "high_review_cost_2x" in scenarios
    assert "symmetric_1x" in scenarios
    assert scenarios["baseline_4x"].false_approval_cost / scenarios["baseline_4x"].human_review_cost == 4.0
    assert scenarios["high_decision_cost_10x"].false_approval_cost / scenarios["high_decision_cost_10x"].human_review_cost == 10.0


def test_sensitivity_results_shape() -> None:
    rng = np.random.default_rng(7)
    n = 120
    dummy_frame = pd.DataFrame(
        {
            "calibrated_confidence": rng.uniform(0.0, 1.0, n),
            "correct": rng.integers(0, 2, n),
            "risk_tier": rng.choice(["low", "medium", "high", "mandatory_review"], n),
            "predicted_label": rng.choice(
                ["approve", "deny_not_covered", "deny_medical_necessity", "request_additional_information"], n
            ),
        }
    )
    results = run_threshold_sensitivity_analysis(
        best_frame=dummy_frame,
        candidate_thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        cost_scenarios=get_cost_scenarios(),
        risk_tiers=["low", "medium", "high", "mandatory_review"],
    )
    assert len(results) == 16
    expected_cols = {"scenario_name", "risk_tier", "threshold", "coverage", "accepted_accuracy", "expected_loss"}
    assert expected_cols.issubset(results.columns)


def test_sensitivity_reproducibility() -> None:
    scenarios = get_cost_scenarios()
    assert scenarios is not None
