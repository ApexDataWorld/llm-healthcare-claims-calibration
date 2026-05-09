import pandas as pd

from llm_claims_calibration.bayesian.expected_loss import optimize_thresholds


def test_threshold_optimizer_returns_one_row_per_risk_tier() -> None:
    frame = pd.DataFrame(
        {
            "risk_tier": ["low", "low", "high", "high"],
            "predicted_label": ["approve", "approve", "deny_not_covered", "deny_medical_necessity"],
            "calibrated_confidence": [0.9, 0.6, 0.8, 0.4],
            "correct": [1, 0, 1, 0],
        }
    )
    result = optimize_thresholds(
        frame,
        candidate_thresholds=[0.5, 0.7, 0.9],
        costs={
            "false_approval_cost": 500.0,
            "false_denial_cost": 1500.0,
            "false_request_info_cost": 200.0,
            "human_review_cost": 75.0,
            "high_risk_delay_cost": 250.0,
        },
        risk_tiers=["low", "high"],
    )
    assert set(result["risk_tier"]) == {"low", "high"}
