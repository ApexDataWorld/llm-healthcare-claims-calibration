from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llm_claims_calibration.bayesian.expected_loss import optimize_thresholds


@dataclass(frozen=True)
class CostScenario:
    """Definition of a cost scenario for sensitivity analysis."""

    name: str
    description: str
    false_approval_cost: float
    false_denial_cost: float
    false_request_info_cost: float
    human_review_cost: float
    high_risk_delay_cost: float
    cost_ratio: str


def get_cost_scenarios() -> dict[str, CostScenario]:
    return {
        "baseline_4x": CostScenario(
            name="baseline_4x",
            description="Wrong decision is 4x more costly than human review.",
            false_approval_cost=4.0,
            false_denial_cost=4.0,
            false_request_info_cost=2.0,
            human_review_cost=1.0,
            high_risk_delay_cost=0.5,
            cost_ratio="4:1",
        ),
        "high_decision_cost_10x": CostScenario(
            name="high_decision_cost_10x",
            description="Wrong decision is 10x more costly than human review.",
            false_approval_cost=10.0,
            false_denial_cost=10.0,
            false_request_info_cost=5.0,
            human_review_cost=1.0,
            high_risk_delay_cost=1.25,
            cost_ratio="10:1",
        ),
        "high_review_cost_2x": CostScenario(
            name="high_review_cost_2x",
            description="Human review is 2x more costly than a wrong decision.",
            false_approval_cost=1.0,
            false_denial_cost=1.0,
            false_request_info_cost=0.5,
            human_review_cost=2.0,
            high_risk_delay_cost=0.25,
            cost_ratio="1:2",
        ),
        "symmetric_1x": CostScenario(
            name="symmetric_1x",
            description="Wrong decision and human review cost the same.",
            false_approval_cost=1.0,
            false_denial_cost=1.0,
            false_request_info_cost=0.5,
            human_review_cost=1.0,
            high_risk_delay_cost=0.25,
            cost_ratio="1:1",
        ),
    }


def run_threshold_sensitivity_analysis(
    best_frame: pd.DataFrame,
    candidate_thresholds: list[float],
    cost_scenarios: dict[str, CostScenario],
    risk_tiers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for scenario_name, scenario in cost_scenarios.items():
        threshold_results = optimize_thresholds(
            best_frame,
            candidate_thresholds=candidate_thresholds,
            costs={
                "false_approval_cost": scenario.false_approval_cost,
                "false_denial_cost": scenario.false_denial_cost,
                "false_request_info_cost": scenario.false_request_info_cost,
                "human_review_cost": scenario.human_review_cost,
                "high_risk_delay_cost": scenario.high_risk_delay_cost,
            },
            risk_tiers=risk_tiers,
        )

        for _, result_row in threshold_results.iterrows():
            risk_tier = str(result_row["risk_tier"])
            subset = best_frame[best_frame["risk_tier"] == risk_tier].copy()
            threshold = float(result_row["threshold"])
            accepted = subset[subset["calibrated_confidence"] >= threshold]
            coverage = float(len(accepted) / len(subset)) if len(subset) else 0.0
            accepted_accuracy = float(accepted["correct"].mean()) if len(accepted) else 0.0

            rows.append(
                {
                    "scenario_name": scenario_name,
                    "risk_tier": risk_tier,
                    "threshold": threshold,
                    "coverage": coverage,
                    "accepted_accuracy": accepted_accuracy,
                    "expected_loss": float(result_row["expected_loss"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["scenario_name", "risk_tier"]).reset_index(drop=True)


def compute_threshold_range_by_scenario(
    sensitivity_results: pd.DataFrame,
    risk_tier: str,
) -> dict[str, tuple[float, float]]:
    filtered = sensitivity_results[sensitivity_results["risk_tier"] == risk_tier]
    result: dict[str, tuple[float, float]] = {}
    for scenario_name, group in filtered.groupby("scenario_name"):
        result[str(scenario_name)] = (float(group["threshold"].min()), float(group["threshold"].max()))
    return result
