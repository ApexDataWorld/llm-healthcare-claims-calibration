from __future__ import annotations

import pandas as pd


def claim_error_cost(predicted_label: str, false_approval_cost: float, false_denial_cost: float, false_request_info_cost: float) -> float:
    if predicted_label == "approve":
        return false_approval_cost
    if predicted_label.startswith("deny"):
        return false_denial_cost
    return false_request_info_cost


def expected_loss_accept(prob_correct: float, error_cost: float) -> float:
    return (1.0 - prob_correct) * error_cost


def expected_loss_escalate(human_review_cost: float, delay_cost: float = 0.0) -> float:
    return human_review_cost + delay_cost


def optimize_thresholds(frame: pd.DataFrame, candidate_thresholds: list[float], costs: dict[str, float], risk_tiers: list[str]) -> pd.DataFrame:
    rows = []
    for risk_tier in risk_tiers:
        subset = frame[frame["risk_tier"] == risk_tier].copy()
        if subset.empty:
            continue

        subset["error_cost"] = subset["predicted_label"].map(
            lambda label: claim_error_cost(
                label,
                false_approval_cost=costs["false_approval_cost"],
                false_denial_cost=costs["false_denial_cost"],
                false_request_info_cost=costs["false_request_info_cost"],
            )
        )
        delay_cost = costs["high_risk_delay_cost"] if risk_tier in {"high", "mandatory_review"} else 0.0

        best_row: dict[str, float | str] | None = None
        for threshold in candidate_thresholds:
            accept = subset["calibrated_confidence"] >= threshold
            accept_loss = expected_loss_accept(subset["calibrated_confidence"], subset["error_cost"]).where(accept, 0.0)
            escalate_loss = expected_loss_escalate(costs["human_review_cost"], delay_cost)
            escalation_component = (~accept).astype(float) * escalate_loss
            total_expected_loss = float((accept_loss + escalation_component).mean())
            accepted_accuracy = float(subset.loc[accept, "correct"].mean()) if accept.any() else 0.0
            row = {
                "risk_tier": risk_tier,
                "threshold": threshold,
                "coverage": float(accept.mean()),
                "accepted_accuracy": accepted_accuracy,
                "expected_loss": total_expected_loss,
            }
            if best_row is None or total_expected_loss < best_row["expected_loss"]:
                best_row = row

        assert best_row is not None
        rows.append(best_row)
    return pd.DataFrame(rows).sort_values("risk_tier").reset_index(drop=True)
