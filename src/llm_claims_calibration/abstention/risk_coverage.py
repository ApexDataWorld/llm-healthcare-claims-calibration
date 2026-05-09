from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        accepted = frame[frame["calibrated_confidence"] >= threshold]
        coverage = len(accepted) / len(frame) if len(frame) else 0.0
        selective_accuracy = accepted["correct"].mean() if len(accepted) else np.nan
        selective_risk = 1.0 - selective_accuracy if len(accepted) else np.nan
        rows.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "selective_accuracy": selective_accuracy,
                "selective_risk": selective_risk,
                "accepted_cases": int(len(accepted)),
            }
        )
    return pd.DataFrame(rows)


def apply_thresholds_by_risk_tier(frame: pd.DataFrame, risk_thresholds: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    threshold_values = output["risk_tier"].map(risk_thresholds)
    output["threshold"] = threshold_values
    output["accept_model"] = output["calibrated_confidence"] >= threshold_values
    output["escalate_reason"] = np.where(
        output["accept_model"],
        "accepted",
        np.where(output["risk_tier"] == "mandatory_review", "mandatory_review", "confidence_below_threshold"),
    )
    return output
