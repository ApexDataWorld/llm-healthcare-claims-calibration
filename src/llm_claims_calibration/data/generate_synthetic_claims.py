from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABELS = [
    "approve",
    "deny_not_covered",
    "deny_medical_necessity",
    "request_additional_information",
    "escalate_to_human_review",
]


@dataclass(frozen=True)
class SyntheticArtifacts:
    claims: pd.DataFrame
    logits: np.ndarray
    label_to_index: dict[str, int]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _amount_band(amount: float) -> str:
    if amount < 250:
        return "low"
    if amount < 1200:
        return "medium"
    return "high"


def _risk_tier(amount_band: str, claim_type: str, true_label: str, appeal_sensitive: bool) -> str:
    if appeal_sensitive or claim_type in {"oncology", "inpatient"}:
        return "mandatory_review"
    if amount_band == "high" or true_label.startswith("deny"):
        return "high"
    if amount_band == "medium" or claim_type in {"imaging", "pharmacy"}:
        return "medium"
    return "low"


def generate_synthetic_claims(n_claims: int, seed: int) -> SyntheticArtifacts:
    rng = np.random.default_rng(seed)

    claim_types = np.array(["outpatient", "inpatient", "pharmacy", "dme", "behavioral_health", "imaging", "oncology"])
    procedure_groups = np.array(["office_visit", "advanced_imaging", "lab", "therapy", "specialty_drug", "surgery", "dme_supply"])
    diagnosis_groups = np.array(["musculoskeletal", "cardiac", "endocrine", "behavioral", "oncology", "respiratory"])
    plan_types = np.array(["commercial", "medicare_advantage", "medicaid", "ppo", "hmo"])
    service_sites = np.array(["office", "facility", "emergency", "telehealth"])

    claim_type = rng.choice(claim_types, size=n_claims, p=[0.26, 0.08, 0.16, 0.10, 0.12, 0.18, 0.10])
    procedure_code_group = rng.choice(procedure_groups, size=n_claims)
    diagnosis_group = rng.choice(diagnosis_groups, size=n_claims)
    plan_type = rng.choice(plan_types, size=n_claims)
    place_of_service = rng.choice(service_sites, size=n_claims, p=[0.45, 0.24, 0.12, 0.19])
    base_amount = rng.lognormal(mean=5.8, sigma=0.8, size=n_claims)
    amount_multiplier = np.select(
        [claim_type == "inpatient", claim_type == "oncology", claim_type == "imaging", claim_type == "pharmacy"],
        [8.0, 6.0, 3.5, 2.3],
        default=1.0,
    )
    claim_amount = np.round(base_amount * amount_multiplier, 2)
    claim_amount_band = np.array([_amount_band(value) for value in claim_amount])
    prior_auth_required = rng.random(n_claims) < np.select(
        [claim_type == "imaging", claim_type == "oncology", claim_type == "inpatient"],
        [0.65, 0.72, 0.58],
        default=0.22,
    )
    prior_auth_present = np.where(prior_auth_required, rng.random(n_claims) < 0.84, False)
    appeal_sensitive = (claim_type == "oncology") | ((claim_type == "behavioral_health") & (rng.random(n_claims) < 0.35))
    policy_conflict_flag = rng.random(n_claims) < np.select(
        [procedure_code_group == "specialty_drug", procedure_code_group == "advanced_imaging"],
        [0.22, 0.16],
        default=0.07,
    )
    documentation_completeness = np.clip(rng.normal(loc=0.78, scale=0.18, size=n_claims), 0.05, 1.0)

    true_labels: list[str] = []
    for idx in range(n_claims):
        if prior_auth_required[idx] and not prior_auth_present[idx]:
            label = "deny_not_covered"
        elif policy_conflict_flag[idx] and claim_type[idx] in {"oncology", "imaging"}:
            label = "deny_medical_necessity"
        elif documentation_completeness[idx] < 0.35:
            label = "request_additional_information"
        elif appeal_sensitive[idx] and documentation_completeness[idx] < 0.55:
            label = "escalate_to_human_review"
        elif claim_type[idx] == "inpatient" and claim_amount_band[idx] == "high" and documentation_completeness[idx] < 0.60:
            label = "escalate_to_human_review"
        else:
            label = "approve"
        true_labels.append(label)

    risk_tier = np.array(
        [_risk_tier(claim_amount_band[idx], claim_type[idx], true_labels[idx], bool(appeal_sensitive[idx])) for idx in range(n_claims)]
    )
    claim_ids = [f"CLM_SYN_{idx:06d}" for idx in range(1, n_claims + 1)]

    policy_text = [
        f"Synthetic policy excerpt for {ct} claims under {pt}; prior authorization required={par}; amount band={ab}."
        for ct, pt, par, ab in zip(claim_type, plan_type, prior_auth_required, claim_amount_band)
    ]
    documentation_summary = [
        f"Synthetic documentation summary for {dg} case with completeness {dc:.2f} and site {pos}."
        for dg, dc, pos in zip(diagnosis_group, documentation_completeness, place_of_service)
    ]

    claims = pd.DataFrame(
        {
            "claim_id": claim_ids,
            "claim_type": claim_type,
            "procedure_code_group": procedure_code_group,
            "diagnosis_group": diagnosis_group,
            "plan_type": plan_type,
            "place_of_service": place_of_service,
            "claim_amount": claim_amount,
            "claim_amount_band": claim_amount_band,
            "prior_auth_required": prior_auth_required,
            "prior_auth_present": prior_auth_present,
            "appeal_sensitive": appeal_sensitive,
            "policy_conflict_flag": policy_conflict_flag,
            "documentation_completeness": documentation_completeness,
            "policy_text": policy_text,
            "documentation_summary": documentation_summary,
            "true_label": true_labels,
            "risk_tier": risk_tier,
        }
    )

    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    logits = rng.normal(loc=0.0, scale=1.0, size=(n_claims, len(LABELS)))

    for row_idx, label in enumerate(true_labels):
        truth_index = label_to_index[label]
        logits[row_idx, truth_index] += 2.5
        if claim_amount_band[row_idx] == "high":
            logits[row_idx] += np.array([0.0, 0.2, 0.25, -0.1, 0.3])
        if not prior_auth_present[row_idx] and prior_auth_required[row_idx]:
            logits[row_idx, label_to_index["deny_not_covered"]] += 1.2
        if documentation_completeness[row_idx] < 0.4:
            logits[row_idx, label_to_index["request_additional_information"]] += 0.8
        if policy_conflict_flag[row_idx]:
            logits[row_idx, label_to_index["deny_medical_necessity"]] += 0.7
        if appeal_sensitive[row_idx]:
            logits[row_idx, label_to_index["escalate_to_human_review"]] += 0.5

        # Intentionally miscalibrate confidence to make calibration meaningful.
        logits[row_idx] *= 1.45

    probs = _softmax(logits)
    predicted_index = probs.argmax(axis=1)
    true_index = claims["true_label"].map(label_to_index).to_numpy()
    raw_confidence = probs.max(axis=1)
    correct = (predicted_index == true_index).astype(int)
    claims["predicted_label_raw"] = [LABELS[index] for index in predicted_index]
    claims["raw_confidence"] = raw_confidence
    claims["raw_correct"] = correct

    return SyntheticArtifacts(claims=claims, logits=logits, label_to_index=label_to_index)


def split_dataset(frame: pd.DataFrame, train_fraction: float, calibration_fraction: float) -> dict[str, pd.DataFrame]:
    n_train = int(len(frame) * train_fraction)
    n_calibration = int(len(frame) * calibration_fraction)
    return {
        "train": frame.iloc[:n_train].reset_index(drop=True),
        "calibration": frame.iloc[n_train : n_train + n_calibration].reset_index(drop=True),
        "test": frame.iloc[n_train + n_calibration :].reset_index(drop=True),
    }
