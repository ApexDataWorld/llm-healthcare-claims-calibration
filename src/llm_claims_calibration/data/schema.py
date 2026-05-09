from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


DecisionLabel = Literal[
    "approve",
    "deny_not_covered",
    "deny_medical_necessity",
    "request_additional_information",
    "escalate_to_human_review",
]

RiskTier = Literal["low", "medium", "high", "mandatory_review"]


class ClaimRecord(BaseModel):
    claim_id: str
    claim_type: str
    procedure_code_group: str
    diagnosis_group: str
    plan_type: str
    place_of_service: str
    claim_amount: float
    claim_amount_band: str
    prior_auth_required: bool
    prior_auth_present: bool
    appeal_sensitive: bool
    policy_conflict_flag: bool
    documentation_completeness: float
    policy_text: str
    documentation_summary: str
    true_label: DecisionLabel
    risk_tier: RiskTier
