# Synthetic Claims Data Dictionary

This repository uses synthetic claims data only. No PHI, member identifiers, or real claim content are included.

| Field | Description |
|---|---|
| `claim_id` | Synthetic unique identifier |
| `claim_type` | Broad workflow category such as outpatient, inpatient, pharmacy, imaging, or oncology |
| `procedure_code_group` | Simplified synthetic procedure grouping |
| `diagnosis_group` | Simplified synthetic diagnosis grouping |
| `plan_type` | Synthetic payer/product category |
| `place_of_service` | Office, facility, emergency, or telehealth |
| `claim_amount` | Synthetic numeric claim amount |
| `claim_amount_band` | Low, medium, or high amount bucket |
| `prior_auth_required` | Whether prior authorization is needed |
| `prior_auth_present` | Whether prior authorization is available |
| `appeal_sensitive` | Whether the case is modeled as appeal-sensitive |
| `policy_conflict_flag` | Synthetic proxy for policy/document conflict |
| `documentation_completeness` | Synthetic completeness score on `[0, 1]` |
| `policy_text` | Synthetic policy excerpt |
| `documentation_summary` | Synthetic documentation summary |
| `true_label` | Ground-truth support label |
| `risk_tier` | Low, medium, high, or mandatory review |
| `predicted_label_raw` | Raw simulated model prediction |
| `raw_confidence` | Raw simulated model confidence |
| `raw_correct` | Whether the raw simulated prediction matches the synthetic ground truth |
