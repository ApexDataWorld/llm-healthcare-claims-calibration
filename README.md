# llm-healthcare-claims-calibration

This repository supports the paper:

"Calibration-Aware LLM Decision Support for Healthcare Claims:
Uncertainty Quantification, Abstention, and Human Escalation"

The codebase is intentionally MVP-sized and publication-oriented. It builds a reproducible synthetic experiment pipeline for:

- synthetic healthcare claims generation
- calibration metrics: ECE, MCE, Brier score, NLL
- post-hoc calibration: temperature scaling, Platt scaling, isotonic regression
- abstention analysis with a risk-coverage curve
- Bayesian beta-binomial uncertainty summaries
- expected-loss threshold optimization for human escalation

The public repository uses synthetic data only. No PHI or real healthcare claims data are included.

## Repo Layout

- `configs/`: experiment YAML files
- `data/`: synthetic outputs and data dictionary
- `scripts/run_experiment.py`: end-to-end experiment runner
- `src/llm_claims_calibration/`: reusable library code
- `tests/`: `pytest` coverage for metrics and pipeline smoke tests
- `paper/tables/`: generated paper tables
- `paper/figures/`: generated paper figures
- `reports/`: experiment summaries

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## Reproduce

Run the baseline experiment:

```bash
python3 scripts/run_experiment.py --config configs/synthetic_baseline.yaml
```

Expected artifacts:

- `paper/tables/calibration_results.csv`
- `paper/tables/threshold_results.csv`
- `paper/figures/reliability_diagram.png`
- `paper/figures/risk_coverage_curve.png`
- `results/bootstrap_results.csv`
- `reports/experiment_summary.md`
- `data/synthetic/claims_synthetic.csv`

## Test

```bash
python3 -m pytest
```

## Notes on Scope

- Predictions are simulated from synthetic claims features to keep the repository reproducible and safe for public release.
- Calibration is evaluated as a probabilistic forecast of prediction correctness, which aligns with the paper's safety and abstention framing.
- The threshold optimizer uses asymmetric costs so denials can be treated as more operationally expensive than routine low-risk approvals.
