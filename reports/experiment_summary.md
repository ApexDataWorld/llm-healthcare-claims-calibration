# Calibration-Aware LLM Decision Support for Healthcare Claims

## Summary
- Synthetic claims generated: 1200
- Best calibration method by ECE: `temperature_scaling`
- Best observed test-set ECE: 0.0300
- Best observed test-set Brier score: 0.0597

## Data Split
- Development/train size: 600
- Calibration size: 300
- Evaluation/test size: 300

## Reproducibility
- Random seed: 20260503
- Python version: 3.13.3
- `numpy`: 2.4.4
- `pandas`: 3.0.2
- `scipy`: 1.17.1
- `scikit-learn`: 1.8.0
- `matplotlib`: 3.10.9
- `pydantic`: 2.13.3
- `PyYAML`: 6.0.3
- Reproduction command: `python3 scripts/run_experiment.py --config configs/synthetic_baseline.yaml`

## Calibration Comparison
```text
             method  accuracy      ece      mce  brier_score      nll
temperature_scaling      0.93 0.030002 0.265525     0.059661 0.197270
      platt_scaling      0.93 0.068983 0.438580     0.075273 0.238935
isotonic_regression      0.93 0.085766 0.609059     0.083572 0.524106
                raw      0.93 0.154557 0.374298     0.089112 0.298552
```

## Bootstrap Confidence Intervals
```text
             method      metric  point_estimate  ci_lower  ci_upper  n_bootstrap  bootstrap_seed  sample_size
                raw         ece        0.154557  0.126316  0.183681         1000        20260503          300
                raw brier_score        0.089112  0.074864  0.103415         1000        20260503          300
                raw         nll        0.298552  0.263617  0.333294         1000        20260503          300
temperature_scaling         ece        0.030002  0.023465  0.063626         1000        20260503          300
temperature_scaling brier_score        0.059661  0.042437  0.077442         1000        20260503          300
temperature_scaling         nll        0.197270  0.146968  0.253163         1000        20260503          300
      platt_scaling         ece        0.068983  0.051705  0.102152         1000        20260503          300
      platt_scaling brier_score        0.075273  0.054742  0.098113         1000        20260503          300
      platt_scaling         nll        0.238935  0.175747  0.305653         1000        20260503          300
isotonic_regression         ece        0.085766  0.056487  0.116769         1000        20260503          300
isotonic_regression brier_score        0.083572  0.058885  0.110111         1000        20260503          300
isotonic_regression         nll        0.524106  0.296661  0.773120         1000        20260503          300
```

## Expected-Loss Thresholds
```text
       risk_tier  threshold  coverage  accepted_accuracy  expected_loss
            high       0.50  0.960784           0.979592      64.003430
             low       0.80  0.692308           0.972222      33.640725
mandatory_review       0.50  0.966667           0.879310      30.851561
          medium       0.85  0.802920           0.981818      23.843227
```

## Bayesian Prior
- alpha0: 1.0
- beta0: 1.0

## Bayesian Posterior by Risk Tier
```text
       risk_tier  cases  posterior_mean  credible_interval_lower  credible_interval_upper  probability_correct_gt_0_80
            high     51        0.943396                 0.867872                 0.987941                     0.999115
             low     52        0.925926                 0.843371                 0.979054                     0.996592
mandatory_review     60        0.838710                 0.738312                 0.918483                     0.804070
          medium    137        0.942446                 0.898281                 0.974643                     0.999999
```

## Reproduction
- `python3 scripts/run_experiment.py --config configs/synthetic_baseline.yaml`

## 8.6 Sensitivity Analysis: Confidence Source Variation

### 8.6.1 Confidence Source Definitions
- `simulated`: the existing synthetic confidence generated in the baseline pipeline.
- `consistency`: agreement across three deterministic perturbed logit samples.
- `logit_margin`: sigmoid-normalized margin between the top two logits.

### 8.6.2 Experiment Procedure
- The same train/calibration/test split was reused for all confidence sources.
- Each non-raw calibration method was re-fit on the variant-specific calibration confidence before test evaluation.
- This robustness check uses the synthetic logits already exposed by `SyntheticArtifacts`; no external model calls were introduced.

### 8.6.3 Results
Table 8.6.1: Calibration Results Across Confidence Sources
```text
Confidence Source              Method  Accuracy      ECE    Brier      NLL
      consistency temperature_scaling      0.93 0.050656 0.064347 0.251820
      consistency       platt_scaling      0.93 0.062441 0.071601 0.277791
      consistency isotonic_regression      0.93 0.062998 0.074515 0.463608
      consistency                 raw      0.93 0.067777 0.068518 0.813920
     logit_margin temperature_scaling      0.93 0.018615 0.056450 0.189827
     logit_margin isotonic_regression      0.93 0.047661 0.072071 0.229776
     logit_margin       platt_scaling      0.93 0.048141 0.065781 0.209820
     logit_margin                 raw      0.93 0.082257 0.064473 0.229604
        simulated       platt_scaling      0.93 0.068983 0.075273 0.238935
        simulated temperature_scaling      0.93 0.077528 0.083543 0.262560
        simulated isotonic_regression      0.93 0.085766 0.083572 0.524106
        simulated                 raw      0.93 0.154557 0.089112 0.298552
```

Key Findings
- Best ECE by confidence source in this synthetic run: consistency -> temperature_scaling (ECE=0.0507), logit_margin -> temperature_scaling (ECE=0.0186), simulated -> platt_scaling (ECE=0.0690).
- Accuracy remained stable because the underlying class predictions were held fixed while only the confidence source changed.
- Consistency and logit-margin confidence provide plausible alternative uncertainty proxies without changing the core adjudication simulation.

### 8.6.4 Threshold Stability
Table 8.6.2: Threshold Stability Across Confidence Sources
```text
Confidence Source  Threshold  Coverage  Accepted Accuracy  Expected Loss
        simulated       0.85  0.795620           0.972477      22.779394
      consistency       0.60  0.948905           0.961538      52.757761
     logit_margin       0.85  0.802920           0.981818      24.291812
```

## 8.7 Sensitivity Analysis: Cost Ratio Variation

### 8.7.1 Cost Scenarios
- `baseline_4x`: wrong decisions are modeled as four times the cost of human review.
- `high_decision_cost_10x`: wrong decisions are modeled as ten times the cost of human review.
- `high_review_cost_2x`: human review is modeled as twice the cost of a wrong decision.
- `symmetric_1x`: wrong decisions and human review have equal cost.

### 8.7.2 Results
Table 8.7.1: Optimal Thresholds Across Cost Scenarios
```text
              Scenario Cost Ratio  High Tier  Low Tier  Mandatory Review  Medium Tier
           baseline_4x        4:1        0.6       0.7               0.5          0.7
high_decision_cost_10x       10:1        0.7       0.9               0.6          0.9
   high_review_cost_2x        1:2        0.5       0.5               0.5          0.5
          symmetric_1x        1:1        0.5       0.5               0.5          0.5
```

Table 8.7.2: Coverage and Accepted Accuracy by Scenario (Medium Tier)
```text
              Scenario Cost Ratio  Threshold  Coverage  Accepted Accuracy  Expected Loss
           baseline_4x        4:1        0.7  0.927007           0.960630       0.252346
high_decision_cost_10x       10:1        0.9  0.751825           0.980583       0.363674
   high_review_cost_2x        1:2        0.5  0.963504           0.962121       0.133599
          symmetric_1x        1:1        0.5  0.963504           0.962121       0.097103
```

### 8.7.3 Implications for Organizations
- The framework recomputes thresholds directly from the selected cost ratio rather than assuming one universal operating point.
- Organizations that treat wrong automated decisions as more expensive will generally prefer lower-coverage operating points.
- Organizations that treat manual review as more expensive will generally prefer higher-coverage operating points.

### 8.7.4 Uncertainty in Cost Assumptions
- These scenarios are illustrative sensitivity checks, not claims that one cost ratio is universally correct.
- In practice, organizations should define local costs with stakeholders and rerun the same analysis using their own assumptions.
