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