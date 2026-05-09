from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llm_claims_calibration.evaluation.metrics import brier_score, expected_calibration_error, negative_log_likelihood


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    bootstrap_seed: int
    sample_size: int


def _bootstrap_metric(
    confidence: np.ndarray,
    correct: np.ndarray,
    metric_name: str,
    metric_fn,
    n_bootstrap: int,
    random_seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(random_seed)
    sample_size = len(confidence)
    point_estimate = float(metric_fn(confidence, correct))
    draws = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, sample_size, size=sample_size)
        draws[idx] = metric_fn(confidence[sample_idx], correct[sample_idx])

    ci_lower, ci_upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapResult(
        metric=metric_name,
        point_estimate=point_estimate,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=random_seed,
        sample_size=sample_size,
    )


def bootstrap_calibration_metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)

    results = [
        _bootstrap_metric(
            confidence=confidence,
            correct=correct,
            metric_name="ece",
            metric_fn=lambda conf, corr: expected_calibration_error(conf, corr, n_bins=n_bins),
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
        _bootstrap_metric(
            confidence=confidence,
            correct=correct,
            metric_name="brier_score",
            metric_fn=brier_score,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
        _bootstrap_metric(
            confidence=confidence,
            correct=correct,
            metric_name="nll",
            metric_fn=negative_log_likelihood,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
    ]
    return pd.DataFrame([result.__dict__ for result in results])
