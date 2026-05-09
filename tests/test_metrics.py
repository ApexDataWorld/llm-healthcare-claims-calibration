import numpy as np

from llm_claims_calibration.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
)


def test_metrics_are_well_behaved() -> None:
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    assert expected_calibration_error(confidence, correct, n_bins=2) >= 0.0
    assert maximum_calibration_error(confidence, correct, n_bins=2) >= 0.0
    assert brier_score(confidence, correct) < 0.1
    assert negative_log_likelihood(confidence, correct) > 0.0
