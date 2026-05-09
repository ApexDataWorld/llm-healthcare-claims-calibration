from __future__ import annotations

import numpy as np


def _clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 1e-12, 1.0 - 1e-12)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    error = 0.0
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        error += mask.mean() * abs(bin_accuracy - bin_confidence)
    return float(error)


def maximum_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        max_error = max(max_error, abs(correct[mask].mean() - confidence[mask].mean()))
    return float(max_error)


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - correct) ** 2))


def negative_log_likelihood(confidence: np.ndarray, correct: np.ndarray) -> float:
    confidence = _clip_probabilities(np.asarray(confidence, dtype=float))
    correct = np.asarray(correct, dtype=float)
    return float(-np.mean(correct * np.log(confidence) + (1.0 - correct) * np.log(1.0 - confidence)))


def binary_forecast_metrics(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(correct)),
        "ece": expected_calibration_error(confidence, correct, n_bins=n_bins),
        "mce": maximum_calibration_error(confidence, correct, n_bins=n_bins),
        "brier_score": brier_score(confidence, correct),
        "nll": negative_log_likelihood(confidence, correct),
    }
