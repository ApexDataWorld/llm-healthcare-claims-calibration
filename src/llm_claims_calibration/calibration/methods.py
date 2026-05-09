from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    minimize = None

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    IsotonicRegression = None


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class TemperatureScaling:
    def __init__(self) -> None:
        self.temperature_ = 1.0

    def fit(self, logits: np.ndarray, true_index: np.ndarray) -> "TemperatureScaling":
        logits = np.asarray(logits, dtype=float)
        true_index = np.asarray(true_index, dtype=int)

        def objective(temp_array: np.ndarray) -> float:
            temperature = float(np.clip(temp_array[0], 1e-3, None))
            probs = softmax(logits / temperature)
            selected = np.clip(probs[np.arange(len(true_index)), true_index], 1e-12, 1.0)
            return float(-np.mean(np.log(selected)))

        if minimize is not None:
            result = minimize(objective, x0=np.array([1.0]), bounds=[(0.05, 10.0)])
            self.temperature_ = float(result.x[0])
        else:
            candidates = np.linspace(0.05, 10.0, 400)
            self.temperature_ = float(min(candidates, key=lambda temp: objective(np.array([temp]))))
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.temperature_)


class PlattScaling:
    def __init__(self) -> None:
        self.a_ = 1.0
        self.b_ = 0.0

    def fit(self, raw_confidence: np.ndarray, correct: np.ndarray) -> "PlattScaling":
        raw_confidence = np.asarray(raw_confidence, dtype=float)
        correct = np.asarray(correct, dtype=float)
        logits = np.log(np.clip(raw_confidence, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - raw_confidence, 1e-8, 1.0))

        def objective(params: np.ndarray) -> float:
            a_value, b_value = params
            calibrated = sigmoid(a_value * logits + b_value)
            calibrated = np.clip(calibrated, 1e-12, 1.0 - 1e-12)
            return float(-np.mean(correct * np.log(calibrated) + (1.0 - correct) * np.log(1.0 - calibrated)))

        if minimize is not None:
            result = minimize(objective, x0=np.array([1.0, 0.0]))
            self.a_, self.b_ = float(result.x[0]), float(result.x[1])
        else:
            best = None
            for a_value in np.linspace(-3.0, 3.0, 81):
                for b_value in np.linspace(-3.0, 3.0, 81):
                    score = objective(np.array([a_value, b_value]))
                    if best is None or score < best[0]:
                        best = (score, a_value, b_value)
            assert best is not None
            self.a_, self.b_ = float(best[1]), float(best[2])
        return self

    def predict(self, raw_confidence: np.ndarray) -> np.ndarray:
        raw_confidence = np.asarray(raw_confidence, dtype=float)
        logits = np.log(np.clip(raw_confidence, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - raw_confidence, 1e-8, 1.0))
        return sigmoid(self.a_ * logits + self.b_)


class IsotonicCorrectnessCalibrator:
    def __init__(self) -> None:
        self.model_ = IsotonicRegression(out_of_bounds="clip") if IsotonicRegression is not None else None
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, raw_confidence: np.ndarray, correct: np.ndarray) -> "IsotonicCorrectnessCalibrator":
        x_values = np.asarray(raw_confidence, dtype=float)
        y_values = np.asarray(correct, dtype=float)
        if self.model_ is not None:
            self.model_.fit(x_values, y_values)
            return self

        order = np.argsort(x_values)
        x_sorted = x_values[order]
        y_sorted = y_values[order]

        blocks = [{"sum": float(y), "count": 1, "min_x": float(x), "max_x": float(x)} for x, y in zip(x_sorted, y_sorted)]
        idx = 0
        while idx < len(blocks) - 1:
            left_mean = blocks[idx]["sum"] / blocks[idx]["count"]
            right_mean = blocks[idx + 1]["sum"] / blocks[idx + 1]["count"]
            if left_mean > right_mean:
                blocks[idx] = {
                    "sum": blocks[idx]["sum"] + blocks[idx + 1]["sum"],
                    "count": blocks[idx]["count"] + blocks[idx + 1]["count"],
                    "min_x": blocks[idx]["min_x"],
                    "max_x": blocks[idx + 1]["max_x"],
                }
                del blocks[idx + 1]
                idx = max(idx - 1, 0)
            else:
                idx += 1

        x_points = []
        y_points = []
        for block in blocks:
            midpoint = (block["min_x"] + block["max_x"]) / 2.0
            x_points.append(midpoint)
            y_points.append(block["sum"] / block["count"])
        self.x_ = np.asarray(x_points, dtype=float)
        self.y_ = np.asarray(y_points, dtype=float)
        return self

    def predict(self, raw_confidence: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_confidence, dtype=float)
        if self.model_ is not None:
            return self.model_.predict(values)
        assert self.x_ is not None and self.y_ is not None
        return np.interp(values, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


@dataclass
class CalibrationOutputs:
    predicted_label: np.ndarray
    raw_confidence: np.ndarray
    calibrated_confidence: np.ndarray


def apply_temperature_scaling(
    calibration_logits: np.ndarray,
    calibration_true_index: np.ndarray,
    test_logits: np.ndarray,
    labels: list[str],
) -> CalibrationOutputs:
    model = TemperatureScaling().fit(calibration_logits, calibration_true_index)
    probs = model.predict(test_logits)
    pred_index = probs.argmax(axis=1)
    return CalibrationOutputs(
        predicted_label=np.array([labels[index] for index in pred_index]),
        raw_confidence=softmax(test_logits).max(axis=1),
        calibrated_confidence=probs.max(axis=1),
    )


def apply_correctness_calibrator(
    method: str,
    calibration_confidence: np.ndarray,
    calibration_correct: np.ndarray,
    test_confidence: np.ndarray,
    predicted_labels: np.ndarray,
) -> CalibrationOutputs:
    if method == "platt_scaling":
        model = PlattScaling().fit(calibration_confidence, calibration_correct)
    elif method == "isotonic_regression":
        model = IsotonicCorrectnessCalibrator().fit(calibration_confidence, calibration_correct)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    return CalibrationOutputs(
        predicted_label=np.asarray(predicted_labels),
        raw_confidence=np.asarray(test_confidence, dtype=float),
        calibrated_confidence=np.clip(model.predict(test_confidence), 1e-6, 0.999999),
    )
