import numpy as np
import pandas as pd

from llm_claims_calibration.confidence.variants import (
    generate_consistency_confidence,
    generate_logit_margin_confidence,
)
from llm_claims_calibration.data.generate_synthetic_claims import generate_perturbed_logits_for_consistency


def test_consistency_confidence_shape() -> None:
    logits = np.random.randn(100, 5)
    frame = pd.DataFrame({"claim_id": np.arange(100)})
    confidence = generate_consistency_confidence(frame, logits, k_samples=3)
    assert confidence.shape == (100,)
    assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)


def test_logit_margin_confidence_shape() -> None:
    logits = np.random.randn(100, 5)
    confidence = generate_logit_margin_confidence(logits)
    assert confidence.shape == (100,)
    assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)


def test_perturbed_logits_shape() -> None:
    logits = np.random.randn(100, 5)
    perturbed = generate_perturbed_logits_for_consistency(logits, k_samples=3)
    assert perturbed.shape == (100, 3, 5)


def test_consistency_confidence_reproducibility() -> None:
    logits = np.random.randn(100, 5)
    frame = pd.DataFrame({"claim_id": np.arange(100)})
    conf1 = generate_consistency_confidence(frame, logits, k_samples=3, seed=42)
    conf2 = generate_consistency_confidence(frame, logits, k_samples=3, seed=42)
    np.testing.assert_array_almost_equal(conf1, conf2, decimal=10)
