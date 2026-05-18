from __future__ import annotations

import numpy as np
import pandas as pd

from llm_claims_calibration.data.generate_synthetic_claims import SyntheticArtifacts, generate_perturbed_logits_for_consistency


def generate_consistency_confidence(
    claims_df: pd.DataFrame,
    logits: np.ndarray,
    k_samples: int = 3,
    seed: int = 20260503,
    perturbation_scale: float = 0.05,
) -> np.ndarray:
    _ = claims_df
    logits = np.asarray(logits, dtype=float)
    base_prediction = logits.argmax(axis=1)
    perturbed = generate_perturbed_logits_for_consistency(
        logits=logits,
        k_samples=k_samples,
        seed=seed,
        perturbation_scale=perturbation_scale,
    )
    sampled_predictions = perturbed.argmax(axis=2)
    agreement = (sampled_predictions == base_prediction[:, None]).sum(axis=1)
    return np.clip(agreement / float(k_samples), 0.0, 1.0)


def generate_logit_margin_confidence(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    top_two = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    return 1.0 / (1.0 + np.exp(-margin))


def get_simulated_confidence(
    artifacts: SyntheticArtifacts,
    splits: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    _ = artifacts
    return {
        "calibration": splits["calibration"]["raw_confidence"].to_numpy(dtype=float),
        "test": splits["test"]["raw_confidence"].to_numpy(dtype=float),
    }
