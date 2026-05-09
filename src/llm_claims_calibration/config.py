from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

from pydantic import BaseModel, Field, model_validator

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


class DataConfig(BaseModel):
    source: str = "synthetic"
    n_claims: int = 1200
    train_fraction: float = 0.5
    calibration_fraction: float = 0.25
    test_fraction: float = 0.25

    @model_validator(mode="after")
    def validate_fractions(self) -> "DataConfig":
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Data split fractions must sum to 1.0")
        return self


class CalibrationConfig(BaseModel):
    methods: List[str] = Field(default_factory=lambda: ["raw", "temperature_scaling", "platt_scaling", "isotonic_regression"])
    bins: int = 10


class AbstentionConfig(BaseModel):
    thresholds: List[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    risk_tiers: Dict[str, float] = Field(
        default_factory=lambda: {"low": 0.8, "medium": 0.88, "high": 0.95, "mandatory_review": 1.01}
    )


class ExpectedLossConfig(BaseModel):
    false_approval_cost: float = 500.0
    false_denial_cost: float = 1500.0
    false_request_info_cost: float = 200.0
    human_review_cost: float = 75.0
    high_risk_delay_cost: float = 250.0


class OutputConfig(BaseModel):
    dataset_path: str = "data/synthetic/claims_synthetic.csv"
    tables_dir: str = "paper/tables"
    figures_dir: str = "paper/figures"
    report_path: str = "reports/experiment_summary.md"


class ExperimentConfig(BaseModel):
    paper_id: int = 3
    title: str
    random_seed: int = 20260503
    data: DataConfig = Field(default_factory=DataConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    expected_loss: ExpectedLossConfig = Field(default_factory=ExpectedLossConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    raw_text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    return ExperimentConfig.model_validate(payload)
