from pathlib import Path
import json
import subprocess
import sys


def test_experiment_runner_generates_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "paper_id": 3,
                "title": "Pipeline Smoke Test",
                "random_seed": 7,
                "data": {
                    "source": "synthetic",
                    "n_claims": 120,
                    "train_fraction": 0.50,
                    "calibration_fraction": 0.25,
                    "test_fraction": 0.25,
                },
                "calibration": {
                    "methods": ["raw", "temperature_scaling", "platt_scaling", "isotonic_regression"],
                    "bins": 5,
                },
                "abstention": {
                    "thresholds": [0.5, 0.7, 0.9],
                    "risk_tiers": {
                        "low": 0.8,
                        "medium": 0.88,
                        "high": 0.95,
                        "mandatory_review": 1.01,
                    },
                },
                "expected_loss": {
                    "false_approval_cost": 500.0,
                    "false_denial_cost": 1500.0,
                    "false_request_info_cost": 200.0,
                    "human_review_cost": 75.0,
                    "high_risk_delay_cost": 250.0,
                },
                "outputs": {
                    "dataset_path": str(tmp_path / "claims.csv"),
                    "tables_dir": str(tmp_path / "tables"),
                    "figures_dir": str(tmp_path / "figures"),
                    "report_path": str(tmp_path / "report.md"),
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--config", str(config_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert (tmp_path / "tables" / "calibration_results.csv").exists()
    assert (tmp_path / "tables" / "threshold_results.csv").exists()
    assert (tmp_path / "figures" / "reliability_diagram.png").exists()
    assert (tmp_path / "figures" / "risk_coverage_curve.png").exists()
    assert (tmp_path / "report.md").exists()
    assert (Path(__file__).resolve().parents[1] / "results" / "bootstrap_results.csv").exists()
