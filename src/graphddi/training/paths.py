"""Per-run filesystem layout.

Every training run gets a unique directory::

    logs/
    ├── tracking/
    │   ├── mlflow.db        SQLite backend store (shared across runs)
    │   └── artifacts/       Artifacts root (shared across runs)
    └── <experiment_name>/
        └── <YYYYMMDDHHMMSS>/
            ├── config.yaml      LightningCLI saved config
            ├── train.log        structlog output
            └── checkpoints/     ModelCheckpoint outputs

Open the UI with::

    mlflow ui \\
        --backend-store-uri sqlite:///logs/tracking/mlflow.db \\
        --default-artifact-root file:logs/tracking/artifacts
"""

from datetime import datetime
from pathlib import Path

LOGS_ROOT = Path("logs")
TRACKING_ROOT = LOGS_ROOT / "tracking"
MLFLOW_DB = TRACKING_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = TRACKING_ROOT / "artifacts"


def mlflow_tracking_uri() -> str:
    """SQLite URI for MLflow (cwd-relative)."""
    return f"sqlite:///{MLFLOW_DB.as_posix()}"


def mlflow_artifact_uri() -> str:
    """`file:` URI MLflow uses as the default artifact root."""
    return f"file:{MLFLOW_ARTIFACTS.as_posix()}"


def make_run_dir(experiment_name: str, timestamp: str | None = None) -> tuple[Path, str]:
    """Create ``logs/<experiment_name>/<YYYYMMDDHHMMSS>/`` and return ``(path, ts)``."""
    ts = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = LOGS_ROOT / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return run_dir, ts
