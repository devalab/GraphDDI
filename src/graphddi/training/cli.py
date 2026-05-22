"""LightningCLI subclass.

Each run gets ``logs/<experiment_name>/<YYYYMMDDHHMMSS>/`` as its home,
containing the resolved config, the structlog file, and the checkpoints.
MLflow tracking is pinned to ``logs/tracking/`` so every run is visible
through ``mlflow ui --backend-store-uri sqlite:///logs/tracking/mlflow.db``.
"""

from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from graphddi.training.logging_setup import configure as configure_logging
from graphddi.training.mlflow import MLFlowLogger
from graphddi.training.paths import make_run_dir, mlflow_artifact_uri, mlflow_tracking_uri

MatmulPrecision = Literal["highest", "high", "medium"]


class _RunDirSaveConfig(SaveConfigCallback):
    """Save the resolved config to an absolute path inside the run dir."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["save_to_log_dir"] = False
        super().__init__(*args, **kwargs)

    def save_config(self, trainer, pl_module, stage) -> None:  # noqa: ARG002
        # Path is fully determined by config_filename; arguments are unused.
        target = Path(self.config_filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not self.overwrite:
            return
        self.parser.save(
            self.config,
            target,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )


class GraphDDILightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        self.run_dir: Path | None = None
        self.run_timestamp: str | None = None
        kwargs.setdefault("save_config_callback", _RunDirSaveConfig)
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            "--matmul_precision",
            type=str,
            default="medium",
            choices=["highest", "high", "medium"],
            help="torch.set_float32_matmul_precision — 'medium' enables TF32 on Tensor Cores.",
        )
        parser.link_arguments("data.task", "model.init_args.task", apply_on="instantiate")
        parser.link_arguments(
            "data.num_classes", "model.init_args.num_classes", apply_on="instantiate"
        )

    def before_instantiate_classes(self) -> None:
        cfg = self.config.get(self.subcommand, self.config)
        torch.set_float32_matmul_precision(getattr(cfg, "matmul_precision", "medium"))

        exp_name = _logger_attr(cfg, "experiment_name", default="default")
        self.run_dir, self.run_timestamp = make_run_dir(exp_name)
        configure_logging(self.run_dir).info(
            "run.start", experiment=exp_name, run_dir=str(self.run_dir)
        )
        self.save_config_kwargs = {
            "config_filename": str(self.run_dir / "config.yaml"),
            "overwrite": False,
        }
        _route_mlflow(cfg, self.run_timestamp)
        _route_checkpoints(cfg, self.run_dir)


def _logger_attr(cfg, name: str, default: str) -> str:
    logger_cfg = getattr(cfg.trainer, "logger", None)
    init = getattr(logger_cfg, "init_args", None) if logger_cfg else None
    return getattr(init, name, None) or default if init else default


def _route_mlflow(cfg, timestamp: str) -> None:
    logger_cfg = getattr(cfg.trainer, "logger", None)
    init = getattr(logger_cfg, "init_args", None) if logger_cfg else None
    if init is None:
        return
    init.tracking_uri = mlflow_tracking_uri()
    init.artifact_location = mlflow_artifact_uri()
    init.run_name = f"{init.run_name}_{timestamp}" if getattr(init, "run_name", None) else timestamp


def _route_checkpoints(cfg, run_dir: Path) -> None:
    """Point any ``ModelCheckpoint`` callback at ``<run_dir>/checkpoints``;
    add a default one if the config doesn't already include it."""
    from jsonargparse import Namespace

    target = str(run_dir / "checkpoints")
    callbacks = list(getattr(cfg.trainer, "callbacks", None) or [])
    has_ckpt = False
    for cb in callbacks:
        init = getattr(cb, "init_args", None)
        if init is None or not getattr(cb, "class_path", "").endswith("ModelCheckpoint"):
            continue
        has_ckpt = True
        if getattr(init, "dirpath", None) is None:
            init.dirpath = target
    if not has_ckpt:
        callbacks.append(
            Namespace(
                class_path="lightning.pytorch.callbacks.ModelCheckpoint",
                init_args=Namespace(dirpath=target, save_last=True, save_top_k=1),
            )
        )
        cfg.trainer.callbacks = callbacks


__all__ = ["GraphDDILightningCLI", "MLFlowLogger"]
