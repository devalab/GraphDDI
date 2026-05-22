"""Optuna sweep with nested MLflow runs and PL pruning.

The sweep YAML has three top-level keys::

    base_config: configs/experiment/smoke_mini.yaml   # any LightningCLI YAML
    sweep:
        n_trials: 20
        sampler: tpe          # tpe | random
        pruner: hyperband     # hyperband | median | none
        monitor: val/f1_macro # also used for direction (maximize for f1/auc)
        timeout: null         # seconds; null = no timeout
    search_space:
        # dotted key in the merged config → suggest spec
        model.init_args.lr:        {type: loguniform, low: 1.0e-5, high: 1.0e-2}
        model.init_args.dropout:   {type: uniform, low: 0.0, high: 0.5}
        model.init_args.encoder.init_args.hidden_channels:
            type: int_categorical
            choices: [64, 128, 256]
    mlflow:
        tracking_uri: ./mlruns
        experiment_name: graphddi-sweep
"""

import argparse
import copy
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml
from lightning.pytorch.loggers import MLFlowLogger
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

from graphddi.training.cli import GraphDDILightningCLI


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind in ("categorical", "int_categorical"):
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown suggest type: {kind!r}")


def _sampler(name: str) -> BaseSampler:
    return {"tpe": TPESampler, "random": RandomSampler}[name]()


def _pruner(name: str) -> BasePruner:
    if name == "hyperband":
        return HyperbandPruner()
    if name == "median":
        return MedianPruner()
    if name == "none":
        return NopPruner()
    raise ValueError(f"unknown pruner: {name!r}")


def _direction(monitor: str) -> str:
    """Most val-side metrics we use are 'higher is better' — be conservative."""
    if any(t in monitor.lower() for t in ("loss", "err", "nll")):
        return "minimize"
    return "maximize"


def run_sweep(sweep_yaml: str) -> None:
    spec = _load_yaml(sweep_yaml)
    base_cfg = _load_yaml(spec["base_config"])
    s = spec["sweep"]
    monitor = s.get("monitor", "val/f1_macro")
    mlflow_cfg = spec.get("mlflow", {})

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "graphddi-sweep"))

    study = optuna.create_study(
        direction=_direction(monitor),
        sampler=_sampler(s.get("sampler", "tpe")),
        pruner=_pruner(s.get("pruner", "hyperband")),
        study_name=mlflow_cfg.get("experiment_name", "graphddi-sweep"),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        for key, sp in spec["search_space"].items():
            _set_dotted(cfg, key, _suggest(trial, key, sp))

        with mlflow.start_run(nested=False, run_name=f"trial-{trial.number}"):
            mlflow.log_params(
                {f"trial.{k}": v for k, v in trial.params.items() if not isinstance(v, dict | list)}
            )

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
                tmp_path = f.name

            pruning_cb = PyTorchLightningPruningCallback(trial, monitor=monitor)
            cli = GraphDDILightningCLI(
                args=["--config", tmp_path],
                run=False,
                save_config_callback=None,
            )
            trainer = cli.trainer
            trainer.callbacks = [*trainer.callbacks, pruning_cb]
            trainer.logger = MLFlowLogger(
                tracking_uri=mlflow_cfg.get("tracking_uri", "./mlruns"),
                experiment_name=mlflow_cfg.get("experiment_name", "graphddi-sweep"),
                run_id=mlflow.active_run().info.run_id,
            )
            trainer.fit(cli.model, datamodule=cli.datamodule)
            score = trainer.callback_metrics.get(monitor)
            if score is None:
                raise optuna.TrialPruned(f"{monitor!r} missing from callback_metrics")
            return float(score)

    study.optimize(
        objective,
        n_trials=int(s.get("n_trials", 20)),
        timeout=s.get("timeout"),
        catch=(optuna.TrialPruned,),
    )

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to sweep YAML")
    args = p.parse_args()
    run_sweep(args.config)


if __name__ == "__main__":
    main()
