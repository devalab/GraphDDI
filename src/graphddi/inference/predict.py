"""Typer CLI: pass SMILES, get DDI predictions.

Loads ``weights/<task>.ckpt`` (a Lightning checkpoint) plus its sibling
``weights/<task>.yaml`` (a copy of the run's resolved config). The YAML
provides the encoder/readout class paths, so the inference side never
needs to hardcode an architecture.

Subcommands:

* ``pair``   score one drug-drug pair on the command line
* ``batch``  score a CSV of pairs and write predictions to disk
"""

import importlib
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from torch_geometric.data import Batch

from graphddi.data.featurizer import smiles_to_graph
from graphddi.models import GraphDDIModule

WEIGHTS_DIR = Path("weights")
Task = Literal["binary", "multiclass"]

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Predict drug-drug interactions from SMILES strings.",
)
console = Console()


# ----- model loading -------------------------------------------------------
def _instantiate(spec: dict):
    module_name, class_name = spec["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**spec.get("init_args", {}) or {})


def _resolve_paths(task: Task, ckpt: Path | None, cfg: Path | None) -> tuple[Path, Path]:
    ckpt_path = ckpt or WEIGHTS_DIR / f"{task}.ckpt"
    cfg_path = cfg or WEIGHTS_DIR / f"{task}.yaml"
    for p, kind in [(ckpt_path, "checkpoint"), (cfg_path, "config")]:
        if not p.exists():
            console.print(
                f"[bold red]✗ Missing {kind}[/]: [cyan]{p}[/]\n"
                f"  Drop a trained ``{task}`` run's last.ckpt and config.yaml "
                f"into [cyan]weights/[/] (or pass --checkpoint/--config)."
            )
            raise typer.Exit(code=2)
    return ckpt_path, cfg_path


def load(task: Task, ckpt: Path | None = None, cfg: Path | None = None) -> GraphDDIModule:
    ckpt_path, cfg_path = _resolve_paths(task, ckpt, cfg)
    spec = yaml.safe_load(cfg_path.read_text())["model"]["init_args"]
    init = dict(spec)
    init["encoder"] = _instantiate(spec["encoder"])
    init["readout"] = _instantiate(spec["readout"])
    model = GraphDDIModule(**init)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model


# ----- output helpers ------------------------------------------------------
def _featurize(s1: str, s2: str) -> tuple[Batch, Batch] | None:
    g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
    if g1 is None or g2 is None:
        return None
    return Batch.from_data_list([g1]), Batch.from_data_list([g2])


def _prob_style(p: float) -> str:
    if p >= 0.85:
        return "bold green"
    if p >= 0.6:
        return "green"
    if p >= 0.4:
        return "yellow"
    return "red"


def _label_map() -> dict[int, str]:
    from tdc.utils import get_label_map

    raw = get_label_map(name="DrugBank", task="DDI")
    if isinstance(raw, list):
        return {i + 1: str(name) for i, name in enumerate(raw)}
    return {int(k): str(v) for k, v in raw.items()}


# ----- commands ------------------------------------------------------------
@app.command()
def pair(
    smiles1: Annotated[str, typer.Argument(help="SMILES of Drug 1")],
    smiles2: Annotated[str, typer.Argument(help="SMILES of Drug 2")],
    task: Annotated[Task, typer.Option("--task", "-t")] = "multiclass",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", "-c")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
) -> None:
    """Score one drug-drug pair."""
    model = load(task, checkpoint, config)
    batches = _featurize(smiles1, smiles2)
    if batches is None:
        console.print("[bold red]✗ RDKit could not parse one of the SMILES strings.[/]")
        raise typer.Exit(code=2)
    b1, b2 = batches

    with torch.no_grad():
        logits = model(b1, b2).squeeze(0)

    table = Table(
        title=f"GraphDDI · task = {task}",
        show_header=True,
        header_style="bold cyan",
    )
    if task == "binary":
        p = float(torch.sigmoid(logits))
        table.add_column("Outcome", style="bold")
        table.add_column("Probability", justify="right")
        table.add_row("Interaction predicted", f"[{_prob_style(p)}]{p:.4f}[/]")
        table.add_row("No interaction", f"{1 - p:.4f}")
    else:
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.numel()))
        labels = _label_map()
        for col, justify in [
            ("Rank", "right"),
            ("Class", "right"),
            ("Probability", "right"),
            ("Description", "left"),
        ]:
            table.add_column(col, justify=justify, style="cyan" if col == "Class" else None)
        for rank, (p, idx) in enumerate(
            zip(top.values.tolist(), top.indices.tolist(), strict=True), 1
        ):
            cls = idx + 1
            table.add_row(
                str(rank), str(cls), f"[{_prob_style(p)}]{p:.4f}[/]", labels.get(cls, "—")
            )

    console.print(
        Panel.fit(
            f"[bold]Drug 1[/]: {smiles1}\n[bold]Drug 2[/]: {smiles2}",
            title="Input",
            border_style="cyan",
        )
    )
    console.print(table)


@app.command()
def batch(
    input_csv: Annotated[Path, typer.Argument(help="CSV with SMILES1, SMILES2 columns")],
    task: Annotated[Task, typer.Option("--task", "-t")] = "multiclass",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", "-c")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    output_csv: Annotated[Path, typer.Option("--out", "-o")] = Path("predictions.csv"),
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 1,
) -> None:
    """Score a CSV of drug pairs."""
    df = pd.read_csv(input_csv).rename(
        columns={
            "Drug1": "SMILES1",
            "drug1": "SMILES1",
            "Drug2": "SMILES2",
            "drug2": "SMILES2",
            "smiles1": "SMILES1",
            "smiles2": "SMILES2",
        }
    )
    if {"SMILES1", "SMILES2"} - set(df.columns):
        console.print(f"[bold red]✗ {input_csv} must have SMILES1 and SMILES2 columns.[/]")
        raise typer.Exit(code=2)

    model = load(task, checkpoint, config)
    labels = _label_map() if task == "multiclass" else {}
    rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        bar = progress.add_task(f"Scoring {len(df)} pairs", total=len(df))
        for _, r in df.iterrows():
            batches = _featurize(r["SMILES1"], r["SMILES2"])
            if batches is None:
                rows.append(
                    {"SMILES1": r["SMILES1"], "SMILES2": r["SMILES2"], "error": "parse_failed"}
                )
                progress.update(bar, advance=1)
                continue
            b1, b2 = batches
            with torch.no_grad():
                logits = model(b1, b2).squeeze(0)
            base = {"SMILES1": r["SMILES1"], "SMILES2": r["SMILES2"]}
            if task == "binary":
                p = float(torch.sigmoid(logits))
                rows.append({**base, "prob_interaction": p, "prediction": int(p >= 0.5)})
            else:
                probs = torch.softmax(logits, dim=-1)
                top = torch.topk(probs, k=min(top_k, probs.numel()))
                for rank, (p, idx) in enumerate(
                    zip(top.values.tolist(), top.indices.tolist(), strict=True), 1
                ):
                    cls = idx + 1
                    rows.append(
                        {
                            **base,
                            "rank": rank,
                            "predicted_class": cls,
                            "probability": p,
                            "description": labels.get(cls, ""),
                        }
                    )
            progress.update(bar, advance=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    console.print(
        Panel.fit(
            f"Wrote [bold green]{len(rows)}[/] rows to [bold]{output_csv}[/]",
            title="Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
