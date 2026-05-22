"""Acetaminophen top-K predicted interactions.

Loads the multiclass GraphDDI checkpoint from ``weights/``, takes
Acetaminophen as Drug 1, drops every drug already known to interact with it
in DrugBank, scores the remaining drugs as Drug 2, and reports the top-K
predictions by softmax probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from tdc.multi_pred import DDI
from torch_geometric.data import Batch

from graphddi.data._loader import quiet_stderr
from graphddi.data.featurizer import smiles_to_graph
from graphddi.inference.predict import load as load_model

ACETAMINOPHEN_ID = "DB00316"
ACETAMINOPHEN_SMILES = "CC(=O)NC1=CC=C(O)C=C1"

KNOWN_NAMES: dict[str, str] = {
    "DB00316": "Acetaminophen",
    "DB08877": "Tofacitinib",
    "DB00460": "Verteporfin",
    "DB09125": "Tetracosactide",
    "DB01078": "Deslanoside",
    "DB01092": "Ouabain",
    "DB06702": "Fesoterodine",
    "DB00364": "Sucralfate",
    "DB00811": "Ribavirin",
    "DB01396": "Digitoxin",
    "DB06786": "Halcinonide",
}

console = Console()


def _drug_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(unique_drugs_df, raw_ddi_df)`` from TDC DrugBank."""
    with quiet_stderr():
        raw = DDI(name="DrugBank").get_data()
    a = raw[["Drug1_ID", "Drug1"]].rename(columns={"Drug1_ID": "id", "Drug1": "smiles"})
    b = raw[["Drug2_ID", "Drug2"]].rename(columns={"Drug2_ID": "id", "Drug2": "smiles"})
    drugs = pd.concat([a, b]).drop_duplicates(subset="id").reset_index(drop=True)
    return drugs, raw


def _partners_of(raw: pd.DataFrame, drug_id: str) -> set[str]:
    mask = (raw["Drug1_ID"] == drug_id) | (raw["Drug2_ID"] == drug_id)
    sub = raw[mask]
    return set(sub["Drug1_ID"]).union(sub["Drug2_ID"]) - {drug_id}


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main(
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 64,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("logs/case_study_acetaminophen.csv"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold cyan]Acetaminophen case study")
    console.print(f"Device: [bold]{device}[/]")

    model = load_model("multiclass").to(device).eval()
    drugs, raw = _drug_universe()
    partners = _partners_of(raw, ACETAMINOPHEN_ID)
    pool = drugs[~drugs["id"].isin(partners) & (drugs["id"] != ACETAMINOPHEN_ID)]

    console.print(
        f"DrugBank universe: [bold]{len(drugs)}[/]  ·  "
        f"known partners: [bold]{len(partners)}[/]  ·  "
        f"to score: [bold green]{len(pool)}[/]"
    )

    g_a = smiles_to_graph(ACETAMINOPHEN_SMILES)
    if g_a is None:
        raise SystemExit("RDKit could not parse the Acetaminophen SMILES.")

    rows: list[dict] = []
    skipped = 0
    pairs = list(pool.itertuples(index=False))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Scoring"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        bar = prog.add_task("scoring", total=len(pairs))
        with torch.no_grad():
            for chunk in _chunks(pairs, batch_size):
                graphs_b: list = []
                ids: list[str] = []
                for drug_id, smi in chunk:
                    g_b = smiles_to_graph(smi)
                    if g_b is None:
                        skipped += 1
                        continue
                    graphs_b.append(g_b)
                    ids.append(drug_id)
                prog.update(bar, advance=len(chunk))
                if not graphs_b:
                    continue
                ba = Batch.from_data_list([g_a] * len(graphs_b)).to(device)
                bb = Batch.from_data_list(graphs_b).to(device)
                probs = torch.softmax(model(ba, bb), dim=-1)
                top = probs.max(dim=-1)
                for drug_id, p, idx in zip(
                    ids, top.values.tolist(), top.indices.tolist(), strict=True
                ):
                    rows.append({"id": drug_id, "predicted_class": idx + 1, "probability": p})

    if skipped:
        console.print(f"[yellow]Skipped {skipped} pairs (RDKit could not parse).[/]")

    ranking = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
    ranking["name"] = ranking["id"].map(KNOWN_NAMES).fillna("—")

    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)

    table = Table(
        title=f"Top-{top_k} predicted interactions of Acetaminophen",
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right")
    table.add_column("DrugBank ID", style="cyan")
    table.add_column("Name")
    table.add_column("Class", justify="right", style="magenta")
    table.add_column("Probability", justify="right")
    for rank, r in enumerate(ranking.head(top_k).itertuples(index=False), 1):
        table.add_row(str(rank), r.id, r.name, str(r.predicted_class), f"{r.probability:.4f}")
    console.print(table)
    console.print(f"Full ranking saved to [bold]{out}[/]")


if __name__ == "__main__":
    typer.run(main)
