"""Parameter-count summary callback that skips Lightning's FlopCounterMode.

Lightning 2.6's ``ModelSummary`` hard-codes ``torch.utils.flop_counter.FlopCounterMode``
initialisation (no flag disables it). On Windows + CUDA installs without
``triton`` it emits a "triton not found" log line per process (main + every
DataLoader worker), and FLOPs come out zero for our PyG ops anyway. Set
``trainer.enable_model_summary: false`` and use this callback instead.
"""

from lightning.pytorch import Callback, LightningModule, Trainer
from rich.table import Table

from graphddi.training.logging_setup import CONSOLE


def _human(n: int) -> str:
    """Format a parameter count, e.g. ``906_326 → '906.3 K'``."""
    x = float(n)
    for unit in ("", "K", "M", "B"):
        if abs(x) < 1000:
            return f"{x:.1f} {unit}".rstrip() if unit else str(n)
        x /= 1000
    return f"{x:.1f} T"


class GraphDDIParamSummary(Callback):
    """One-shot param-count table printed at ``on_fit_start`` (top-level children only)."""

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        table = Table(title="GraphDDI parameters", title_style="bold cyan")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Params", justify="right")

        for i, (name, module) in enumerate(pl_module.named_children()):
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            table.add_row(str(i), name, type(module).__name__, _human(n))

        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        table.add_section()
        table.add_row("", "[bold]Trainable[/]", "", f"[bold]{_human(trainable)}[/]")
        table.add_row("", "[bold]Total[/]", "", f"[bold]{_human(total)}[/]")

        CONSOLE.print(table)
