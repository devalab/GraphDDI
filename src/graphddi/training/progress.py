"""Aesthetic progress display for GraphDDI training.

Layout::

    [console prints land above this region — Rich Live redirects them]
    Training        ━━━━━━━━━━━━━━╸━━━━━━━━━ 12/60 epochs
    Epoch 12        ━━━━━━━━╸━━━━━━━━━━━━━━━ 144/662 steps   ← transient
                                                          val/macro_f1 0.772

The top "Training" task is persistent for the whole fit. The lower task is
swapped between train/validate and hidden when each phase ends.
"""

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.progress.rich_progress import (
    CustomProgress,
    MetricsTextColumn,
    RichProgressBar,
    RichProgressBarTheme,
)

from graphddi.training.logging_setup import CONSOLE

_THEME = RichProgressBarTheme(
    description="bold cyan",
    progress_bar="#6206E0",
    progress_bar_finished="#6206E0",
    progress_bar_pulse="#6206E0",
    batch_progress="bold",
    time="dim",
    processing_speed="dim italic",
    metrics="bold magenta",
    metrics_text_delimiter="  ",
    metrics_format=".3f",
)


class GraphDDIProgressBar(RichProgressBar):
    """RichProgressBar with a persistent total-epochs task at the top."""

    def __init__(self, refresh_rate: int = 10) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=False, theme=_THEME)
        self._epoch_task_id: int | None = None

    def _init_progress(self, trainer: Trainer) -> None:
        # Reuse our shared Rich console so structlog lines land above the live region.
        if not (self.is_enabled and (self.progress is None or self._progress_stopped)):
            return
        self._reset_progress_bar_ids()
        self._console = CONSOLE
        self._metric_component = MetricsTextColumn(
            trainer,
            self.theme.metrics,
            self.theme.metrics_text_delimiter,
            self.theme.metrics_format,
        )
        self.progress = CustomProgress(
            *self.configure_columns(trainer),
            self._metric_component,
            auto_refresh=True,
            refresh_per_second=self.refresh_rate if self.is_enabled else 1,
            disable=self.is_disabled,
            console=self._console,
        )
        self.progress.start()
        self._progress_stopped = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        if self.progress is None or self.is_disabled:
            return
        total = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None
        self._epoch_task_id = self.progress.add_task(
            f"[{self.theme.description}]Training",
            total=total,
            completed=trainer.current_epoch,
        )
        self.refresh()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self.progress is not None and self._epoch_task_id is not None:
            self.progress.update(self._epoch_task_id, completed=trainer.current_epoch + 1)
            self.refresh()

    def _get_train_description(self, current_epoch: int) -> str:
        return f"Epoch {current_epoch + 1:>3}"

    @property
    def validation_description(self) -> str:
        return "Validate"

    @property
    def test_description(self) -> str:
        return "Test"

    @property
    def sanity_check_description(self) -> str:
        return "Sanity"

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> dict[str, Any]:
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)  # already encoded in the run dir name
        return metrics
