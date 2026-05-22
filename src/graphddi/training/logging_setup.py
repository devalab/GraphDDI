"""structlog → RichHandler for the console, JSONL for the file.

The shared :data:`CONSOLE` is the one Rich object every other component
(progress bar, traceback, etc.) writes through, so log lines automatically
land *above* the live progress region instead of fighting it.
"""

import logging
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

CONSOLE = Console(
    highlight=False,
    theme=Theme(
        {
            "logging.level.info": "bold cyan",
            "logging.level.warning": "bold yellow",
            "logging.level.error": "bold red",
            "logging.level.debug": "dim",
        }
    ),
)

_NOISY = ("urllib3", "matplotlib", "PIL", "git", "filelock", "fsspec")


class _DropLightningTips(logging.Filter):
    """Lightning prints a ``💡 Tip: ...`` line for litlogger every run — drop it."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.getMessage()).lstrip().startswith("💡 Tip")


def configure(run_dir: Path, level: int = logging.INFO) -> structlog.stdlib.BoundLogger:
    """Pretty Rich output on the console, JSONL in ``<run_dir>/train.log``."""
    run_dir.mkdir(parents=True, exist_ok=True)

    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*pre_chain, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Console: RichHandler renders timestamp + level; structlog just gives clean text.
    console_handler = RichHandler(
        console=CONSOLE,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="[%H:%M:%S]",
    )
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False, pad_event=24),
            ],
        )
    )
    console_handler.addFilter(_DropLightningTips())

    # File: include level + ISO timestamp inside the JSON payload.
    file_handler = logging.FileHandler(run_dir / "train.log", mode="a")
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=False),
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level)

    # Lightning + MLflow ship their own StreamHandlers with propagate=False.
    # Importing mlflow first ensures its handler exists before we strip it.
    import mlflow  # noqa: F401
    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "mlflow"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return structlog.get_logger("graphddi")
