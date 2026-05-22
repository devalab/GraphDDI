"""Thin re-export of Lightning's MLFlowLogger so configs can keep using
``graphddi.training.MLFlowLogger`` while we own a single import surface.

Tracking URI defaults are wired in :mod:`graphddi.training.cli`.
"""

from lightning.pytorch.loggers import MLFlowLogger

__all__ = ["MLFlowLogger"]
