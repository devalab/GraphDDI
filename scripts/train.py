"""Train GraphDDI via LightningCLI.

Usage::

    uv run python scripts/train.py fit --config configs/experiment/drugbank_multiclass.yaml
"""

import warnings

# Silence chatty third-party warnings that don't apply to our pinned stack.
warnings.filterwarnings(
    "ignore",
    message=r".*scatter\(reduce='(min|max)'\).*can be accelerated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*'train_dataloader' does not have many workers.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*'val_dataloader' does not have many workers.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\).*",
)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"mlflow\..*")

from graphddi.training import GraphDDILightningCLI  # noqa: E402


def main() -> None:
    GraphDDILightningCLI()


if __name__ == "__main__":
    main()
