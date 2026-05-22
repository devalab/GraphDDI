# GraphDDI

[![Paper](https://img.shields.io/badge/AIiH%202024-LNCS%2014812-b31b1b)](https://doi.org/10.1007/978-3-031-67278-1_2)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--67278--1__2-0a7bbb)](https://doi.org/10.1007/978-3-031-67278-1_2)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyG](https://img.shields.io/badge/PyG-2.7-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-792EE5)](https://lightning.ai/)
[![uv](https://img.shields.io/badge/managed_by-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![ruff](https://img.shields.io/badge/lint-ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Graph neural network for drug-drug interaction (DDI) prediction. Predicts whether two drugs interact (binary), and which of 86 DDI event types they trigger (multi-class), from SMILES alone.

A writeup of this work is published in *Lecture Notes in Computer Science* vol. 14812, pp. 17–30 ([AIiH 2024, Swansea](https://doi.org/10.1007/978-3-031-67278-1_2)).

## Model

Two molecular graphs go in, one DDI prediction comes out. Three stages, trained end-to-end:

1. **Featurize.** A 6-layer PNA encoder (Corso et al., 2020) with per-layer `GraphNorm` and a `GRUCell` state update produces per-atom features for each drug. Hidden width 64, edge MLP width 128.
2. **Interact.** An atom-by-atom interaction map `I = tanh(A Bᵀ / √H)` lets each drug attend to the other's atoms; the influence vectors `A' = IB`, `B' = IᵀA` get concatenated back into the node features.
3. **Predict.** A Set2Set readout (2 processing steps) with a `Linear → ReLU` projection collapses each drug to a 128-d graph vector. The two are concatenated and a `Linear → GELU → Dropout → Linear` head outputs either a binary probability or a softmax over the 86 DDI event types from DrugBank. Training uses cross-entropy with `label_smoothing=0.1` for the multi-class task.

The production multi-class model is ~906 K parameters.

## Results

DrugBank uses the Ryu et al. (2018) split via Therapeutics Data Commons, random 70/10/20 with under-represented classes oversampled in the training split only.

**DrugBank multi-class — test set, current weights**

| Metric          |  Value |
| --------------- | -----: |
| Macro-F1        | 0.9159 |
| Macro-Precision | 0.9079 |
| Macro-Recall    | 0.9294 |
| Micro-F1        | 0.9408 |
| PR-AUC          | 0.9621 |
| Per-class F1 ≥ 1.0 | 20 classes |
| Per-class F1 = 0.0 |  2 classes |

The shipped checkpoint (`weights/multiclass.ckpt`) is the best-val epoch from a 60-epoch run on the default config. Reproduce the test row with `uv run python scripts/train.py test --config configs/experiment/drugbank_multiclass.yaml --ckpt_path weights/multiclass.ckpt`.

Binary checkpoints are not currently shipped; train your own with `configs/experiment/drugbank_binary.yaml` or `configs/experiment/biosnap_binary.yaml`.

## Quickstart

```bash
uv sync                                            # locks the environment from uv.lock (PyTorch CUDA 13.0 wheel)
uv run python scripts/prepare_test_fixture.py      # ~200-pair real slice for tests + smoke runs
uv run python scripts/download_biosnap.py          # pulls the CASTER BioSNAP splits (binary task only)
uv run pytest                                      # 31 tests, around 5 s

# Smoke train on the committed mini fixture (CPU OK)
uv run python scripts/train.py fit --config configs/experiment/smoke_mini.yaml

# Full multi-class run
uv run python scripts/train.py fit --config configs/experiment/drugbank_multiclass.yaml

# Live experiment tracking (SQLite backend, see below)
uv run mlflow ui \
  --backend-store-uri sqlite:///logs/tracking/mlflow.db \
  --default-artifact-root file:logs/tracking/artifacts
```

For a single 8 GB laptop GPU there's `configs/experiment/laptop_5epoch.yaml` (batch 128, 5 epochs).

## Inference

A Typer CLI loads `weights/<task>.ckpt` and its sibling `weights/<task>.yaml` (a copy of the run's resolved config), so the architecture is reconstructed exactly from the run that produced the checkpoint — no need to hardcode encoder/readout choices on the inference side.

`weights/multiclass.ckpt` + `weights/multiclass.yaml` are committed and ready to use. For binary, train your own and drop them in:

```bash
cp logs/<exp>/<TS>/checkpoints/last.ckpt   weights/binary.ckpt
cp logs/<exp>/<TS>/config.yaml             weights/binary.yaml
```

Then:

```bash
# One pair
uv run python -m graphddi.inference.predict pair \
  "CC(=O)NC1=CC=C(O)C=C1" "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN1CCCCCC1" --top-k 5

# A whole CSV of pairs (SMILES1, SMILES2 columns)
uv run python -m graphddi.inference.predict batch pairs.csv --out predictions.csv

# Case study — score Acetaminophen against every non-partner in DrugBank
uv run python scripts/case_study.py
```

`--checkpoint` / `--config` flags override the default `weights/<task>.{ckpt,yaml}` lookup if you want to score from an arbitrary run directory.

## Run directory layout

Every run gets its own dated folder; MLflow points at a single shared SQLite store so the UI sees every run at once.

```
logs/
├── tracking/
│   ├── mlflow.db                           SQLite backend
│   └── artifacts/                          MLflow artifact root
└── <experiment_name>/
    └── <YYYYMMDDHHMMSS>/
        ├── config.yaml                     resolved LightningCLI config
        ├── train.log                       structlog JSONL
        └── checkpoints/                    ModelCheckpoint outputs
```

The console gets a single persistent **Training** bar (epoch count) with a transient per-epoch bar below it; structlog lines and Lightning info messages print above the live region. No `print` chatter — TDC's stderr noise is captured per-call with `contextlib.redirect_stderr`, RDKit's C++ parse-error logger is muted at module import via the public `RDLogger.DisableLog` API.

## Repository layout

```
src/graphddi/
  data/         featurizer, DrugBank / BioSNAP / mini DataModules, paired-graph collate
  models/       PNA + MPNN encoders, interaction map, Set2Set / SetTransformer readouts,
                GraphDDIModule (the LightningModule)
  training/     LightningCLI subclass, MLflow wiring, Rich progress bar, structlog setup,
                param-count summary callback
  inference/    Typer CLI for single-pair and batch prediction
  sweeps/       Optuna runner with nested MLflow runs and PL pruning
  analysis/     data-distribution plot, per-class F1 polar plot, Acetaminophen case study

configs/
  base.yaml                       shared trainer defaults (incl. matmul_precision: medium)
  experiment/                     full runs (drugbank_multiclass, drugbank_binary, biosnap_binary, …)
  experiment/ablations/           MPNN encoder, no-IMAP, tGraphDDI (set-transformer readout)
  sweep/                          Optuna sweep specifications

scripts/
  train.py, sweep.py              entry points
  prepare_test_fixture.py         materialises tests/fixtures/drugbank_mini.csv from TDC
  download_biosnap.py             pulls + splits the CASTER BioSNAP CSVs
  eval_per_class_f1.py            dumps test-set per-class F1 for the polar plot
  data_dist.py, f1_polar.py       figure generators (vector PDF, drop into LaTeX directly)
  case_study.py                   Acetaminophen top-K predicted interactions

weights/
  multiclass.ckpt + multiclass.yaml   shipped multi-class checkpoint
  README.md                            drop your own <task>.ckpt + <task>.yaml here
```

`data/` and `logs/` are gitignored. The only data in the repo is `tests/fixtures/drugbank_mini.csv` (200 pairs across 52 classes); everything else downloads on first run.

## Configs

A run is a single YAML. Everything composes via `class_path` / `init_args`, so the same model definition reuses across datasets by swapping the data block:

```yaml
seed_everything: 42
matmul_precision: medium            # 'medium' enables TF32 on Tensor Cores
trainer:
  accelerator: gpu
  precision: bf16-mixed
  max_epochs: 60
  gradient_clip_val: 1.0
  logger:
    class_path: graphddi.training.MLFlowLogger
    init_args:
      experiment_name: graphddi-drugbank-multiclass
      log_model: false
  callbacks:
    - class_path: graphddi.training.GraphDDIParamSummary
    - class_path: graphddi.training.GraphDDIProgressBar
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: { monitor: val/f1_macro, mode: max, save_top_k: 3, save_last: true }
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args: { monitor: val/f1_macro, mode: max, patience: 10, min_delta: 0.001 }
data:
  class_path: graphddi.data.DrugBankDataModule
  init_args: { task: multiclass, batch_size: 256, oversample_min_fraction: 0.005 }
model:
  class_path: graphddi.models.GraphDDIModule
  init_args:
    encoder:
      class_path: graphddi.models.PNAEncoder
      init_args: { in_channels: 108, hidden_channels: 64, edge_dim: 9, edge_hidden_channels: 128, num_layers: 6, dropout: 0.1 }
    readout:
      class_path: graphddi.models.Set2SetReadout
      init_args: { in_channels: 128, processing_steps: 2, project_to: 128 }
    use_interaction_map: true
    hidden_dim: 64
    mlp_hidden_dim: 128
    dropout: 0.1
    label_smoothing: 0.1
    lr: 0.0003
    weight_decay: 0.01
    warmup_steps: 1000
    max_steps: 100000
```

The MLflow `tracking_uri` and `ModelCheckpoint(dirpath=...)` are filled in automatically at run time — every run goes under `logs/<experiment_name>/<timestamp>/` and tracking goes to `logs/tracking/mlflow.db`. `GraphDDILightningCLI` also links `data.task` and `data.num_classes` into the model at instantiation, so the model is always consistent with the DataModule it's paired with.

## Sweeps

```bash
uv run python scripts/sweep.py --config configs/sweep/drugbank_mini_optuna.yaml
```

A sweep YAML names a base experiment config and a `search_space` of dotted hyperparameter paths. Each trial deep-overrides the base config, instantiates the CLI with `run=False`, attaches a `PyTorchLightningPruningCallback` watching the chosen `monitor`, and reports the final validation metric. MLflow nests every trial under one parent experiment.

Don't sweep shape-coupled hparams like `encoder.hidden_channels` or `use_interaction_map` directly; the downstream `readout.in_channels` and `readout.project_to` must move with them. For architecture exploration, run separate studies with different base configs.

## Figures and analysis

All figures land in `outputs/figures/` as vector PDF, ready to drop into LaTeX with `\includegraphics{...}`.

```bash
# DDI class frequency distribution
uv run python scripts/data_dist.py --out outputs/figures/data_dist.pdf
# (--mini plots from the committed fixture instead of the full TDC dataset)

# Per-class F1 polar plot (one panel per model variant)
for cfg in drugbank_multiclass ablations/mpnn_imap ablations/pna_no_imap ablations/tgraphddi; do
  name=$(basename "$cfg")
  uv run python scripts/eval_per_class_f1.py \
    --config configs/experiment/$cfg.yaml \
    --ckpt <path/to/your/last.ckpt> \
    --out outputs/eval/$name.json
done

uv run python scripts/f1_polar.py \
  --per-class-f1 \
    GraphDDI=outputs/eval/drugbank_multiclass.json \
    "MPNN+IMAP+set2set=outputs/eval/mpnn_imap.json" \
    "PNA+set2set No IMAP=outputs/eval/pna_no_imap.json" \
    tGraphDDI=outputs/eval/tgraphddi.json \
  --out outputs/figures/f1_per_class_polar.pdf

# Case study — Acetaminophen top-K predicted interactions
uv run python scripts/case_study.py
```

Novel predictions from the case study should be cross-checked against drugbank.com / drugs.com to confirm.

## Training details

- **Optimizer.** AdamW, `lr=3e-4`, `weight_decay=1e-2`, **non-fused** (see *Things worth knowing*).
- **Schedule.** Linear warmup (1000 steps) into cosine decay (`SequentialLR`).
- **Precision.** `bf16-mixed` on GPU; `matmul_precision: medium` enables TF32 on Ampere+.
- **Batch size.** 256 for the full DrugBank runs, 128 on a laptop GPU.
- **Gradient clipping.** Global-norm at 1.0.
- **Label smoothing.** 0.1 on the multi-class cross-entropy.
- **Early stopping.** `val/f1_macro` (multi-class, patience 10) or `val/f1` (binary, patience 8); `min_delta` 0.001.
- **Oversampling.** For the multi-class task, each class is brought up to at least `0.5 % × N_train_initial` samples in a single pass over the training split. Val and test are never oversampled.
- **Featurization.** Atom features are 108-dimensional (atom type, chiral tag, degree, formal charge, H count, hybridization, aromaticity, ring membership) and bond features are 9-dimensional (bond type, stereochemistry, conjugation). Hydrogens are not added; this trades a small accuracy hit for a ~10× speedup.
- **DrugBank split.** TDC `random(seed=42, frac=[0.7, 0.1, 0.2])`. TDC doesn't expose a stratified DDI split for this dataset.
- **Metrics.** Pattern-A torchmetrics 1.x — logits go in, the collection applies argmax / softmax internally. The full micro+macro precision/recall/F1 suite plus AUROC and PR-AUC is logged at train/val/test, with `val/loss_best` and `val/f1_macro_best` tracked across epochs.

## Things worth knowing

- **`PNAConv` log-degree buffer.** PNA sizes a log-degree scaler buffer from the `deg` histogram passed at construction time. The encoder ships with a small molecule-shaped default, which is fine for DrugBank/BioSNAP; if you swap to a dataset with a meaningfully different degree distribution, pass an updated `deg` tensor or macro-F1 will silently lag.
- **Readout output width.** `Set2SetReadout` produces `2 × in_channels` natively; with `project_to` set, the trailing `Linear → ReLU` projects to that width instead. The head sizes itself from `readout.out_channels`, so keep `project_to` aligned with `mlp_hidden_dim` if you change either. `SetTransformerReadout` works the same way.
- **`use_interaction_map: false` is a fair-shape ablation.** When IMAP is off, `A'` is set to `A` so `[h | A'] = [h | h]` keeps `readout.in_channels = 2 * hidden_channels`. Hold the readout config identical between IMAP on/off so the two variants stay parameter-matched.
- **Don't enable `AdamW(fused=True)` with `bf16-mixed`.** Lightning's AMP plugin refuses to clip gradients when the fused optimizer unscales internally, so `fused=True` + `bf16-mixed` + `gradient_clip_val` crashes at step 1. The default unfused AdamW is intentional.
- **`*/loss` is shifted by label smoothing.** With `label_smoothing=0.1` on 86 classes the irreducible CE floor is `ε · log K ≈ 0.45`, so a `test/loss` around 0.95 is a near-perfect fit, not a regression. Don't compare loss values across runs that disagree on label smoothing — compare F1 / accuracy instead.
- **`paired_collate` mirrors PyG batching.** The DrugBank/BioSNAP datasets emit `(graph_a, graph_b, label)` triples; the collate function batches each side with `Batch.from_data_list` so `batch_a.batch` and `batch_b.batch` line up for the interaction map.

## Stack

`uv` manages the environment and lockfile (PyTorch is pinned to the CUDA 13.0 wheel index). `ruff` and `ty` enforce style and types (both clean). Lightning runs the training loop, MLflow stores the metrics, Optuna drives the sweeps with mid-trial Hyperband pruning. `structlog` + `rich` handle the console output; `typer` powers the inference CLI.

## Citation

```bibtex
@inproceedings{gupta2024graphddi,
  author    = {Gupta, Suyash and Laghuvarapu, Siddhartha and Priyakumar, U. Deva},
  title     = {{GraphDDI}: Graph Neural Network for Prediction of Drug-Drug Interaction},
  editor    = {Xie, Xianghua and Styles, Iain and Powathil, Gibin and Ceccarelli, Marco},
  booktitle = {Artificial Intelligence in Healthcare (AIiH 2024)},
  series    = {Lecture Notes in Computer Science},
  volume    = {14812},
  pages     = {17--30},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  isbn      = {978-3-031-67278-1},
  doi       = {10.1007/978-3-031-67278-1_2}
}
```

## Acknowledgements

Supported by IHub-Data, IIIT Hyderabad, and DST-SERB (Grant No. CRG/2021/008036).

Built on PyTorch Geometric, PyTorch Lightning, MLflow, Optuna, RDKit, and Therapeutics Data Commons.
