# weights/

The multi-class checkpoint (`multiclass.ckpt` + `multiclass.yaml`) ships
with the repo and is ready to use via `graphddi.inference.predict`. Binary
checkpoints are not shipped — train your own with `drugbank_binary.yaml`
or `biosnap_binary.yaml` and drop the resulting Lightning checkpoint +
resolved config here. The inference CLI looks up `weights/<task>.{ckpt,yaml}`
by default.

## Layout

```
weights/
├── binary.ckpt        # Lightning checkpoint for the binary model
├── binary.yaml        # the config.yaml from that run
├── multiclass.ckpt    # Lightning checkpoint for the multiclass model
└── multiclass.yaml    # the config.yaml from that run
```

Both files are needed:

- `*.ckpt` — the trained weights (`model.state_dict()` plus Lightning bookkeeping).
- `*.yaml` — the resolved LightningCLI config, including encoder/readout
  `class_path` and `init_args`. Inference reads this to reconstruct the
  module before loading the state dict, so the architecture is captured at
  training time rather than hardcoded on the inference side.

## After a training run

```bash
cp logs/<exp>/<TS>/checkpoints/last.ckpt   weights/multiclass.ckpt
cp logs/<exp>/<TS>/config.yaml             weights/multiclass.yaml
```

(Replace `multiclass` with `binary` for the binary-task models.)

## Sanity-check

```bash
uv run python -m graphddi.inference.predict pair \
  "CC(=O)NC1=CC=C(O)C=C1" "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN1CCCCCC1" \
  --task multiclass --top-k 5
```

If the checkpoint or config is missing the CLI prints an explicit "Drop a
trained run's `last.ckpt` and `config.yaml` into `weights/`" message and
exits with code 2 — no silent failure.

## Pointing at a specific run

Skip the copy step entirely by passing the run dir directly:

```bash
uv run python -m graphddi.inference.predict pair \
  "<smiles1>" "<smiles2>" \
  --task multiclass \
  --checkpoint logs/graphddi-drugbank-multiclass/20260522123045/checkpoints/last.ckpt \
  --config     logs/graphddi-drugbank-multiclass/20260522123045/config.yaml
```
