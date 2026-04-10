# Training Guide

## Prerequisites

1. Processed training data (see [Data Format](data_format.md))
2. Pre-fitted scalers (`feature_scaler.pkl`, etc.) in the data directory
3. GPU recommended for full-dataset training; CPU works for small runs

---

## Step 1: Process Raw MONC Data

```bash
python -m ml_sgs.data.processor \
    --data-dir /path/to/netcdf_files/ \
    --output-dir processed_data/ \
    --sampling-strategy stratified \
    --sampling-fraction 0.5 \
    --n-workers 8
```

This produces `processed_data/{features,visc_coeff,diff_coeff,richardson,regime}.npy`
and the scaler `.pkl` files.

---

## Step 2: Train Models

### Baseline MLP

```bash
python -m ml_sgs.training.train_baseline \
    --data-dir processed_data/ \
    --output-dir models/baseline_mlp/ \
    --epochs 200 \
    --batch-size 2048 \
    --lr 1e-3
```

### Ri-Conditioned Variants

```bash
python -m ml_sgs.training.train_ri_conditioned \
    --data-dir processed_data/ \
    --output-dir models/ri_conditioned/ \
    --arch mlp          # mlp | resmlp | tab_transformer
    --epochs 200
```

### Q1–Q4 Quadrant Models

Train each quadrant model separately (they can be run in parallel):

```bash
for q in 1 2 3 4; do
    python -m ml_sgs.training.train_q${q}_models \
        --data-dir processed_data/ \
        --output-dir models/q${q}/ \
        --epochs 200 &
done
wait
```

---

## Step 3: Verify Training

Each training script saves:

```
models/baseline_mlp/
├── best_model.pt          # Checkpoint with lowest validation loss
├── final_model.pt         # End-of-training checkpoint
├── training_history.json  # Per-epoch loss components
└── config.yaml            # Hyperparameters used
```

Plot training curves:
```python
import json, matplotlib.pyplot as plt

with open("models/baseline_mlp/training_history.json") as f:
    history = json.load(f)

plt.plot(history["val_loss"], label="Validation")
plt.plot(history["train_loss"], label="Train")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.savefig("training_curve.png")
```

---

## Hyperparameter Reference

All hyperparameters have documented defaults in `configs/default.yaml`.
Pass a custom config with `--config my_config.yaml`; any key not present
in your file falls back to the default.

| Parameter             | Default | Notes                                   |
|-----------------------|---------|-----------------------------------------|
| `batch_size`          | 2048    | Increase for GPU memory > 16 GB         |
| `max_epochs`          | 200     | Early stopping usually triggers earlier |
| `learning_rate`       | 1e-3    | Cosine-annealed to 1e-6                 |
| `dropout`             | 0.3     | Applied to backbone only                |
| `patience`            | 20      | Early stopping patience (epochs)        |
| `grad_clip`           | 1.0     | Max-norm gradient clipping              |
| `loss_weights.visc`   | 1.0     | Relative weight of viscosity MSE        |
| `loss_weights.diff`   | 1.0     | Relative weight of diffusivity MSE      |
| `loss_weights.richardson` | 0.5 | Auxiliary Ri prediction weight        |
| `loss_weights.regime` | 0.3     | Regime classification cross-entropy weight |

---

## Reproducibility

Set `training.seed` in your config (default 42). All random operations
(weight initialisation, data splitting, data augmentation) are seeded.

```yaml
# configs/my_run.yaml
training:
  seed: 123
```
