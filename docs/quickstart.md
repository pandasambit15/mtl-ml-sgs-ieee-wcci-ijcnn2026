# Quick Start

This page walks through the minimum steps to go from raw MONC output to a
published comparison figure.

## 1 – Process raw data

```bash
python -m ml_sgs.data.processor \
    --data-dir  /data/monc_output/ \
    --output-dir processed_data/ \
    --sampling-strategy stratified \
    --sampling-fraction 0.5 \
    --n-workers 8
```

## 2 – Train the baseline MLP

```bash
python -m ml_sgs.training.train_baseline \
    --data-dir   processed_data/ \
    --output-dir models/baseline_mlp/ \
    --epochs     200
```

## 3 – Train the Ri-conditioned MLP

```bash
python -m ml_sgs.training.train_ri_conditioned \
    --data-dir   processed_data/ \
    --output-dir models/ri_mlp/ \
    --arch       mlp
```

## 4 – Compare models

```bash
python scripts/compare_models.py \
    --data-dir      /data/monc_output/ \
    --output        results/ \
    --baseline-mlp  models/baseline_mlp/best_model.pt \
    --ri-mlp        models/ri_mlp/best_model.pt \
    --scaler-dir    models/baseline_mlp/ \
    --save-predictions
```

## 5 – Inspect results

```
results/
├── comparison_table.csv   ← paste into your paper
├── metrics_all.json       ← full metrics + best-model summary
└── plots/
    ├── metrics_comparison_r2.png
    ├── scatter_grid_visc_coeff_part1.png
    └── summary_figure.png
```

For interactive exploration, open:

```bash
jupyter notebook notebooks/quickstart.ipynb
```
