# Model Architecture

## Overview

The framework implements three neural network **backbone architectures**, each
available in two **conditioning variants** and four **quadrant configurations**,
giving 3 × (2 + 4) = **18 trained model configurations** in total.

---

## Backbone Architectures

### MLP

A three-hidden-layer feedforward network with ReLU activations and dropout.

```
Input (54)  →  Linear(256)  →  ReLU  →  Dropout(0.3)
            →  Linear(512)  →  ReLU  →  Dropout(0.3)
            →  Linear(256)  →  ReLU  →  Dropout(0.3)
            →  Prediction heads
```

### ResMLP

Residual MLP with layer normalisation.

```
Input (54)  →  Linear(256)  →  ReLU
            →  ResidualBlock × 4   (256 → 256, LayerNorm)
            →  Prediction heads
```

Each `ResidualBlock` applies `LayerNorm(x + layers(x))`.

### TabTransformer

Attention-based architecture treating each input feature as a token.

```
Input (54)  →  FeatureEmbedding: (54, 1) → (54, 32)
            →  TransformerBlock × 4
            →  Flatten → (54 × 32 = 1728)
            →  Prediction heads
```

---

## Prediction Heads

All architectures share a common four-head output structure:

| Head         | Input dim    | Output dim | Task                               |
|--------------|-------------|------------|-------------------------------------|
| `visc_head`  | 256 (+1 Ri) | 1          | SGS eddy viscosity coefficient      |
| `diff_head`  | 256 (+1 Ri) | 1          | SGS eddy diffusivity coefficient    |
| `ri_head`    | 256          | 1          | Richardson number (auxiliary)       |
| `regime_head`| 256 (+1 Ri) | 3          | Stability regime (stable/neutral/unstable) |

---

## Conditioning Variants

### Baseline (no physics conditioning)

All four heads receive the same backbone features independently.

```
backbone_features → visc_head  → ŷ_visc
                 → diff_head  → ŷ_diff
                 → ri_head    → ŷ_Ri
                 → regime_head→ ŷ_regime
```

### Richardson-Number Cascade Conditioning

The Richardson number is predicted **first** and then concatenated to the
backbone features before the coefficient heads, enforcing a physically
motivated information flow.

```
backbone_features → ri_head         → ŷ_Ri
[backbone; ŷ_Ri]  → visc_head       → ŷ_visc
[backbone; ŷ_Ri]  → diff_head       → ŷ_diff
[backbone; ŷ_Ri]  → regime_head     → ŷ_regime
```

The coefficient head input dimension increases by 1 (from 256 to 257).

---

## Quadrant (Q1–Q4) Configurations

Samples are pre-assigned to one of four stability quadrants based on the
Richardson number and buoyancy flux sign. A separate model is trained per
quadrant, allowing regime-specific learning without explicit conditioning.

| Quadrant | Description                |
|----------|----------------------------|
| Q1       | Strongly stable            |
| Q2       | Weakly stable / near-neutral |
| Q3       | Weakly unstable            |
| Q4       | Strongly convective        |

At inference time, the `Q1Q4UnifiedEngine` routes each sample to its
corresponding quadrant model.

---

## Parameter Counts (approximate)

| Variant              | Parameters |
|----------------------|-----------|
| Baseline MLP         | ~560 k    |
| Ri-conditioned MLP   | ~565 k    |
| Baseline ResMLP      | ~690 k    |
| Ri-conditioned ResMLP| ~695 k    |
| Baseline TabTransformer | ~420 k |
| Ri-conditioned TabTransformer | ~425 k |

Ri-conditioning adds roughly **5 k parameters** per architecture
(+1 input neuron × 3 heads × 128 neurons).
