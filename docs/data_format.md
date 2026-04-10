# Data Format

## Input: MONC NetCDF Files

The pipeline expects output files from the
[Met Office NERC Cloud model (MONC)](https://code.metoffice.gov.uk/trac/monc).

### Required Variables

| Variable                  | Dimensions                  | Description                          |
|---------------------------|-----------------------------|--------------------------------------|
| `zu`                      | `(time, zn, y, x)`          | u-velocity on scalar grid            |
| `zv`                      | `(time, zn, y, x)`          | v-velocity on scalar grid            |
| `zw`                      | `(time, z, y, x)`           | w-velocity on w-grid                 |
| `zth`                     | `(time, zn, y, x)`          | Potential temperature                |
| `zq_vapour`               | `(time, zn, y, x)`          | Water vapour mixing ratio            |
| `zq_cloud_liquid_mass`    | `(time, zn, y, x)`          | Cloud liquid water mixing ratio      |
| `visc_coeff`              | `(time, z, y, x)`           | **Target**: SGS viscosity coefficient|
| `diff_coeff`              | `(time, z, y, x)`           | **Target**: SGS diffusivity coefficient |

### Required Global Attributes

| Attribute | Description                      |
|-----------|----------------------------------|
| `dx`      | Grid spacing in x-direction (m)  |
| `dy`      | Grid spacing in y-direction (m)  |
| `thref`   | Reference potential temperature (K) |

### Time Dimension

MONC output uses a dynamic time dimension name of the form `time_series_N`
(e.g. `time_series_0`). The processor auto-detects this.

---

## Feature Engineering

The processor extracts **54 features** per grid point from the raw 3D fields.
Feature categories:

| Category              | Count | Examples                                         |
|-----------------------|-------|--------------------------------------------------|
| Velocity gradients    | 9     | `dU/dz`, `dV/dz`, `dW/dx`, `dW/dy`, …           |
| Thermodynamic         | 6     | `dθ/dz`, virtual potential temperature, …        |
| Moisture              | 4     | `q_v`, `q_l`, `dq_v/dz`, …                       |
| Richardson number     | 2     | Ri, saturated Ri                                 |
| Strain rate tensor    | 6     | `S₁₁`, `S₁₂`, `S₁₃`, `S₂₂`, `S₂₃`, `S₃₃`       |
| Deformation invariants| 3     | First, second, third invariants                  |
| Stability parameters  | 6     | Obukhov length, TKE, mixing length, …            |
| Spatial context       | 9     | Height, height², neighbourhood averages, …       |
| Buoyancy              | 5     | Brunt-Väisälä frequency, buoyancy flux, …        |
| Cross terms           | 4     | Ri × shear, moisture × stability, …              |

---

## Processed Numpy Arrays

After running the data processor, the following `.npy` files are saved:

```
processed_data/
├── features.npy           # (N, 54)   float32 – input features
├── visc_coeff.npy         # (N,)      float32 – viscosity target
├── diff_coeff.npy         # (N,)      float32 – diffusivity target
├── richardson.npy         # (N,)      float32 – Richardson number
├── regime.npy             # (N,)      int8    – stability regime (0/1/2)
├── feature_scaler.pkl     #           RobustScaler fitted on training split
├── visc_scaler.pkl        #           RobustScaler for viscosity
├── diff_scaler.pkl        #           RobustScaler for diffusivity
└── richardson_scaler.pkl  #           RobustScaler for Richardson number
```

---

## Prediction Storage (HDF5)

Saved predictions are stored in a compressed HDF5 file:

```
results/predictions/
├── predictions.h5
│   ├── /baseline/MLP/visc_coeff     # (N,)
│   ├── /baseline/MLP/diff_coeff     # (N,)
│   ├── /ri_conditioned/MLP/visc_coeff
│   └── /q1q4/MLP/visc_coeff
└── metadata.json
```

Load predictions with:
```python
from ml_sgs.evaluation import PredictionStorage

store = PredictionStorage("results/predictions/predictions.h5")
preds = store.load("baseline/MLP/visc_coeff")
```
