# L/D Surrogate — integrated aerodynamics

A small, **torch-free** MLP that maps a BWB planform + a flight condition to the
integrated aerodynamic coefficients and lift-to-drag ratio:

```
(9 planform variables + root chord C1) + (altitude, KCAS, AoA)  ->  CL, CD, L/D
```

It is the forward model you use to **hit `L/D_target`** in the inverse-design loop.
Because it is pure NumPy it imports in milliseconds and vectorises over a whole
optimizer population as a single matmul.

## Files

| File | Purpose |
|---|---|
| `predict_ld.py` | **Start here.** Clean wrapper: `predict_ld(geom, alt_kft, kcas, aoa)` and a batched `predict_ld_batch(...)`. Run `python predict_ld.py --demo`. |
| `regressor.py` | The `NumpyMLP` architecture + checkpoint loader + trainer. |
| `reg_full.json` | Trained weights (trained on the **full** dataset — most accurate). Loaded by default. |
| `reg_feasible.json` | Alternate weights trained on the feasible subset only. |
| `flight_conversion.py` | ISA atmosphere, compressible CAS→Mach, `Re_L`, and feature assembly. |
| `aero_design_space.json` | Planform + flight bounds of the aero training data (search box). |

## The conversion (why C1 is special)

Geometry is evaluated at a **fixed 1000 mm reference chord**, so the root chord
`C1` never enters the aero feature vector directly. It reaches aerodynamics
**only through Reynolds number**:

```
ISA(altitude)          -> rho, p, a, mu
KCAS                   -> M_inf              (compressible CAS relation)
Re_L = rho * (M*a) * (C1/1000) / mu
features = [ 6 length ratios × 1000,  S1, S3, X3/C1,  Re_L, M_inf, alpha ]  (12)
NumpyMLP(features)     -> (CL, CD)  ;  L/D = CL / CD
```

That makes `C1` a **near-pure scale knob**: it buys payload/fuel volume and costs
structural mass, while perturbing L/D only weakly through Re. Closing that
`C1 → Re → L/D` loop is exactly the multidisciplinary coupling the challenge asks
you to handle.

## Usage

```python
from predict_ld import predict_ld

geom = {"B1/C1":0.15, "B2/C1":0.12, "B3/C1":0.52, "C2/C1":0.70, "C3/C1":0.23,
        "C4/C1":0.075, "S1":50.0, "S3":30.0, "X3/C1":0.575, "C1":3000.0}

r = predict_ld(geom, alt_kft=15.0, kcas=180.0, aoa=6.0)
print(r["LD"], r["CL"], r["CD"])          # e.g. 12.20  0.2032  0.01665
print(r["warnings"])                       # non-empty => you are extrapolating
```

Batched over many designs at one flight condition (fast — atmosphere is fixed and
`Re_L` is linear in `C1`):

```python
import pandas as pd
from predict_ld import predict_ld_batch
df = pd.read_csv("../../data/bwb_structures_dataset.csv")
ld = predict_ld_batch(df, alt_kft=15.0, kcas=180.0, aoa=6.0)   # (N,) array
```

Geometry keys use the **nTop ratio convention** — the six length ratios
(`B1/C1 … C4/C1`) are dimensionless, `S1/S3` are sweep angles in degrees, `X3/C1`
is a dimensionless fraction (~0.5–0.65), and `C1` is the true root chord in mm.

## Trained envelope

Predictions outside these ranges are extrapolation (`predict_ld` returns a
`warnings` list when you cross them):

- `Re_L` ∈ [5.1e4, 1.0e8],  `M_inf` ∈ [0.05, 0.50],  `alpha` ∈ [−8°, 16°]
- planform bounds in `aero_design_space.json`

`aero_design_space.json` covers a slightly wider box than the shipped structures
CSV — its `S3` ceiling is ≈45° where the CSV stops at 40° — and its
`n_feasible = 3860` is unrelated to the structural 335 MPa allowable.

Every structures-CSV design sits inside the **geometry** envelope, so planform
never triggers a warning. The **flight** envelope is the one to watch: evaluating
CSV rows at their own flight conditions warns on ~0.6 % of them, always
`M_inf` marginally over the 0.50 ceiling (fast, low-altitude points). Check the
`warnings` list rather than assuming a mission is in-envelope.

## Retraining

```bash
python regressor.py --data full --epochs 4000        # writes reg_full.json
```
(Requires the `regressor_data_full.npz` produced at data-prep time; the shipped
`reg_full.json` is ready to use as-is.)
