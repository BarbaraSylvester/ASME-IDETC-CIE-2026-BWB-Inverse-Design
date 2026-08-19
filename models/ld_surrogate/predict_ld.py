#!/usr/bin/env python3
"""L/D surrogate  —  (planform geometry + flight condition) -> CL, CD, L/D.

This is the integrated-aerodynamics forward model for the BWB inverse-design
challenge.  It is a small, torch-free Adam-trained MLP (see ``regressor.py``)
trained on the BlendedNet++ CFD ground truth, wrapped here with the ISA / CAS->
Mach / Reynolds conversion so it can be driven directly from a mission profile.

INPUTS
------
geom : dict with the 9 planform variables (nTop ratio convention) + root chord:
        "B1/C1","B2/C1","B3/C1","C2/C1","C3/C1","C4/C1"   (span/chord ratios)
        "S1","S3"                                          (sweep angles, deg)
        "X3/C1"                                            (outboard break, fraction)
        "C1"                                               (root chord, mm — scale)
alt_kft : altitude               [thousands of feet]
kcas    : calibrated airspeed    [knots]
aoa     : angle of attack        [degrees]

PIPELINE
--------
    ISA(alt)        -> rho, p, a, mu
    KCAS            -> M_inf                     (compressible CAS relation)
    Re_L            =  rho * (M*a) * (C1/1000) / mu
    features_aero   -> 12-vector [9 scaled geom, Re_L, M_inf, alpha]
    NumpyMLP        -> (CL, CD)
    L/D             =  CL / CD

C1 does NOT enter the aero feature vector directly — geometry is evaluated at a
fixed 1000 mm reference chord — so C1 reaches aerodynamics ONLY through Re_L.
It is therefore a near-pure scale knob: it buys payload/fuel volume and costs
mass, while perturbing L/D only weakly through Reynolds number.

USAGE
-----
    from predict_ld import predict_ld
    r = predict_ld(
        {"B1/C1":0.15,"B2/C1":0.12,"B3/C1":0.52,"C2/C1":0.70,"C3/C1":0.23,
         "C4/C1":0.075,"S1":50.0,"S3":30.0,"X3/C1":0.575,"C1":3000.0},
        alt_kft=15.0, kcas=180.0, aoa=6.0)
    print(r["LD"], r["CL"], r["CD"])

    # or from the command line:
    python predict_ld.py --demo
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regressor as aero              # noqa: E402  (pure-numpy MLP + checkpoint loader)
import flight_conversion as FC        # noqa: E402  (ISA, CAS->Mach, feature assembly)

# The 9 planform keys the aero model consumes, plus C1 (used only for Re_L).
GEOM_KEYS = FC.GEOM_ORDER_B + ["C1"]

_MODEL, _STD = aero.load("full")      # loads reg_full.json sitting beside this file


def predict_ld(geom: dict, alt_kft: float, kcas: float, aoa: float) -> dict:
    """One design at one flight condition -> {CL, CD, LD, Re_L, M_inf, warnings}."""
    missing = [k for k in GEOM_KEYS if k not in geom]
    if missing:
        raise KeyError(f"geom is missing keys: {missing}")

    fs = FC.convert(alt_kft, kcas, aoa, C1_mm=float(geom["C1"]))
    feats = np.asarray(FC.features_aero(geom, fs), dtype=np.float64)   # 12 features
    geom9, flight3 = feats[:9], feats[9:]
    CL, CD = aero.predict(_MODEL, _STD, geom9, flight3)[0]
    return {
        "CL": float(CL),
        "CD": float(CD),
        "LD": float(CL / CD),
        "Re_L": float(fs.Re_L),
        "M_inf": float(fs.M_inf),
        "warnings": FC.check_ranges(fs),      # non-empty => extrapolating outside trained envelope
    }


def predict_ld_batch(rows, alt_kft, kcas, aoa) -> np.ndarray:
    """Vectorised L/D over many designs at ONE flight condition.

    `rows` is an (N, 10) array whose columns are GEOM_KEYS in order, OR a
    pandas DataFrame containing those columns. Returns an (N,) array of L/D.
    Efficient because at a fixed mission the atmosphere is constant and
    Re_L is linear in C1.
    """
    try:
        import pandas as pd
        if isinstance(rows, pd.DataFrame):
            rows = rows[GEOM_KEYS].to_numpy(dtype=np.float64)
    except ImportError:
        pass
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))

    # scale the six length ratios to the 1000 mm reference chord; S1/S3/X3 pass through
    scaled = rows[:, :9].copy()
    scaled[:, :6] *= FC.AERO_C1_REF_MM
    C1 = rows[:, 9]

    # atmosphere is mission-fixed; Re_L = Re_coef * C1
    fs = FC.convert(alt_kft, kcas, aoa, C1_mm=1000.0)
    Re_coef = fs.rho * fs.V_true / (fs.mu * 1000.0)
    ReL = Re_coef * C1
    flight3 = np.column_stack([ReL, np.full_like(ReL, fs.M_inf), np.full_like(ReL, aoa)])
    out = aero.predict(_MODEL, _STD, scaled, flight3)     # (N,2) = CL, CD
    return out[:, 0] / out[:, 1]


def _demo():
    missions = [
        dict(name="cruise",  alt_kft=15.0, kcas=180.0, aoa=6.0),
        dict(name="loiter",  alt_kft=8.0,  kcas=140.0, aoa=4.0),
        dict(name="dash",    alt_kft=12.0, kcas=220.0, aoa=2.0),
    ]
    geom = {"B1/C1": 0.15, "B2/C1": 0.12, "B3/C1": 0.52, "C2/C1": 0.70,
            "C3/C1": 0.23, "C4/C1": 0.075, "S1": 50.0, "S3": 30.0,
            "X3/C1": 0.575, "C1": 3000.0}
    print("L/D surrogate demo — one planform across three flight conditions\n")
    print(f"  geom: {geom}\n")
    for m in missions:
        r = predict_ld(geom, m["alt_kft"], m["kcas"], m["aoa"])
        w = "  [!] " + "; ".join(r["warnings"]) if r["warnings"] else ""
        print(f"  {m['name']:7} alt {m['alt_kft']:5.1f} kft  KCAS {m['kcas']:5.1f}  "
              f"AoA {m['aoa']:4.1f}  ->  CL {r['CL']:+.4f}  CD {r['CD']:.5f}  "
              f"L/D {r['LD']:6.2f}   (Re_L {r['Re_L']:.2e}, M {r['M_inf']:.3f}){w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BWB L/D surrogate (geometry + flight -> L/D)")
    ap.add_argument("--demo", action="store_true", help="run three example missions")
    ap.add_argument("--alt_kft", type=float); ap.add_argument("--kcas", type=float)
    ap.add_argument("--aoa", type=float)
    for k in ["B1/C1", "B2/C1", "B3/C1", "C2/C1", "C3/C1", "C4/C1", "S1", "S3", "X3/C1", "C1"]:
        ap.add_argument("--" + k.replace("/", "_"), type=float)
    a = ap.parse_args()

    if a.demo or a.alt_kft is None:
        _demo()
    else:
        geom = {k: getattr(a, k.replace("/", "_")) for k in GEOM_KEYS}
        r = predict_ld(geom, a.alt_kft, a.kcas, a.aoa)
        import json
        print(json.dumps(r, indent=2))
