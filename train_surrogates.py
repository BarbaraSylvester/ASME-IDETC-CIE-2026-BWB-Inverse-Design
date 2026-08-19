"""
Train surrogate models for the outputs NOT already covered by the provided
L/D surrogate: Aircraft Empty Weight (mass), Payload Volume, Fuel Volume,
and Max Hotspot Stress.

Input features (24): 10 planform + 3 flight condition + 11 structural
Output targets (4): mass, payload volume, fuel volume, max hotspot stress

We use gradient-boosted trees (HistGradientBoostingRegressor) — a strong,
fast, no-GPU-needed choice for tabular data like this, and one that doesn't
require any deep learning background to use or reason about.

Run: python train_surrogates.py
Produces: surrogates.joblib (dict of trained models + feature/target lists)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import joblib
import os

STRESS_LIMIT_MPA = 335.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "bwb_structures_dataset.csv")

FEATURE_COLS = [
    # planform (10)
    "C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1", "B3/C1", "X3/C1", "S1", "S3", "C1",
    # flight condition (3)
    "Altitude", "KCAS", "AOA",
    # structure (11)
    "Skin Thickness", "Front Spar Chord %", "Rear Spar Chord %", "Spar Thickness",
    "# of Ribs", "Rib Thickness", "Wingbox Cutout", "# of Fuselage Ribs",
    "# of Fuselage Spars", "Fuselage Struct Thickness", "Fuselage Struct Width",
]

TARGET_COLS = ["Aircraft Empty Weight", "Payload Volume", "Fuel Volume", "Max Hotspot Stress"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    models = {}

    # Max Hotspot Stress is extremely right-skewed (median ~228 MPa, max ~3.9M MPa,
    # driven by a heavy tail of highly-stressed outlier designs). Training a
    # regressor directly on raw stress lets a few huge outliers dominate the loss,
    # which is why it fit worse than the mean (negative R2). We fit log10(stress)
    # instead, which is well-behaved, and exponentiate back at prediction time.
    # This matters because the 335 MPa threshold sits in the dense, well-sampled
    # part of the distribution (between the 50th and 75th percentile) -- exactly
    # where we need the surrogate to be accurate for the hard constraint check.
    LOG_TARGETS = {"Max Hotspot Stress"}

    print(f"{'target':<25}{'R2':>10}{'MAE':>16}")
    for target in TARGET_COLS:
        y_raw = df[target].values
        y = np.log10(y_raw) if target in LOG_TARGETS else y_raw

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.08, random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # report R2/MAE in the ORIGINAL units (undo log for reporting only)
        if target in LOG_TARGETS:
            pred_report, y_test_report = 10 ** pred, 10 ** y_test
        else:
            pred_report, y_test_report = pred, y_test
        r2 = r2_score(y_test_report, pred_report)
        mae = mean_absolute_error(y_test_report, pred_report)
        print(f"{target:<25}{r2:>10.4f}{mae:>16.3f}")

        # refit on ALL data for the final deployed model (more data = better surrogate)
        model_full = HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.08, random_state=42
        )
        model_full.fit(X, y)
        models[target] = model_full

    # --- Dedicated feasibility classifier -----------------------------------
    # The regression-derived feasibility check (predicted stress <= 335) gets
    # ~87% accuracy on held-out data. A classifier trained directly on the
    # binary feasibility label performs comparably, but exposes a class
    # probability we can threshold conservatively -- e.g. only accepting a
    # design as "feasible" if P(feasible) >= 0.75, not just >= 0.5. Since a
    # false "feasible" is the dangerous error (an invalid design reported as
    # valid), we use this classifier with a conservative threshold as the
    # hard-constraint gate in the optimizer, and use the log-regression above
    # only to report an estimated stress *value* for context.
    y_feas = (df["Max Hotspot Stress"].values <= STRESS_LIMIT_MPA).astype(int)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X, y_feas, test_size=0.15, random_state=42, stratify=y_feas
    )
    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
    clf.fit(Xf_train, yf_train)
    pred_f = clf.predict(Xf_test)
    print(f"\n{'Feasibility classifier':<25}{'acc':>10}{'F1':>16}")
    print(f"{'(stress<=335 MPa)':<25}{accuracy_score(yf_test, pred_f):>10.4f}{f1_score(yf_test, pred_f):>16.4f}")

    clf_full = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
    clf_full.fit(X, y_feas)
    models["Feasibility"] = clf_full

    out_path = os.path.join(HERE, "surrogates.joblib")
    joblib.dump(
        {
            "models": models,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "log_targets": LOG_TARGETS,
            "stress_limit_mpa": STRESS_LIMIT_MPA,
        },
        out_path,
    )
    print(f"\nSaved surrogates -> {out_path}")


if __name__ == "__main__":
    main()
