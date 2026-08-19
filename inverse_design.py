"""
BWB Inverse Design — optimization pipeline.

Given a mission profile (L/D target, payload/fuel volume minima, flight
condition), search the 21-variable design space (10 planform + 11 structural)
for the design that:
  - satisfies the hard constraint: predicted max hotspot stress <= 335 MPa
  - minimizes structural mass
  - gets as close as possible to the L/D target and the volume minima

Method: differential evolution (a population-based global optimizer from
scipy). It doesn't need gradients, handles the mixed continuous/near-integer
search space fine, and is simple to reason about -- a good fit for a
black-box objective built on top of ML surrogates.

Usage:
    python inverse_design.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models", "ld_surrogate"))
import predict_ld as ld_surrogate  # noqa: E402

# ---------------------------------------------------------------------------
# Design variable bounds (from the dataset / README, verified against the CSV)
# ---------------------------------------------------------------------------
# order: 10 planform, 11 structural  (flight condition is fixed per test case,
# not something we're designing, so it's not part of the search vector)
PLANFORM_VARS = ["C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1", "B3/C1", "X3/C1", "S1", "S3", "C1"]
STRUCT_VARS = [
    "Skin Thickness", "Front Spar Chord %", "Rear Spar Chord %", "Spar Thickness",
    "# of Ribs", "Rib Thickness", "Wingbox Cutout", "# of Fuselage Ribs",
    "# of Fuselage Spars", "Fuselage Struct Thickness", "Fuselage Struct Width",
]
DESIGN_VARS = PLANFORM_VARS + STRUCT_VARS

BOUNDS = {
    "C2/C1": (0.55, 0.85), "C3/C1": (0.18, 0.28), "C4/C1": (0.06, 0.09),
    "B1/C1": (0.10, 0.20), "B2/C1": (0.05, 0.20), "B3/C1": (0.35, 0.70),
    "X3/C1": (0.50, 0.65), "S1": (40.0, 60.0), "S3": (20.0, 40.0), "C1": (2500.0, 4000.0),
    "Skin Thickness": (0.0003, 0.0050),
    "Front Spar Chord %": (0.18, 0.35),
    "Rear Spar Chord %": (0.55, 0.75),
    "Spar Thickness": (0.00098, 0.0080),
    "# of Ribs": (3, 14),                 # integer
    "Rib Thickness": (0.0015, 0.015),
    "Wingbox Cutout": (0.01, 0.05),
    "# of Fuselage Ribs": (3, 11),        # odd integer only: {3,5,7,9,11}
    "# of Fuselage Spars": (3, 12),       # integer
    "Fuselage Struct Thickness": (0.002, 0.025),
    "Fuselage Struct Width": (0.0010, 0.015),
}
INTEGER_VARS = {"# of Ribs", "# of Fuselage Spars"}
ODD_INTEGER_VARS = {"# of Fuselage Ribs"}

STRESS_LIMIT_MPA = 335.0

# ---------------------------------------------------------------------------
# Test cases (from the nTop README, exact values)
# ---------------------------------------------------------------------------
TEST_CASES = {
    "Case 1 - High Speed Dash": dict(LD_target=6.0, V_payload_min=0.75, V_fuel_min=0.45,
                                      altitude_kft=15, kcas=120, aoa=1.0),
    "Case 2 - Max Endurance":   dict(LD_target=10.0, V_payload_min=0.80, V_fuel_min=0.45,
                                      altitude_kft=15, kcas=45, aoa=8.0),
    "Case 3 - Max Capacity":    dict(LD_target=15.0, V_payload_min=1.00, V_fuel_min=0.65,
                                      altitude_kft=5, kcas=220, aoa=4.5),
}

# ---------------------------------------------------------------------------
def snap_design_vector(x):
    """Round the integer-constrained variables to their valid values."""
    d = dict(zip(DESIGN_VARS, x))
    d["# of Ribs"] = int(round(d["# of Ribs"]))
    d["# of Fuselage Spars"] = int(round(d["# of Fuselage Spars"]))
    # odd-only: snap to nearest odd integer in {3,5,7,9,11}
    odd_choices = np.array([3, 5, 7, 9, 11])
    d["# of Fuselage Ribs"] = int(odd_choices[np.argmin(np.abs(odd_choices - d["# of Fuselage Ribs"]))])
    return d


class Surrogates:
    def __init__(self, path=os.path.join(HERE, "surrogates.joblib")):
        bundle = joblib.load(path)
        self.models = bundle["models"]
        self.feature_cols = bundle["feature_cols"]
        self.log_targets = bundle["log_targets"]

    def predict(self, design_dict, altitude_kft, kcas, aoa):
        row = {**design_dict, "Altitude": altitude_kft, "KCAS": kcas, "AOA": aoa}
        x = np.array([[row[c] for c in self.feature_cols]])
        out = {}
        for target in ["Aircraft Empty Weight", "Payload Volume", "Fuel Volume", "Max Hotspot Stress"]:
            pred = self.models[target].predict(x)[0]
            out[target] = 10 ** pred if target in self.log_targets else pred
        out["Feasible_prob"] = self.models["Feasibility"].predict_proba(x)[0, 1]
        return out


SURR = Surrogates()


def evaluate_design(x, mission, feasibility_threshold=0.92, stress_margin_mpa=300.0, return_details=False):
    """Full evaluation of a design vector against a mission profile.
    Returns a scalar penalty/cost (lower is better) for the optimizer,
    or a details dict if return_details=True.

    IMPORTANT — updated per the organizers' clarification: submitted designs
    are re-evaluated by the organizers' own hidden ground-truth analysis, not
    our surrogates, and a design that exceeds the 335 MPa stress limit scores
    ZERO for that entire test case. Since we cannot self-validate against
    their real analysis, feasibility is checked TWO independent ways, both of
    which must pass:
      1. The feasibility CLASSIFIER's probability must clear a high bar
         (feasibility_threshold=0.92, raised from an earlier, less
         conservative 0.75) -- not just "more likely than not" feasible.
      2. The stress REGRESSOR's point estimate must independently sit below
         a hard buffer (stress_margin_mpa=300, i.e. ~10% below the stated
         335 MPa allowable) -- not just "the classifier is confident."
    Requiring agreement from two different models trained on the same data
    is materially stronger evidence than trusting either alone, and directly
    answers the organizers' explicit ask to make stress estimates
    conservative given we can't validate them ourselves.
    """
    design = snap_design_vector(x)

    geom = {k: design[k] for k in ["B1/C1", "B2/C1", "B3/C1", "C2/C1", "C3/C1", "C4/C1", "S1", "S3", "X3/C1", "C1"]}
    aero = ld_surrogate.predict_ld(geom, alt_kft=mission["altitude_kft"], kcas=mission["kcas"], aoa=mission["aoa"])
    struct = SURR.predict(design, mission["altitude_kft"], mission["kcas"], mission["aoa"])

    mass = struct["Aircraft Empty Weight"]
    payload_vol_m3 = struct["Payload Volume"] / 1e9
    fuel_vol_m3 = struct["Fuel Volume"] / 1e9
    feasible_prob = struct["Feasible_prob"]
    stress_est_mpa = struct["Max Hotspot Stress"]
    ld = aero["LD"]

    # --- hard constraint: TWO independent conservative checks, both must pass
    classifier_fails = feasible_prob < feasibility_threshold
    regressor_fails = stress_est_mpa > stress_margin_mpa
    infeasible = classifier_fails or regressor_fails

    # Heavy penalty for infeasible designs, scaled by how far each check is
    # from passing (gives the optimizer gradient information to climb back
    # toward feasibility instead of hitting a flat wall). Both terms are
    # included whenever their respective check fails, so a design failing
    # both checks gets a proportionally larger penalty.
    feas_penalty = 0.0
    if classifier_fails:
        feas_penalty += 1e6 * (feasibility_threshold - feasible_prob)
    if regressor_fails:
        feas_penalty += 1e4 * (stress_est_mpa - stress_margin_mpa)

    # --- soft targets: shortfall-only penalties (no reward for exceeding) --
    ld_shortfall = max(0.0, mission["LD_target"] - ld)
    payload_shortfall = max(0.0, mission["V_payload_min"] - payload_vol_m3)
    fuel_shortfall = max(0.0, mission["V_fuel_min"] - fuel_vol_m3)

    # Weights chosen so shortfalls are penalized meaningfully relative to
    # typical mass values (tens-hundreds of kg), tunable in the technical writeup.
    cost = (
        mass
        + 50.0 * ld_shortfall
        + 200.0 * payload_shortfall     # volumes are O(1) m^3, scale up
        + 200.0 * fuel_shortfall
        + feas_penalty
    )

    if return_details:
        return dict(design=design, mass=mass, LD=ld, payload_vol_m3=payload_vol_m3,
                    fuel_vol_m3=fuel_vol_m3, feasible_prob=feasible_prob,
                    stress_est_mpa=stress_est_mpa, classifier_fails=classifier_fails,
                    regressor_fails=regressor_fails,
                    infeasible=infeasible, cost=cost,
                    ld_shortfall=ld_shortfall, payload_shortfall=payload_shortfall,
                    fuel_shortfall=fuel_shortfall)
    return cost


def _objective(x, mission):
    """Module-level (picklable) objective wrapper for parallel workers."""
    return evaluate_design(x, mission)


def optimize_for_mission(mission, seed=42, maxiter=150, popsize=25, workers=1):
    bounds_list = [BOUNDS[v] for v in DESIGN_VARS]
    result = differential_evolution(
        _objective,
        bounds_list,
        args=(mission,),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-7,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=True,
        workers=workers,        # >1 needs the surrogates to be re-loaded per worker; keep 1 unless profiled
        updating="immediate" if workers == 1 else "deferred",
    )
    details = evaluate_design(result.x, mission, return_details=True)
    return details


def main():
    all_results = []
    for name, mission in TEST_CASES.items():
        print(f"\nOptimizing: {name} ...")
        details = optimize_for_mission(mission)
        d = details["design"]
        row = {
            "Case": name,
            **{f"design.{k}": v for k, v in d.items()},
            "Mass_kg": details["mass"],
            "LD_achieved": details["LD"],
            "LD_target": mission["LD_target"],
            "Payload_Volume_m3": details["payload_vol_m3"],
            "Payload_Volume_min_m3": mission["V_payload_min"],
            "Fuel_Volume_m3": details["fuel_vol_m3"],
            "Fuel_Volume_min_m3": mission["V_fuel_min"],
            "Estimated_Stress_MPa": details["stress_est_mpa"],
            "Stress_Limit_MPa": 335.0,
            "Feasible_prob": details["feasible_prob"],
            "Classifier_Check_Failed": details["classifier_fails"],
            "Regressor_Margin_Check_Failed": details["regressor_fails"],
            "Infeasible_flag": details["infeasible"],
        }
        all_results.append(row)
        print(f"  mass={details['mass']:.1f} kg | L/D={details['LD']:.2f} (target {mission['LD_target']}) "
              f"| payload={details['payload_vol_m3']:.3f} m3 (min {mission['V_payload_min']}) "
              f"| fuel={details['fuel_vol_m3']:.3f} m3 (min {mission['V_fuel_min']}) "
              f"| feasible_prob={details['feasible_prob']:.3f} "
              f"| est_stress={details['stress_est_mpa']:.1f} MPa (margin limit 300)")

    df = pd.DataFrame(all_results)
    out_csv = os.path.join(HERE, "optimal_designs.csv")
    df.to_csv(out_csv, index=False)
    with open(os.path.join(HERE, "optimal_designs.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results -> {out_csv}")


if __name__ == "__main__":
    main()