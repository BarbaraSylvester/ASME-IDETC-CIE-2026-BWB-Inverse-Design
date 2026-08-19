# BWB Inverse Design — Project Summary

**Challenge:** AI-Driven Multidisciplinary Design Optimization (nTop × MIT DeCoDE Lab), ASME IDETC/CIE 2026 Student Hackathon
**Language:** Python 3
**Core libraries:** numpy, pandas, scikit-learn, scipy, joblib, matplotlib (all CPU-only, no GPU needed)

---

## 1. What the challenge actually asks for

Given a mission profile (target L/D, minimum payload volume, minimum fuel
volume, flight condition) and a hard structural safety limit (335 MPa max
stress), automatically produce a **21-number design vector** — 10 external
planform parameters (chords, sweep angles, span fractions) and 11 internal
structural parameters (skin thickness, spar/rib placement and thickness) —
that is as light as possible while meeting the stress limit and getting as
close as possible to the mission targets.

Critically, the organizers explicitly forbid bringing in your own CFD or FEA.
You're required to build on **only** what they provide:
- `data/bwb_structures_dataset.csv` — 13,720 pre-computed finite-element
  designs (24 input columns, 4 output columns: mass, payload volume, fuel
  volume, max hotspot stress)
- `models/ld_surrogate/predict_ld.py` — a working, pre-trained aerodynamics
  surrogate (geometry + flight condition → CL, CD, L/D)

So the actual task isn't "do aerospace engineering from scratch" — it's
**"build an inverse-design search on top of surrogates,"** which is a
data/optimization problem, not a physics-simulation problem. That framing
drove every decision below.

---

## 2. Step 1 — Understand what we're working with

Before writing any code, I cloned the repo and inspected the dataset and the
provided surrogate directly (`predict_ld.py --demo`) to confirm it ran and
understand its input/output interface. Then I looked at the distribution of
each target column, because **you can't safely model something you haven't
looked at.**

Mass, payload volume, and fuel volume all turned out to be well-behaved,
roughly continuous, reasonably-scaled numbers — good candidates for a
standard regression model straight out of the box.

**Max Hotspot Stress was not well-behaved.** Its distribution is extremely
right-skewed: median ≈ 228 MPa, but the maximum in the dataset is
**3.9 million MPa** — a few designs with catastrophic, mesh-driven stress
spikes. This matters a lot, because stress is your one *hard* constraint —
if the model that decides feasibility is bad, the whole pipeline is
untrustworthy.

---

## 3. Step 2 — Train the "forward" surrogates

Since we're not allowed to run our own FEA, we need to learn the mapping
`design + flight condition → mass, volumes, stress` directly from the
13,720-row dataset. I used **gradient-boosted trees**
(`HistGradientBoostingRegressor` from scikit-learn) rather than a neural
network, deliberately:

- It's a strong, standard, well-documented choice for tabular data like this
- It needs no GPU, no architecture tuning, and is fast to train (seconds,
  not hours)
- It doesn't require any deep learning background to use correctly —
  reasonable defaults work well out of the box, which matched the "no ML
  background" starting point for this project

**First attempt, naive approach:** train one regressor per target directly
on the raw values. Mass, payload, and fuel volume fit beautifully
(R² > 0.98 on held-out data). **The stress model came back with a *negative*
R²** — meaning it did worse than just guessing the average stress every
time. This is a direct consequence of the skew described above: a
squared-error loss lets a handful of million-MPa outliers dominate training,
so the model contorts itself to reduce error on those few extreme points
instead of learning the shape of the distribution where it actually matters
(near the 335 MPa threshold).

**Fix:** train on `log10(stress)` instead of raw stress, and exponentiate
back at prediction time. In log space the distribution is well-behaved
(std ≈ 0.68 instead of spanning orders of magnitude), and — conveniently —
the 335 MPa threshold sits right in the dense, well-sampled middle of the
distribution (between the 50th and 75th percentile), which is exactly where
we most need the model to be accurate. This brought the log-space R² to
**0.80**.

**Going one step further — a dedicated feasibility classifier.** Since what
the optimizer actually needs is a yes/no answer ("is this design under
335 MPa?"), not a precise stress value, I also trained a direct binary
classifier (`HistGradientBoostingClassifier`) on the feasibility label. This
gives:
1. A comparable accuracy (~87%) to deriving feasibility from the regression
2. A **class probability** (`P(feasible)`), which lets us set a
   **conservative decision threshold** instead of the naive 50/50 cutoff

Because a false "feasible" (an actually-unsafe design slipping through as
valid) is the dangerous error for a hard safety constraint, the optimizer
uses `P(feasible) ≥ 0.75`, not `≥ 0.50`, as its gate. This trades away a bit
of coverage (fewer borderline designs get accepted) for a lower rate of
accidentally validating an unsafe design — the right trade for a stated hard
constraint, and something worth stating explicitly in any write-up since
it's a real methodological choice, not an arbitrary one.

---

## 4. Step 3 — Build the inverse-design optimizer

With forward surrogates in hand (mass, volumes, stress-feasibility, plus the
provided L/D surrogate), the "inverse design" problem becomes: **search the
21-dimensional design space for the point that minimizes mass, subject to
the feasibility gate, while penalizing any shortfall against the L/D and
volume targets.**

**Why differential evolution:** the objective function is a black box built
on tree-ensemble surrogates — it has no usable gradient, may have multiple
local optima, and mixes continuous variables with two integer-constrained
ones (rib count, fuselage spar count) and one odd-integer-only variable
(fuselage rib count, must be one of {3,5,7,9,11}). Differential evolution
(`scipy.optimize.differential_evolution`) is a population-based, gradient-free
global optimizer that handles exactly this kind of messy, mixed search space
well, and it's simple to reason about: maintain a population of candidate
designs, mutate/recombine them, keep what improves, repeat.

**Objective function design.** Rather than a single blended score, the cost
function is built from clearly separable pieces:

```
cost = mass
     + 50  × max(0, L/D_target − L/D_achieved)        # L/D shortfall only
     + 200 × max(0, V_payload_min − V_payload_achieved) # payload shortfall only
     + 200 × max(0, V_fuel_min − V_fuel_achieved)        # fuel shortfall only
     + large_penalty if P(feasible) < 0.75               # hard constraint
```

Two deliberate choices here:
- **Shortfall-only penalties** (via `max(0, ...)`) mean the optimizer is
  never rewarded for *exceeding* a target — it only pays a price for falling
  short. This matches the spec's framing of these as "soft targets" to
  approach, not maximize.
- **The feasibility penalty is scaled by how far below the probability
  threshold a design is**, not just a flat wall. This gives the optimizer a
  gradient to climb back toward feasibility instead of hitting a cliff with
  no directional information.

Integer-constrained variables are searched as continuous values (within their
bounds) and **snapped** to the nearest valid integer (or nearest valid odd
integer, for fuselage ribs) only when the design is evaluated — this keeps
the search space continuous (which differential evolution handles natively)
while still respecting the discrete constraints at evaluation time.

---

## 5. Step 4 — Run it on the three official test cases

The nTop README specifies three fixed mission profiles (High Speed Dash, Max
Endurance, Max Capacity) with a shared 335 MPa stress allowable. Running the
optimizer independently for each produced:

| Case | Mass (kg) | L/D achieved | L/D target | Payload vol | Fuel vol | Feasible prob |
|---|---|---|---|---|---|---|
| 1 – High Speed Dash | 47.6 | 7.79 | 6.0 ✅ | 0.785 / 0.75 ✅ | 0.442 / 0.45 (short) | 0.933 |
| 2 – Max Endurance | 84.7 | 9.83 (short) | 10.0 | 0.807 / 0.80 ✅ | 0.237 / 0.45 (short) | 0.964 |
| 3 – Max Capacity | 104.0 | 15.03 | 15.0 ✅ | 0.855 / 1.00 (short) | 0.403 / 0.65 (short) | 0.773 |

All three designs are feasible under the conservative gate, and mass scales
sensibly with mission demand — the heaviest-duty mission (Case 3) produces
the heaviest structure, which is the kind of physically-plausible pattern
you want to see out of a surrogate-driven search, not proof of correctness
on its own, but a good sanity check. Fuel volume is the target that's
hardest to hit across the board — a real, reportable finding about where
this design space is most constrained.

**Honest caveat:** Case 3's feasibility probability (0.773) sits close to
the 0.75 acceptance threshold. That's worth flagging rather than glossing
over — it means this particular design is closer to the edge of what the
classifier is confident about, not as safely inside the feasible region as
Cases 1 and 2.

---

## 6. Step 5 — Pareto front (mass vs. L/D trade-off)

The rubric explicitly asks for a visualization of the mass/aerodynamic
efficiency trade-off space (part of the 20-point "multidisciplinary
parameter coupling" criterion). A single optimal point per mission doesn't
show *how* mass and L/D trade off against each other — it just shows one
answer.

To generate an actual trade-off curve, I re-ran the optimizer multiple times
per mission, each time changing the *relative weight* on the L/D-shortfall
term in the cost function (from barely caring about L/D to chasing it hard
even at a mass cost). Each run lands at a different point on the mass-vs-L/D
curve; plotting them together approximates the Pareto front for that
mission. This is a standard, legitimate technique for approximating
multi-objective trade-offs using a single-objective optimizer, without
needing to implement a full multi-objective algorithm (e.g. NSGA-II) under
time pressure.

**Honest caveat:** to fit within the compute budget available while building
this, each Pareto point used a much smaller optimizer budget (10 iterations,
population 8) than the main results (40 iterations, population 15), so the
curves in `pareto_front.png` are noisier than they would be with more
compute — you can see this in a couple of non-monotonic points. For a final
submission, re-running the sweep with the same settings as the main
optimization (or higher) would produce a cleaner, more trustworthy curve.
The code (`pareto_front.py`) is already set up to do this — just increase
`maxiter`/`popsize` when you re-run it.

---

## 7. What's genuinely solid vs. what to double check before submitting

**Solid:**
- Mass/payload/fuel surrogates are excellent fits (R² > 0.98)
- The stress-skew diagnosis and log-transform fix is a real, defensible
  methodological finding, not a workaround — worth highlighting in your
  write-up as evidence of "methodological rigor" (20 rubric points)
- The feasibility-classifier-with-conservative-threshold approach directly
  addresses how to treat a hard safety constraint under model uncertainty,
  which is a meaningful design decision, not a default
- All three test cases converge to feasible, physically-plausible designs

**Worth tightening before you submit:**
- Increase `maxiter`/`popsize` in `inverse_design.py` (try 150/25) if you
  have more compute than this sandbox did — the current results are already
  good, but less likely to be sitting exactly at the true optimum
- Re-run the Pareto sweep with a larger budget for a cleaner curve
- Double-check Case 3, whose feasibility probability is closer to the
  acceptance threshold than the other two
- The submission still needs: a 3-page technical summary PDF, and a
  README with exact reproduction commands (this document plus the code
  comments cover the content — happy to help format the PDF next)

---

## 8. How to run this yourself

See `README.md` in this folder for exact setup and run commands. Short
version: this is plain Python 3 (3.10+ recommended), no paid dependencies,
no GPU required — you can run the entire pipeline on a laptop.
