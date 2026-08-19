# AI-Driven BWB Inverse Design

**ASME IDETC/CIE 2026 Student Hackathon — nTop × MIT DeCoDE Lab Challenge**

An automated inverse design pipeline that takes a Blended Wing Body (BWB)
aircraft's mission profile (target L/D, payload/fuel volume minimums,
flight condition) and directly outputs both the optimal external planform
geometry and internal structural configuration — without running any new
CFD or FEA, built entirely on the organizers' provided dataset and
aerodynamics surrogate.

**Author(s):** Barbara Sylvester (The Tarleton Texans)

\---

## Overview

Modern aircraft design faces a "lock-in trap": teams either iterate fast
with low-fidelity tools and get cartoonish results, or use high-fidelity
tools that are too slow to explore more than one design. This project
addresses that by treating the problem as **surrogate-driven optimization**
rather than physics simulation:

1. Train fast machine-learning surrogates (mass, payload volume, fuel
volume, structural feasibility) on the organizers' 13,720-design
finite-element dataset.
2. Use the organizers' own trained neural-network surrogate for
aerodynamics (L/D).
3. Run a global, gradient-free optimizer (differential evolution) on top
of these surrogates to search the full 21-variable design space for
each mission profile.

This makes each full mission's inverse design a matter of minutes, not
days — while still respecting a hard 335 MPa structural safety limit.

## Why this matters: the hard constraint problem

The organizers validate submitted designs against their own **hidden**
ground-truth analysis — a submission has no way to self-check against the
real answer. A design that actually exceeds the 335 MPa stress limit
scores **zero** for that entire test case, no matter how good everything
else about it is. This shaped the core methodology decision in this
project: feasibility is checked with **two independent, conservative
criteria that must both pass** — a trained classifier's confidence (≥0.92)
*and* a separately trained regressor's point estimate (≤300 MPa, a real
margin below the stated 335 MPa limit) — rather than trusting a single
model's best guess. See `technical\_summary.pdf` for the full reasoning.

## Repository structure

```
├── train\_surrogates.py          # trains mass/volume/stress/feasibility models
├── inverse\_design.py            # main optimizer — run this to reproduce results
├── pareto\_front.py              # mass-vs-L/D trade-off sweep
├── requirements.txt
├── data/
│   └── bwb\_structures\_dataset.csv     # organizers' provided FE dataset (13,720 designs)
├── models/
│   └── ld\_surrogate/                  # organizers' provided L/D neural network + physics
├── results/
│   ├── optimal\_designs.csv            # final optimized design, all 3 test cases
│   ├── optimal\_designs.json
│   └── pareto\_front.png
├── technical\_summary.pdf        # 3-page engineering report (methodology + trade-offs)
└── presentation/
    └── ASME\_BWB\_Presentation.pptx     # Round 1 slides
```

## Setup

Plain Python 3.10+, no GPU required.

```bash
git clone <this-repo-url>
cd <this-repo>

python3 -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

pip install -r requirements.txt
```

## Running the pipeline

Run in this exact order — each step depends on the previous one's output.

```bash
python3 train\_surrogates.py   # \~10-30s — trains and saves surrogates.joblib
python3 inverse\_design.py     # several minutes — runs the optimizer, saves results/optimal\_designs.csv
python3 pareto\_front.py       # several minutes — saves results/pareto\_front.png
```

Expect to see, at each step:

* `train\_surrogates.py`: R² scores for mass/payload/fuel (should be >0.97),
and a feasibility classifier accuracy (\~0.85–0.90). The raw-units R² for
stress will look poor/near-zero — that's expected, not a bug (see
`technical\_summary.pdf`, the stress target is trained in log-space).
* `inverse\_design.py`: mass, L/D, volumes, and feasibility status printed
per mission case, then written to `results/optimal\_designs.csv`.

## Methodology summary

|Component|Method|Why|
|-|-|-|
|Mass / payload / fuel volume surrogates|Gradient-boosted trees (`HistGradientBoostingRegressor`)|Strong, fast, no-GPU baseline for tabular data|
|Stress surrogate|Same, trained on log10(stress)|Raw stress is extremely right-skewed; log-transform fixes a broken fit|
|Feasibility gate|Dedicated classifier + conservative dual-check|Submitted designs can't be self-validated — false "feasible" is the costly error|
|Aerodynamics (L/D)|Organizers' provided trained MLP surrogate|Not allowed to run our own CFD|
|Optimizer|`scipy.optimize.differential\_evolution`|Gradient-free, handles the mixed continuous/integer 21-variable space|

## Results

See `results/optimal\_designs.csv` for the full 21-parameter design per
mission case, and `technical\_summary.pdf` for the mass-vs-L/D Pareto
trade-off analysis and full discussion.

## Acknowledgments

Dataset, L/D surrogate, and challenge design provided by nTop and the MIT
DeCoDE Lab for the ASME IDETC/CIE 2026 Student Hackathon. See the original
challenge repository: https://github.com/nicksungg/nTop---ASME-IDETC-CIE-Student-Hackathon

