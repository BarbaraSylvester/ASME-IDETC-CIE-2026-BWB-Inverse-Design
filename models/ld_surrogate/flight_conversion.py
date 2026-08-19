"""
flight_conversion.py -- mission profile -> surrogate feature vectors.

Mission inputs (the only things the profile gives you):
    alt_kft, V_CAS (kt), alpha (deg)   [+ L/D_target, V_payload_min, V_fuel_min]

  Consumer A -- 3 structural surrogates, 24 features, NO flight conversion.
  Consumer B -- aero regressor, 12 features, flight conversion REQUIRED.

Reference length for Re_L is the root chord C1, in mm, so L = C1_mm / 1000.
Validated against bnpp/data/case_with_geom_params.csv: back-solving L from the
published (alt_kft, Re_L, M_inf) triples gives median 1.0011 m against the
declared C1_constant of 1000 mm, with corr(log10 L, alt) = -0.0002 -- i.e. this
atmosphere model reproduces the one used to generate the CFD cases.

GEOMETRY CONVENTION
-------------------
The canonical design vector uses the *nTop ratio* convention, matching both the
dataset columns and metadata_v2.json bounds:

    C2/C1 C3/C1 C4/C1 B1/C1 B2/C1 B3/C1 X3/C1   (dimensionless)
    S1 S3                                        (degrees)
    C1                                           (mm, the true root chord)

Consumer A consumes those ratios directly. Consumer B needs lengths in mm at a
FIXED 1000 mm reference chord, so the six length ratios are scaled by 1000.

X3 is dimensionless in BOTH models (~0.5-0.65) -- it is never scaled. Note that
bnpp/norm_stats.json ships a shape_mean whose 9th entry is 5.75e-4, i.e. the
field models divided X3 by 1000 anyway; do not copy that convention.

C1 does NOT appear in the aero feature vector. It reaches aerodynamics only
through Re_L, which is why it behaves as a near-pure scale knob: it buys
payload/fuel volume and mass at almost no L/D cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants (ICAO Standard Atmosphere, 1976)
# ---------------------------------------------------------------------------
G0      = 9.80665        # m/s^2      standard gravity
R_AIR   = 287.05287      # J/(kg*K)   specific gas constant, dry air
GAMMA   = 1.4            # -          ratio of specific heats

T0_SL   = 288.15         # K          sea-level temperature
P0_SL   = 101325.0       # Pa         sea-level pressure
RHO0_SL = 1.225          # kg/m^3     sea-level density
A0_SL   = 340.294        # m/s        sea-level speed of sound

LAPSE   = 0.0065         # K/m        troposphere lapse rate
H_TROP  = 11000.0        # m          tropopause
T_TROP  = 216.65         # K          temperature above tropopause
P_TROP  = 22632.06       # Pa         pressure at tropopause

MU_REF  = 1.716e-5       # Pa*s       Sutherland reference viscosity
T_REF   = 273.15         # K          Sutherland reference temperature
S_SUTH  = 110.4          # K          Sutherland constant

FT_TO_M  = 0.3048
KT_TO_MS = 0.514444444444  # 1 knot = 1852 m / 3600 s

# Trained ranges -- inputs outside these are extrapolation, not prediction.
RANGE_A = {  # structural surrogates, raw mission units
    "Altitude_kft": (0.0015, 18.00),
    "KCAS_kt":      (33.02, 249.95),
    "AOA_deg":      (-8.00, 16.00),
}
RANGE_B = {  # aero regressor, converted units
    "Re_L":      (5.07e4, 1.00e8),
    "M_inf":     (0.0500, 0.5000),
    "alpha_deg": (-8.0, 16.0),
}


# ---------------------------------------------------------------------------
# 1. Atmosphere
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Atmosphere:
    """ISA state at a geopotential altitude."""
    h_m: float; T: float; p: float; rho: float; a: float; mu: float


def isa(alt_kft: float) -> Atmosphere:
    """ISA properties at `alt_kft` thousands of feet.

    Troposphere (h < 11 km) uses the linear-lapse solution; above that the
    isothermal-layer exponential. The trained envelope tops out at 18 kft
    (5.49 km) so the second branch is defensive only.
    """
    h = alt_kft * 1000.0 * FT_TO_M

    if h < H_TROP:
        T = T0_SL - LAPSE * h
        # p/p0 = (T/T0)^(g0 / (R * lapse));  exponent ~= 5.255877
        p = P0_SL * (T / T0_SL) ** (G0 / (R_AIR * LAPSE))
    else:
        T = T_TROP
        p = P_TROP * math.exp(-G0 * (h - H_TROP) / (R_AIR * T_TROP))

    rho = p / (R_AIR * T)
    a   = math.sqrt(GAMMA * R_AIR * T)
    mu  = MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)

    return Atmosphere(h_m=h, T=T, p=p, rho=rho, a=a, mu=mu)


# ---------------------------------------------------------------------------
# 2. Calibrated airspeed -> Mach  (compressible, subsonic)
# ---------------------------------------------------------------------------
def cas_to_mach(kcas: float, p_static: float) -> float:
    """Convert calibrated airspeed [kt] to free-stream Mach at `p_static` [Pa].

    Two steps, both from isentropic compressible flow:

      1. CAS is *defined* as the speed producing impact pressure qc at sea
         level, so invert the sea-level relation:
             qc = p0 * [ (1 + (g-1)/2 * (V_cas/a0)^2)^(g/(g-1)) - 1 ]

      2. The same qc at the real static pressure gives the real Mach:
             M = sqrt( 2/(g-1) * [ (qc/p + 1)^((g-1)/g) - 1 ] )

    The incompressible shortcut M = V_cas*sqrt(rho0/rho)/a is NOT used: at
    250 kt / 18 kft it is off by 1.67%, real error in a Re spanning 4 decades.
    """
    if kcas < 0:
        raise ValueError(f"KCAS must be non-negative, got {kcas}")

    v_cas = kcas * KT_TO_MS
    exp_c = GAMMA / (GAMMA - 1.0)          # 3.5
    exp_i = (GAMMA - 1.0) / GAMMA          # 2/7

    qc = P0_SL * ((1.0 + 0.5 * (GAMMA - 1.0) * (v_cas / A0_SL) ** 2) ** exp_c - 1.0)
    m_sq = 2.0 / (GAMMA - 1.0) * ((qc / p_static + 1.0) ** exp_i - 1.0)
    return math.sqrt(max(m_sq, 0.0))


# ---------------------------------------------------------------------------
# 3. The converter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlightState:
    """Everything both consumers need, plus intermediates for auditing."""
    alt_kft: float; kcas: float; alpha_deg: float     # raw (Consumer A)
    Re_L: float; M_inf: float                         # converted (Consumer B)
    L_ref_m: float
    T: float; p: float; rho: float; a: float; mu: float; V_true: float


def convert(alt_kft: float, kcas: float, alpha_deg: float, C1_mm: float) -> FlightState:
    """(alt_kft, KCAS, alpha, C1_mm) -> (Re_L, M_inf) plus intermediates.

        ISA(alt)   -> T, p, rho, a = sqrt(gRT), mu = Sutherland(T)
        KCAS -> qc -> M_inf                (compressible CAS relation)
        Re_L = rho * M*a * (C1_mm/1000) / mu

    C1_mm is a DESIGN variable, so Re_L is not a per-mission constant -- it
    moves whenever the optimiser touches root chord. Call this inside the
    objective, not once per mission.
    """
    if C1_mm <= 0:
        raise ValueError(f"C1_mm must be positive, got {C1_mm}")

    atm    = isa(alt_kft)
    M      = cas_to_mach(kcas, atm.p)
    V_true = M * atm.a
    L_ref  = C1_mm / 1000.0
    Re_L   = atm.rho * V_true * L_ref / atm.mu

    return FlightState(
        alt_kft=alt_kft, kcas=kcas, alpha_deg=alpha_deg,
        Re_L=Re_L, M_inf=M, L_ref_m=L_ref,
        T=atm.T, p=atm.p, rho=atm.rho, a=atm.a, mu=atm.mu, V_true=V_true,
    )


# ---------------------------------------------------------------------------
# 4. Range checking
# ---------------------------------------------------------------------------
def check_ranges(fs: FlightState) -> list[str]:
    """Human-readable warnings for out-of-envelope values.

    The known offender is M_inf: high KCAS at low altitude crosses 0.50 (e.g.
    250 kt / 18 kft -> M = 0.526). That is ~0.5% of the dataset envelope.
    """
    warnings = []

    def _chk(name, value, lo, hi):
        if not (lo <= value <= hi):
            warnings.append(f"{name} = {value:.6g} outside trained range [{lo:.6g}, {hi:.6g}]")

    _chk("Altitude_kft", fs.alt_kft,   *RANGE_A["Altitude_kft"])
    _chk("KCAS_kt",      fs.kcas,      *RANGE_A["KCAS_kt"])
    _chk("AOA_deg",      fs.alpha_deg, *RANGE_A["AOA_deg"])
    _chk("Re_L",         fs.Re_L,      *RANGE_B["Re_L"])
    _chk("M_inf",        fs.M_inf,     *RANGE_B["M_inf"])
    return warnings


# ---------------------------------------------------------------------------
# 5. Feature assembly
# ---------------------------------------------------------------------------
# Canonical design keys -- the nTop ratio convention (see module docstring).
# These are exactly the dataset column names, so a CSV row is already a `geom`.
GEOM_ORDER_A = ["C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1", "B3/C1", "X3/C1",
                "S1", "S3", "C1"]

# Aero block: 9 entries, C1 excluded (it enters via Re_L only).
GEOM_ORDER_B = ["B1/C1", "B2/C1", "B3/C1", "C2/C1", "C3/C1", "C4/C1",
                "S1", "S3", "X3/C1"]

# The six LENGTH ratios are scaled to the fixed 1000 mm reference chord.
# S1/S3 are angles and X3/C1 is already dimensionless in both models.
AERO_SCALED_KEYS = {"B1/C1", "B2/C1", "B3/C1", "C2/C1", "C3/C1", "C4/C1"}
AERO_C1_REF_MM = 1000.0


def features_structural(struct_11: list[float], geom: dict[str, float],
                        fs: FlightState) -> list[float]:
    """Consumer A -- 24 features: 11 struct -> 10 geom -> 3 flight.

    `geom` is the canonical ratio dict, so the geom block is a straight
    lookup: ratios pass through and C1 goes last in mm. The flight block is
    the raw mission triple -- no conversion.
    """
    if len(struct_11) != 11:
        raise ValueError(f"expected 11 structural features, got {len(struct_11)}")

    geom_block = [geom[k] for k in GEOM_ORDER_A]
    flight_block = [fs.alt_kft, fs.kcas, fs.alpha_deg]

    vec = list(struct_11) + geom_block + flight_block
    assert len(vec) == 24, len(vec)
    return vec


def features_aero(geom: dict[str, float], fs: FlightState,
                  log_re: bool = False) -> list[float]:
    """Consumer B -- 12 features: geom[9] + (Re_L, M_inf, alpha_deg).

    Length ratios are scaled to the 1000 mm reference chord; S1/S3/X3 pass
    through. `log_re=True` emits log10(Re_L) -- the forward regressor wants
    the raw value, so only set this for conditioners that ask for the log.
    """
    geom_block = [geom[k] * AERO_C1_REF_MM if k in AERO_SCALED_KEYS else geom[k]
                  for k in GEOM_ORDER_B]
    re = math.log10(fs.Re_L) if log_re else fs.Re_L

    vec = geom_block + [re, fs.M_inf, fs.alpha_deg]
    assert len(vec) == 12, len(vec)
    return vec
