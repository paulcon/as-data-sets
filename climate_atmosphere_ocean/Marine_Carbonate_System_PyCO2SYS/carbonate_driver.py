"""Seawater carbonate system with uncertain measurements and equilibrium constants, via PyCO2SYS.

as-data-sets: climate_atmosphere_ocean/Marine_Carbonate_System_PyCO2SYS. Code: MIT (this file).

This module is a thin wrapper around PyCO2SYS v1.8.3 (Humphreys et al., Geosci. Model Dev. 15:15-43,
2022, doi:10.5194/gmd-15-15-2022; https://github.com/mvdh7/PyCO2SYS). PyCO2SYS is GPLv3 and is NOT
copied here: it is imported, and must be installed separately (`pip install PyCO2SYS==1.8.3 pandas`).
The notebook does not need it; it reads the stored CSVs. Without PyCO2SYS this module still imports (its
constants are usable), and `evaluate` and `gradients` raise ImportError with the install command.

The model. From total alkalinity A_T and dissolved inorganic carbon T_C (the measured pair), salinity,
temperature, pressure and nutrients, PyCO2SYS solves the carbonate system and returns pH (total scale),
pCO2 and the aragonite and calcite saturation states. This is a stored-data set: the module drives an
outside code (hence `_driver.py`), and the shipped CSVs are the record. The 14 inputs are

    0  alkalinity       A_T                       umol/kg
    1  dic              T_C                       umol/kg
    2  temperature      T                         deg C
    3  salinity         S                         (practical salinity, no unit)
    4  total_silicate   Si_T                      umol/kg
    5  total_phosphate  P_T                       umol/kg
    6  dpk_CO2          offset of pK0*  (CO2 solubility)               pK units
    7  dpk_carbonic_1   offset of pK1*  (carbonic acid, 1st)           pK units
    8  dpk_carbonic_2   offset of pK2*  (carbonic acid, 2nd)           pK units
    9  dpk_borate       offset of pKB*  (boric acid)                   pK units
   10  dpk_water        offset of pKw*  (water)                        pK units
   11  dpk_aragonite    offset of pKsp* (aragonite solubility product) pK units
   12  dpk_calcite      offset of pKsp* (calcite solubility product)   pK units
   13  f_total_borate   multiplier on the salinity-derived total borate (1 = nominal)

An offset d means the constant used is K = K_param(T, S, P) * 10**(-d), with K_param PyCO2SYS's own
parameterisation at the sample's T, S and P: the constant's uncertainty is an error of its
parameterisation, so the offset is what is uncertain (Orr et al., 2018, give standard uncertainties of
the pK* values; PyCO2SYS ships them as `pyco2.uncertainty_OEDG18`). Total borate is
B_T = B_T,param(S) * f_total_borate. Every other constant and total (sulfate, fluoride, calcium,
phosphoric and silicic acid, ammonia, sulfide) follows PyCO2SYS's parameterisations at T, S, P.

Evaluation is two vectorised PyCO2SYS calls: one without overrides to get K_param and B_T,param at the
sample's T, S, P; one with the offset constants and scaled borate passed as PyCO2SYS's override
arguments (`k_CO2`, `k_carbonic_1`, ..., `total_borate`). With every offset 0 and the multiplier 1 the
second call reproduces the first (checked in generate_data.py).
"""
import warnings

import numpy as np

try:
    import PyCO2SYS as pyco2
except ImportError:            # constants stay usable without PyCO2SYS
    pyco2 = None

INSTALL = ("PyCO2SYS is not installed. Install PyCO2SYS 1.8.3, the version that wrote the shipped files "
           "(pip install PyCO2SYS==1.8.3 pandas; pandas is an undeclared import of PyCO2SYS 1.8.3), and rerun.")


def _need_pyco2():
    if pyco2 is None:
        raise ImportError(INSTALL)

NAMES = ("alkalinity", "dic", "temperature", "salinity", "total_silicate", "total_phosphate",
         "dpk_CO2", "dpk_carbonic_1", "dpk_carbonic_2", "dpk_borate", "dpk_water",
         "dpk_aragonite", "dpk_calcite", "f_total_borate")
UNITS = ("umol/kg", "umol/kg", "degC", "1", "umol/kg", "umol/kg",
         "pK", "pK", "pK", "pK", "pK", "pK", "pK", "1")
# PyCO2SYS names of the overridden constants, in the order of inputs 6-12
K_KEYS = ("k_CO2", "k_carbonic_1", "k_carbonic_2", "k_borate", "k_water", "k_aragonite", "k_calcite")

OUTPUTS = ("pH_total", "pCO2", "saturation_aragonite", "saturation_calcite")
OUTPUT_UNITS = ("1 (total scale)", "uatm", "1", "1")

# Held settings: the conditions of Humphreys et al. (2022), Table 4 (p. 26), and the options of PyCO2SYS
# v1.8.0, the version of the paper, whose defaults are the paper's "default parameterisation options"
# (v1.8.3 changed some defaults, so they are set explicitly). The paper's round-robin script
# (manuscript/round_robin.py in the v1.8.0 release) also adds two hypothetical acid-base systems
# (total_alpha/k_alpha, total_beta/k_beta) that are not seawater constituents; they are NOT used here
# (see DATA.md). The measurement study uses HELD as is; the ocean-wide study is a surface-ocean study
# and overrides the pressure: HELD_OCEANWIDE.
HELD = dict(
    pressure=1234.0,        # dbar (Table 4)
    total_ammonia=2.0,      # umol/kg
    total_sulfide=3.0,      # umol/kg
    opt_k_carbonic=16,      # Sulpis et al. (2020); v1.8.0 default
    opt_total_borate=1,     # Uppstrom (1974); v1.8.0 default
    opt_k_bisulfate=1,      # Dickson (1990); v1.8.0 default
    opt_k_fluoride=1,       # Dickson and Riley (1979); v1.8.0 default
    opt_gas_constant=3,     # 2018 CODATA; v1.8.0 default
    opt_pressured_kCO2=0,   # no pressure correction of K0 and the fugacity factor (v1.8.3 option; v1.8.0 behaviour):
                            # pCO2 (and fCO2, xCO2) are then valid only for the surface ocean (paper p. 17, Sect. 5.3)
    opt_pH_scale=1,         # total scale
)
# The ocean-wide study is a surface-ocean study: pressure 0 dbar (temperatures 6-37 degC do not occur at
# 1234 dbar). Everything else as HELD.
HELD_OCEANWIDE = dict(pressure=0.0)

# Nominal point: Table 4 of Humphreys et al. (2022): A_T 2300, T_C 2100, S 33, T 22 degC, P 1234 dbar,
# Si_T 10, P_T 1, NH3_T 2, H2S_T 3 umol/kg. Offsets 0, borate multiplier 1.
NOMINAL = np.array([2300.0, 2100.0, 22.0, 33.0, 10.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0])

# Standard deviations of the two studies (see DATA.md for sources and choices).
# Constants: Orr et al. (2018) via pyco2.uncertainty_OEDG18 (v1.8.3), for both studies. Orr et al. (2018)
# predate the Sulpis et al. (2020) K1*, K2* used here, so applying their pK1*, pK2* uncertainties to these
# constants is our assumption.
SD_CONSTANTS = np.array([0.002, 0.0075, 0.015, 0.01, 0.01, 0.02, 0.02, 0.02])
# "measurement": measurement uncertainties of the six measured inputs (our choice).
SD_MEASUREMENT = np.concatenate([[2.0, 2.0, 0.05, 0.01, 0.4, 0.04], SD_CONSTANTS])
# "oceanwide": a wide spread of the measured inputs around the same centre (our choice), at the sea
# surface (HELD_OCEANWIDE).
SD_OCEANWIDE = np.concatenate([[50.0, 50.0, 5.0, 1.0, 2.0, 0.2], SD_CONSTANTS])

# Central-difference steps for the gradients, in physical units: 1e-3 of the measurement sd.
STEPS = 1e-3 * SD_MEASUREMENT


def _base_kwargs(X, held=None):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != len(NAMES):
        raise ValueError(f"expected {len(NAMES)} columns, got {X.shape[1]}")
    return X, dict(par1=X[:, 0], par2=X[:, 1], par1_type=1, par2_type=2,
                   temperature=X[:, 2], salinity=X[:, 3],
                   total_silicate=X[:, 4], total_phosphate=X[:, 5], **{**HELD, **(held or {})})


def evaluate(X, outputs=OUTPUTS, full=False, held=None):
    """Model outputs at the rows of X (M, 14), physical units. Returns an (M, len(outputs)) array.

    `held` overrides entries of HELD (e.g. {"pressure": 0.0}). With full=True also returns the
    PyCO2SYS results dict of the second (override) call.
    """
    _need_pyco2()
    X, kw = _base_kwargs(X, held)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = pyco2.sys(**kw)
        over = {k: np.asarray(r0[k]) * 10.0 ** (-X[:, 6 + i]) for i, k in enumerate(K_KEYS)}
        over["total_borate"] = np.asarray(r0["total_borate"]) * X[:, 13]
        r = pyco2.sys(**kw, **over)
    Y = np.column_stack([np.broadcast_to(np.asarray(r[o], dtype=float), (X.shape[0],)) for o in outputs])
    if not np.all(np.isfinite(Y)):
        bad = np.where(~np.all(np.isfinite(Y), axis=1))[0]
        raise FloatingPointError(f"non-finite PyCO2SYS output at rows {bad[:10].tolist()}")
    return (Y, r) if full else Y


def gradients(X, outputs=OUTPUTS, steps=STEPS, held=None):
    """Central-difference gradients d(output)/d(input), physical units. Returns (M, len(outputs), 14).

    `held` overrides entries of HELD, as in evaluate."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    M, m = X.shape
    G = np.empty((M, len(outputs), m))
    for j in range(m):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, j] += steps[j]
        Xm[:, j] -= steps[j]
        G[:, :, j] = (evaluate(Xp, outputs, held=held) - evaluate(Xm, outputs, held=held)) / (2.0 * steps[j])
    return G
