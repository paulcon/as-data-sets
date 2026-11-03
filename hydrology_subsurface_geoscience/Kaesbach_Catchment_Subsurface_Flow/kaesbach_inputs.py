"""Input and output definitions for the Kaesbach catchment HydroGeoSphere ensemble.

Erdal and Cirpka (2019, HESS 23:3787-3805, doi:10.5194/hess-23-3787-2019) ran a
steady-state HydroGeoSphere model with 32 uniformly sampled parameters (Table 1,
p. 3794). Their supplement stores every input as a coordinate u in [0, 1]
(variable ``Rorg`` in ``activeSubSpPilotData.mat``). This module records how u
maps to physical units and back.

Mapping, column by column (``SCALE`` and ``REVERSED``):

* linear:   x = lo + u (hi - lo)
* log:      ln x = ln lo + u (ln hi - ln lo)
* reversed: u is replaced by 1 - u in the formula above.

``STATUS`` says how each mapping was established:

* "verified": the marginal histograms of u match the paper's Fig. 3 (p. 3796),
  which shows these nine inputs on physical axes (ln K for conductivities).
  The five horizontal conductivities are log-uniform and stored REVERSED
  (u = 1 means the lower bound).
* "sign checked": not in Fig. 3; the orientation agrees with the sign of the
  Spearman correlation with a flow output that physics fixes (recharge raises
  gauge-C discharge and net boundary outflow; a larger vertical anisotropy
  ratio of the weathered Gipskeuper raises flooding and stream-A flow). The
  linear scale is assumed.
* "assumed": linear (log for specific storage, which spans two decades), not
  reversed. Input 27 is "riverbed thickness" in Table 1 but is labelled
  ``River-Elev.`` in the supplement, and p. 3793 says the stream-drain head is
  drawn from a uniform distribution; it may be that head (see DATA.md). The supplement and the paper do not say; these inputs have
  negligible influence on every stored output (paper p. 3797).

Whatever the true physical map, the analysis coordinate used in the notebook,
``to_analysis(X)``, is an affine function of the stored u (ln x for log inputs,
x otherwise), so the active subspace and activity scores computed from it are
the ones the stored data define; only the physical labels depend on the map.

Only numpy is imported.
"""

import numpy as np

# Python identifiers, in the column order of the supplement's Rorg matrix.
NAMES = [
    "bottom_head_offset", "fault_offset", "contact_offset_ku_km1", "km1w_thickness",
    "K_km1w", "aniso_km1w", "K_ku", "aniso_ku", "K_km1", "aniso_km1", "K_mo", "K_Q",
    "alpha_km1w", "alpha_ku", "alpha_km1", "alpha_mo", "alpha_Q",
    "n_km1w", "n_ku", "n_km1", "n_mo", "n_Q",
    "Ss_km1w", "Ss_ku", "Ss_km1", "Ss_mo", "Ss_Q",
    "riverbed_thickness", "cauchy_head", "recharge_crop", "recharge_grass", "recharge_forest",
]

# Labels used by the supplement (xlbl), for cross-reference.
SUPPLEMENT_LABELS = [
    "bottomP", "offsetMOW", "offsetkukm1", "km1Wthickness", "K-km1W-XY", "K-km1W-Z",
    "K-ku-XY", "K-ku-Z", "K-km1-XY", "K-km1-Z", "K-mo", "K-Q",
    "A-km1W", "A-ku", "A-km1", "A-mo", "A-Q", "n-km1W", "n-ku", "n-km1", "n-mo", "n-Q",
    "SS-km1W", "SS-ku", "SS-km1", "SS-mo", "SS-Q", "River-Elev.", "AmmerExit",
    "R-crop", "R-grass", "R-forest",
]

# Short labels for plots.
LABELS = [
    "bottom head offset", "fault offset", "contact offset ku-km1", "km1-w thickness",
    "K km1-w", "aniso km1-w", "K ku", "aniso ku", "K km1", "aniso km1", "K mo", "K Q",
    "alpha km1-w", "alpha ku", "alpha km1", "alpha mo", "alpha Q",
    "n km1-w", "n ku", "n km1", "n mo", "n Q",
    "Ss km1-w", "Ss ku", "Ss km1", "Ss mo", "Ss Q",
    "riverbed thickness (?)", "Cauchy boundary head", "recharge crop", "recharge grass",
    "recharge forest",
]

UNITS = (["m"] * 4 + ["m/s", "-", "m/s", "-", "m/s", "-", "m/s", "m/s"]
         + ["1/m"] * 5 + ["-"] * 5 + ["1/m"] * 5 + ["m", "m"] + ["mm/yr"] * 3)

# Table 1, p. 3794 of the paper.
LOWER = np.array(
    [-5.0, 0.0, -20.0, 5.0,
     1e-7, 1.0, 1e-8, 1.0, 1e-9, 1.0, 1e-7, 1e-7,
     0.5, 0.5, 0.5, 0.5, 0.5,
     1.5, 1.5, 1.5, 1.5, 1.5,
     1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
     0.005, 335.0, 100.0, 80.0, 100.0])
UPPER = np.array(
    [5.0, 100.0, 20.0, 50.0,
     5e-5, 50.0, 1e-5, 50.0, 1e-7, 50.0, 1e-5, 1e-5,
     5.0, 5.0, 5.0, 5.0, 5.0,
     9.0, 9.0, 9.0, 9.0, 9.0,
     1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
     0.2, 355.0, 150.0, 130.0, 150.0])

_K = {4, 6, 8, 10, 11}
_SS = {22, 23, 24, 25, 26}
SCALE = ["log" if j in _K or j in _SS else "linear" for j in range(32)]
REVERSED = np.array([j in _K for j in range(32)])
STATUS = []
for _j in range(32):
    if _j in _K or _j in (0, 1, 3, 28):
        STATUS.append("verified")
    elif _j in (5, 29, 30, 31):
        STATUS.append("sign checked")
    else:
        STATUS.append("assumed")

# Outputs, in the column order of the supplement's obsMTRX matrix.
OUTPUT_NAMES = [
    "flooding", "discharge_gauge_C", "flow_stream_A", "flow_stream_B",
    "ratio_stream_to_recharge", "net_boundary_outflow",
    "residence_time_ku", "residence_time_km1", "residence_time_km1w",
]
# The residence-time unit is not stated by the authors. The paper's Fig. 8 (p. 3799, in
# years) equals the stored values divided by RESIDENCE_PER_YEAR = 17,520: the colour bars span
# the behavioral min-max of stored/17,520 (ku 0.24-79.1, km1 0.51-303.6, km1-w 2.13-37.6 yr),
# twice the stored/8,760 range would be needed for hours. 17,520 = 2 x 8,760, so the stored
# unit is probably half-hours (365-day year), but that is an inference, not a stated fact.
OUTPUT_UNITS = ["m^3/s"] * 4 + ["-", "m^3/s"] + ["stored unit; yr = value/17520"] * 3
RESIDENCE_PER_YEAR = 17520.0

_LOG = np.array([s == "log" for s in SCALE])


def _bounds_analysis():
    lo, hi = LOWER.copy(), UPPER.copy()
    lo[_LOG], hi[_LOG] = np.log(LOWER[_LOG]), np.log(UPPER[_LOG])
    return lo, hi


def to_physical(U):
    """Map stored coordinates u in [0, 1] (shape (M, 32)) to physical units."""
    U = np.asarray(U, dtype=float)
    V = np.where(REVERSED, 1.0 - U, U)
    lo, hi = _bounds_analysis()
    Z = lo + V * (hi - lo)
    Z[..., _LOG] = np.exp(Z[..., _LOG])
    return Z


def to_analysis(X):
    """Physical inputs -> analysis coordinates: ln x for log inputs, x otherwise."""
    Z = np.array(X, dtype=float, copy=True)
    Z[..., _LOG] = np.log(Z[..., _LOG])
    return Z


def analysis_bounds():
    """Lower and upper bounds of the analysis coordinates (ln of Table 1 for log inputs)."""
    return _bounds_analysis()


def to_unit(X):
    """Physical inputs -> stored coordinates u in [0, 1] (inverse of ``to_physical``)."""
    lo, hi = _bounds_analysis()
    V = (to_analysis(X) - lo) / (hi - lo)
    return np.where(REVERSED, 1.0 - V, V)


def behavioral(Y):
    """Stage-two acceptance of the paper (Sect. 3.4 and p. 3795): each target must be
    on the permitted side of the midpoint between its limit and its outer point.

    Thresholds are those hard-coded as defaults in the supplement's app
    (activeSubspacePilot_v3.mlapp, checkIX): flooding <= 3e-3 m^3/s, gauge C >= 4e-3
    m^3/s, stream A <= 1.5e-3 m^3/s, stream B >= 7.5e-7 m^3/s and 0.225 <= ratio <= 0.67.
    The paper's midpoint for the upper ratio limit is 0.675; no stored run lies in
    (0.67, 0.675], so both give the same set.
    """
    Y = np.asarray(Y, dtype=float)
    return ((Y[:, 0] <= 3e-3) & (Y[:, 1] >= 4e-3) & (Y[:, 2] <= 1.5e-3)
            & (Y[:, 3] >= 7.5e-7) & (Y[:, 4] >= 0.225) & (Y[:, 4] <= 0.67))
