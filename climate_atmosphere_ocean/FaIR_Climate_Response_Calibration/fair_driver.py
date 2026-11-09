"""Drive FaIR v2.1.3 with the fair-calibrate v1.4.1 posterior (not a numpy/scipy model).

The module has two parts.

* Constants and helpers that need only numpy: the 45 parameter names, units and
  short labels; the coordinate change between the physical parameters and the
  analysis coordinates u used for the stored gradients (``to_analysis``,
  ``from_analysis``, ``analysis_jacobian``); the coordinates v of the notebook, in
  which the aerosol-cloud scale beta is replaced by the ERFaci of 2005-2014
  (``to_erfaci_coordinates``, which also converts gradients by the chain rule,
  and ``from_erfaci_coordinates``); loaders for the shipped CSV files.
* ``FairEnsemble``, which runs FaIR (``pip install fair==2.1.3``) for SSP2-4.5 with
  any set of parameter vectors and returns the quantities of interest, and
  ``fd_gradients``, the central-difference gradients in analysis coordinates.
  Importing the module without FaIR works; constructing ``FairEnsemble`` then
  raises an error that says what to install.

The FaIR set-up follows fair-calibrate v1.4.1,
``input/fair-2.1.3/v1.4/all-2022/constraining/05_constrained-ssp-projections.py``
(commit 5a31144b58d6b9e2b23a845f68f73c7eda98701c, MIT licence), restricted to
SSP2-4.5 and to the years 1750-2101.
"""
import os
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

try:  # FaIR is needed only to run the model
    import fair as _fair
    from fair import FAIR
    from fair.interface import fill, initialise
    from fair.io import read_properties
    FAIR_INSTALLED_VERSION = _fair.__version__
except ImportError:  # pragma: no cover - depends on the environment
    FAIR = None
    FAIR_INSTALLED_VERSION = None

FAIR_VERSION = "2.1.3"          # the version that fair-calibrate v1.4.1 targets (paper p. 8570)
START, END = 1750, 2101         # time bounds of every run (first and last)
SCENARIO = "ssp245"

# ---- the 45 sampled parameters, in the column order of calibrated_constrained_parameters.csv ----
PARAM_NAMES = [
    "clim_gamma", "clim_c1", "clim_c2", "clim_c3", "clim_kappa1", "clim_kappa2", "clim_kappa3",
    "clim_epsilon", "clim_sigma_eta", "clim_sigma_xi", "clim_F_4xCO2",
    "cc_r0", "cc_rU", "cc_rT", "cc_rA",
    "ari_BC", "ari_OC", "ari_Sulfur", "ari_NOx", "ari_VOC", "ari_NH3", "ari_CH4", "ari_N2O",
    "ari_Equivalent effective stratospheric chlorine",
    "aci_shape_so2", "aci_shape_bc", "aci_shape_oc", "aci_beta",
    "o3_CH4", "o3_N2O", "o3_Equivalent effective stratospheric chlorine", "o3_CO", "o3_VOC", "o3_NOx",
    "fscale_CH4", "fscale_N2O", "fscale_minorGHG", "fscale_Stratospheric water vapour",
    "fscale_Land use", "fscale_Volcanic", "fscale_solar_amplitude", "fscale_solar_trend",
    "fscale_Light absorbing particles on snow and ice", "fscale_CO2", "cc_co2_concentration_1750",
]
SHORT = [
    "gamma", "C1", "C2", "C3", "kappa1", "kappa2", "kappa3", "epsilon", "sigma_eta", "sigma_xi", "F4xCO2",
    "r0", "rU", "rT", "rA",
    "ari_BC", "ari_OC", "ari_SO2", "ari_NOx", "ari_VOC", "ari_NH3", "ari_CH4", "ari_N2O", "ari_EESC",
    "ln_s_SO2", "ln_s_BC", "ln_s_OC", "asinh_beta",
    "o3_CH4", "o3_N2O", "o3_EESC", "o3_CO", "o3_VOC", "o3_NOx",
    "fs_CH4", "fs_N2O", "fs_minorGHG", "fs_stratH2O", "fs_landuse", "fs_volcanic", "fs_solar_amp",
    "solar_trend", "fs_LAPSI", "fs_CO2", "CO2_1750",
]
# physical units of the parameters (paper Sect. 3.1-3.2 and Tables 2-5; FaIR 2.1.3 for the carbon cycle)
UNITS = (
    ["yr-1", "W yr m-2 K-1", "W yr m-2 K-1", "W yr m-2 K-1", "W m-2 K-1", "W m-2 K-1", "W m-2 K-1", "1",
     "W m-2 yr-1/2", "W m-2 yr-1/2", "W m-2",
     "yr", "yr (Gt CO2)-1", "yr K-1", "yr (Gt CO2)-1",
     "W m-2 (Mt BC yr-1)-1", "W m-2 (Mt OC yr-1)-1", "W m-2 (Mt SO2 yr-1)-1", "W m-2 (Mt NO2 yr-1)-1",
     "W m-2 (Mt VOC yr-1)-1", "W m-2 (Mt NH3 yr-1)-1", "W m-2 ppb-1", "W m-2 ppb-1", "W m-2 ppt-1",
     "(Mt SO2 yr-1)-1", "(Mt BC yr-1)-1", "(Mt OC yr-1)-1", "W m-2",
     "W m-2 ppb-1", "W m-2 ppb-1", "W m-2 ppt-1", "W m-2 (Mt CO yr-1)-1", "W m-2 (Mt VOC yr-1)-1",
     "W m-2 (Mt NO2 yr-1)-1"]
    + ["1"] * 7 + ["W m-2", "1", "1", "ppm"]
)
GROUP = (["climate response"] * 11 + ["carbon cycle"] * 4 + ["aerosol-radiation"] * 9
         + ["aerosol-cloud"] * 4 + ["ozone"] * 6 + ["forcing scaling"] * 10 + ["carbon cycle"])
M_PARAMS = len(PARAM_NAMES)
assert M_PARAMS == 45 and len(SHORT) == 45 and len(UNITS) == 45 and len(GROUP) == 45

# ---- analysis coordinates u: identity except four aerosol-cloud parameters ----
LOG_IDX = [PARAM_NAMES.index(n) for n in ("aci_shape_so2", "aci_shape_bc", "aci_shape_oc")]
ASINH_IDX = [PARAM_NAMES.index("aci_beta")]


def to_analysis(X):
    """Physical parameters (M, 45) -> analysis coordinates u: ln s for the three aerosol-cloud
    shape parameters, asinh(beta / 1 W m-2) for the aerosol-cloud scale, the rest unchanged."""
    U = np.array(X, dtype=float, copy=True)
    U[..., LOG_IDX] = np.log(U[..., LOG_IDX])
    U[..., ASINH_IDX] = np.arcsinh(U[..., ASINH_IDX])
    return U


def from_analysis(U):
    """Inverse of ``to_analysis``."""
    X = np.array(U, dtype=float, copy=True)
    X[..., LOG_IDX] = np.exp(X[..., LOG_IDX])
    X[..., ASINH_IDX] = np.sinh(X[..., ASINH_IDX])
    return X


def analysis_jacobian(X):
    """du_j/dx_j at physical points X (M, 45), so that df/dx_j = df/du_j * du_j/dx_j."""
    X = np.asarray(X, dtype=float)
    J = np.ones_like(X)
    J[..., LOG_IDX] = 1.0 / X[..., LOG_IDX]
    J[..., ASINH_IDX] = 1.0 / np.sqrt(1.0 + X[..., ASINH_IDX] ** 2)
    return J


# ---- quantities of interest ----
QOI_NAMES = ["dT_ssp245_2081-2100", "ECS", "TCR", "GMST_2003-2022"]
QOI_LONG = [
    "GMST change 2081-2100 relative to 1995-2014 under SSP2-4.5 (harmonised to 2022)",
    "equilibrium climate sensitivity",
    "transient climate response",
    "GMST 2003-2022 relative to 1850-1900 (historical emissions)",
]
EXTRA_NAMES = ["CO2_2022"]       # a check quantity: CO2 concentration in 2022 (ppm)

# ---- shipped file names ----
F_POSTERIOR = "posterior_parameters.csv"
F_FIXED = "fixed_calibrations.csv"
F_EMIS = "ssp245_emissions_1750-2100.csv"
F_FORCING = "solar_volcanic_forcing_1750-2101.csv"
F_QOI = "qoi.csv"
F_QOI_IV = "qoi_internal_variability.csv"
GRAD_FILES = {q: f"gradients_{q}.csv" for q in QOI_NAMES}

FIXED_NAMES = ["ch4_lifetime_base", "ch4_sens_CH4", "ch4_sens_NOx", "ch4_sens_VOC", "ch4_sens_HC",
               "ch4_sens_N2O", "ch4_sens_temp", "landuse_CO2_AFOLU", "lapsi_BC"]

# 1750 baseline emissions set in the v1.4.1 projection script (05_constrained-ssp-projections.py)
BASELINE_EMISSIONS = [
    ("CH4", 38.246272), ("N2O", 0.92661989), ("NOx", 19.41683292), ("Sulfur", 2.293964929),
    ("CO", 348.4549732), ("VOC", 60.62284009), ("BC", 2.096765609), ("OC", 15.44571911),
    ("NH3", 6.656462698), ("CCl4", 0.02129917), ("CHCl3", 202.7251231), ("CH2Cl2", 211.0095537),
    ("CH3Cl", 4544.519056), ("CH3Br", 111.4920237), ("Halon-1211", 0.008146006),
    ("SO2F2", 0.000010554155), ("CF4", 0),
]
MINOR_GHG = [
    "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115", "HCFC-22", "HCFC-141b", "HCFC-142b", "CCl4",
    "CHCl3", "CH2Cl2", "CH3Cl", "CH3CCl3", "CH3Br", "Halon-1211", "Halon-1301", "Halon-2402", "CF4",
    "C2F6", "C3F8", "c-C4F8", "C4F10", "C5F12", "C6F14", "C7F16", "C8F18", "NF3", "SF6", "SO2F2",
    "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", "HFC-227ea", "HFC-23", "HFC-236fa", "HFC-245fa",
    "HFC-32", "HFC-365mfc", "HFC-4310mee",
]


# ---- second analysis coordinates v: as u, but aci_beta replaced by ERFaci 2005-2014 ----
ACI_IDX = [PARAM_NAMES.index(n) for n in ("aci_shape_so2", "aci_shape_bc", "aci_shape_oc")]
BETA_IDX = PARAM_NAMES.index("aci_beta")
SHORT_V = SHORT[:BETA_IDX] + ["ERFaci"] + SHORT[BETA_IDX + 1:]
ACI_SPECIES = ("Sulfur", "BC", "OC")
ACI_WINDOW = (2005, 2015)       # emission time points 2005.5, ..., 2014.5


def erfaci_unscaled(X, directory=HERE):
    """D(s) = mean over the emission time points 2005.5-2014.5 of log(1 + sum_i s_i A_i(t)) minus
    log(1 + sum_i s_i A_i,1750), with A_i the SO2, BC and OC emissions of the shipped file and the
    1750 baseline emissions (FaIR 2.1.3's ERFaci, Eq. 8 of the paper, without the scale beta).
    Returns D (n,) and dD/d(ln s_i) (n, 3)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tp, names, E = load_emissions(directory)
    base = dict(BASELINE_EMISSIONS)
    win = (tp >= ACI_WINDOW[0]) & (tp < ACI_WINDOW[1])
    A = np.array([E[win, names.index(n)] for n in ACI_SPECIES]).T       # (10, 3)
    Ab = np.array([base[n] for n in ACI_SPECIES])
    s = X[:, ACI_IDX]
    sa = 1.0 + s @ A.T                                                   # (n, 10)
    sb = 1.0 + s @ Ab                                                    # (n,)
    D = np.log(sa).mean(axis=1) - np.log(sb)
    dD = (s[:, None, :] * A[None, :, :] / sa[:, :, None]).mean(axis=1) - s * Ab / sb[:, None]
    return D, dD


def to_erfaci_coordinates(X, G=None, directory=HERE):
    """Physical parameters X (n, 45) -> coordinates v: u (``to_analysis``) with u_beta = asinh(beta)
    replaced by ERFaci_2005-2014 = beta * D(s) in W m-2, the quantity whose prior sets beta in
    fair-calibrate (paper Sect. 3.2.2, p. 8580). With G (n, 45) = df/du, also returns df/dv by the
    chain rule (u_beta = asinh(ERFaci / D(s)); u_j = ln s_j for the shape parameters)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    D, dD = erfaci_unscaled(X, directory)
    beta = X[:, BETA_IDX]
    V = to_analysis(X)
    V[:, BETA_IDX] = beta * D
    if G is None:
        return V
    G = np.atleast_2d(np.asarray(G, dtype=float))
    dub = 1.0 / np.sqrt(1.0 + beta ** 2)                                 # d asinh(beta) / d beta
    Gv = G.copy()
    Gv[:, BETA_IDX] = G[:, BETA_IDX] * dub / D
    for i, j in enumerate(ACI_IDX):                                      # d beta / d ln s_j at fixed ERFaci
        Gv[:, j] = G[:, j] - G[:, BETA_IDX] * dub * beta * dD[:, i] / D
    return V, Gv


def from_erfaci_coordinates(V, directory=HERE):
    """Inverse of ``to_erfaci_coordinates`` (physical parameters from v)."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    U = V.copy()
    U[:, BETA_IDX] = 0.0
    X = from_analysis(U)
    D, _ = erfaci_unscaled(X, directory)
    X[:, BETA_IDX] = V[:, BETA_IDX] / D
    return X


def _header_names(path, line=1):
    with open(path) as fh:
        lines = [fh.readline() for _ in range(2)]
    return [c.strip() for c in lines[line].lstrip("#").split(",")]


def load_posterior(directory=HERE):
    """Return (config_id, X, seed): prior-ensemble index (841,), parameters (841, 45), seeds (841,)."""
    A = np.loadtxt(os.path.join(directory, F_POSTERIOR), delimiter=",")
    return A[:, 0].astype(np.int64), A[:, 1:46], A[:, 46].astype(np.int64)


def load_fixed(directory=HERE):
    v = np.loadtxt(os.path.join(directory, F_FIXED), delimiter=",")
    return dict(zip(FIXED_NAMES, np.atleast_1d(v)))


def load_emissions(directory=HERE):
    """Return (timepoints, species, E): mid-year times (351,), FaIR species names (51,), E (351, 51)."""
    path = os.path.join(directory, F_EMIS)
    A = np.loadtxt(path, delimiter=",")
    names = [c.rsplit(" [", 1)[0] for c in _header_names(path)[1:]]
    return A[:, 0], names, A[:, 1:]


def load_forcing(directory=HERE):
    """Return (timebounds, solar, volcanic) in W m-2, 1750-2101."""
    A = np.loadtxt(os.path.join(directory, F_FORCING), delimiter=",")
    return A[:, 0], A[:, 1], A[:, 2]


class FairEnsemble:
    """Run FaIR 2.1.3 for SSP2-4.5, 1750-2101, with the v1.4.1 fixed calibrations."""

    def __init__(self, directory=HERE):
        if FAIR is None:
            raise ImportError(f"FaIR is not installed. Install it with pip install fair=={FAIR_VERSION} "
                              "(the version that produced the shipped files) and rerun.")
        if FAIR_INSTALLED_VERSION != FAIR_VERSION:
            warnings.warn(f"FaIR {FAIR_INSTALLED_VERSION} is installed; the shipped files were made "
                          f"with FaIR {FAIR_VERSION}.")
        self.fixed = load_fixed(directory)
        self.tp, self.species_emis, self.E = load_emissions(directory)
        tb, self.solar, self.volcanic = load_forcing(directory)
        self.trend = np.ones(len(tb))
        self.trend[:271] = np.linspace(0, 1, 271)     # 0 in 1750 to 1 in 2020, as in fair-calibrate

    def run(self, X, seeds=None, stochastic=False, return_model=False):
        """Run FaIR at the parameter vectors X (n, 45, physical). Returns a dict of QoI arrays (n,).

        stochastic=False switches internal variability off (the deterministic energy balance model);
        stochastic=True switches it on with the given seeds, as in the paper's runs (p. 8580).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        P = {name: X[:, j] for j, name in enumerate(PARAM_NAMES)}
        fx = self.fixed
        if seeds is None:
            seeds = np.zeros(n, dtype=np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = FAIR(ch4_method="Thornhill2021")
            f.define_time(START, END, 1)
            f.define_scenarios([SCENARIO])
            f.define_configs(list(range(n)))
            species, properties = read_properties()
            for s in ("Halon-1202", "NOx aviation", "Contrails"):
                species.remove(s)
            f.define_species(species, properties)
            f.allocate()
            for k, sp in enumerate(self.species_emis):
                fill(f.emissions, self.E[:, k][:, None], scenario=SCENARIO, specie=sp)
            fill(f.forcing, self.volcanic[:, None, None] * P["fscale_Volcanic"], specie="Volcanic")
            fill(f.forcing, self.solar[:, None, None] * P["fscale_solar_amplitude"]
                 + self.trend[:, None, None] * P["fscale_solar_trend"], specie="Solar")
            # climate response
            fill(f.climate_configs["ocean_heat_capacity"], np.c_[P["clim_c1"], P["clim_c2"], P["clim_c3"]])
            fill(f.climate_configs["ocean_heat_transfer"],
                 np.c_[P["clim_kappa1"], P["clim_kappa2"], P["clim_kappa3"]])
            fill(f.climate_configs["deep_ocean_efficacy"], P["clim_epsilon"])
            fill(f.climate_configs["gamma_autocorrelation"], P["clim_gamma"])
            fill(f.climate_configs["sigma_eta"], P["clim_sigma_eta"])
            fill(f.climate_configs["sigma_xi"], P["clim_sigma_xi"])
            fill(f.climate_configs["seed"], np.asarray(seeds))
            fill(f.climate_configs["stochastic_run"], bool(stochastic))
            fill(f.climate_configs["use_seed"], True)
            fill(f.climate_configs["forcing_4co2"], P["clim_F_4xCO2"])
            f.fill_species_configs()
            # carbon cycle
            fill(f.species_configs["iirf_0"], P["cc_r0"], specie="CO2")
            fill(f.species_configs["iirf_airborne"], P["cc_rA"], specie="CO2")
            fill(f.species_configs["iirf_uptake"], P["cc_rU"], specie="CO2")
            fill(f.species_configs["iirf_temperature"], P["cc_rT"], specie="CO2")
            # aerosol-cloud interactions
            fill(f.species_configs["aci_scale"], P["aci_beta"])
            fill(f.species_configs["aci_shape"], P["aci_shape_so2"], specie="Sulfur")
            fill(f.species_configs["aci_shape"], P["aci_shape_bc"], specie="BC")
            fill(f.species_configs["aci_shape"], P["aci_shape_oc"], specie="OC")
            # methane lifetime (fixed calibration, Table 1)
            fill(f.species_configs["unperturbed_lifetime"], fx["ch4_lifetime_base"], specie="CH4")
            for sp, key in (("CH4", "ch4_sens_CH4"), ("N2O", "ch4_sens_N2O"), ("VOC", "ch4_sens_VOC"),
                            ("NOx", "ch4_sens_NOx"),
                            ("Equivalent effective stratospheric chlorine", "ch4_sens_HC")):
                fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], fx[key], specie=sp)
            fill(f.species_configs["lifetime_temperature_sensitivity"], fx["ch4_sens_temp"])
            fill(f.species_configs["land_use_cumulative_emissions_to_forcing"], fx["landuse_CO2_AFOLU"],
                 specie="CO2 AFOLU")
            fill(f.species_configs["lapsi_radiative_efficiency"], fx["lapsi_BC"], specie="BC")
            for sp, v in BASELINE_EMISSIONS:
                fill(f.species_configs["baseline_emissions"], v, specie=sp)
            # aerosol-radiation interactions
            for sp, col in (("BC", "ari_BC"), ("CH4", "ari_CH4"), ("N2O", "ari_N2O"), ("NH3", "ari_NH3"),
                            ("NOx", "ari_NOx"), ("OC", "ari_OC"), ("Sulfur", "ari_Sulfur"),
                            ("VOC", "ari_VOC"),
                            ("Equivalent effective stratospheric chlorine",
                             "ari_Equivalent effective stratospheric chlorine")):
                fill(f.species_configs["erfari_radiative_efficiency"], P[col], specie=sp)
            # forcing scale factors
            for sp in ("CO2", "CH4", "N2O", "Stratospheric water vapour",
                       "Light absorbing particles on snow and ice", "Land use"):
                fill(f.species_configs["forcing_scale"], P[f"fscale_{sp}"], specie=sp)
            for sp in MINOR_GHG:
                fill(f.species_configs["forcing_scale"], P["fscale_minorGHG"], specie=sp)
            # ozone
            for sp in ("CH4", "N2O", "Equivalent effective stratospheric chlorine", "CO", "VOC", "NOx"):
                fill(f.species_configs["ozone_radiative_efficiency"], P[f"o3_{sp}"], specie=sp)
            fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")
            fill(f.species_configs["baseline_concentration"], P["cc_co2_concentration_1750"], specie="CO2")
            initialise(f.concentration, f.species_configs["baseline_concentration"])
            initialise(f.forcing, 0)
            initialise(f.temperature, 0)
            initialise(f.cumulative_emissions, 0)
            initialise(f.airborne_emissions, 0)
            f.run(progress=False)
        T = f.temperature[:, 0, :, 0].values            # (352 timebounds, n), surface layer, K
        out = qois_from_temperature(T)
        out["ECS"] = f.ebms.ecs.values.copy()
        out["TCR"] = f.ebms.tcr.values.copy()
        out["CO2_2022"] = f.concentration[272:274, 0, :, 2].values.mean(axis=0)  # mid-2022, ppm
        if return_model:
            return out, f
        return out


def qois_from_temperature(T):
    """Period means as in fair-calibrate: timebound series, end points weighted 1/2.
    T has shape (352, n) for the time bounds 1750, ..., 2101."""
    w21 = np.ones(21)
    w21[[0, -1]] = 0.5
    w52 = np.ones(52)
    w52[[0, -1]] = 0.5
    tb = lambda y: y - START  # noqa: E731  index of time bound y
    mean = lambda a, b, w: np.average(T[tb(a):tb(b) + 1], weights=w, axis=0)  # noqa: E731
    return {
        "dT_ssp245_2081-2100": mean(2081, 2101, w21) - mean(1995, 2015, w21),
        "GMST_2003-2022": mean(2003, 2023, w21) - mean(1850, 1901, w52),
    }


def fd_gradients(ens, X, delta, sd_u, params=None, max_configs=900, log=None, cache=None):
    """Central differences of the four QoIs in analysis coordinates u = to_analysis(X).

    The step for parameter j is h_j = delta * sd_u[j] (sd_u: posterior standard deviation of u_j).
    Returns {qoi: (n, 45) array of df/du_j}; parameters not in ``params`` get NaN.
    ``cache`` (a directory) keeps one .npy file per parameter, so an interrupted run resumes.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    U = to_analysis(X)
    params = list(range(M_PARAMS) if params is None else params)
    G = {q: np.full((n, M_PARAMS), np.nan) for q in QOI_NAMES}
    fname = lambda j: None if cache is None else os.path.join(cache, f"grad_param{j:02d}.npy")  # noqa: E731
    todo = []
    for j in params:
        fn = fname(j)
        if fn is not None and os.path.exists(fn):
            g = np.load(fn)
            for k, q in enumerate(QOI_NAMES):
                G[q][:, j] = g[k]
        else:
            todo.append(j)
    per_call = max(1, max_configs // (2 * n))      # parameters per FaIR call (small n: many at once)
    for c in range(0, len(todo), per_call):
        js = todo[c:c + per_call]
        blocks = []
        for j in js:
            for sgn in (1.0, -1.0):
                Uj = U.copy()
                Uj[:, j] += sgn * delta * sd_u[j]
                blocks.append(Uj)
        Z = from_analysis(np.vstack(blocks))
        res = {q: [] for q in QOI_NAMES}
        for a in range(0, len(Z), max_configs):
            r = ens.run(Z[a:a + max_configs])
            for q in QOI_NAMES:
                res[q].append(r[q])
        res = {q: np.concatenate(res[q]).reshape(len(js), 2, n) for q in QOI_NAMES}
        for i, j in enumerate(js):
            h = delta * sd_u[j]
            g = np.array([(res[q][i, 0] - res[q][i, 1]) / (2 * h) for q in QOI_NAMES])
            for k, q in enumerate(QOI_NAMES):
                G[q][:, j] = g[k]
            if fname(j) is not None:
                np.save(fname(j), g)
            if log is not None:
                log(f"parameter {j:2d} {PARAM_NAMES[j]} done")
    return G
