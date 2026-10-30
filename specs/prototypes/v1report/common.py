"""Shared loaders for the v1 requirements repros (as-data-sets data, numpy only)."""
import numpy as np

R = "/Users/paulconstantine/PycharmProjects/as-data-sets"


def _rd(fn):
    with open(fn) as fh:
        hdr = [h.strip('"') for h in fh.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)


ORDER_G = ['dewmx', 'lai', 'lai0', 'sai', 'xl', 'dleaf', 'rhol_vis', 'rhol_nir', 'rhos_vis',
           'rhos_nir', 'taul_vis', 'taul_nir', 'taus_vis', 'taus_nir', 'fieldcap', 'wiltpt',
           'z0m', 'soilcol', 'watertable']
YL = np.array([.05, 1, .05, .5, -.3, 0, .06, .35, .3, .5, .01, .1, .15, .3, .3, .1, 1, -10])
YU = np.array([.15, 4, 3, 4, .4, .16, .13, .55, .4, .6, .09, .45, .25, .4, 1, .3, 8, -1])


def hydrology_grass():
    """Physical grass inputs (z0m column 16 raw), outputs (300, k)."""
    h, X = _rd(f"{R}/hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/SC_ScatterParameters_grass.txt")
    X = X[:, [h.index(n) for n in ORDER_G]]
    F = np.loadtxt(f"{R}/hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/ScatterSummary_grass.txt")
    return X, F


def stomatal():
    h, X = _rd(f"{R}/hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/stomata_realization_inputs.txt")
    _, B = _rd(f"{R}/hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/param_ranges.txt")
    out = {}
    for fn in ['mo_avg_rssha', 'mo_avg_rssun', 'mo_avg_tran_veg', 'JJA_hr_avg_rssha',
               'JJA_hr_avg_rssun', 'JJA_hr_avg_tran_veg']:
        hh, F = _rd(f"{R}/hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/Tonzi_Stomata_SC.{fn}.txt")
        out[fn] = (hh, F)
    return h, X, B, out


def subsurface():
    X = np.loadtxt(f"{R}/hydrology_subsurface_geoscience/Runoff_Permeability_Field/inputs-ShortCorr.GW02.txt")
    fs = np.loadtxt(f"{R}/hydrology_subsurface_geoscience/Runoff_Permeability_Field/RunOff02_Saturated_10k.txt")
    fu = np.loadtxt(f"{R}/hydrology_subsurface_geoscience/Runoff_Permeability_Field/RunOff02_Unsaturated_10k.txt")
    return X, fs, fu


def battery(rate="0.25C"):
    d = np.loadtxt(f"{R}/energy/Lithium_Ion_Battery/{rate}.txt", delimiter=",", skiprows=1)
    if rate == "4C":
        d = np.delete(d, 1537, 0)
    capfac = {'0.25C': 7.5 / 36000., '1C': 30. / 36000., '4C': 120. / 36000.}[rate]
    return d[:, 1:20], d[:, 20:70], capfac * d[:, 70:120]


BATTERY_NAMES = ["eps_a", "eps_s", "eps_c", "brugg_a", "brugg_s", "brugg_c", "t_plus", "D",
                 "D_a", "D_c", "sig_a", "sig_c", "k_a", "k_c", "r_a", "r_c", "L_a", "L_s", "L_c"]


def wind():
    d = np.loadtxt(f"{R}/energy/Offshore_Wind_Turbine_Fatigue/NRELwind_testing.txt", delimiter=",", skiprows=1)
    return d


def pv():
    d = np.loadtxt(f"{R}/energy/Single_Diode_Solar_Cell/SingleDiodePV-Pmax.txt", delimiter=",", skiprows=1)
    return d


def airfoil(name, m):
    fn = f"{R}/{name}/{name}.txt"
    d = np.loadtxt(fn, delimiter=",", skiprows=1)
    X, fl, fd = d[:, 1:m + 1], d[:, m + 1], d[:, m + 2]
    Gl, Gd = d[:, m + 3:2 * m + 3], d[:, 2 * m + 3:]
    return X, fl, fd, Gl, Gd


HYSHOT_XL = np.array([2.6, 0.1, 122.5, 1.6448e7, 3.0551e6, 0.03, 0.087])
HYSHOT_XU = np.array([4.6, 1.9, 367.5, 1.9012e7, 3.4280e6, 0.07, 0.203])


def hyshot30():
    d30 = np.loadtxt(f"{R}/aerospace_cfd/HyShotII_Scramjet/HyShotII30.txt", delimiter=",")
    Z = d30[:-2, :7]
    X = HYSHOT_XL + (Z + 1) / 2 * (HYSHOT_XU - HYSHOT_XL)
    return X, d30[:-2, 7]


def hiv():
    G = np.loadtxt(f"{R}/epidemiology_systems_biology/HIV_Long_Term_Dynamics/gradients.csv", skiprows=1, delimiter=",")[:, 1:]
    X = np.loadtxt(f"{R}/epidemiology_systems_biology/HIV_Long_Term_Dynamics/inputs.csv", skiprows=1, delimiter=",")[:, 1:]
    F = np.loadtxt(f"{R}/epidemiology_systems_biology/HIV_Long_Term_Dynamics/outputs.csv", skiprows=1, delimiter=",")[:, 1:]
    return X, F, G


def codes(w):
    """Codes and categories of recorded warnings."""
    return [(type(x.message).__name__, getattr(x.message, "code", None)) for x in w]
