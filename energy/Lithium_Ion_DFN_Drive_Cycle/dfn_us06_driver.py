"""Driver for the PyBaMM DFN model of a lithium-ion cell under a scaled US06 drive cycle.

Rebuilds the setup of Zonta et al., "Time-dependent global sensitivity analysis of
the Doyle-Fuller-Newman model", J. Energy Storage (2026), doi:10.1016/j.est.2026.121751,
Sections 3.1-3.3 and Table A.2. PyBaMM is an outside code, not a numpy/scipy model, so this
module only sets it up and runs it. Running the model needs PyBaMM (the paper used v24.9.0; the
stored data were made with PyBaMM 24.9.0 and casadi 3.6.7). PyBaMM is imported only inside
`parameter_values` and `voltage`, so the constants (INPUTS, NAMES, LOWER, UPPER, LOG, SCALE, ...)
can be imported with numpy alone.

Base parameter set: Marquis et al. (2019), PyBaMM's "Marquis2019" (Kokam SLPB78205130H).
The 24 inputs of Table A.2 overwrite the corresponding entries; everything else (OCV
curves, electrode area, temperature, ...) stays at the Marquis values.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = 96485.33212  # C/mol

# (name, symbol, units, lower, upper, distribution) in the order of Table A.2, p. 18
INPUTS = [
    ("cs_max_pos", "c_s,pos^max", "mol m^-3", 23900.0, 51765.0, "uniform"),
    ("L_pos", "L_pos", "m", 6.00e-6, 6.60e-5, "uniform"),
    ("eps_pos", "eps_l,pos", "-", 0.171, 0.648, "uniform"),
    ("R_pos", "R_pos", "m", 5.00e-7, 1.00e-5, "uniform"),
    ("k0_pos", "k_0,pos", "m^2.5 mol^-0.5 s^-1", 2.10e-12, 2.41e-5, "loguniform"),
    ("Ds_pos", "D_s,pos", "m^2 s^-1", 9.59e-19, 2.51e-12, "loguniform"),
    ("sigma_pos", "sigma_pos", "S m^-1", 5.20e-6, 1.00e3, "loguniform"),
    ("b_pos", "b_pos", "-", 1.44, 2.44, "uniform"),
    ("cs_max_neg", "c_s,neg^max", "mol m^-3", 16100.0, 31920.0, "uniform"),
    ("L_neg", "L_neg", "m", 4.60e-5, 7.20e-5, "uniform"),
    ("eps_neg", "eps_l,neg", "-", 0.26, 0.50, "uniform"),
    ("R_neg", "R_neg", "m", 1.00e-5, 1.26e-5, "uniform"),
    ("k0_neg", "k_0,neg", "m^2.5 mol^-0.5 s^-1", 1.00e-11, 3.00e-3, "loguniform"),
    ("Ds_neg", "D_s,neg", "m^2 s^-1", 2.00e-16, 9.07e-12, "loguniform"),
    ("sigma_neg", "sigma_neg", "S m^-1", 1.11e-1, 2.20e3, "loguniform"),
    ("b_neg", "b_neg", "-", 1.50, 4.10, "uniform"),
    ("L_sep", "L_sep", "m", 1.60e-5, 5.00e-5, "uniform"),
    ("eps_sep", "eps_l,sep", "-", 0.37, 0.60, "uniform"),
    ("b_sep", "b_sep", "-", 1.50, 2.57, "uniform"),
    ("Theta", "Theta", "-", 1.00, 1.86, "uniform"),
    ("c_l0", "c_l,0", "mol m^-3", 500.0, 1500.0, "uniform"),
    ("D_l", "D_l", "m^2 s^-1", 3.60e-11, 1.09e-9, "loguniform"),
    ("kappa", "kappa", "S m^-1", 4.45e-3, 2.45, "uniform"),
    ("t_plus", "t+", "-", -0.37, 0.51, "uniform"),
]
NAMES = [r[0] for r in INPUTS]
LOWER = np.array([r[3] for r in INPUTS])
UPPER = np.array([r[4] for r in INPUTS])
LOG = np.array([r[5] == "loguniform" for r in INPUTS])

# Drive cycle: PyBaMM's US06 current profile [A] (as shipped with PyBaMM up to v24.1 and in
# the TDGSA examples folder), 1 Hz, t = 0..600 s.
US06 = np.loadtxt(os.path.join(HERE, "US06.csv"), delimiter=",", comments="#")
T_EVAL = US06[:, 0]

# Electrode area of the Marquis cell (held fixed, Section 3.3.1)
AREA = 0.137 * 0.207  # m^2, electrode height x width, one electrode pair


def theoretical_capacity(cs_max, L, eps_l):
    """Theoretical areal capacity [A h m^-2], Eq. (46) with eps_s = 1 - eps_l."""
    return F * cs_max * L * (1.0 - eps_l) / 3600.0


# Global current scaling (Section 3.3.2): one factor for all samples, chosen so that the
# peak C_theo-rate is at most 2 1/h for every point of the Table A.2 box. The smallest
# limiting capacity in the box is the positive electrode at (cs_max_pos, L_pos) lower bounds
# and eps_pos upper bound.
C_THEO_MIN = theoretical_capacity(LOWER[0], LOWER[1], UPPER[2])  # A h m^-2
PEAK_RATE = 2.0  # 1/h
PEAK_CURRENT_DENSITY = PEAK_RATE * C_THEO_MIN  # A m^-2
SCALE = PEAK_CURRENT_DENSITY * AREA / US06[:, 1].max()  # dimensionless factor on US06 [A]


def _j0(k0):
    def j0(c_e, c_s_surf, c_s_max, T):
        # Eq. (31) with dimensional concentrations, alpha = 0.5; isothermal (no Arrhenius)
        return F * k0 * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

    return j0


def parameter_values(x, peak_current_density=PEAK_CURRENT_DENSITY):
    """Marquis2019 values with the 24 inputs x (physical units, order NAMES) set.

    The US06 current [A] is multiplied by peak_current_density * AREA / max(US06 current);
    the default gives the factor SCALE used for the stored data.
    """
    import pybamm

    p = dict(zip(NAMES, np.asarray(x, dtype=float)))
    pv = pybamm.ParameterValues("Marquis2019")
    scale = SCALE if peak_current_density == PEAK_CURRENT_DENSITY else (
        peak_current_density * AREA / US06[:, 1].max())
    current = pybamm.Interpolant(US06[:, 0], scale * US06[:, 1], pybamm.t)
    pv.update(
        {
            "Current function [A]": current,
            # positive electrode
            "Maximum concentration in positive electrode [mol.m-3]": p["cs_max_pos"],
            "Positive electrode thickness [m]": p["L_pos"],
            "Positive electrode porosity": p["eps_pos"],
            "Positive electrode active material volume fraction": 1.0 - p["eps_pos"],
            "Positive particle radius [m]": p["R_pos"],
            "Positive electrode exchange-current density [A.m-2]": _j0(p["k0_pos"]),
            "Positive particle diffusivity [m2.s-1]": p["Ds_pos"],
            "Positive electrode conductivity [S.m-1]": p["sigma_pos"],
            "Positive electrode Bruggeman coefficient (electrolyte)": p["b_pos"],
            "Initial concentration in positive electrode [mol.m-3]": 0.5 * p["cs_max_pos"],
            # negative electrode
            "Maximum concentration in negative electrode [mol.m-3]": p["cs_max_neg"],
            "Negative electrode thickness [m]": p["L_neg"],
            "Negative electrode porosity": p["eps_neg"],
            "Negative electrode active material volume fraction": 1.0 - p["eps_neg"],
            "Negative particle radius [m]": p["R_neg"],
            "Negative electrode exchange-current density [A.m-2]": _j0(p["k0_neg"]),
            "Negative particle diffusivity [m2.s-1]": p["Ds_neg"],
            "Negative electrode conductivity [S.m-1]": p["sigma_neg"],
            "Negative electrode Bruggeman coefficient (electrolyte)": p["b_neg"],
            "Initial concentration in negative electrode [mol.m-3]": 0.5 * p["cs_max_neg"],
            # separator
            "Separator thickness [m]": p["L_sep"],
            "Separator porosity": p["eps_sep"],
            "Separator Bruggeman coefficient (electrolyte)": p["b_sep"],
            # electrolyte (scalars, independent of concentration, Section 2.2)
            "Thermodynamic factor": p["Theta"],
            "Initial concentration in electrolyte [mol.m-3]": p["c_l0"],
            "Electrolyte diffusivity [m2.s-1]": p["D_l"],
            "Electrolyte conductivity [S.m-1]": p["kappa"],
            "Cation transference number": p["t_plus"],
        }
    )
    return pv


def voltage(x, rtol=1e-6, atol=1e-6, peak_current_density=PEAK_CURRENT_DENSITY, solver=None):
    """Terminal voltage [V] at t = 0, 1, ..., 600 s. Raises RuntimeError on failure.

    PyBaMM's DFN with its default mesh (20 points in each of the three x-subdomains and
    in r of each particle), CasADi solver in "fast" mode (no event detection, so no
    voltage cut-off is applied, as in the TDGSA example), tolerances rtol, atol. `solver`
    replaces the CasADi solver by another PyBaMM solver (used only by check_accuracy.py).
    """
    import pybamm

    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(
        model,
        parameter_values=parameter_values(x, peak_current_density),
        solver=solver if solver is not None else pybamm.CasadiSolver(mode="fast", rtol=rtol, atol=atol),
    )
    sol = sim.solve(T_EVAL)
    t = sol["Time [s]"].entries
    if t.shape != T_EVAL.shape or abs(t[-1] - T_EVAL[-1]) > 1e-8:
        raise RuntimeError(f"solution stopped at t = {t[-1]:.3g} s")
    v = sol["Voltage [V]"](T_EVAL)
    if not np.all(np.isfinite(v)):
        raise RuntimeError("non-finite voltage")
    return np.asarray(v, dtype=float)


def nominal():
    """Geometric centre of the box (arithmetic for uniform, geometric for log-uniform)."""
    return np.where(LOG, np.sqrt(np.abs(LOWER * UPPER)), 0.5 * (LOWER + UPPER))
