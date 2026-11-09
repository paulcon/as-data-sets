"""Ignition delay and laminar flame speed of NH3 mixtures with 52 uncertain rate constants (Cantera).

The study of Huang, Wang, Tian, Haidn and Slavinskaya, "Uncertainty quantification and
reduction for combustion kinetic Modeling: A case study of NH3/H2 models", Fuel 400 (2025)
135810, doi:10.1016/j.fuel.2025.135810: the pre-exponential factors A of 52 highly sensitive
reactions of the authors' NH3/H2 mechanism (42 species, 346 reactions), the low- and
high-pressure limits of falloff reactions counted separately, are varied within their initial
uncertainty bounds (paper Sec. 2.3 and 3.4; Table 4 for 25 of them, the supplement's table of
all 52 for the rest).

Inputs. x_i = log10(A_i / A_i0), the base-10 log of the factor that multiplies the
pre-exponential factor of entry i, with |x_i| <= f_i, where f_i = f_l = f_u is the initial
uncertainty factor of Eqs. (5)-(6) (supplement table, column "Initial uncertainty bounds").
Because only A changes, k_i(T) is multiplied by 10**x_i at every temperature. For a falloff
reaction, the low- or the high-pressure limit A is changed, not the whole rate. A_i0 is the
value in the authors' Cantera file NH3_H2.yaml, which is the nominal model.

Quantities of interest (conditions from the paper's Fig. 10 and the supplement's data table):
  idt   ignition delay tau (s) of 4.375 % NH3 / 3.281 % O2 / 92.344 % Ar (phi = 1) at
        T5 = 1587.982151 K, p5 = 1013250 Pa (Chen et al. 2021 point in the supplement), in an
        adiabatic constant-volume reactor. tau is the first time at which T reaches T5 + 400 K.
  lfs   laminar flame speed Su (m/s) of NH3/H2 (50/50 by mole) with air (O2 : N2 = 1 : 3.76),
        phi = 1.0, Tu = 298 K, p = 101325 Pa; freely propagating flame, mixture-averaged
        transport, grid refined with ratio 3, slope 0.1, curve 0.2.

Gradients. idt: forward differences in x with step 1e-3 (53 solves). lfs: Cantera's discrete
adjoint (Sim1D.solve_adjoint) on the converged grid, with one parameter per uncertain A,
perturbing the low- or high-pressure A of falloff entries separately. lfs_grad_fd gives central
differences on the same grid for checking.

The mechanism file is not distributed with this module: its header restricts copying and
distribution. Download the supplement of the paper (Appendix A, file mmc3.zip), take
NH3_H2.yaml from it and put it next to this module, or pass its path (mech=...) or set the
environment variable NH3H2_MECH. The file is checked against MECH_SHA256.

The constants need only numpy; the models need Cantera (the shipped data were written with
Cantera 3.2.0: pip install cantera==3.2.0). Cantera is an outside code, so this module is a
driver and the data set is stored data with a regeneration script.
"""
import hashlib
import os

import numpy as np

try:
    import cantera as ct
except ImportError:          # constants stay usable without Cantera
    ct = None

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_NAME = "NH3_H2.yaml"
MECH_SHA256 = "23289472b013db8b4095448419c62996035b0f02265e0d5569b88f2a5d87284a"

# The 52 entries of the supplement's table (mmc1.xlsx), in its row order:
# (row, reaction as printed, A0 as printed [cm, mol, s], f_ini, f_red_l, f_red_u,
#  temperature range as printed). f_ini = f_l = f_u (the table's column heading).
TABLE = [
    (1, 'H + O2 <=> O + OH', 1.04e+14, 0.698970000000001, 0.120051143, 0.188345126, '298 - 3550'),
    (2, 'H + O2 (+M) <=> HO2 (+M) low', 6.366e+20, 0.30102999999999724, 0.087099093, 0.084608517, '298 - 3000'),
    (3, 'H + O2 (+M) <=> HO2 (+M)', 4650943897164.62, 0.301030000000001, 0.086742414, 0.08560417, '298 - 3000'),
    (4, 'H2 + O <=> OH + H', 50800.0, 0.47712125500000013, 0.137855087, 0.136184237, '298 - 3000'),
    (5, 'H2 + OH <=> H + H2O', 216000000.0, 0.30102999600000047, 0.086672808, 0.084407796, '298 - 3000'),
    (6, 'H2O2 (+M) <=> 2 OH (+M) low', 2.49e+24, 1.0, 0.286874015, 0.285343312, '600 - 3000'),
    (7, 'H2O2 (+M) <=> 2 OH (+M) high', 2000000000000.0, 1.4771212499999997, 0.423094656, 0.41975943, '600 - 2000'),
    (8, 'NH3 + H <=> NH2 + H2', 2890000.0, 1.1760912590000006, 0.320190636, 0.338462449, '298 - 3070'),
    (9, 'NH3 + O <=> NH2 + OH', 279.0, 1.301029996, 0.359055408, 0.360633988, '298 - 2500'),
    (10, 'NH3 + OH <=> NH2 + H2O', 155900.0, 1.1760912589999997, 0.308616949, 0.34120249, '200 - 3000'),
    (11, 'NH3 + HO2 <=> NH2 + H2O2', 1.2, 1.477121255, 0.420100144, 0.39515854, '500 - 2000'),
    (12, '2 NH2 <=> NH3 + NH', 5.64, 1.0, 0.287319468, 0.280036809, '298 - 2500'),
    (13, 'NH3 + NO <=> HNO + NH2', 1463.0, 1.4771212550000001, 0.417502856, 0.410939946, '1100 - 3000'),
    (14, 'NH3 + NH2 <=> N2H3 + H2', 15000000000000.0, 2.0, 0.571354694, 0.568609153, '1000 - 3000'),
    (15, 'NH2 + H (+M) <=> NH3 (+M) low', 3.61e+22, 0.8450980399999999, 0.237141549, 0.240629857, '290 - 550'),
    (16, 'NH2 + H (+M) <=> NH3 (+M) high', 150000000000000.0, 1.0, 0.285268305, 0.286245974, '350 - 710'),
    (17, 'NH2 + H <=> NH + H2', 109000.0, 1.0, 0.286668516, 0.286630682, '298 - 3000'),
    (18, 'NH2 + O <=> H + HNO', 27800000000000.0, 0.602059989999999, 0.170432691, 0.173358204, '200 - 2000'),
    (19, 'NH2 + O <=> H2 + NO', 2380000000000.0, 1.7781512500000005, 0.509730181, 0.488841388, '200 - 2000'),
    (20, 'NH2 + O2 <=> H2NO + O', 260000000000.0, 1.1760912599999997, 0.327593631, 0.322091488, '500 - 2300'),
    (21, 'NH2 + HO2 <=> NH3 + O2', 918000.0, 1.0, 0.287882717, 0.268197634, '298 - 2000'),
    (22, 'NH2 + HO2 <=> H2NO + OH', 29500000000000.0, 1.4771212499999997, 0.430622056, 0.409751735, '200 - 3000'),
    (23, 'NH2 + HO2 <=> HNO + H2O', 255000000000.0, 1.0, 0.286599971, 0.286591936, '298 - 3500'),
    (24, 'NH2 + NO <=> NNH + OH', 42900000000.0, 0.47712126000000055, 0.133334175, 0.136537622, '300 - 2300'),
    (25, 'NH2 + NO <=> H2O + N2', 2.6e+19, 1.1760912599999997, 0.325184522, 0.304026389, '200 - 3000'),
    (26, 'NH2 + NH <=> N2H2 + H', 213000000000000.0, 0.69897001, 0.196362833, 0.188133938, '200 - 2500'),
    (27, 'NH2 + NO2 <=> H2NO + NO', 860000000000.0, 0.69897001, 0.200729484, 0.201955255, '300-1350'),
    (28, 'NH + O <=> NO + H', 141000000000000.0, 0.69897001, 0.199487401, 0.200699695, '250 - 3000'),
    (29, 'NH + O2 <=> NO + OH', 451000000.0, 1.6020599880000006, 0.463859552, 0.457967763, '200 - 3300'),
    (30, 'NH + O2 <=> HNO + O', 3320000000.0, 1.3010299960000005, 0.35084133, 0.372337754, '298 - 3500'),
    (31, 'NH + NO <=> H + N2O', 2700000000000000.0, 1.3010300000000008, 0.357401625, 0.371085462, '298 - 3000'),
    (32, 'NNH <=> N2 + H', 300000000.0, 1.477121254, 0.419674785, 0.418904277, '298 - 3000'),
    (33, 'NNH + O2 <=> HO2 + N2', 560000000000000.0, 2.4771212500000015, 0.706173067, 0.706976562, '500 - 2000'),
    (34, 'N2H2 + M <=> NNH + H + M', 1.9e+27, 2.0, 0.556970244, 0.575519307, '1000 - 3000'),
    (35, 'N2H2 + H <=> NNH + H2', 482000000.0, 1.0, 0.285126671, 0.288541737, '298 - 3000'),
    (36, 'N2H2 + NH2 <=> NH3 + NNH', 271000.0, 1.602059991, 0.454400208, 0.459685728, '298 - 3000'),
    (37, 'N2H3 (+M) <=> N2H2 + H (+M) low', 3.8e+40, 1.6020599900000008, 0.463462936, 0.454195277, '250 - 3000'),
    (38, 'N2H3 (+M) <=> N2H2 + H (+M) high', 130000000000.0, 1.6020599900000008, 0.46130715, 0.4606981, '250 - 3000'),
    (39, 'N2H3 + M <=> NH2 + NH + M', 5e+16, 1.0, 0.284236402, 0.286941322, '1000 - 3000'),
    (40, 'N2H4 + OH <=> NH3 + H2NO', 13100000000000.0, 0.4771212499999997, 0.138443531, 0.136425272, '200 - 950'),
    (41, 'HNO + H <=> NO + H2', 16600000000.0, 1.6020599900000008, 0.457778283, 0.448672353, '200 - 2500'),
    (42, 'H + NO (+M) <=> HNO (+M)', 236000000000000.0, 1.0, 0.269306054, 0.280032537, '230 - 3000'),
    (43, 'H + NO (+M) <=> HNO (+M)', 1520000000000000.0, 1.0, 0.284556512, 0.280690269, '230 - 3000'),
    (44, 'H2NO + O2 <=> HNO + HO2', 230.0, 1.0, 0.286513497, 0.282357436, '600 - 3000'),
    (45, 'H2NO + NH2 <=> HNO + NH3', 1800000.0, 1.3010299960000005, 0.374293849, 0.368593681, '298 - 3000'),
    (46, 'N2O (+M) <=> N2 + O (+M) low', 1040000000000000.0, 1.0000000000000018, 0.281320129, 0.279646498, '1000 - 3000'),
    (47, 'N2O (+M) <=> N2 + O (+M) high', 169000000000.0, 1.1760912599999997, 0.33725627, 0.340031083, '1200 - 3000'),
    (48, 'N2O + H <=> N2 + OH', 158.0, 1.301029996, 0.37471557, 0.368697559, '350 - 4000'),
    (49, 'NO + HO2 <=> NO2 + OH', 2400000000000.0, 1.54406805, 0.437623345, 0.441460492, '200 - 2000'),
    (50, 'NH2 + OH <=> NH + H2O', 3300000.0, 1.0, 0.284757983, 0.282243386, '250 - 3000'),
    (51, 'NH2 + O2 <=> HNO + OH', 0.029, 2.0, 0.566058615, 0.567371916, '600 - 2300'),
    (52, 'NH + H <=> N + H2', 165000000000.0, 0.7781512500000005, 0.22302509, 0.214882702, '298 - 2500'),
]
M_INPUTS = len(TABLE)

# Index of each entry's reaction in NH3_H2.yaml (0-based, Cantera order), matched by
# reactants and products, and which falloff limit it is (None: not a falloff reaction).
CANTERA_INDEX = [2, 1, 1, 5, 6, 13, 13, 131, 132, 133, 134, 128, 135, 136, 129, 129, 107, 111,
                 110, 117, 114, 116, 115, 120, 119, 124, 123, 91, 98, 97, 99, 137, 142, 150, 152,
                 159, 172, 172, 173, 195, 205, 204, 204, 228, 227, 51, 51, 54, 48, 113, 118, 90]
LIMIT = [None, "low", "high", None, None, "low", "high", None, None, None, None, None, None,
         None, "low", "high", None, None, None, None, None, None, None, None, None, None, None,
         None, None, None, None, None, None, None, None, None, "low", "high", None, None, None,
         "low", "high", None, None, "low", "high", None, None, None, None, None]

# Table 4 (paper p. 7) labels of the 25 entries it lists; the others are only in the supplement.
TABLE4_LABEL = {1: "R1", 4: "R2", 5: "R3", 8: "R4", 9: "R5", 10: "R6", 11: "R7", 14: "R8",
                15: "R9L", 16: "R9H", 20: "R10", 21: "R11", 22: "R12", 24: "R13", 25: "R14",
                26: "R15", 28: "R16", 29: "R17", 31: "R18", 32: "R19", 34: "R20", 41: "R21",
                42: "R22L", 43: "R22H", 48: "R23"}

NAMES = [f"x{r[0]:02d}" for r in TABLE]
EQUATIONS = [r[1] for r in TABLE]
F_INI = np.array([r[3] for r in TABLE])
F_RED_L = np.array([r[4] for r in TABLE])
F_RED_U = np.array([r[5] for r in TABLE])
LOWER = -F_INI
UPPER = F_INI
NOMINAL = np.zeros(M_INPUTS)


def label(i):
    """Short label of entry i (0-based): the Table 4 name where there is one, else S<row>."""
    row = TABLE[i][0]
    return TABLE4_LABEL.get(row, f"S{row}")


# -- conditions ---------------------------------------------------------------------------
IDT_COND = {"T": 1587.982151, "P": 1013250.0, "X": {"NH3": 0.04375, "O2": 0.03281, "AR": 0.92344}}
IDT_RISE = 400.0        # K: tau is the first time T = T5 + IDT_RISE
IDT_TMAX = 1.0          # s: no ignition before this raises an error
LFS_COND = {"T": 298.0, "P": 101325.0, "phi": 1.0, "fuel": {"NH3": 0.5, "H2": 0.5},
            "oxidizer": {"O2": 1.0, "N2": 3.76}}
LFS_REFINE = {"ratio": 3.0, "slope": 0.1, "curve": 0.2}
LFS_WIDTH = 0.03        # m, initial domain width
FD_STEP = 1e-3          # forward-difference step in x (log10 units) for the ignition delay


def mech_path(mech=None):
    """Locate NH3_H2.yaml: the argument, $NH3H2_MECH, or the file next to this module."""
    for p in (mech, os.environ.get("NH3H2_MECH"), os.path.join(HERE, MECH_NAME)):
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "NH3_H2.yaml not found. It is not distributed with this data set. Download the supplementary "
        "data of Huang et al., Fuel 400 (2025) 135810 from https://doi.org/10.1016/j.fuel.2025.135810 "
        "(Appendix A, file mmc3.zip), unzip it, and put NH3_H2.yaml next to nh3h2_kinetics_driver.py, "
        "or pass --mech PATH, or set NH3H2_MECH to its path; then rerun.")


def check_mech(path):
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != MECH_SHA256:
        raise ValueError(f"{os.path.basename(path)} has sha256 {h}; the shipped data were made with "
                         f"the supplement's file, sha256 {MECH_SHA256}")
    return h


class KineticsModel:
    """The 52-input model. Not thread-safe (one Cantera Solution per instance)."""

    def __init__(self, mech=None, rtol=1e-9, atol=1e-15):
        if ct is None:
            raise ImportError("KineticsModel needs Cantera: pip install cantera==3.2.0 (the version "
                              "that wrote the shipped data), then rerun")
        path = mech_path(mech)
        check_mech(path)
        ct.suppress_thermo_warnings()
        self.gas = ct.Solution(path)
        self.rtol, self.atol = rtol, atol
        self._orig = {j: self.gas.reaction(j).rate for j in set(CANTERA_INDEX)}
        self.A0 = np.empty(M_INPUTS)      # nominal A of each entry in the file (SI units)
        for i, j in enumerate(CANTERA_INDEX):
            rt = self._orig[j]
            if LIMIT[i] == "low":
                self.A0[i] = rt.low_rate.pre_exponential_factor
            elif LIMIT[i] == "high":
                self.A0[i] = rt.high_rate.pre_exponential_factor
            else:
                self.A0[i] = rt.pre_exponential_factor
        self._x = None
        self._flame = None
        self.n_fresh_starts = 0     # flames that had to be solved from scratch

    # -- parameters -----------------------------------------------------------------------
    def set_x(self, x):
        """Multiply A of entry i by 10**x[i] (falloff entries: the named limit only)."""
        x = np.asarray(x, float)
        if x.shape != (M_INPUTS,):
            raise ValueError(f"x must have {M_INPUTS} entries")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        self._x = x.copy()
        self._apply(np.log(10.0) * x)

    def _apply(self, lnm):
        """Apply natural-log multipliers lnm (length 52) to the nominal file."""
        gas = self.gas
        gas.set_multiplier(1.0)
        fall = {}
        for i, j in enumerate(CANTERA_INDEX):
            if LIMIT[i] is None:
                gas.set_multiplier(float(np.exp(lnm[i])), j)
            else:
                fall.setdefault(j, {"low": 0.0, "high": 0.0})[LIMIT[i]] = lnm[i]
        for j, d in fall.items():
            o = self._orig[j]
            lo, hi = o.low_rate, o.high_rate
            rxn = gas.reaction(j)
            rxn.rate = ct.TroeRate(
                low=ct.ArrheniusRate(lo.pre_exponential_factor * np.exp(d["low"]),
                                     lo.temperature_exponent, lo.activation_energy),
                high=ct.ArrheniusRate(hi.pre_exponential_factor * np.exp(d["high"]),
                                      hi.temperature_exponent, hi.activation_energy),
                falloff_coeffs=o.falloff_coeffs)
            gas.modify_reaction(j, rxn)

    # -- ignition delay -------------------------------------------------------------------
    def idt(self, x=None, t_rise=IDT_RISE, return_history=False):
        """Ignition delay tau (s) at IDT_COND: first time T = T5 + t_rise."""
        self.set_x(NOMINAL if x is None else x)
        gas = self.gas
        saved = gas.state      # restored on return: a flame of this instance takes its pressure from gas
        try:
            return self._idt(gas, t_rise, return_history)
        finally:
            gas.state = saved

    def _idt(self, gas, t_rise, return_history):
        gas.TPX = IDT_COND["T"], IDT_COND["P"], IDT_COND["X"]
        T0 = IDT_COND["T"]
        r = ct.IdealGasReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol
        net.max_steps = 500000
        Tc = T0 + t_rise
        T_near = T0 + 0.25 * t_rise

        def tdot():   # constant volume: dT/dt = -sum(u_k w_k) / (rho cv)
            g = r.phase
            return -np.dot(g.partial_molar_int_energies, g.net_production_rates) / (g.density * g.cv_mass)

        t_a, T_a, d_a = 0.0, T0, None
        hist = [(t_a, T_a)]
        while True:
            t_b = net.step()
            T_b = r.T
            if not np.isfinite(T_b):
                raise RuntimeError("non-finite temperature")
            d_b = tdot() if T_b >= T_near else None
            if return_history:
                hist.append((t_b, T_b))
            if T_b >= Tc:
                break
            if t_b > IDT_TMAX:
                raise RuntimeError(f"no ignition before {IDT_TMAX} s")
            t_a, T_a, d_a = t_b, T_b, d_b
        if d_a is None:
            raise RuntimeError("ignition front crossed in a single step; tighten rtol")
        h = t_b - t_a

        def T_of(t):   # cubic Hermite interpolation of T between the bracketing steps
            s = (t - t_a) / h
            return ((2 * s**3 - 3 * s**2 + 1) * T_a + (s**3 - 2 * s**2 + s) * h * d_a
                    + (-2 * s**3 + 3 * s**2) * T_b + (s**3 - s**2) * h * d_b - Tc)

        from scipy.optimize import brentq
        tau = brentq(T_of, t_a, t_b, xtol=1e-15 * max(t_b, 1e-12), rtol=4 * np.finfo(float).eps)
        if return_history:
            return tau, np.array(hist)
        return tau

    def idt_grad(self, x, step=FD_STEP, tau=None):
        """tau and d tau / d x_i by forward differences with step `step` in x (s per log10 unit)."""
        x = np.asarray(x, float)
        tau = self.idt(x) if tau is None else tau
        g = np.empty(M_INPUTS)
        for i in range(M_INPUTS):
            e = np.zeros(M_INPUTS)
            e[i] = step
            g[i] = (self.idt(x + e) - tau) / step
        return tau, g

    def idt_grad_central(self, x, step=FD_STEP):
        x = np.asarray(x, float)
        g = np.empty(M_INPUTS)
        for i in range(M_INPUTS):
            e = np.zeros(M_INPUTS)
            e[i] = step
            g[i] = (self.idt(x + e) - self.idt(x - e)) / (2 * step)
        return g

    # -- laminar flame speed --------------------------------------------------------------
    def _new_flame(self, refine=None):
        gas = self.gas
        gas.set_equivalence_ratio(LFS_COND["phi"], LFS_COND["fuel"], LFS_COND["oxidizer"])
        gas.TP = LFS_COND["T"], LFS_COND["P"]
        f = ct.FreeFlame(gas, width=LFS_WIDTH)
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(**(LFS_REFINE if refine is None else refine), prune=0.0)
        return f

    def lfs(self, x=None, refine=None):
        """Laminar flame speed Su (m/s) at LFS_COND and the flame object.

        Every call starts from the same initial guess, the converged nominal-model flame, and
        refines the grid from it with LFS_REFINE (or `refine`). The result still depends on the
        flames this instance solved before: the Sim1D object keeps solver state between calls, so
        the refined grid, and with it Su, can differ by up to about 3 % at slow and mid-range
        flames (DATA.md, "Path dependence"). A fresh instance gives a reproducible result."""
        if self._flame is None:
            self.set_x(NOMINAL)
            f = self._new_flame()
            f.solve(loglevel=0, auto=True)
            self._flame = f
            self._nominal_state = f.to_array()
        self.set_x(NOMINAL if x is None else x)
        f = self._flame
        f.from_array(self._nominal_state)
        if refine is not None:       # after from_array, which restores the stored refine criteria
            f.set_refine_criteria(**refine, prune=0.0)
        try:
            f.solve(loglevel=0, refine_grid=True)
        except ct.CanteraError:          # fall back to a fresh start
            f = self._new_flame(refine)
            f.solve(loglevel=0, auto=True)
            self._flame = f
            self.n_fresh_starts += 1
        if refine is not None:
            f.set_refine_criteria(**LFS_REFINE, prune=0.0)
        return f.velocity[0], f

    def lfs_grad(self, x, refine=None, dp=1e-5):
        """Su (m/s) and d Su / d x_i (m/s per log10 unit) from Cantera's discrete adjoint.

        Parameter i scales the (limit) A of entry i by (1 + p_i), so the adjoint returns
        dSu/dln A_i; times ln 10 gives dSu/dx_i."""
        Su, f = self.lfs(x, refine=refine)
        lnm0 = np.log(10.0) * self._x
        Nvars = sum(D.n_components * D.n_points for D in f.domains)
        i_Su = f.inlet.n_components + f.flame.component_index("velocity")
        dgdx = np.zeros(Nvars)
        dgdx[i_Su] = 1.0
        model = self

        def perturb(sim, i, p):
            if LIMIT[i] is None:
                sim.gas.set_multiplier(float(np.exp(lnm0[i]) * (1.0 + p)), CANTERA_INDEX[i])
            else:
                lnm = lnm0.copy()
                lnm[i] += np.log1p(p)
                model._apply_falloff_only(lnm, CANTERA_INDEX[i])

        g = f.solve_adjoint(perturb, M_INPUTS, dgdx, dp=dp)
        self._apply(lnm0)
        return Su, np.log(10.0) * np.asarray(g)

    def _apply_falloff_only(self, lnm, j):
        """Re-set the falloff reaction j from the multipliers lnm (both limits)."""
        o = self._orig[j]
        lo, hi = o.low_rate, o.high_rate
        dl = dh = 0.0
        for i, jj in enumerate(CANTERA_INDEX):
            if jj == j:
                if LIMIT[i] == "low":
                    dl = lnm[i]
                elif LIMIT[i] == "high":
                    dh = lnm[i]
        rxn = self.gas.reaction(j)
        rxn.rate = ct.TroeRate(
            low=ct.ArrheniusRate(lo.pre_exponential_factor * np.exp(dl), lo.temperature_exponent,
                                 lo.activation_energy),
            high=ct.ArrheniusRate(hi.pre_exponential_factor * np.exp(dh), hi.temperature_exponent,
                                  hi.activation_energy),
            falloff_coeffs=o.falloff_coeffs)
        self.gas.modify_reaction(j, rxn)

    def lfs_grad_fd(self, x, step=1e-2, entries=None):
        """Su (m/s) and central differences of Su in x on the grid of the flame at x.

        Each perturbed flame is solved from the flame at x with steady tolerances rtol 1e-9,
        atol 1e-15, and solved a second time from its own result: at small perturbations a single
        call can stop after one Newton step, which biases the difference. `entries` limits the
        differences to some inputs (nan elsewhere)."""
        Su, f = self.lfs(x)
        state = f.to_array()
        f.flame.set_steady_tolerances(default=(1e-9, 1e-15))
        g = np.full(M_INPUTS, np.nan)
        x = np.asarray(x, float)
        for i in (range(M_INPUTS) if entries is None else entries):
            v = []
            for s in (step, -step):
                e = np.zeros(M_INPUTS)
                e[i] = s
                self.set_x(x + e)
                f.from_array(state)
                f.solve(loglevel=0, refine_grid=False)
                f.solve(loglevel=0, refine_grid=False)
                v.append(f.velocity[0])
            g[i] = (v[0] - v[1]) / (2 * step)
        f.flame.set_steady_tolerances(default=(1e-4, 1e-9))
        self.set_x(x)
        f.from_array(state)
        return Su, g
