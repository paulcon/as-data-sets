"""Constant-pressure H2/O2 ignition delay with perturbed Arrhenius parameters (Cantera).

The model of Vohra, Alexanderian, Guy and Mahadevan, "Active subspace-based dimension
reduction for chemical kinetics applications with epistemic uncertainty", Combust. Flame
204 (2019) 152-161, doi:10.1016/j.combustflame.2019.03.006: homogeneous adiabatic ignition
of an H2/O2 mixture at constant pressure, with the 19-reaction H2/O2 mechanism (paper
Table 1). The paper used the Yetter, Dryer and Rabitz (1991) rate parameters in TChem; this
module uses the same 19 reactions, in the same order, with the rate parameters of Li, Zhao,
Kazakov and Dryer (2004), Table I (file h2o2_li2004.yaml), solved with Cantera. One
reaction is written the other way round: R4 is "OH + OH -> O + H2O" in the paper's Table 1
but "O + H2O <=> 2 OH" here (Li et al.'s direction), so A*_4 and Ea*_4 belong to that
direction. The reverse rate comes from the equilibrium constant, so a multiplier on A_4
scales both directions alike; only the nominal values, and hence the widths of the ranges
on ln A_4 and Ea_4, depend on the written direction.

Ignition delay tau (s) is the time at which T first reaches T0 + 400 K (the paper says only
"the time required for the rate of temperature increase to exceed a given threshold").

Inputs (physical units):
  lnA_i  natural log of the pre-exponential factor of reaction Ri, A in cm-mol-s units
         (for a reaction with two Arrhenius expressions, the first one listed in Li et al.
         Table I; every expression of Ri is multiplied by A_i / A*_i).
  Ea_i   activation energy of Ri in kcal/mol (36-input case; every expression of Ri is
         scaled by Ea_i / Ea*_i).
  P0     initial pressure, atm;  T0 initial temperature, K;  phi equivalence ratio
         (mixture H2 : O2 = 2 phi : 1 by moles, no diluent).

The constants (names, nominal values, bounds) need only numpy; IgnitionModel needs cantera
(tested with Cantera 3.2.0) and scipy. Because the solver is Cantera, an outside code, the
data set is "stored data" in as-data-sets' sense (generated with this module and
generate_data.py; provenance in DATA.md), not a numpy/scipy "Python model".
"""
import os
import numpy as np

try:
    import cantera as ct
except ImportError:          # constants stay usable without Cantera
    ct = None

MECH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2o2_li2004.yaml")

REACTIONS = [
    "H + O2 <=> O + OH", "O + H2 <=> H + OH", "H2 + OH <=> H2O + H", "O + H2O <=> 2 OH",
    "H2 + M <=> 2 H + M", "2 O + M <=> O2 + M", "O + H + M <=> OH + M",
    "H + OH + M <=> H2O + M", "H + O2 (+M) <=> HO2 (+M)", "HO2 + H <=> H2 + O2",
    "HO2 + H <=> 2 OH", "HO2 + O <=> OH + O2", "HO2 + OH <=> H2O + O2",
    "2 HO2 <=> H2O2 + O2", "H2O2 (+M) <=> 2 OH (+M)", "H2O2 + H <=> H2O + OH",
    "H2O2 + H <=> H2 + HO2", "H2O2 + O <=> OH + HO2", "H2O2 + OH <=> H2O + HO2",
]
# Cantera reaction indices in h2o2_li2004.yaml for paper reactions R1..R19.
CANTERA_INDEX = [[0], [1], [2], [3], [4, 5], [6, 7], [8], [9], [10], [11], [12], [13],
                 [14], [15, 16], [17], [18], [19], [20], [21, 22]]
# Reference A (first expression listed in Li et al. 2004 Table I; cm-mol-s units) and
# reference Ea (kcal/mol) of R1..R19, as printed there.
A_REF = np.array([3.55e15, 5.08e4, 2.16e8, 2.97e6, 4.58e19, 6.16e15, 4.71e18, 3.8e22,
                  6.37e20, 1.66e13, 7.08e13, 3.25e13, 2.89e13, 4.20e14, 1.20e17, 2.41e13,
                  4.82e13, 9.55e6, 1.00e12])
EA_REF = np.array([16.6, 6.29, 3.43, 13.4, 104.38, 0.0, 0.0, 0.0, 0.52, 0.82, 0.30, 0.0,
                   -0.50, 11.98, 45.5, 3.97, 7.95, 3.97, 0.0])
LNA_REF = np.log(A_REF)
# Reactions whose reference Ea is nonzero: their Ea is uncertain in the 36-input case.
EA_SET = [i + 1 for i in range(19) if EA_REF[i] != 0.0]   # R numbers

NOMINAL_IC = {"P0": 1.0, "T0": 900.0, "phi": 2.0}
T_RISE = 400.0      # K, ignition criterion T(tau) = T0 + T_RISE
T_MAX = 20.0        # s, integration limit; no ignition before it raises an error


def _bounds(nominal, frac):
    a, b = (1 - frac) * nominal, (1 + frac) * nominal
    return np.minimum(a, b), np.maximum(a, b)


# 19-input case (paper Sec. 4): lnA_i uniform on [0.97, 1.03] x lnA*_i.
NAMES19 = [f"lnA{i}" for i in range(1, 20)]
NOMINAL19 = LNA_REF.copy()
LOWER19, UPPER19 = _bounds(NOMINAL19, 0.03)

# 36-input case (paper Sec. 5): lnA_i, Ea_i (i in EA_SET), P0, T0, phi uniform on +/- 2 %.
NAMES36 = NAMES19 + [f"Ea{i}" for i in EA_SET] + ["P0", "T0", "phi"]
NOMINAL36 = np.concatenate([LNA_REF, EA_REF[np.array(EA_SET) - 1],
                            [NOMINAL_IC["P0"], NOMINAL_IC["T0"], NOMINAL_IC["phi"]]])
LOWER36, UPPER36 = _bounds(NOMINAL36, 0.02)
UNITS36 = ["ln(cm-mol-s)"] * 19 + ["kcal/mol"] * len(EA_SET) + ["atm", "K", "-"]

KCAL = 4.184e6      # J/kmol per kcal/mol


class IgnitionModel:
    """Ignition delay solver. Not thread-safe (one Cantera Solution per instance)."""

    def __init__(self, rtol=1e-9, atol=1e-18, mech=MECH):
        if ct is None:
            raise ImportError("IgnitionModel needs Cantera (pip install cantera)")
        self.gas = ct.Solution(mech)
        self.rtol, self.atol = rtol, atol
        self._orig = [self.gas.reaction(j).rate for j in range(self.gas.n_reactions)]
        self._ea_scaled = set()

    # -- parameters -------------------------------------------------------------------
    def _set_lnA(self, lnA):
        lnA = np.asarray(lnA, float)
        if lnA.shape != (19,):
            raise ValueError("lnA must have 19 entries")
        self.gas.set_multiplier(1.0)
        for i in range(19):
            m = np.exp(lnA[i] - LNA_REF[i])
            for j in CANTERA_INDEX[i]:
                self.gas.set_multiplier(m, j)

    def _set_Ea(self, Ea):
        """Ea: dict {R number: Ea in kcal/mol} or None (nominal)."""
        Ea = {} if Ea is None else dict(Ea)
        todo = set(Ea) | self._ea_scaled
        for i in sorted(todo):
            if EA_REF[i - 1] == 0.0:
                raise ValueError(f"R{i} has zero reference Ea and cannot be scaled")
            s = Ea.get(i, EA_REF[i - 1]) / EA_REF[i - 1]
            for j in CANTERA_INDEX[i - 1]:
                rxn = self.gas.reaction(j)
                o = self._orig[j]
                if isinstance(o, ct.TroeRate):
                    lo, hi = o.low_rate, o.high_rate
                    rxn.rate = ct.TroeRate(
                        low=ct.ArrheniusRate(lo.pre_exponential_factor, lo.temperature_exponent,
                                             lo.activation_energy * s),
                        high=ct.ArrheniusRate(hi.pre_exponential_factor, hi.temperature_exponent,
                                              hi.activation_energy * s),
                        falloff_coeffs=o.falloff_coeffs)
                else:
                    rxn.rate = ct.ArrheniusRate(o.pre_exponential_factor, o.temperature_exponent,
                                                o.activation_energy * s)
                self.gas.modify_reaction(j, rxn)
        self._ea_scaled = {i for i in Ea if Ea[i] != EA_REF[i - 1]}

    # -- solve ------------------------------------------------------------------------
    def ignition_delay(self, lnA=None, Ea=None, P0=1.0, T0=900.0, phi=2.0, t_rise=T_RISE,
                       return_history=False):
        """Ignition delay tau (s). lnA: 19 values (None = nominal); Ea: dict R -> kcal/mol;
        t_rise: the temperature rise (K) that defines ignition."""
        if not (P0 > 0 and T0 > 0 and phi > 0 and t_rise > 0):
            raise ValueError("P0, T0, phi and t_rise must be positive")
        self._set_lnA(LNA_REF if lnA is None else lnA)
        self._set_Ea(Ea)
        gas = self.gas
        gas.TPX = T0, P0 * ct.one_atm, {"H2": 2.0 * phi, "O2": 1.0}
        r = ct.IdealGasConstPressureReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol
        net.max_steps = 200000
        Tc = T0 + t_rise

        T_near = T0 + 0.25 * t_rise   # dT/dt is evaluated only above this temperature

        def tdot():
            g = r.phase
            return -np.dot(g.partial_molar_enthalpies, g.net_production_rates) / (
                g.density * g.cp_mass)

        t_a, T_a, d_a = 0.0, T0, None
        hist = [(t_a, T_a)]
        while True:
            t_b = net.step()
            T_b = r.T
            d_b = tdot() if T_b >= T_near else None
            if return_history:
                hist.append((t_b, T_b))
            if not np.isfinite(T_b):
                raise RuntimeError("non-finite temperature")
            if T_b >= Tc:
                break
            if t_b > T_MAX:
                raise RuntimeError(f"no ignition before {T_MAX} s")
            t_a, T_a, d_a = t_b, T_b, d_b
        if d_a is None:   # the step that crossed Tc started below T_near: refuse
            raise RuntimeError("ignition front crossed in a single step; tighten rtol")
        # cubic Hermite interpolation of T on [t_a, t_b], root of T(t) = Tc
        h = t_b - t_a

        def T_of(t):
            s = (t - t_a) / h
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            return h00 * T_a + h10 * h * d_a + h01 * T_b + h11 * h * d_b - Tc

        from scipy.optimize import brentq
        tau = brentq(T_of, t_a, t_b, xtol=1e-15 * max(t_b, 1e-12), rtol=4 * np.finfo(float).eps)
        if return_history:
            return tau, np.array(hist)
        return tau

    # -- vector interfaces ------------------------------------------------------------
    def tau19(self, x):
        x = np.asarray(x, float)
        return self.ignition_delay(lnA=x)

    def tau36(self, x):
        x = np.asarray(x, float)
        nE = len(EA_SET)
        Ea = {r: x[19 + k] for k, r in enumerate(EA_SET)}
        return self.ignition_delay(lnA=x[:19], Ea=Ea, P0=x[19 + nE], T0=x[20 + nE],
                                   phi=x[21 + nE])

    def grad(self, f, x, steps):
        """Central differences of f at x with per-input steps (physical units)."""
        x = np.asarray(x, float)
        g = np.empty(x.size)
        for k in range(x.size):
            e = np.zeros(x.size)
            e[k] = steps[k]
            g[k] = (f(x + e) - f(x - e)) / (2 * steps[k])
        return g


# Finite-difference steps: 1e-3 of each input's half-range (equal steps in [-1, 1] coords).
FD_REL = 1e-3
STEPS19 = FD_REL * 0.5 * (UPPER19 - LOWER19)
STEPS36 = FD_REL * 0.5 * (UPPER36 - LOWER36)
