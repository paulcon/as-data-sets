"""Burke et al. (2012) H2/O2 kinetics with the 31 calibration parameters of Bell et al. (2019), in Cantera.

The study of J. Bell, M. Day, J. Goodman, R. Grout and M. Morzfeld, "A Bayesian approach to
calibrating hydrogen flame kinetics using many experiments and parameters", Combust. Flame
(2019), doi:10.1016/j.combustflame.2019.04.023: the H2/O2 mechanism of Burke, Chaos, Ju,
Dryer and Klippenstein (Int. J. Chem. Kinet. 44 (2012) 444-474) plus reactions X1 and X6 of
You, Packard and Frenklach (Int. J. Chem. Kinet. 44 (2012) 101-116), with 31 pre-exponential
factors and third-body efficiencies of the HO2 and H2O2 reactions as uncertain inputs (paper
Table 1, p. 6). Each input has a Gaussian prior with mean = sd = its Table 1 value, truncated
to the bounds L:U of Table 1 (paper Sec. 2.2, p. 5).

Inputs (physical units; order of Table 1):
  A factors in cm-mol-s units (cm^3/mol/s for bimolecular reactions and falloff high-pressure
  limits, cm^6/mol^2/s for termolecular reactions and low-pressure limits); third-body
  efficiencies dimensionless. Each A multiplies its own Arrhenius expression only (the other
  limit of a falloff reaction, and every reaction not listed, keep their Table 1 values).

Quantities of interest (conditions chosen for this data set; see DATA.md):
  tau    ignition delay (s) of a constant-volume H2/O2/Ar reactor (IGNITION)
  Su1    laminar flame speed (m/s) of H2/air, phi = 1, 1 atm, 298 K (FLAMES["Su1"])
  Su20   laminar flame speed (m/s) of H2/air, phi = 0.4, 20 atm, 298 K (FLAMES["Su20"]), the
         mixture and pressure of the paper's 2-D flame kernel (p. 11)

The constants (names, Table 1 values, bounds, prior moments) need only numpy. The solvers need
Cantera (the shipped data were written with Cantera 3.2.0: pip install cantera==3.2.0) and
scipy. Cantera is an outside code, so this module is a driver, not a numpy/scipy model, and the
data set is stored data with a regeneration script in as-data-sets' sense.
"""
import math
import os
import re

import numpy as np

try:
    import cantera as ct
except ImportError:          # constants stay usable without Cantera
    ct = None

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "burke2012_x1x6.yaml")

# --- the 31 inputs of Table 1 (p. 6) ---------------------------------------------------------
# (name, reaction, part, species, Table 1 value, lower bound L, upper bound U)
#   part: "A" (Arrhenius A), "Ainf"/"A0" (falloff high/low-pressure A), "eff" (efficiency)
PARAMS = [
    ("R9_Ainf",    "R9",  "Ainf", None,         4.65084e12, 2e12, 1e13),
    ("R9_A0",      "R9",  "A0",   None,         6.366e20,   0.0,  8e20),
    ("R9_eff_H2",  "R9",  "eff",  ("H2",),      2.0,        0.0,  6.0),
    ("R9_eff_H2O", "R9",  "eff",  ("H2O",),     14.0,       0.0,  28.0),
    ("R9_eff_O2",  "R9",  "eff",  ("O2",),      0.78,       0.0,  3.0),
    ("R9_eff_Ar",  "R9",  "eff",  ("AR",),      0.67,       0.0,  3.0),
    ("R9_eff_He",  "R9",  "eff",  ("HE",),      0.8,        0.0,  3.0),
    ("R10_A",      "R10", "A",    None,         2.750e6,    1e6,  5e16),
    ("R11_A",      "R11", "A",    None,         7.079e13,   2e13, 1e14),
    ("R12_A",      "R12", "A",    None,         2.850e10,   1e9,  1e11),
    ("R13_A",      "R13", "A",    None,         2.890e13,   1e13, 6e13),
    ("R14a_A",     "R14a", "A",   None,         4.200e14,   1e14, 2e15),
    ("R14b_A",     "R14b", "A",   None,         1.300e11,   5e10, 4e11),
    ("R15_Ainf",   "R15", "Ainf", None,         2.00e12,    5e11, 1e12),
    ("R15_A0",     "R15", "A0",   None,         2.49e24,    1e23, 1e25),
    ("R15_eff_H2", "R15", "eff",  ("H2",),      3.7,        0.0,  15.0),
    ("R15_eff_H2O", "R15", "eff", ("H2O",),     7.5,        0.0,  20.0),
    ("R15_eff_H2O2", "R15", "eff", ("H2O2",),   7.7,        0.0,  20.0),
    ("R15_eff_O2", "R15", "eff",  ("O2",),      1.2,        0.0,  5.0),
    ("R15_eff_N2", "R15", "eff",  ("N2",),      1.5,        0.0,  5.0),
    ("R15_eff_He", "R15", "eff",  ("HE",),      0.65,       0.0,  4.0),
    ("R16_A",      "R16", "A",    None,         2.410e13,   5e11, 1e14),
    ("R17_A",      "R17", "A",    None,         4.820e13,   1e12, 9e13),
    ("R18_A",      "R18", "A",    None,         9.550e6,    1e5,  3e7),
    ("R19a_A",     "R19a", "A",   None,         1.740e12,   5e10, 5e12),
    ("R19b_A",     "R19b", "A",   None,         7.590e13,   4e12, 4e14),
    ("X1_A",       "X1",  "A",    None,         3.97e12,    1e12, 9e12),
    ("X6_A",       "X6",  "A",    None,         8.000e15,   2e15, 2e16),
    ("X6_eff_H2",  "X6",  "eff",  ("H2",),      2.0,        0.0,  6.0),
    ("X6_eff_H2O", "X6",  "eff",  ("H2O",),     12.0,       0.0,  35.0),
    ("X6_eff_ArHe", "X6", "eff",  ("AR", "HE"), 0.7,        0.0,  3.0),
]
NAMES = [p[0] for p in PARAMS]
NOMINAL = np.array([p[4] for p in PARAMS])      # Table 1 value = prior mean = prior sd
LOWER = np.array([p[5] for p in PARAMS])
UPPER = np.array([p[6] for p in PARAMS])


def _unit(p):
    """Units of A (cm-mol-s; T in K). R15 (H2O2 (+M) <=> 2 OH (+M)) is unimolecular: s^-1 at
    high pressure, cm^3/mol/s at low pressure. R9 and X6 add a third body at low pressure."""
    name, rxn, part = p[0], p[1], p[2]
    if part == "eff":
        return "-"
    if rxn == "R15":
        return "s^-1" if part == "Ainf" else "cm^3/mol/s"
    if part == "A0" or rxn == "X6":
        return "cm^6/mol^2/s"
    return "cm^3/mol/s"


UNITS = [_unit(p) for p in PARAMS]
M_INPUTS = len(PARAMS)

# Paper reaction labels -> (equation in burke2012_x1x6.yaml, occurrence among duplicates)
REACTION_EQ = {
    "R9": ("H + O2 (+M) <=> HO2 (+M)", 0), "R10": ("H + HO2 <=> H2 + O2", 0),
    "R11": ("H + HO2 <=> 2 OH", 0), "R12": ("HO2 + O <=> O2 + OH", 0),
    "R13": ("HO2 + OH <=> H2O + O2", 0), "R14a": ("2 HO2 <=> H2O2 + O2", 0),
    "R14b": ("2 HO2 <=> H2O2 + O2", 1), "R15": ("H2O2 (+M) <=> 2 OH (+M)", 0),
    "R16": ("H + H2O2 <=> H2O + OH", 0), "R17": ("H + H2O2 <=> H2 + HO2", 0),
    "R18": ("H2O2 + O <=> HO2 + OH", 0), "R19a": ("H2O2 + OH <=> H2O + HO2", 0),
    "R19b": ("H2O2 + OH <=> H2O + HO2", 1), "X1": ("H + HO2 <=> H2O + O", 0),
    "X6": ("O + OH + M <=> HO2 + M", 0),
}


# --- prior: N(mu, mu^2) truncated to [L, U] --------------------------------------------------
def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def prior_moments():
    """Mean and sd of each truncated-Gaussian prior (closed form)."""
    mean, sd = np.empty(M_INPUTS), np.empty(M_INPUTS)
    for k, (mu, lo, up) in enumerate(zip(NOMINAL, LOWER, UPPER)):
        s = mu
        a, b = (lo - mu) / s, (up - mu) / s
        Z = _Phi(b) - _Phi(a)
        r = (_phi(a) - _phi(b)) / Z
        mean[k] = mu + s * r
        sd[k] = s * math.sqrt(1.0 + (a * _phi(a) - b * _phi(b)) / Z - r * r)
    return mean, sd


PRIOR_MEAN, PRIOR_SD = prior_moments()


def sample_prior(M, rng):
    """M independent draws from the 31 truncated-Gaussian priors (inverse-CDF transform of
    uniforms from rng.random((M, 31))). Needs scipy."""
    from scipy.special import ndtr, ndtri
    U = rng.random((M, M_INPUTS))
    a = ndtr((LOWER - NOMINAL) / NOMINAL)
    b = ndtr((UPPER - NOMINAL) / NOMINAL)
    X = NOMINAL + NOMINAL * ndtri(a + U * (b - a))
    return np.clip(X, LOWER, UPPER)


# --- experimental conditions (this data set's choice; DATA.md) -------------------------------
IGNITION = {"T0": 1100.0, "P0_atm": 2.0, "X": "H2:2, O2:1, AR:97"}   # constant volume
FLAMES = {
    "Su1": {"P_atm": 1.0, "phi": 1.0, "T": 298.0, "oxidizer": "O2:1, N2:3.76", "width": 0.03},
    "Su20": {"P_atm": 20.0, "phi": 0.4, "T": 298.0, "oxidizer": "O2:1, N2:3.76", "width": 0.02},
}
REFINE = {"ratio": 2.0, "slope": 0.02, "curve": 0.04, "prune": 0.0}
TRANSPORT = "mixture-averaged"
FD_REL = 1e-3          # ignition-delay gradient step: 1e-3 of each prior sd
ADJ_DP = 1e-5          # flame adjoint: parameter perturbation, 1e-5 of each prior sd
FD_FLAME_STEP = 1e-3   # flame finite differences (FlameSpeed.fd_grad): step, 1e-3 of each prior sd
FD_FLAME_CAP = 0.02    # ... and at most 2 % of the value of an A factor
TIGHT = (1e-9, 1e-14)  # steady-state (rtol, atol) of the fixed-grid solves of fd_grad

# Status codes stored with every flame value
STATUS_OK, STATUS_RETRY, STATUS_FAILED = 0, 1, 2
EFF_PARAMS = [k for k, p in enumerate(PARAMS) if p[2] == "eff"]
SHADOW_REL = 1e-3      # shadow-reaction efficiency offset: 1e-3 of the prior sd (see FlameSpeed)


class H2Kinetics:
    """Builds Cantera Solutions of burke2012_x1x6.yaml with the 31 inputs applied.

    Cantera 3.2 cannot change third-body efficiencies of an existing Solution in place
    (Kinetics.modify_reaction replaces only the rate), so every input vector gets a new Solution
    (about 5 ms). A factors can also be changed in place (set_A)."""

    def __init__(self, mech=MECH):
        if ct is None:
            raise ImportError("this solver needs Cantera: pip install cantera==3.2.0 (the version "
                              "that wrote the shipped data), then rerun")
        ct.suppress_thermo_warnings()
        base = ct.Solution(mech)
        self.species = base.species()
        self.reactions = [base.reaction(j) for j in range(base.n_reactions)]
        eqs = [r.equation for r in self.reactions]
        self.index = {}
        for lab, (eq, occ) in REACTION_EQ.items():
            hits = [j for j, e in enumerate(eqs) if e == eq]
            if len(hits) <= occ:
                raise RuntimeError(f"reaction {lab} ({eq}) not found in {mech}")
            self.index[lab] = hits[occ]
        self.by_rxn = {}
        for k, p in enumerate(PARAMS):
            self.by_rxn.setdefault(p[1], []).append(k)

    def reaction(self, lab, x, eff_bump=None):
        """Reaction lab with inputs x applied; eff_bump = (parameter index, offset) adds the
        offset to that efficiency (the shadow reactions of FlameSpeed)."""
        o = self.reactions[self.index[lab]]
        ks = self.by_rxn[lab]
        rate = o.rate
        if isinstance(rate, ct.TroeRate):
            s_lo = s_hi = 1.0
            for k in ks:
                if PARAMS[k][2] == "A0":
                    s_lo = x[k] / NOMINAL[k]
                elif PARAMS[k][2] == "Ainf":
                    s_hi = x[k] / NOMINAL[k]
            lo, hi = rate.low_rate, rate.high_rate
            new_rate = ct.TroeRate(
                low=ct.ArrheniusRate(lo.pre_exponential_factor * s_lo, lo.temperature_exponent,
                                     lo.activation_energy),
                high=ct.ArrheniusRate(hi.pre_exponential_factor * s_hi, hi.temperature_exponent,
                                      hi.activation_energy),
                falloff_coeffs=rate.falloff_coeffs)
        else:
            s = 1.0
            for k in ks:
                if PARAMS[k][2] == "A":
                    s = x[k] / NOMINAL[k]
            new_rate = ct.ArrheniusRate(rate.pre_exponential_factor * s, rate.temperature_exponent,
                                        rate.activation_energy)
        tb = None
        if o.third_body is not None:
            eff = dict(o.third_body.efficiencies)
            for k in ks:
                if PARAMS[k][2] == "eff":
                    v = float(x[k])
                    if eff_bump is not None and eff_bump[0] == k:
                        v += eff_bump[1]
                    for sp in PARAMS[k][3]:
                        eff[sp] = v
            tb = ct.ThirdBody(efficiencies=eff, default_efficiency=o.third_body.default_efficiency)
        r = ct.Reaction(equation=o.equation, rate=new_rate, third_body=tb)
        r.duplicate = o.duplicate
        return r

    def make_gas(self, x, shadows=False):
        """A new Solution with inputs x (physical units). With shadows=True, two copies of the
        parent reaction of every efficiency input k are appended, with that efficiency raised by
        Delta_k and 2 Delta_k (Delta_k = SHADOW_REL * prior sd), multiplier 0; their indices are
        in self.shadow_index[k]. The shadows change nothing until a multiplier is set."""
        x = np.asarray(x, float)
        if x.shape != (M_INPUTS,):
            raise ValueError(f"x must have {M_INPUTS} entries")
        if np.any(x < 0):
            raise ValueError("A factors and efficiencies must be nonnegative")
        rs = list(self.reactions)
        for lab, j in self.index.items():
            rs[j] = self.reaction(lab, x)
        shadow_index = {}
        if shadows:
            for k in EFF_PARAMS:
                lab = PARAMS[k][1]
                rs[self.index[lab]].duplicate = True
                d = SHADOW_REL * PRIOR_SD[k]
                idx = []
                for mult in (1.0, 2.0):
                    r = self.reaction(lab, x, eff_bump=(k, mult * d))
                    r.duplicate = True
                    idx.append(len(rs))
                    rs.append(r)
                shadow_index[k] = idx
        gas = ct.Solution(thermo="ideal-gas", kinetics="gas", transport_model=TRANSPORT,
                          species=self.species, reactions=rs)
        for js in shadow_index.values():
            for j in js:
                gas.set_multiplier(0.0, j)
        if shadows:
            self.shadow_index = shadow_index
        return gas

    def set_A(self, gas, x, k, value):
        """In place on gas (built with inputs x): set A input k (not an efficiency) to value."""
        if PARAMS[k][2] == "eff":
            raise ValueError("efficiencies cannot be changed in place; build a new gas")
        xp = np.array(x, float)
        xp[k] = value
        lab = PARAMS[k][1]
        gas.modify_reaction(self.index[lab], self.reaction(lab, xp))


class IgnitionDelay(H2Kinetics):
    """Constant-volume ignition delay (s) of IGNITION's mixture. tau is the first time T reaches
    T0 + dT/2, dT = T_eq(UV) - T0 (kinetics-independent), located by cubic Hermite interpolation
    of T between solver steps with the exact dT/dt."""

    def __init__(self, mech=MECH, rtol=1e-9, atol=1e-18, cond=IGNITION):
        super().__init__(mech)
        self.rtol, self.atol, self.cond = rtol, atol, dict(cond)
        g = self.make_gas(NOMINAL)
        g.TPX = cond["T0"], cond["P0_atm"] * ct.one_atm, cond["X"]
        g.equilibrate("UV")
        self.T_eq = g.T
        self.T_c = cond["T0"] + 0.5 * (self.T_eq - cond["T0"])

    def tau(self, x, t_max=10.0):
        g, c = self.make_gas(x), self.cond
        g.TPX = c["T0"], c["P0_atm"] * ct.one_atm, c["X"]
        r = ct.IdealGasReactor(g, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol
        net.max_steps = 500000
        T0, Tc = c["T0"], self.T_c
        T_near = T0 + 0.25 * (Tc - T0)

        def tdot():   # constant volume: dT/dt = -sum(u_k w_k) / (rho cv)
            ph = r.phase
            return -np.dot(ph.partial_molar_int_energies, ph.net_production_rates) / (
                ph.density * ph.cv_mass)

        t_a, T_a, d_a = 0.0, T0, None
        while True:
            t_b = net.step()
            T_b = r.T
            d_b = tdot() if T_b >= T_near else None
            if not np.isfinite(T_b):
                raise RuntimeError("non-finite temperature")
            if T_b >= Tc:
                break
            if t_b > t_max:
                raise RuntimeError(f"no ignition before {t_max} s")
            t_a, T_a, d_a = t_b, T_b, d_b
        if d_a is None:
            raise RuntimeError("ignition front crossed in a single step; tighten rtol")
        h = t_b - t_a

        def T_of(t):
            s = (t - t_a) / h
            return ((2 * s**3 - 3 * s**2 + 1) * T_a + (s**3 - 2 * s**2 + s) * h * d_a
                    + (-2 * s**3 + 3 * s**2) * T_b + (s**3 - s**2) * h * d_b - Tc)

        from scipy.optimize import brentq
        return brentq(T_of, t_a, t_b, xtol=1e-15 * max(t_b, 1e-12), rtol=4 * np.finfo(float).eps)

    def grad(self, x, rel=FD_REL):
        """Central differences in the physical inputs, step rel * prior sd; a second-order
        one-sided difference where x - step would be negative."""
        x = np.asarray(x, float)
        g = np.empty(M_INPUTS)
        f0 = None
        for k in range(M_INPUTS):
            h = rel * PRIOR_SD[k]
            e = np.zeros(M_INPUTS)
            e[k] = h
            if x[k] - h >= 0:
                g[k] = (self.tau(x + e) - self.tau(x - e)) / (2 * h)
            else:
                if f0 is None:
                    f0 = self.tau(x)
                g[k] = (-3 * f0 + 4 * self.tau(x + e) - self.tau(x + 2 * e)) / (2 * h)
        return g


class FlameSpeed(H2Kinetics):
    """Laminar speed Su (m/s) of a freely propagating, unstrained, adiabatic premixed flame
    (Cantera FreeFlame, mixture-averaged transport, no Soret term), and dSu/dx by central finite
    differences on the fixed grid of the flame (fd_grad, the method of the stored gradients) or
    from Cantera's discrete adjoint (adjoint_grad, Sim1D.solve_adjoint; kept for comparison: for
    these flames its scale is wrong, see DATA.md, "Accuracy").

    The adjoint needs the derivative of the discrete residual with respect to each input, which
    solve_adjoint takes by perturbing the input in place. A factors are perturbed in place. An
    efficiency cannot be (see H2Kinetics), so the adjoint is evaluated on a copy of the flame whose
    gas carries two shadow copies of the parent reaction, with that efficiency raised by Delta and
    2 Delta; the rate of progress at efficiency eps + delta is then
        q(eps) (1 - 3 t / 2) + q(eps + Delta) 2 t - q(eps + 2 Delta) t / 2,   t = delta / Delta,
    set with reaction multipliers: the second-order one-sided difference of q in the efficiency
    (truncation error O(Delta^2), Delta = 1e-3 prior sd). With all shadow multipliers 0 the
    residual is that of the plain mechanism, so the converged plain solution is restored into the
    copy unchanged."""

    def __init__(self, case, mech=MECH, refine=REFINE):
        super().__init__(mech)
        self.case, self.c, self.refine = case, dict(FLAMES[case]), dict(refine)
        self._nominal_solution = None     # solved on first use (the retry start)
        self.flame = None

    @property
    def nominal_solution(self):
        """The converged flame at the Table 1 values (SolutionArray), the start of a retry."""
        if self._nominal_solution is None:
            f = self.new_flame(self.make_gas(NOMINAL))
            f.solve(loglevel=0, auto=True)
            self._nominal_solution = f.to_array()
        return self._nominal_solution

    def new_flame(self, gas):
        c = self.c
        gas.TP = c["T"], c["P_atm"] * ct.one_atm
        gas.set_equivalence_ratio(c["phi"], "H2", c["oxidizer"])
        f = ct.FreeFlame(gas, width=c["width"])
        f.transport_model = TRANSPORT
        f.set_refine_criteria(**self.refine)
        f.max_time_step_count = 50000
        return f

    def _ok(self, f):
        su = float(f.velocity[0])
        return bool(np.isfinite(su) and su > 0 and f.T[-1] > self.c["T"] + 300.0)

    def solve(self, x):
        """Su at x and a status code: STATUS_OK (solved from scratch with auto=True),
        STATUS_RETRY (that failed; solved from the nominal solution with time stepping and grid
        refinement) or STATUS_FAILED (both failed; Su is NaN). The converged flame is kept in
        self.flame for adjoint_grad."""
        self.x = np.array(x, float)
        for status in (STATUS_OK, STATUS_RETRY):
            try:
                f = self.new_flame(self.make_gas(self.x))
                if status == STATUS_OK:
                    f.solve(loglevel=0, auto=True)
                else:
                    f.from_array(self.nominal_solution)
                    f.solve(loglevel=0, refine_grid=True, auto=False)
                if self._ok(f):
                    self.flame = f
                    return float(f.velocity[0]), status
            except ct.CanteraError:
                pass
        self.flame = None
        return float("nan"), STATUS_FAILED

    def adjoint_grad(self, dp=ADJ_DP):
        """dSu/dx (physical units) at the flame of the last successful solve."""
        if self.flame is None:
            raise RuntimeError("no converged flame")
        x0 = self.x
        gas = self.make_gas(x0, shadows=True)
        f = self.new_flame(gas)
        f.from_array(self.flame.to_array())
        nvars = sum(d.n_components * d.n_points for d in f.domains)
        dgdx = np.zeros(nvars)
        dgdx[f.inlet.n_components + f.flame.component_index("velocity")] = 1.0
        base_index = {k: self.index[PARAMS[k][1]] for k in range(M_INPUTS)}

        def perturb(sim, i, d):
            delta = d * PRIOR_SD[i]
            if PARAMS[i][2] == "eff":
                t = delta / (SHADOW_REL * PRIOR_SD[i])
                j1, j2 = self.shadow_index[i]
                gas.set_multiplier(1.0 - 1.5 * t, base_index[i])
                gas.set_multiplier(2.0 * t, j1)
                gas.set_multiplier(-0.5 * t, j2)
            else:
                self.set_A(gas, x0, i, x0[i] + delta)

        g = np.array(f.solve_adjoint(perturb, M_INPUTS, dgdx, dp=dp), float)
        return g / PRIOR_SD

    def fd_grad(self, x, step=FD_FLAME_STEP, rel_cap=FD_FLAME_CAP):
        """Central finite differences dSu/dx (physical units) on the fixed grid of the flame at x.

        1. Su from scratch exactly as solve() (su_auto, status).
        2. The same from-scratch solve with the shadow gas (make_gas(x, shadows=True); the shadows
           have multiplier 0, so the kinetics and the grid are those of step 1, checked in
           grid_match), then tightened in place on its own grid (steady-state tolerances TIGHT):
           su_base. No from_array restore: a restored flame is re-anchored by time stepping and can
           land on a different discrete solution (Su shifted by about 1e-3).
        3. Each input perturbed IN PLACE on that one flame and re-solved on the fixed grid, then reset
           and re-solved back to the base (drift recorded): an A factor by modify_reaction with
           A +- h, h = step * prior sd, at most rel_cap * A; an efficiency by the multipliers
           (parent, shadow 1) = (0, 1) for +h and (2, -1) for -h, h = SHADOW_REL * prior sd (exact
           for plain three-body reactions, which are linear in the efficiency; for the falloff
           reactions R9 and R15 it is a forward difference of the rate in the efficiency, first order
           in h, with a relative error of the rate derivative below 1e-4). An A factor equal to 0 gets the forward difference
           (flag 1); a failed solve gives NaN (flag 3).
        Also returned: the forward difference and a central difference with a step ten times larger
        (efficiencies by linear extrapolation of the multipliers), as a step check.
        Returns a dict; "grad" is the result (NaN entries where flag is 3)."""
        x = np.array(x, float)
        out = {}
        su_auto, status = self.solve(x)
        out.update(su_auto=su_auto, status=status)
        if status == STATUS_FAILED:
            return dict(out, ok=False, grad=np.full(M_INPUTS, np.nan))
        plain_grid = np.array(self.flame.grid)
        gas = self.make_gas(x, shadows=True)
        f = self.new_flame(gas)
        base_mode = 0
        try:
            f.solve(loglevel=0, auto=True)
            if not self._ok(f):
                raise ct.CanteraError("no flame")
        except ct.CanteraError:
            base_mode = 1                      # fallback: restore the plain flame
            f = self.new_flame(gas)
            f.from_array(self.flame.to_array())
            f.inlet.mdot = self.flame.inlet.mdot
        su_shadow_auto = float(f.velocity[0]) if base_mode == 0 else float("nan")
        grid_match = bool(base_mode == 0 and len(f.grid) == len(plain_grid)
                          and np.array_equal(f.grid, plain_grid))
        fixed_T_auto = float(f.fixed_temperature)
        f.flame.set_steady_tolerances(default=TIGHT)

        def resolve():
            f.solve(loglevel=0, refine_grid=False, auto=False)
            return float(f.velocity[0])

        su_base = resolve()
        if not self._ok(f):
            return dict(out, ok=False, grad=np.full(M_INPUTS, np.nan))
        m, sd = M_INPUTS, PRIOR_SD
        vp, vm, vp10, vm10 = (np.full(m, np.nan) for _ in range(4))
        flag, drift, hh = np.zeros(m, int), np.zeros(m), np.empty(m)
        for k in range(m):
            eff = PARAMS[k][2] == "eff"
            h = SHADOW_REL * sd[k] if eff else step * sd[k]
            if not eff and x[k] > 0:
                h = min(h, rel_cap * x[k])
            hh[k] = h
            j0 = self.index[PARAMS[k][1]]
            try:
                if eff:
                    j1 = self.shadow_index[k][0]
                    for (m0, m1), store in (((0.0, 1.0), vp), ((2.0, -1.0), vm), ((-9.0, 10.0), vp10),
                                            ((11.0, -10.0), vm10)):
                        gas.set_multiplier(m0, j0)
                        gas.set_multiplier(m1, j1)
                        store[k] = resolve()
                    gas.set_multiplier(1.0, j0)
                    gas.set_multiplier(0.0, j1)
                else:
                    for dx, store in ((h, vp), (-h, vm), (10 * h, vp10), (-10 * h, vm10)):
                        if x[k] + dx < 0:
                            continue
                        self.set_A(gas, x, k, x[k] + dx)
                        store[k] = resolve()
                    self.set_A(gas, x, k, x[k])
                drift[k] = resolve() / su_base - 1
            except ct.CanteraError:
                flag[k] = 3
                vp[k] = vm[k] = np.nan
                try:                           # back to the base state before the next input
                    if eff:
                        gas.set_multiplier(1.0, j0)
                        gas.set_multiplier(0.0, self.shadow_index[k][0])
                    else:
                        self.set_A(gas, x, k, x[k])
                    drift[k] = resolve() / su_base - 1
                except ct.CanteraError:
                    flag[k + 1:] = 3
                    break
        g = (vp - vm) / (2 * hh)
        fwd = (vp - su_base) / hh
        one = ~np.isfinite(vm) & np.isfinite(vp) & (flag != 3)
        g[one], flag[one] = fwd[one], 1
        out.update(ok=bool(np.all(np.isfinite(g)) and np.all(flag != 3)), su_base=su_base, grad=g,
                   grad_forward=fwd, grad_central10=(vp10 - vm10) / (20 * hh), h=hh, flag=flag,
                   drift=drift, su_plus=vp, su_minus=vm, su_plus10=vp10, su_minus10=vm10,
                   n_points=len(f.grid), base_mode=base_mode, su_shadow_auto=su_shadow_auto,
                   grid_match=grid_match, fixed_T_auto=fixed_T_auto,
                   fixed_T_base=float(f.fixed_temperature))
        return out

    def fixed_grid_su(self, x, flame=None):
        """Su with inputs x re-solved on the grid of a converged flame (no refinement), starting
        from that flame's solution (for finite-difference checks of adjoint_grad)."""
        src = self.flame if flame is None else flame
        f = self.new_flame(self.make_gas(x))
        f.from_array(src.to_array())
        f.inlet.mdot = src.inlet.mdot      # from_array does not restore the inlet mass flux
        f.solve(loglevel=0, refine_grid=False, auto=False)
        return float(f.velocity[0])


# --- mechanism file ----------------------------------------------------------------------------
def build_mechanism(src, out):
    """Write burke2012_x1x6.yaml from PelePhysics' Mechanisms/BurkeDryer/mechanism.yaml (Burke et
    al. 2012, version 6-10-2011, N2-bath R9 block): drop the inert species CO and CO2 and their
    collision efficiencies, and append reactions X1 and X6 with the values of Bell et al.
    (2019) Table 1. Text edits only; every other line is the source's."""
    t = open(src).read()
    t = re.sub(r"- name: CO\n.*?(?=\nreactions:)", "", t, flags=re.S)
    t = t.replace("elements: [H, O, N, Ar, He, C]", "elements: [H, O, N, Ar, He]")
    t = t.replace(", CO, CO2]", "]")
    t = re.sub(r",\s*CO2?: [0-9.]+", "", t)
    t = re.sub(r"\{CO2?: [0-9.]+,\s*", "{", t)
    t = re.sub(r"\s*CO2?: [0-9.]+,", "", t)
    head = ("# Burke et al. (2012) H2/O2 mechanism (version 6-10-2011) as distributed in PelePhysics\n"
            "# (Mechanisms/BurkeDryer/mechanism.yaml), with the inert species CO and CO2 and their\n"
            "# collision efficiencies removed and reactions X1 and X6 of Bell et al. (2019) Table 1\n"
            "# (from You et al. 2012) appended. Written by h2_flame_driver.build_mechanism.\n")
    t = head + t.rstrip("\n") + "\n" + (
        "- equation: HO2 + H <=> H2O + O  # X1 of Bell et al. (2019) Table 1, from You et al. (2012)\n"
        "  rate-constant: {A: 3.97e+12, b: 0.0, Ea: 671.0}\n"
        "- equation: O + OH + M <=> HO2 + M  # X6 of Bell et al. (2019) Table 1, from You et al. (2012)\n"
        "  type: three-body\n"
        "  rate-constant: {A: 8.0e+15, b: 0.0, Ea: 0.0}\n"
        "  efficiencies: {H2: 2.0, H2O: 12.0, AR: 0.7, HE: 0.7}\n")
    with open(out, "w") as fh:
        fh.write(t)
