"""Ammonia ignition delay times with perturbed rate constants of 41 reactions (Cantera).

The model is the optimized ammonia/syngas mechanism of Li, Zou and Yao, "An optimized model
for ammonia/syngas combustion", React. Chem. Eng. 8 (2023) 2071-2085,
doi:10.1039/D3RE00160A (37 species; the CHEMKIN files of the article's electronic
supplementary information, converted with Cantera's ck2yaml by convert_mechanism.py to
ammonia_syngas_optimized.yaml). The mechanism file is not part of this data set: see MECH and
MECH_HELP below for the download and the conversion. The paper optimized
the Arrhenius parameters (A, b, E) of 41 reactions (123 parameters; Table 2, p. 2076; for the
falloff reactions, those of the low-pressure limit). Each reaction has a temperature-dependent
uncertainty factor f(T) (ESI Table S3) and the feasible region of eqn (4)-(5), p. 2074:

    | log10( k(T) / k0(T) ) | <= f(T)   for 300 K <= T <= 3000 K,
    k(T) = A T^b exp(-E / (R T)).

The paper's Table 2 prints E* = 3.564e4 cal/mol for NH2+H=NH+H2; the supplementary mechanism file,
used here, has 3.5639e3 (check_accuracy.py prints both).

The paper's nominal (trial-model) rate constants k0 are not published. This module centres
the region on the optimized rate constants k* (the ESI mechanism file), which is this data set's
choice:  k(T) = k*(T) exp(alpha + beta ln T - eps / T)  with  alpha = ln(A/A*),
beta = b - b*,  eps = (E - E*)/R,  the paper's eqn (5) variables measured from k*.

Coordinates. For each reaction r the three inputs are the log10 rate deviations at three
node temperatures, x_{r,i} = log10(k_r(T_i) / k*_r(T_i)), T = (300, 750, 3000) K. They
determine (alpha, beta, eps) by a 3 x 3 linear solve (to_theta). Bounds are
|x_{r,i}| <= f_r(T_i); the feasible region is the subset of that box where the constraint
holds at every temperature in [300, 3000] K (checked on a 1 K grid, feasible()).

Quantities of interest: ignition delay times tau (s) at three conditions (CONDITIONS),
adiabatic constant-volume reactor, the ignition criterion of the experiment named for each.

Needs numpy; IgnitionModel also needs Cantera (the shipped data were written with Cantera
3.2.0: pip install cantera==3.2.0). Cantera is an outside code, so this module is a driver;
the data set is stored data with a regeneration script (DATA.md).
"""
import os

import numpy as np

try:
    import cantera as ct
except ImportError:          # constants and geometry stay usable without Cantera
    ct = None

HERE = os.path.dirname(os.path.abspath(__file__))
# The mechanism is a conversion of the article's supplementary file, which is not part of this
# data set (DATA.md, "Regenerating"). It is read from the path in the environment variable
# AMMONIA_SYNGAS_MECH if that is set, or else from this file name next to the driver.
MECH = os.environ.get("AMMONIA_SYNGAS_MECH", os.path.join(HERE, "ammonia_syngas_optimized.yaml"))
MECH_HELP = (f"the mechanism file was not found at {MECH}. It is not part of this data set. "
             "Download d3re00160a1_suppl.zip (65 231 bytes, sha256 "
             "ee493371ad68aef892269bc3d19951609caa364d4ccd7d83112c3d0414a86a6b) from the article page "
             "https://doi.org/10.1039/D3RE00160A, link 'Electronic supplementary information'; convert it "
             "with 'python convert_mechanism.py --zip d3re00160a1_suppl.zip --out PATH' (Cantera 3.2.0); "
             "then set AMMONIA_SYNGAS_MECH=PATH, or use as PATH ammonia_syngas_optimized.yaml next to "
             "ammonia_idt_driver.py, and rerun")

# The 41 reactions of Table 2 (p. 2076) and ESI Table S3, in that order, as printed there.
REACTIONS = [
    "HNO+OH=NO+H2O", "HNO+O2=HO2+NO", "NO+H(+M)=HNO(+M)", "N2O(+M)=N2+O(+M)", "N2O+H=N2+OH",
    "NH2+H(+M)=NH3(+M)", "NH3+H=NH2+H2", "NH3+HO2=NH2+H2O2", "NH2+H=NH+H2", "NH2+O=HNO+H",
    "NH2+HO2=H2NO+OH", "NH2+O2=H2NO+O", "NH2+NH=N2H2+H", "NH2+NO=N2+H2O", "NH2+NO=NNH+OH",
    "NH2+HONO=NH3+NO2", "NH2+NO2=N2O+H2O", "NH2+NO2=H2NO+NO", "NH+O2=HNO+O", "NH+NO=N2O+H",
    "N+NO=N2+O", "NH2+NH2(+M)=N2H4(+M)", "N2H4+NH2=N2H3+NH3", "N2H3+M=N2H2+H+M",
    "N2H3+HO2=N2H4+O2", "N2H3+NH2=H2NN+NH3", "N2H2+M=NNH+H+M", "N2H2+H=NNH+H2",
    "N2H2+NO=N2O+NH2", "NH2+NH2=N2H2+H2", "H2NN=N2+H2", "H2NN+O2=NH2+NO2",
    "H2NO+HO2=HNO+H2O2", "H2NO+O2=HNO+HO2", "HNOH+NH2=H2NN+H2O", "H2NO+NH2=HNO+NH3",
    "H2NO+H=HNO+H2", "H2NO+H=NH2+OH", "H2NO+O=HNO+OH", "H2NO+OH=HNO+H2O", "H2NO+NO2=HONO+HNO",
]
# Cantera reaction indices in ammonia_syngas_optimized.yaml. N2O+H=N2+OH has two DUPLICATE
# Arrhenius entries (Table 2 lists both with one adjustment); one (alpha, beta, eps) acts on both.
CANTERA_INDEX = [
    [60], [61], [65], [97], [98, 99], [105], [106], [109], [110], [111], [238], [119], [122],
    [126], [127], [128], [129], [237], [135], [141], [148], [173], [180], [184], [194], [196],
    [198], [199], [205], [207], [209], [218], [239], [236], [233], [240], [241], [242], [243],
    [244], [245],
]
# Uncertainty factor f(T) of ESI Table S3: a constant, or (T [K], f) nodes, piecewise linear
# in T between nodes and constant outside them (Table S3 footnote).
F_TABLE = [
    0.48, 0.5, [(400, 0.5), (900, 0.55), (2000, 1.0)], [(700, 1.0), (1200, 0.75), (3200, 1.0)],
    [(500, 1.60), (1200, 0.70), (2600, 0.60)], 0.4,
    [(400, 1.2), (650, 0.9), (1500, 0.45), (2500, 0.9)], 0.7,
    [(300, 0.15), (800, 0.9), (1700, 0.15), (2800, 0.3)], 0.7, 1.0, 0.3, 0.7,
    [(298, 0.3), (1200, 0.3), (2000, 0.5), (3300, 0.5)],
    [(298, 0.3), (1200, 0.3), (2000, 0.5), (3300, 0.5)], 0.7, 0.7, 0.7, 0.7, 0.3, 0.2, 0.3,
    0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.3, 0.5, 0.5, 0.5, 0.6, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7,
    0.7,
]
N_REACTIONS = 41
T_NODES = np.array([300.0, 750.0, 3000.0])          # K, node temperatures of the coordinates
T_RANGE = (300.0, 3000.0)                           # K, range of the constraint (eqn (4))
T_GRID = np.arange(T_RANGE[0], T_RANGE[1] + 0.5, 1.0)


def f_of_T(r, T):
    """Uncertainty factor f_r(T) of reaction r (0-based), ESI Table S3."""
    spec = F_TABLE[r]
    T = np.asarray(T, float)
    if np.isscalar(spec):
        return np.full(T.shape, float(spec))
    a = np.array(spec, float)
    return np.interp(T, a[:, 0], a[:, 1])


def basis(T):
    """Rows (1, ln T, -1/T): ln(k/k*) = basis(T) @ (alpha, beta, eps)."""
    T = np.asarray(T, float)
    return np.stack([np.ones_like(T), np.log(T), -1.0 / T], axis=-1)


NODE_BASIS = basis(T_NODES)                         # 3 x 3
NODE_INV = np.linalg.inv(NODE_BASIS)
LN10 = np.log(10.0)

NAMES = [f"R{r + 1:02d}_{int(T)}K" for r in range(N_REACTIONS) for T in T_NODES]
UPPER = np.array([f_of_T(r, T) for r in range(N_REACTIONS) for T in T_NODES], float)
LOWER = -UPPER
NOMINAL = np.zeros(3 * N_REACTIONS)                 # x = 0 is the optimized model


def to_theta(x):
    """Node coordinates x (123,) -> (alpha, beta [-], eps [K]) per reaction, shape (41, 3)."""
    X = np.asarray(x, float).reshape(N_REACTIONS, 3)
    return (LN10 * X) @ NODE_INV.T


def from_theta(theta):
    th = np.asarray(theta, float).reshape(N_REACTIONS, 3)
    return (th @ NODE_BASIS.T / LN10).ravel()


def violation(x):
    """max_T |log10(k/k*)| / f(T) per reaction (<= 1 means inside the feasible region)."""
    th = to_theta(x)
    B = basis(T_GRID)
    out = np.empty(N_REACTIONS)
    for r in range(N_REACTIONS):
        out[r] = np.max(np.abs(B @ th[r]) / LN10 / f_of_T(r, T_GRID))
    return out


def feasible(x):
    return bool(np.all(violation(x) <= 1.0))


def sample_feasible(n, rng):
    """n points uniform on the feasible region: each reaction's 3 node coordinates are drawn
    uniformly in their box and redrawn until that reaction satisfies eqn (4) on [300, 3000] K.
    Returns (X, draws), draws[r] = number of box draws used for reaction r."""
    B = basis(T_GRID)
    fgrid = [f_of_T(r, T_GRID) for r in range(N_REACTIONS)]
    X = np.empty((n, 3 * N_REACTIONS))
    draws = np.zeros(N_REACTIONS, int)
    for k in range(n):
        for r in range(N_REACTIONS):
            ub = UPPER[3 * r:3 * r + 3]
            while True:
                u = ub * (2.0 * rng.random(3) - 1.0)
                draws[r] += 1
                th = NODE_INV @ (LN10 * u)
                if np.all(np.abs(B @ th) <= LN10 * fgrid[r]):
                    X[k, 3 * r:3 * r + 3] = u
                    break
    return X, draws


# Ignition conditions (initial state of the adiabatic constant-volume reactor).
#   key: (T0 [K], P0 [Pa], mole fractions, criterion, description)
CONDITIONS = {
    "A": (1850.0, 1.4 * 101325.0,
          {"NH3": 0.02 * 4.0 / 7.0, "O2": 0.02 * 3.0 / 7.0, "AR": 0.98}, "OH*",
          "shock tube, stoichiometric NH3/O2 in 98 % Ar, 1.4 atm, 1850 K (Mathieu and Petersen "
          "2015; ESI Table S1, Fig. S14a); ignition at max d[OH*]/dt"),
    "B": (1400.0, 20.0 * 101325.0,
          {"NH3": 1.0, "O2": 0.75, "N2": 0.75 * 79.0 / 21.0}, "P",
          "shock tube, stoichiometric NH3/air, 20 atm, 1400 K (Shu et al. 2019; ESI Table S1); "
          "ignition at max dP/dt"),
    "C": (1100.0, 40.0e5,
          {"NH3": 0.12, "O2": 0.18, "AR": 0.70}, "P",
          "rapid compression machine end-of-compression state, NH3/O2/Ar = 0.12/0.18/0.70, "
          "40 bar, 1100 K (He et al. 2019; ESI Table S1, Fig. S14b), adiabatic constant volume "
          "(no volume profile); ignition at max dP/dt"),
}
QOIS = list(CONDITIONS)

RTOL, ATOL = 1e-8, 1e-20     # CVODES tolerances of the shipped data
T_MAX = 1.0                  # s; no ignition before it raises an error
FD_REL = 1e-3                # forward-difference step: 1e-3 of each input's half-range
STEPS = FD_REL * UPPER


class IgnitionModel:
    """Ignition delay solver. Not thread-safe (one Cantera Solution per instance)."""

    def __init__(self, rtol=RTOL, atol=ATOL, mech=MECH):
        if ct is None:
            raise ImportError("IgnitionModel needs Cantera: pip install cantera==3.2.0 (the version "
                              "that wrote the shipped data), then rerun")
        if not os.path.exists(mech):
            raise FileNotFoundError(MECH_HELP)
        import warnings
        with warnings.catch_warnings():       # NASA-polynomial discontinuity notice for OH*
            warnings.simplefilter("ignore")
            self.gas = ct.Solution(mech)
            eq = ct.Solution(mech)
        self.rtol, self.atol = rtol, atol
        self.orig = {j: self.gas.reaction(j).rate for idx in CANTERA_INDEX for j in idx}
        self.k_oh_star = self.gas.species_index("OH*")
        self.T_eq = {}
        for key, (T0, P0, X, _, _) in CONDITIONS.items():
            eq.TPX = T0, P0, X
            eq.equilibrate("UV")
            self.T_eq[key] = eq.T
        self._x = None

    def set_x(self, x):
        """Apply node coordinates x (123,) to the 41 reactions."""
        x = np.asarray(x, float)
        if x.shape != (3 * N_REACTIONS,):
            raise ValueError("x must have 123 entries")
        if self._x is not None and np.array_equal(x, self._x):
            return
        theta = to_theta(x)
        R = ct.gas_constant                     # J/(kmol K)
        for r, idx in enumerate(CANTERA_INDEX):
            a, b, e = theta[r]
            for j in idx:
                rxn = self.gas.reaction(j)
                o = self.orig[j]
                if hasattr(o, "low_rate"):      # falloff: perturb the low-pressure limit only
                    lo = o.low_rate
                    low = ct.ArrheniusRate(lo.pre_exponential_factor * np.exp(a),
                                           lo.temperature_exponent + b,
                                           lo.activation_energy + R * e)
                    if isinstance(o, ct.TroeRate):
                        rxn.rate = ct.TroeRate(low=low, high=o.high_rate,
                                               falloff_coeffs=o.falloff_coeffs)
                    else:
                        rxn.rate = ct.LindemannRate(low=low, high=o.high_rate)
                else:
                    rxn.rate = ct.ArrheniusRate(o.pre_exponential_factor * np.exp(a),
                                                o.temperature_exponent + b,
                                                o.activation_energy + R * e)
                self.gas.modify_reaction(j, rxn)
        self._x = x.copy()

    def ignition_delay(self, key, x=None, return_history=False):
        """tau (s) at condition key for node coordinates x (None = optimized model)."""
        self.set_x(NOMINAL if x is None else x)
        T0, P0, X, crit, _ = CONDITIONS[key]
        gas = self.gas
        gas.TPX = T0, P0, X
        r = ct.IdealGasReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol
        net.max_steps = 1000000
        T_done = T0 + 0.9 * (self.T_eq[key] - T0)
        ts, ss, Ts = [0.0], [0.0], [T0]
        smax = 0.0
        T_watch = T0 + 0.02 * (self.T_eq[key] - T0)   # the signal is evaluated above this T
        while True:
            t = net.step()
            g = r.phase
            if g.T < T_watch:
                if t > T_MAX:
                    raise RuntimeError(f"no ignition before {T_MAX} s")
                ts.append(t)
                ss.append(0.0)
                Ts.append(g.T)
                continue
            w = g.net_production_rates
            if crit == "OH*":
                s = w[self.k_oh_star]          # d[OH*]/dt at constant volume, kmol/m3/s
            else:
                Tdot = -np.dot(g.partial_molar_int_energies, w) / (g.density * g.cv_mass)
                s = g.P * (Tdot / g.T + w.sum() / g.density_mole)     # dP/dt, Pa/s
            if not np.isfinite(s):
                raise RuntimeError("non-finite state")
            ts.append(t)
            ss.append(s)
            Ts.append(g.T)
            smax = max(smax, s)
            if g.T > T_done and s < 0.2 * smax:
                break
            if t > T_MAX:
                raise RuntimeError(f"no ignition before {T_MAX} s")
        ts, ss = np.array(ts), np.array(ss)
        i = int(np.argmax(ss))
        if i < 1 or i > len(ts) - 2:
            raise RuntimeError("maximum of the ignition signal at the end of the record")
        t1, t2, t3 = ts[i - 1:i + 2]
        s1, s2, s3 = ss[i - 1:i + 2]
        d1 = (s2 - s1) / (t2 - t1)
        d2 = (s3 - s2) / (t3 - t2)
        curv = (d2 - d1) / (t3 - t1)
        tau = 0.5 * (t1 + t2) - d1 / (2.0 * curv)   # vertex of the parabola through 3 steps
        if not (t1 <= tau <= t3):
            raise RuntimeError("parabolic peak outside its bracket")
        if return_history:
            return tau, np.column_stack([ts, ss, np.array(Ts)])
        return tau

    def taus(self, x):
        return np.array([self.ignition_delay(k, x) for k in QOIS])

    def log10_tau_and_grad(self, x, steps=STEPS):
        """log10 tau at every condition and forward-difference gradients d log10 tau / dx."""
        x = np.asarray(x, float)
        tau0 = self.taus(x)
        f0 = np.log10(tau0)
        G = np.empty((len(QOIS), x.size))
        for j in range(x.size):
            xp = x.copy()
            xp[j] += steps[j]
            G[:, j] = (np.log10(self.taus(xp)) - f0) / steps[j]
        return tau0, G

    def central_grad(self, x, steps=STEPS):
        x = np.asarray(x, float)
        G = np.empty((len(QOIS), x.size))
        for j in range(x.size):
            xp, xm = x.copy(), x.copy()
            xp[j] += steps[j]
            xm[j] -= steps[j]
            G[:, j] = (np.log10(self.taus(xp)) - np.log10(self.taus(xm))) / (2.0 * steps[j])
        return G
