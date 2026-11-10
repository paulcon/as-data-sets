"""Cantera driver for the H2/air case of Ji et al. (2018), Combust. Flame 190:146-157.

Mechanism: A. A. Konnov, Combust. Flame 152 (2008) 507-528 (10 species, 33 reactions),
converted to Cantera YAML by build_mechanism.py from the publisher's supplementary files
(not shipped). Uncertainty factors UF_j: Konnov (2008) Table 1, pp. 509-510.

Inputs x_j (j = 1..33), one per reaction in mechanism order, normalized as in Ji et al.
Eq. (2): ln(k_j / k_j0) = x_j * ln(UF_j) / 3, x_j ~ N(0, 1) truncated to [-3, 3].
The perturbation multiplies the whole rate of reaction j (Cantera's rate multiplier),
which scales falloff and three-body rates exactly; the reverse rate follows through the
equilibrium constant.

Quantities of interest
  ignition_delay(P_atm, T0, x): constant-volume adiabatic ignition of stoichiometric H2/air
      (H2 : O2 : N2 = 2 : 1 : 3.76); tau = first time T reaches T0 + 400 K (s).
  flame_speed(P_atm, x): freely propagating, adiabatic, stoichiometric H2/air premixed
      flame at T_u = 300 K; S_L in m/s (mixture-averaged transport).

The constants (NAMES, UF, ...) import without Cantera; the models need Cantera 3.2.0.
"""
import os

import numpy as np

try:
    import cantera as ct
except ImportError:  # constants stay importable
    ct = None

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "konnov2008_h2.yaml")
ATM = 101325.0

# (Konnov Table 1 number, equation, UF) in the order of the mechanism file (33 reactions).
# Duplicate expressions are separate Cantera reactions and separate inputs; for R16, R17a and
# R21 each expression takes the UF printed on its own line (R21 prints one UF for both).
REACTIONS = [
    ("1a", "H+H+M=H2+M", 2.0),
    ("1b", "H+H+H2=H2+H2", 2.5),
    ("1c", "H+H+N2=H2+N2", 3.2),
    ("1d", "H+H+H=H2+H", 3.2),
    ("2", "O+O+M=O2+M", 2.0),
    ("3", "O+H+M=OH+M", 3.0),
    ("4a", "H2O+M=H+OH+M", 2.0),
    ("4b", "H2O+H2O=H+OH+H2O", 2.0),
    ("5a", "H+O2(+M)=HO2(+M)", 1.2),
    ("5b", "H+O2(+Ar)=HO2(+Ar)", 1.2),
    ("5c", "H+O2(+O2)=HO2(+O2)", 1.2),
    ("5d", "H+O2(+H2O)=HO2(+H2O)", 1.4),
    ("6a", "OH+OH(+M)=H2O2(+M)", 2.5),
    ("6b", "OH+OH(+H2O)=H2O2(+H2O)", 2.5),
    ("7", "O+H2=OH+H", 1.3),
    ("8", "H+O2=OH+O", 1.5),
    ("9", "H2+OH=H2O+H", 2.0),
    ("10", "OH+OH=H2O+O", 1.5),
    ("11", "HO2+O=OH+O2", 1.2),
    ("12", "H+HO2=OH+OH", 2.0),
    ("13", "H+HO2=H2O+O", 3.0),
    ("14", "H+HO2=H2+O2", 2.0),
    ("15", "H2+O2=OH+OH", 3.0),
    ("16 (1st)", "HO2+OH=H2O+O2", 3.0),
    ("16 (2nd)", "HO2+OH=H2O+O2", 3.0),
    ("17a (1st)", "HO2+HO2=H2O2+O2", 2.5),
    ("17a (2nd)", "HO2+HO2=H2O2+O2", 1.4),
    ("17b", "HO2+HO2+M=H2O2+O2+M", 1.4),
    ("18", "H2O2+H=HO2+H2", 3.0),
    ("19", "H2O2+H=H2O+OH", 2.0),
    ("20", "H2O2+O=HO2+OH", 3.0),
    ("21 (1st)", "H2O2+OH=HO2+H2O", 2.0),
    ("21 (2nd)", "H2O2+OH=HO2+H2O", 2.0),
]
M_IN = len(REACTIONS)
NAMES = [f"x{j + 1}" for j in range(M_IN)]          # x_j <-> mechanism reaction j+1 (R1..R33)
KONNOV_NO = [r[0] for r in REACTIONS]
EQUATIONS = [r[1] for r in REACTIONS]
UF = np.array([r[2] for r in REACTIONS])
SIGMA = np.log(UF) / 3.0                             # sd of ln k_j (Ji et al. Eq. 1)
LOWER = -3.0 * np.ones(M_IN)                         # truncation k0/UF .. k0*UF
UPPER = 3.0 * np.ones(M_IN)

# Conditions (Ji et al. Table 2, p. 151; the pressures and temperatures of Sec. 3.2, p. 152)
IDT_P = [1.0, 2.0, 5.0, 10.0, 20.0]                  # atm
IDT_T = [1000.0, 1400.0, 1800.0]                     # K
SL_P = [1.0, 2.0, 5.0, 10.0, 20.0]                   # atm, T_u = 300 K
FIG1 = (1.59e5 / ATM, 1000.0)                        # Fig. 1: 1.59 bar, 1000 K
IDT_CONDITIONS = [FIG1] + [(p, t) for p in IDT_P for t in IDT_T]
T_RISE = 400.0
RTOL, ATOL = 1e-10, 1e-20
DELTA = 1e-3                                        # Ji et al. Eq. (20)
SL_TOL = (1e-9, 1e-14)                              # flame Newton (rtol, atol)


def _need_cantera():
    if ct is None:
        raise ImportError("Cantera is not installed: pip install cantera==3.2.0")
    if not os.path.exists(MECH):
        raise FileNotFoundError(
            "konnov2008_h2.yaml is missing: build it with build_mechanism.py from the "
            "supplementary files of Konnov (2008), doi:10.1016/j.combustflame.2007.10.024")


class H2Model:
    def __init__(self, mech=MECH):
        _need_cantera()
        self.gas = ct.Solution(mech)
        assert self.gas.n_reactions == M_IN
        self.flames = {}
        self.nominal = {}

    def _set_mult(self, x):
        lnm = SIGMA * np.asarray(x, dtype=float)
        for j in range(M_IN):
            self.gas.set_multiplier(float(np.exp(lnm[j])), j)

    # ---------- 0-D constant-volume ignition ----------
    def ignition_delay(self, P_atm, T0, x, rtol=RTOL, atol=ATOL):
        self._set_mult(x)
        g = self.gas
        g.TPX = T0, P_atm * ATM, "H2:2, O2:1, N2:3.76"
        r = ct.IdealGasReactor(g, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = rtol, atol
        Tc = T0 + T_RISE
        t_old, T_old, dT_old = 0.0, T0, self._dTdt(r)
        while True:
            t = net.step()
            T = r.T
            dT = self._dTdt(r)
            if T >= Tc:
                return self._hermite_root(t_old, t, T_old, T, dT_old, dT, Tc)
            t_old, T_old, dT_old = t, T, dT
            if t > 10.0:
                raise RuntimeError("no ignition within 10 s")

    def _dTdt(self, r):
        g = self.gas
        # constant volume: rho cv dT/dt = -sum(u_k w_k)
        wdot = g.net_production_rates  # kmol/m3/s
        u = g.partial_molar_int_energies
        return -float(np.dot(u, wdot)) / (g.density * g.cv_mass)

    @staticmethod
    def _hermite_root(t0, t1, y0, y1, d0, d1, yc):
        h = t1 - t0

        def H(s):
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1 - yc
        a, b = 0.0, 1.0
        for _ in range(80):
            c = 0.5 * (a + b)
            if H(a) * H(c) <= 0:
                b = c
            else:
                a = c
        return t0 + 0.5 * (a + b) * h

    def idt_grad(self, P_atm, T0, x, delta=DELTA, central=False):
        """tau (s) and dtau/dln k_j (s): Ji et al. Eq. (20), k_j -> (1 + delta) k_j, forward
        differences (central=True: also k_j / (1 + delta), for checking)."""
        x = np.asarray(x, dtype=float)
        tau = self.ignition_delay(P_atm, T0, x)
        g = np.empty(M_IN)
        for j in range(M_IN):
            xp = x.copy(); xp[j] += np.log1p(delta) / SIGMA[j]
            tp = self.ignition_delay(P_atm, T0, xp)
            if central:
                xm = x.copy(); xm[j] -= np.log1p(delta) / SIGMA[j]
                g[j] = (tp - self.ignition_delay(P_atm, T0, xm)) / (2 * np.log1p(delta))
            else:
                g[j] = (tp - tau) / np.log1p(delta)
        return tau, g

    # ---------- 1-D freely propagating flame ----------
    def _flame(self, P_atm):
        if P_atm not in self.flames:
            g = self.gas
            for j in range(M_IN):
                g.set_multiplier(1.0, j)
            g.TPX = 300.0, P_atm * ATM, "H2:2, O2:1, N2:3.76"
            width = 0.03 / P_atm ** 0.7
            f = ct.FreeFlame(g, width=width)
            f.transport_model = "mixture-averaged"
            f.set_refine_criteria(ratio=2.5, slope=0.04, curve=0.08, prune=0.01)
            f.set_max_grid_points(f.flame, 2000)
            f.flame.set_steady_tolerances(default=SL_TOL)
            f.flame.set_transient_tolerances(default=SL_TOL)
            f.solve(loglevel=0, auto=True)
            self.flames[P_atm] = f
            self.nominal[P_atm] = f.to_array()
        return self.flames[P_atm]

    def flame_speed(self, P_atm, x, refine=True, restart=True):
        """S_L (m/s). restart=True starts from the nominal solution (so each sample's
        result does not depend on the samples solved before it)."""
        f = self._flame(P_atm)
        if restart:
            f.from_array(self.nominal[P_atm])
        f.P = P_atm * ATM          # the pressure is not restored by from_array
        self._set_mult(x)
        f.solve(loglevel=0, refine_grid=refine)
        return float(f.velocity[0])

    def sl_grad(self, P_atm, x, delta=DELTA, central=False):
        """S_L (m/s) and dS_L/dln k_j (m/s): the base solve refines the grid; each perturbed
        solve (k_j -> (1 + delta) k_j, forward differences; central=True adds k_j / (1 + delta))
        starts from the base solution on that fixed grid."""
        x = np.asarray(x, dtype=float)
        f = self._flame(P_atm)
        sl = self.flame_speed(P_atm, x, refine=True)
        base = f.to_array()
        g = np.empty(M_IN)
        for j in range(M_IN):
            xp = x.copy(); xp[j] += np.log1p(delta) / SIGMA[j]
            f.from_array(base)
            sp = self.flame_speed(P_atm, xp, refine=False, restart=False)
            if central:
                xm = x.copy(); xm[j] -= np.log1p(delta) / SIGMA[j]
                f.from_array(base)
                g[j] = (sp - self.flame_speed(P_atm, xm, refine=False, restart=False)) / (2 * np.log1p(delta))
            else:
                g[j] = (sp - sl) / np.log1p(delta)
        f.from_array(base)
        self._set_mult(x)
        return sl, g

    def sl_adjoint(self, P_atm, x, builtin=False):
        """S_L and dS_L/dln k_j (m/s) from Cantera's adjoint solve, at the solution for x
        (builtin=True: S_L times the built-in result, i.e. dS_L/dm_j, see below).

        The default perturbs multiplier j as m_j (1 + dp), which gives d ln S_L/d ln k_j
        directly. builtin=True calls FreeFlame.get_flame_speed_reaction_sensitivities(),
        whose perturbation sets multiplier j to 1 + dp; the residual is linear in each
        multiplier, so that is the derivative with respect to the multiplier,
        (1/S_L) dS_L/dm_j, and it must be multiplied by m_j = exp(sigma_j x_j) (chain rule)
        to give d ln S_L/d ln k_j. At x = 0 (every m_j = 1) the two are the same."""
        x = np.asarray(x, dtype=float)
        f = self._flame(P_atm)
        sl = self.flame_speed(P_atm, x, refine=True)
        if builtin:
            s = f.get_flame_speed_reaction_sensitivities()
        else:
            mult = np.exp(SIGMA * x)
            nvars = sum(D.n_components * D.n_points for D in f.domains)
            dgdx = np.zeros(nvars)
            dgdx[f.inlet.n_components + f.flame.component_index("velocity")] = 1.0

            def perturb(sim, i, dp):
                sim.gas.set_multiplier(float(mult[i] * (1.0 + dp)), i)
            s = f.solve_adjoint(perturb, M_IN, dgdx) / f.velocity[0]
        self._set_mult(x)
        return sl, np.asarray(s) * sl


# ---------- numpy-only post-processing shared with the notebook ----------
N_IDT = len(IDT_CONDITIONS)
N_OUT = N_IDT + len(SL_P)
SHARED_TOL = 0.05                # max |u_i^T v - 1| for the truncated-SVD shared direction
I_R16 = 15                      # x16: H + O2 = OH + O (Konnov No. 8)


def normalized_gradients(F, G):
    """Gradients of the analysed QoIs w.r.t. x, shape (M, 21, 33): log10(tau) for the
    ignition outputs (Ji et al. p. 151), S_L in m/s; chain rule of Eq. (19)."""
    F = np.atleast_2d(F)
    G = np.asarray(G).reshape(F.shape[0], N_OUT, M_IN)
    scale = np.ones(F.shape)
    scale[:, :N_IDT] = 1.0 / (F[:, :N_IDT] * np.log(10.0))
    return G * scale[:, :, None] * SIGMA[None, None, :]


def shared_directions(F, G, train=None, tol=SHARED_TOL):
    """Per-output first eigenvectors and the shared direction.

    u_i: first eigenvector of C_i = mean(grad grad^T) for output i, in x (sign: R16
    component positive). A = [u_i, i in train] (default: outputs 2..21, all but the Fig. 1
    condition). Returns U (21 x 33), v_eq10 = A (A^T A)^-1 1 (Ji et al. Eq. 10, exact
    u_i^T v = 1), and v_tsvd: the minimum-norm solution of A^T v = 1 restricted to the k
    leading singular vectors of A, with the smallest k for which max_i |u_i^T v - 1| <= tol
    (this data set's rule; Eq. 10 is ill-conditioned when the u_i are nearly parallel)."""
    Gx = normalized_gradients(F, G)
    U = np.empty((N_OUT, M_IN))
    for i in range(N_OUT):
        C = Gx[:, i, :].T @ Gx[:, i, :] / Gx.shape[0]
        w, V = np.linalg.eigh(C)
        u = V[:, -1]
        U[i] = u if u[I_R16] > 0 else -u
    if train is None:
        train = list(range(1, N_OUT))
    A = U[train].T
    one = np.ones(len(train))
    v_eq10 = A @ np.linalg.solve(A.T @ A, one)
    P, s, Qt = np.linalg.svd(A.T, full_matrices=False)    # A^T = P diag(s) Qt
    for k in range(1, len(s) + 1):
        v_tsvd = Qt[:k].T @ ((P[:, :k].T @ one) / s[:k])
        if np.max(np.abs(A.T @ v_tsvd - 1.0)) <= tol:
            break
    return U, v_eq10, v_tsvd
