"""Driver for the impedance misfit of the grouped-parameter SPMe (Hallemans et al., JES 2025).

PyBaMM and PyBOP are outside codes, not numpy/scipy models; this module sets them up and adds
the impedance solve and its adjoint gradient.

The model is PyBOP's ``GroupedSPMe`` with double-layer capacitance (Appendix C of the
paper), discretised by PyBaMM with the paper's mesh (100 radial points in each particle,
100 points across each electrode, 20 across the separator; Section 2). Its impedance at
an equilibrium operating point is the paper's Eq. (9),

    Z_m(w, theta) = -v / c,   (j w M - J_theta,m) x = b,   v = O_v x,  c = O_c x,

computed with PyBaMM's frequency-domain EIS method (``pybamm.EISSimulation``, the former
pybamm-eis). The quantity of interest is the paper's least-squares cost, Eq. (7),

    Phi(theta) = sum_{m=1}^{9} sum_{k=1}^{60} |Z_m(w_k) - Z_m(w_k, theta)|^2   [Ohm^2],

over the 9 SOCs 10 %, ..., 90 % and 60 log-spaced frequencies in [200 uHz, 1 kHz],
against synthetic reference spectra Z_m(w_k) computed at the true parameters of Table 3.

Gradients of Phi with respect to the 18 grouped parameters are exact (to round-off):
an adjoint solve at every frequency, A^T lam = g, gives dZ = lam^T dJ x + (output
terms), and one reverse-mode CasADi sweep through J(y0(theta), theta) and the output
Jacobian gives the whole gradient for each SOC. ``generate_data.py --gradcheck`` compares it
with central finite differences.

Every failure (singular system, non-finite impedance, out-of-range input) raises.
Running the model requires pybamm and pybop (tested with PyBaMM 26.10.1.1, PyBOP 26.3,
CasADi 3.8.1); the module-level constants need numpy only.
"""

import numpy as np

# PyBaMM, PyBOP, CasADi and scipy.sparse are imported where they are used, so that the
# constants below (PARAMETERS, SOCS, FREQUENCIES, MESH) can be read with numpy alone.

# (python name, PyBOP parameter name, symbol, units, lower, upper) -- Table 3, p. 12
PARAMETERS = [
    ("tau_d_p", "Positive particle diffusion time scale [s]", "tau_d+", "s", 5e2, 1e4),
    ("tau_d_n", "Negative particle diffusion time scale [s]", "tau_d-", "s", 5e2, 1e4),
    ("tau_e_p", "Positive electrode electrolyte diffusion time scale [s]", "tau_e+", "s", 2e2, 1e3),
    ("tau_e_n", "Negative electrode electrolyte diffusion time scale [s]", "tau_e-", "s", 2e2, 1e3),
    ("tau_e_sep", "Separator electrolyte diffusion time scale [s]", "tau_e^sep", "s", 2e2, 1e3),
    ("zeta_p", "Positive electrode relative porosity", "zeta+", "-", 0.5, 1.5),
    ("zeta_n", "Negative electrode relative porosity", "zeta-", "-", 0.5, 1.5),
    ("Q_e", "Reference electrolyte capacity [A.s]", "Q_e", "A s", 5e2, 1e3),
    ("tau_ct_p", "Positive electrode charge transfer time scale [s]", "tau_ct+", "s", 1e3, 5e4),
    ("tau_ct_n", "Negative electrode charge transfer time scale [s]", "tau_ct-", "s", 1e3, 5e4),
    ("C_p", "Positive electrode capacitance [F]", "C+", "F", 0.0, 1.0),
    ("C_n", "Negative electrode capacitance [F]", "C-", "F", 0.0, 1.0),
    ("c0_p", "Maximum positive stoichiometry", "c+^0%", "-", 0.8, 0.9),
    ("c0_n", "Minimum negative stoichiometry", "c-^0%", "-", 0.0, 0.1),
    ("c100_p", "Minimum positive stoichiometry", "c+^100%", "-", 0.2, 0.3),
    ("c100_n", "Maximum negative stoichiometry", "c-^100%", "-", 0.85, 0.95),
    ("t_plus", "Cation transference number", "t+", "-", 0.2, 0.5),
    ("R0", "Series resistance [Ohm]", "R0", "Ohm", 0.0, 0.05),
]
NAMES = [p[0] for p in PARAMETERS]
PYBOP_NAMES = [p[1] for p in PARAMETERS]
LOWER = np.array([p[4] for p in PARAMETERS])
UPPER = np.array([p[5] for p in PARAMETERS])
M_INPUTS = len(PARAMETERS)

SOCS = np.linspace(0.1, 0.9, 9)                       # Section 6.1: 10 %, ..., 90 %
FREQUENCIES = np.logspace(np.log10(2e-4), 3.0, 60)    # Section 6.1: 60 log-spaced, Hz
R0_TRUE = 0.01                                         # Table 3 (Chen2020 + 10 mOhm contact)
MESH = {"r_n": 100, "r_p": 100, "x_n": 100, "x_s": 20, "x_p": 100}   # Section 2


def chen2020_grouped():
    """Grouped Chen2020 parameter values from PyBOP (Table 3 'theta'), with R0 = 0.01 Ohm."""
    import pybop
    pv = pybop.lithium_ion.GroupedSPMe(
        options={"surface form": "differential"}).default_parameter_values
    pv["Series resistance [Ohm]"] = R0_TRUE
    return pv


def true_theta():
    pv = chen2020_grouped()
    return np.array([float(pv[n]) for n in PYBOP_NAMES])


class SPMeImpedance:
    """Grouped SPMe built once, with the 18 grouped parameters and the SOC as inputs."""

    def __init__(self, mesh=None, frequencies=None, socs=None):
        import casadi
        import pybamm
        import pybop
        from scipy.sparse import csc_matrix
        self.mesh = dict(MESH if mesh is None else mesh)
        self.frequencies = FREQUENCIES if frequencies is None else np.asarray(frequencies)
        self.socs = SOCS if socs is None else np.asarray(socs)
        self.omega = 2 * np.pi * self.frequencies

        model = pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"})
        pv = chen2020_grouped()
        self.fixed = {n: float(pv[n]) for n in ("Measured cell capacity [A.s]",
                      "Positive electrode relative thickness",
                      "Negative electrode relative thickness", "Initial temperature [K]")}
        for n in PYBOP_NAMES + ["Initial SoC"]:
            pv[n] = "[input]"
        var_pts = {k: self.mesh[k.name] for k in model.default_var_pts}
        sim = pybamm.EISSimulation(model, parameter_values=pv, var_pts=var_pts)
        self.input_names = PYBOP_NAMES + ["Initial SoC"]
        inputs0 = self._inputs(true_theta(), 0.5)
        sim.build(inputs=inputs0)
        bm = sim._built_model
        pybamm.BaseSolver().set_up(bm, inputs=inputs0)
        self.n_states = bm.concatenated_initial_conditions.shape[0]

        # Symbolic inputs in the order the CasADi functions were built with
        p = casadi.MX.sym("p", len(self.input_names))
        pdict = {n: p[i] for i, n in enumerate(self.input_names)}
        y = casadi.MX.sym("y", self.n_states)
        y0 = bm.concatenated_initial_conditions.to_casadi(t=0, y=y, inputs=pdict)
        J = bm.jac_rhs_algebraic_eval(0, y0, p)
        outputs = pybamm.numpy_concatenation(
            bm.get_processed_variable("Voltage [V]"),
            bm.get_processed_variable("Current [A]"))
        O = casadi.jacobian(outputs.to_casadi(t=0, y=y, inputs=pdict), y)
        O = casadi.substitute(O, y, y0)
        if casadi.depends_on(y0, y):
            raise RuntimeError("initial conditions depend on the state")
        self._Jsp = J.sparsity()
        self._Osp = O.sparsity()
        Jnz, Onz = J.nz[:], O.nz[:]
        self._JO = casadi.Function("JO", [p], [Jnz, Onz])
        wJ = casadi.MX.sym("wJ", Jnz.shape[0])
        wO = casadi.MX.sym("wO", Onz.shape[0])
        s = casadi.dot(wJ, Jnz) + casadi.dot(wO, Onz)
        self._grad = casadi.Function("gradJO", [p, wJ, wO], [casadi.gradient(s, p)])
        self._Jrow, self._Jcol = self._Jsp.get_triplet()
        self._Orow, self._Ocol = self._Osp.get_triplet()
        self._Jrow, self._Jcol = np.array(self._Jrow), np.array(self._Jcol)
        self._Orow, self._Ocol = np.array(self._Orow), np.array(self._Ocol)
        Mm = csc_matrix(bm.mass_matrix.entries)
        self.M = Mm
        cur = {v.name: s_[0].start for v, s_ in bm.y_slices.items()}["Current variable [A]"]
        self.b = np.zeros(self.n_states)
        self.b[cur] = -1.0
        self.versions = {"pybamm": pybamm.__version__, "pybop": pybop.__version__,
                         "casadi": casadi.__version__, "numpy": np.__version__}

    def _inputs(self, theta, soc):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (M_INPUTS,) or not np.all(np.isfinite(theta)):
            raise ValueError("theta must be 18 finite numbers")
        d = {n: float(v) for n, v in zip(PYBOP_NAMES, theta)}
        d["Initial SoC"] = float(soc)
        return d

    def _pvec(self, theta, soc):
        return np.concatenate([np.asarray(theta, float), [float(soc)]])

    def _matrices(self, pvec):
        from scipy.sparse import csc_matrix
        Jnz, Onz = self._JO(pvec)
        Jnz, Onz = np.array(Jnz).ravel(), np.array(Onz).ravel()
        n = self.n_states
        J = csc_matrix((Jnz, (self._Jrow, self._Jcol)), shape=(n, n))
        O = np.zeros((2, n))
        O[self._Orow, self._Ocol] = Onz
        return J, O

    def impedance(self, theta, soc, weight_fn=None):
        """Complex impedance (Ohm) at self.frequencies for one SOC.

        If weight_fn is given, a = weight_fn(Z) (complex, one per frequency) and the
        real gradient d/dtheta of sum_k Re(a_k Z_k) is returned too (adjoint method)."""
        from scipy.sparse.linalg import splu
        pvec = self._pvec(theta, soc)
        J, O = self._matrices(pvec)
        nf = len(self.omega)
        X = np.empty((nf, self.n_states), complex)
        lus = []
        b = self.b.astype(complex)
        for k, w in enumerate(self.omega):
            lu = splu(((1j * w) * self.M - J).tocsc())
            X[k] = lu.solve(b)
            lus.append(lu)
        v_all = X @ O[0]
        c_all = X @ O[1]
        if not (np.all(np.isfinite(v_all)) and np.all(np.isfinite(c_all))
                and np.all(c_all != 0)):
            raise FloatingPointError("non-finite impedance")
        Z = -v_all / c_all
        if weight_fn is None:
            return Z
        a = np.asarray(weight_fn(Z), complex)
        cv = -1.0 / c_all                    # dZ/dv
        cc = v_all / c_all**2                # dZ/dc
        L = np.empty_like(X)
        for k, lu in enumerate(lus):
            g = cv[k] * O[0] + cc[k] * O[1]
            L[k] = lu.solve(g.astype(complex), trans="T")   # A^T lam = g
        # dZ_k = lam_k^T dJ x_k + [cv_k dO_v + cc_k dO_c] x_k
        wJ = np.real(np.einsum("k,ki,ki->i", a, L[:, self._Jrow], X[:, self._Jcol]))
        coef = np.where(self._Orow[None, :] == 0, cv[:, None], cc[:, None])
        wO = np.real(np.einsum("k,ki,ki->i", a, coef, X[:, self._Ocol]))
        gp = np.array(self._grad(pvec, wJ, wO)).ravel()
        return Z, gp[:M_INPUTS]


class ImpedanceMisfit:
    """Phi(theta) of Eq. (7) against synthetic data at the true theta (Table 3)."""

    def __init__(self, mesh=None, theta_ref=None):
        self.eis = SPMeImpedance(mesh=mesh)
        self.theta_ref = true_theta() if theta_ref is None else np.asarray(theta_ref)
        self.Zref = np.array([self.eis.impedance(self.theta_ref, s) for s in self.eis.socs])

    def check_bounds(self, theta):
        theta = np.asarray(theta, float)
        if np.any(theta < LOWER) or np.any(theta > UPPER):
            raise ValueError("theta outside the Table 3 bounds")
        if theta[10] <= 0 or theta[11] <= 0:
            raise ValueError("double-layer capacitance must be > 0")

    def misfit(self, theta, gradient=False, check=True):
        """Return (Phi, Phi_m) or (Phi, Phi_m, grad Phi, grad Phi_m); Ohm^2 and Ohm^2/unit."""
        if check:
            self.check_bounds(theta)
        phis, grads = [], []
        for m, s in enumerate(self.eis.socs):
            zr = self.Zref[m]
            if gradient:
                Z, g = self.eis.impedance(theta, s, weight_fn=lambda Z: 2 * np.conj(Z - zr))
                grads.append(g)
            else:
                Z = self.eis.impedance(theta, s)
            r = Z - zr
            phis.append(float(np.sum(np.abs(r) ** 2)))
        phis = np.array(phis)
        if not np.all(np.isfinite(phis)):
            raise FloatingPointError("non-finite misfit")
        if gradient:
            grads = np.array(grads)
            return phis.sum(), phis, grads.sum(axis=0), grads
        return phis.sum(), phis
