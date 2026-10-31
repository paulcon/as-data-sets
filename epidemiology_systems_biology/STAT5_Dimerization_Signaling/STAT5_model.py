"""STAT5A/STAT5B phosphorylation and dimerization model (Boehm et al. 2014), as in the
benchmark collection of Hass et al. (2019).

The model is the Boehm_JProteomeRes2014 problem of the benchmark collection of
H. Hass, C. Loos, E. Raimundez-Alvarez, J. Timmer, J. Hasenauer and C. Kreutz,
"Benchmark problems for dynamic modeling of intracellular processes", Bioinformatics
35(17):3073-3082 (2019), doi:10.1093/bioinformatics/btz020, originally from M. E. Boehm
et al., J. Proteome Res. 13(12):5685-5694 (2014), doi:10.1021/pr5006923. The equations
were rewritten by hand from the SBML file of the PEtab version of the problem
(github.com/Benchmarking-Initiative/Benchmark-Models-PEtab, problems/
Boehm_JProteomeRes2014/v1/model.xml; BSD-3-Clause repository); no code was copied.

States (concentrations, model units; time t in minutes), cytoplasm volume
V_c = 1.4, nucleus volume V_n = 0.45:

    A, B         unphosphorylated STAT5A, STAT5B (cytoplasm)
    pAB, pAA, pBB phosphorylated hetero- and homodimers (cytoplasm)
    nAA, nAB, nBB the same dimers in the nucleus

with the Epo stimulus Epo(t) = 1.25e-7 exp(-k_deg t) and

    dA/dt   = -2 w_AA - w_AB + (V_n/V_c) (2 k_exp_homo nAA + k_exp_hetero nAB)
    dB/dt   = -2 w_BB - w_AB + (V_n/V_c) (2 k_exp_homo nBB + k_exp_hetero nAB)
    dpAB/dt = w_AB - k_imp_hetero pAB
    dpAA/dt = w_AA - k_imp_homo pAA
    dpBB/dt = w_BB - k_imp_homo pBB
    dnAB/dt = (V_c/V_n) k_imp_hetero pAB - k_exp_hetero nAB
    dnAA/dt = (V_c/V_n) k_imp_homo pAA - k_exp_homo nAA
    dnBB/dt = (V_c/V_n) k_imp_homo pBB - k_exp_homo nBB

where w_AA = k_phos Epo A^2, w_AB = k_phos Epo A B, w_BB = k_phos Epo B^2, and
A(0) = 207.6 ratio, B(0) = 207.6 (1 - ratio), all dimers 0 at t = 0.

Observables (PEtab observables.tsv, in per cent; s = specC17, c = 1 - s):

    pSTAT5A_rel = (100 pAB + 200 s pAA) / (pAB + s A + 2 s pAA)
    pSTAT5B_rel = (100 pAB + 200 c pBB) / (pAB + c B + 2 c pBB)
    rSTAT5A_rel = (100 pAB + 100 s A + 200 s pAA) / (2 pAB + s A + 2 s pAA + c B + 2 c pBB)

Inputs: the 6 kinetic parameters that the benchmark estimates, as log10 values
xi = log10(theta) (`NAMES`, in the PEtab order). The 3 noise parameters the benchmark
also estimates enter only the likelihood, which is not computed here. `ratio` = 0.693
and `specC17` = 0.107 are fixed in the benchmark and held here.

Study box (`LOWER`, `UPPER`, log10): the PEtab best-fit value +- 1 decade, clipped to
the PEtab bounds [1e-5, 1e5]. Not from the paper; see DATA.md.

Quantities of interest (`QOI_NAMES`): the time averages over [0, 240] min of the three
observables, then the three observables at the 15 measurement times 2.5, ..., 240 min.
`qoi_and_grad` also returns the gradients with respect to xi = log10(theta), from forward
sensitivity equations solved with the model.

Solver: scipy.integrate.solve_ivp, Radau, rtol = 1e-10, atol = 1e-10 (states and
sensitivities are of order 1 to 200). Failed solves raise RuntimeError.

Only numpy and scipy are imported.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp

#: Input names (PEtab parameterId), in column order. Inputs are log10 of these.
NAMES = ('Epo_degradation_BaF3', 'k_exp_hetero', 'k_exp_homo', 'k_imp_hetero',
         'k_imp_homo', 'k_phos')

#: Short names for plots and tables.
SHORT = ('k_deg', 'k_exp,het', 'k_exp,hom', 'k_imp,het', 'k_imp,hom', 'k_phos')

#: Units of theta (not log10). Time in min; concentrations in the model's units (the
#: SBML defines only the time unit).
UNITS = ('1/min', '1/min', '1/min', '1/min', '1/min', '1/(conc^2 min)')

#: PEtab best-fit values (parameters.tsv, nominalValue).
NOMINAL_THETA = np.array([0.026982514033029, 1.00067973851508e-05, 0.006170228086381,
                          0.0163679184468, 97749.3794024716, 15766.5070195731])
NOMINAL = np.log10(NOMINAL_THETA)

#: PEtab bounds (parameters.tsv), log10.
PETAB_LOWER = np.full(6, -5.0)
PETAB_UPPER = np.full(6, 5.0)

#: Study box in log10: best fit +- 1 decade, clipped to the PEtab bounds.
LOWER = np.maximum(NOMINAL - 1.0, PETAB_LOWER)
UPPER = np.minimum(NOMINAL + 1.0, PETAB_UPPER)

#: Held settings (PEtab / SBML).
RATIO = 0.693          # STAT5A fraction of total STAT5 (parameters.tsv, fixed)
SPECC17 = 0.107        # constant in the observation functions (parameters.tsv, fixed)
STAT5_TOTAL = 207.6    # total STAT5 (SBML initial assignments)
EPO0 = 1.25e-7         # Epo at t = 0 (SBML assignment rule)
V_CYT, V_NUC = 1.4, 0.45
T_END = 240.0
RTOL, ATOL = 1e-10, 1e-10

#: Measurement times (min) after t = 0 (PEtab measurements.tsv; at t = 0 every
#: observable is fixed by ratio and specC17 alone).
TIMES = np.array([2.5, 5., 10., 15., 20., 30., 40., 50., 60., 80., 100., 120., 160., 200., 240.])

OBS = ('pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel')
STATES = ('A', 'B', 'pAB', 'pAA', 'pBB', 'nAA', 'nAB', 'nBB')
QOI_NAMES = tuple(f'{o}_mean' for o in OBS) + tuple(
    f'{o}_t{t:g}' for o in OBS for t in TIMES)

_NY, _NP, _NQ = 8, 6, 3
LN10 = math.log(10.0)


def _obs_coeffs(s=SPECC17):
    """Numerator and denominator coefficients (rows: observables; cols: states)."""
    c = 1.0 - s
    N = np.zeros((3, 8)); D = np.zeros((3, 8))
    # states: A B pAB pAA pBB nAA nAB nBB
    N[0, [2, 3]] = 100., 200.*s
    D[0, [2, 0, 3]] = 1., s, 2.*s
    N[1, [2, 4]] = 100., 200.*c
    D[1, [2, 1, 4]] = 1., c, 2.*c
    N[2, [2, 0, 3]] = 100., 100.*s, 200.*s
    D[2, [2, 0, 3, 1, 4]] = 2., s, 2.*s, c, 2.*c
    return N, D


_N, _D = _obs_coeffs()


def _coeffs(spec):
    return (_N, _D) if spec == SPECC17 else _obs_coeffs(spec)


def observables(y, spec=SPECC17):
    """The three observables (per cent) for states y of shape (8,) or (8, k)."""
    N, D = _coeffs(spec)
    return (N @ y) / (D @ y)


def _obs_grad(y, spec=SPECC17):
    """d(observables)/dy, shape (3, 8), for one state vector y."""
    N, D = _coeffs(spec)
    n = N @ y; d = D @ y
    return (N - (n/d)[:, None]*D)/d[:, None]


def _f_and_jac(t, y, th):
    kdeg, kexhe, kexho, kimhe, kimho, kp = th
    A, B, pAB, pAA, pBB, nAA, nAB, nBB = y
    epo = EPO0*math.exp(-kdeg*t)
    r = V_NUC/V_CYT
    q = V_CYT/V_NUC
    wAA = kp*epo*A*A; wAB = kp*epo*A*B; wBB = kp*epo*B*B
    f = np.array([
        -2*wAA - wAB + r*(2*kexho*nAA + kexhe*nAB),
        -2*wBB - wAB + r*(2*kexho*nBB + kexhe*nAB),
        wAB - kimhe*pAB,
        wAA - kimho*pAA,
        wBB - kimho*pBB,
        q*kimho*pAA - kexho*nAA,
        q*kimhe*pAB - kexhe*nAB,
        q*kimho*pBB - kexho*nBB])
    ke = kp*epo
    J = np.zeros((8, 8))
    J[0, 0] = -4*ke*A - ke*B; J[0, 1] = -ke*A; J[0, 5] = 2*r*kexho; J[0, 6] = r*kexhe
    J[1, 0] = -ke*B; J[1, 1] = -4*ke*B - ke*A; J[1, 7] = 2*r*kexho; J[1, 6] = r*kexhe
    J[2, 0] = ke*B; J[2, 1] = ke*A; J[2, 2] = -kimhe
    J[3, 0] = 2*ke*A; J[3, 3] = -kimho
    J[4, 1] = 2*ke*B; J[4, 4] = -kimho
    J[5, 3] = q*kimho; J[5, 5] = -kexho
    J[6, 2] = q*kimhe; J[6, 6] = -kexhe
    J[7, 4] = q*kimho; J[7, 7] = -kexho
    # d f / d theta (8 x 6), columns in NAMES order
    Fp = np.zeros((8, 6))
    Fp[:, 0] = -t*np.array([-2*wAA - wAB, -2*wBB - wAB, wAB, wAA, wBB, 0, 0, 0])
    Fp[0, 1] = r*nAB; Fp[1, 1] = r*nAB; Fp[6, 1] = -nAB
    Fp[0, 2] = 2*r*nAA; Fp[1, 2] = 2*r*nBB; Fp[5, 2] = -nAA; Fp[7, 2] = -nBB
    Fp[2, 3] = -pAB; Fp[6, 3] = q*pAB
    Fp[3, 4] = -pAA; Fp[4, 4] = -pBB; Fp[5, 4] = q*pAA; Fp[7, 4] = q*pBB
    Fp[:, 5] = np.array([-2*wAA - wAB, -2*wBB - wAB, wAB, wAA, wBB, 0, 0, 0])/kp
    return f, J, Fp


def _y0(ratio=RATIO):
    y = np.zeros(8)
    y[0] = STAT5_TOTAL*ratio
    y[1] = STAT5_TOTAL - STAT5_TOTAL*ratio
    return y


def _theta(xi):
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,) or not np.all(np.isfinite(xi)):
        raise ValueError(f'expected 6 finite log10 parameters, got {xi!r}')
    return 10.0**xi


# --- plain solve (with time averages as quadrature states) -------------------------
def _rhs_plain(t, z, th, spec=SPECC17):
    y = z[:8]
    f, _, _ = _f_and_jac(t, y, th)
    return np.concatenate([f, observables(y, spec)/T_END])


def _jac_plain(t, z, th, spec=SPECC17):
    y = z[:8]
    _, J, _ = _f_and_jac(t, y, th)
    Jz = np.zeros((11, 11))
    Jz[:8, :8] = J
    Jz[8:, :8] = _obs_grad(y, spec)/T_END
    return Jz


def _check(sol, xi):
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed at log10 theta = {np.array2string(np.asarray(xi), precision=4)}: '
                           f'{sol.message}')
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f'non-finite solution at log10 theta = {np.asarray(xi)}')
    if np.min(sol.y[:8]) < -1e-6*STAT5_TOTAL:
        raise RuntimeError(f'negative concentration {np.min(sol.y[:8]):.3e} at log10 theta = {np.asarray(xi)}')


def trajectory(xi, times, rtol=RTOL, atol=ATOL, ratio=RATIO, spec=SPECC17):
    """States (8, len(times)) and observables (3, len(times)) at the given times."""
    th = _theta(xi)
    times = np.asarray(times, dtype=float)
    sol = solve_ivp(_rhs_plain, (0.0, max(T_END, times.max())), np.concatenate([_y0(ratio), np.zeros(3)]),
                    method='Radau', t_eval=times, args=(th, spec), jac=_jac_plain, rtol=rtol, atol=atol)
    _check(sol, xi)
    return sol.y[:8], observables(sol.y[:8], spec)


def qoi(xi, rtol=RTOL, atol=ATOL, ratio=RATIO, spec=SPECC17):
    """The 48 quantities of interest (`QOI_NAMES`) at log10 parameters xi, shape (48,)."""
    th = _theta(xi)
    sol = solve_ivp(_rhs_plain, (0.0, T_END), np.concatenate([_y0(ratio), np.zeros(3)]),
                    method='Radau', t_eval=TIMES, args=(th, spec), jac=_jac_plain, rtol=rtol, atol=atol)
    _check(sol, xi)
    out = np.concatenate([sol.y[8:, -1], observables(sol.y[:8], spec).ravel()])
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f'non-finite output at log10 theta = {np.asarray(xi)}')
    return out


# --- forward sensitivities with respect to xi = log10(theta) -----------------------
# augmented state: y (8), S (8 x 6, row-major), Q (3), SQ (3 x 6, row-major)
_NZ = 8 + 48 + 3 + 18


def _rhs_sens(t, z, th):
    y = z[:8]
    S = z[8:56].reshape(8, 6)
    f, J, Fp = _f_and_jac(t, y, th)
    dS = J @ S + Fp*(th*LN10)
    H = _obs_grad(y)
    return np.concatenate([f, dS.ravel(), observables(y)/T_END, (H @ S).ravel()/T_END])


def _jac_sens(t, z, th):
    # Newton-iteration Jacobian; the second-derivative coupling of S to y is omitted,
    # which affects only the convergence of Radau's Newton iterations, not accuracy.
    y = z[:8]
    _, J, _ = _f_and_jac(t, y, th)
    H = _obs_grad(y)/T_END
    Jz = np.zeros((_NZ, _NZ))
    Jz[:8, :8] = J
    for j in range(6):
        idx = 8 + np.arange(8)*6 + j
        Jz[np.ix_(idx, idx)] = J
        qidx = 59 + np.arange(3)*6 + j
        Jz[np.ix_(qidx, idx)] = H
    Jz[56:59, :8] = H
    return Jz


def qoi_and_grad(xi, rtol=RTOL, atol=ATOL, ratio=RATIO):
    """QoIs (48,) and their gradients (48, 6) with respect to xi = log10(theta)."""
    th = _theta(xi)
    z0 = np.zeros(_NZ); z0[:8] = _y0(ratio)
    sol = solve_ivp(_rhs_sens, (0.0, T_END), z0, method='Radau', t_eval=TIMES, args=(th,),
                    jac=_jac_sens, rtol=rtol, atol=atol)
    _check(sol, xi)
    Y = sol.y[:8]                                   # (8, 15)
    S = sol.y[8:56].reshape(8, 6, -1)              # (8, 6, 15)
    f = np.concatenate([sol.y[56:59, -1], observables(Y).ravel()])
    G = np.empty((48, 6))
    G[:3] = sol.y[59:77, -1].reshape(3, 6)
    for k in range(len(TIMES)):
        H = _obs_grad(Y[:, k])
        G[3 + k + np.arange(3)*len(TIMES)] = H @ S[:, :, k]
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(G))):
        raise RuntimeError(f'non-finite output at log10 theta = {np.asarray(xi)}')
    return f, G
