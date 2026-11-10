"""Three-stage HIV model of Clarke and Pankavich (2024), Eq. (1), with forward sensitivities.

Clarke C, Pankavich S. Three-stage modeling of HIV infection and implications for
antiretroviral therapy. J Math Biol 88:34 (2024), doi:10.1007/s00285-024-02056-1.
The model itself is that of Hernandez-Vargas and Middleton (2013), J Theor Biol 320:33-40.

States y = (T, T*, M, M*, V): uninfected and infected CD4+ T cells and macrophages
(cells/mm^3) and free virus (copies/mm^3, the units of c_T and c_M in Table 1):

    T'  = sT + rhoT V/(cT + V) T - kT T V - dT T
    T*' = kT T V - dT* T*
    M'  = sM + rhoM V/(cM + V) M - kM M V - dM M
    M*' = kM M V - dM* M*
    V'  = pT T* + pM M* - dV V

Time is in days; a year is 365 days (this module's choice). Initial state (paper p. 6):
T(0) = 1000 cells/mm^3, M(0) = 150 cells/mm^3, T*(0) = M*(0) = 0 and V(0) = 10 copies/ml, which is
0.01 copies/mm^3 in the units of Table 1 (the conversion is this module's reading; see DATA.md).

Quantities of interest (paper p. 12):
    T5  = T(5 years)      cells/mm^3
    T10 = T(10 years)     cells/mm^3
    F   = max{t >= 0 : T(t) > 200}, the time to AIDS, in years

Gradients come from the forward sensitivity equations S' = J S + df/dp, integrated with the
states; for F, dF/dp = -(dT/dp)/(dT/dt) at the last downward crossing of T = 200.

Only numpy and scipy are used.
"""
import numpy as np
from scipy.integrate import solve_ivp

NAMES = ['sT', 'sM', 'kT', 'kM', 'pT', 'pM', 'deltaT', 'deltaTs', 'deltaM', 'deltaMs',
         'deltaV', 'rhoT', 'rhoM', 'cT', 'cM']
LATEX = [r'$s_T$', r'$s_M$', r'$k_T$', r'$k_M$', r'$p_T$', r'$p_M$', r'$\delta_T$', r'$\delta_{T^*}$',
         r'$\delta_M$', r'$\delta_{M^*}$', r'$\delta_V$', r'$\rho_T$', r'$\rho_M$', r'$c_T$', r'$c_M$']
UNITS = ['cells/mm^3/day', 'cells/mm^3/day', 'mm^3/day/copy', 'mm^3/day/copy', 'copies/cell/day',
         'copies/cell/day', '1/day', '1/day', '1/day', '1/day', '1/day', '1/day', '1/day',
         'copies/mm^3', 'copies/mm^3']
DESCRIPTIONS = ['T-cell source', 'macrophage source', 'T-cell infection rate', 'macrophage infection rate',
                'virus production by infected T cells', 'virus production by infected macrophages',
                'T-cell death rate', 'infected T-cell death rate', 'macrophage death rate',
                'infected macrophage death rate', 'virion clearance rate', 'T-cell maximum proliferation',
                'macrophage maximum proliferation', 'T-cell half-velocity constant',
                'macrophage half-velocity constant']
# Table 1, p. 6 (nominal values and the printed 10 % ranges)
NOMINAL = np.array([10.0, 0.15, 4.57e-5, 4.33e-8, 38.0, 35.0, 0.01, 0.4, 1e-3, 1e-3, 2.4,
                    0.01, 0.003, 300.0, 220.0])
LOWER = np.array([9.0, 0.135, 4.11e-5, 3.90e-8, 34.2, 31.5, 0.009, 0.36, 9e-4, 9e-4, 2.16,
                  0.009, 0.0027, 270.0, 198.0])
UPPER = np.array([11.0, 0.165, 5.02e-5, 4.76e-8, 41.8, 38.5, 0.011, 0.44, 1.1e-3, 1.1e-3, 2.64,
                  0.011, 0.0033, 330.0, 242.0])
Y0 = np.array([1000.0, 0.0, 150.0, 0.0, 0.01])   # V(0) = 10 copies/ml = 0.01 copies/mm^3
STATE_NAMES = ['T', 'Ts', 'M', 'Ms', 'V']
DAYS_PER_YEAR = 365.0
T_AIDS = 200.0                     # cells/mm^3, the clinical AIDS threshold (p. 12)
HORIZON_YEARS = 40.0               # integration window used to locate the last crossing of 200
QOI_NAMES = ['T5', 'T10', 'F']
QOI_UNITS = ['cells/mm^3', 'cells/mm^3', 'years']
RTOL = 1e-10
ATOL = 1e-10
M_PAR = 15


def rhs(t, y, p):
    sT, sM, kT, kM, pT, pM, dT, dTs, dM, dMs, dV, rT, rM, cT, cM = p
    T, Ts, M, Ms, V = y[:5]
    return np.array([sT + rT * V / (cT + V) * T - kT * T * V - dT * T,
                     kT * T * V - dTs * Ts,
                     sM + rM * V / (cM + V) * M - kM * M * V - dM * M,
                     kM * M * V - dMs * Ms,
                     pT * Ts + pM * Ms - dV * V])


def jac(t, y, p):
    sT, sM, kT, kM, pT, pM, dT, dTs, dM, dMs, dV, rT, rM, cT, cM = p
    T, Ts, M, Ms, V = y[:5]
    J = np.zeros((5, 5))
    J[0, 0] = rT * V / (cT + V) - kT * V - dT
    J[0, 4] = rT * T * cT / (cT + V) ** 2 - kT * T
    J[1, 0] = kT * V
    J[1, 1] = -dTs
    J[1, 4] = kT * T
    J[2, 2] = rM * V / (cM + V) - kM * V - dM
    J[2, 4] = rM * M * cM / (cM + V) ** 2 - kM * M
    J[3, 2] = kM * V
    J[3, 3] = -dMs
    J[3, 4] = kM * M
    J[4, 1] = pT
    J[4, 3] = pM
    J[4, 4] = -dV
    return J


def dfdp(y, p):
    """Partial derivatives of the right-hand side with respect to the 15 parameters, (5, 15)."""
    sT, sM, kT, kM, pT, pM, dT, dTs, dM, dMs, dV, rT, rM, cT, cM = p
    T, Ts, M, Ms, V = y[:5]
    P = np.zeros((5, 15))
    P[0, 0] = 1.0
    P[0, 2] = -T * V
    P[0, 6] = -T
    P[0, 11] = V * T / (cT + V)
    P[0, 13] = -rT * V * T / (cT + V) ** 2
    P[1, 2] = T * V
    P[1, 7] = -Ts
    P[2, 1] = 1.0
    P[2, 3] = -M * V
    P[2, 8] = -M
    P[2, 12] = V * M / (cM + V)
    P[2, 14] = -rM * V * M / (cM + V) ** 2
    P[3, 3] = M * V
    P[3, 9] = -Ms
    P[4, 4] = Ts
    P[4, 5] = Ms
    P[4, 10] = -V
    return P


def _rhs_sens(t, z, p):
    """States and relative sensitivities S~ = dy/d(ln p) (5 x 15, row-major after the states)."""
    y = z[:5]
    S = z[5:].reshape(5, 15)
    dS = jac(t, y, p) @ S + dfdp(y, p) * p
    return np.concatenate([rhs(t, y, p), dS.ravel()])


def _jac_sens(t, z, p):
    """Jacobian of the augmented system without the second-derivative coupling (block-diagonal);
    it is used only in the Newton iteration of the stiff solver, so it affects cost, not accuracy."""
    J = jac(t, z[:5], p)
    A = np.zeros((80, 80))
    A[:5, :5] = J
    A[5:, 5:] = np.kron(J, np.eye(15))
    return A


def _event(t, z, p):
    return z[0] - T_AIDS
_event.direction = -1


def _check(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (15,):
        raise ValueError(f'expected 15 parameters, got shape {p.shape}')
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError('all 15 parameters must be finite and positive')
    return p


def trajectory(p, times_days, rtol=RTOL, atol=ATOL):
    """States (5, len(times_days)) at the given times (days), without sensitivities."""
    p = _check(p)
    times_days = np.asarray(times_days, dtype=float)
    s = solve_ivp(rhs, (0.0, float(times_days.max())), Y0, args=(p,), method='LSODA', jac=jac,
                  rtol=rtol, atol=atol, t_eval=times_days)
    if not s.success:
        raise RuntimeError(f'ODE solve failed: {s.message}')
    return s.y


def qoi(p, grad=False, rtol=RTOL, atol=ATOL, horizon_years=HORIZON_YEARS):
    """(T5, T10, F) at the physical parameters p; with grad=True also their gradients with
    respect to the physical parameters, shape (3, 15).

    Raises RuntimeError if the solve fails, if T never falls to 200 within the horizon, or if T is
    above 200 at the end of the horizon (the last crossing would be beyond the window)."""
    p = _check(p)
    tq = np.array([5.0, 10.0]) * DAYS_PER_YEAR
    tend = horizon_years * DAYS_PER_YEAR
    if grad:
        z0 = np.concatenate([Y0, np.zeros(75)])
        s = solve_ivp(_rhs_sens, (0.0, tend), z0, args=(p,), method='LSODA', jac=_jac_sens,
                      rtol=rtol, atol=atol, events=_event, dense_output=True)
    else:
        s = solve_ivp(rhs, (0.0, tend), Y0, args=(p,), method='LSODA', jac=jac,
                      rtol=rtol, atol=atol, events=_event, dense_output=True)
    if not s.success:
        raise RuntimeError(f'ODE solve failed: {s.message}')
    if s.y[0, -1] > T_AIDS:
        raise RuntimeError(f'T = {s.y[0, -1]:.1f} > {T_AIDS} at the end of the {horizon_years:g}-year window')
    if len(s.t_events[0]) == 0:
        raise RuntimeError('T never falls to 200 within the window')
    tF = s.t_events[0][-1]
    zF = s.y_events[0][-1]
    zq = s.sol(tq)
    f = np.array([zq[0, 0], zq[0, 1], tF / DAYS_PER_YEAR])
    if not grad:
        return f
    G = np.empty((3, 15))
    G[0] = zq[5:20, 0] / p             # row T of S~ = dT/d(ln p); divide by p for dT/dp
    G[1] = zq[5:20, 1] / p
    Tdot = rhs(tF, zF[:5], p)[0]
    if not Tdot < 0:
        raise RuntimeError('T is not decreasing at its last crossing of 200')
    G[2] = -(zF[5:20] / p) / Tdot / DAYS_PER_YEAR
    return f, G


def qois(P, grad=False, **kw):
    """qoi for each row of P (M, 15); returns f (M, 3) and, with grad=True, G (M, 3, 15)."""
    P = np.atleast_2d(P)
    out = [qoi(p, grad=grad, **kw) for p in P]
    if grad:
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])
    return np.array(out)


def reproduction_numbers(p):
    """R_T and R_M of Theorem 3, p. 9 (R_T = 1.809 and R_M = 0.09 at the nominal values, p. 10)."""
    sT, sM, kT, kM, pT, pM, dT, dTs, dM, dMs, dV, rT, rM, cT, cM = _check(p)
    return pT * kT * sT / (dT * dTs * dV), pM * kM * sM / (dM * dMs * dV)
