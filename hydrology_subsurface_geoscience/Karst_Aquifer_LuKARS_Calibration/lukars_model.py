"""LuKARS lumped karst model of the Kerschbaum spring (Waidhofen a.d. Ybbs, Austria),
with the Bayesian data misfit of Teixeira Parente et al. (2019) and its exact gradient.

A numpy port of the authors' R script `scens/scen_main/main_exe.R` and of the
parameter transform in `scens/scen_main/env.py` (HydroShare resource
4901a0d654334c259f4ff9b49dc0a74e, CC BY 4.0, by D. Bittner and M. Teixeira
Parente). The port reproduces the R script line by line, including its forcing
pre-processing (Thornthwaite evapotranspiration, interception, degree-day snow),
so the numbers are those of the authors' code; see DATA.md for the checks.

Three calibrated hydrotopes (paper's Hyd 1-3 = the script's hydrotopes 2-4),
seven parameters each (paper Table 1). The dolomite quarry hydrotope (script's
hydrotope 1) has no quickflow and no recharge, so it never reaches the spring
and is omitted here. Time step 1 day, 2001-01-01 to 2016-12-31 (5844 days),
spin-up from 2001; the misfit uses the calibration window 2006-01-01 to
2008-12-31 (1096 days, the script's `cal_p <- c(1827, 2922)`).

Main functions
--------------
to_physical(x)              normalized calibration parameters (paper Eq. 33) -> Table 1
to_calibration(p)           inverse of to_physical
discharge(p)                simulated spring discharge (l/s) in the window, batched
misfit(x) / misfit_and_grad(x)   f_d(x) = 1/2 ||(d - G(x)) / (0.05 d)||^2 and df_d/dx

Only numpy is imported.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- parameters
KINDS = ('khyd', 'emin', 'emax', 'alpha', 'kis', 'ksec', 'esec')
#: physical parameter names, paper Table 1 order (No. 1-21)
NAMES = tuple(f'{k}_{i}' for i in (1, 2, 3) for k in KINDS)
UNITS = {'khyd': 'm^2/day', 'emin': 'mm', 'emax': 'mm', 'alpha': '-',
         'kis': 'm mm^-1 day', 'ksec': 'm mm^-1 day', 'esec': 'mm'}  # as printed in Table 1
#: calibration parameter names, paper Eq. (33) order
CAL_NAMES = ('log_khyd_1', 'emin_1', 'de_1', 'alpha_1', 'log_kis_1', 'log_ksec_1', 'esec_1',
             'D_log_khyd_12', 'D_emin_12', 'de_2', 'D_alpha_12', 'D_log_kis_12', 'D_log_ksec_12',
             'D_esec_12',
             'D_log_khyd_23', 'D_emin_23', 'de_3', 'D_alpha_23', 'D_log_kis_23', 'D_log_ksec_23',
             'D_esec_23')

#: Table 1 bounds, rows = hydrotope 1-3, columns = KINDS
LB = np.array([[9.0, 10.0, 15.0, 0.7, 0.002, 0.0095, 25.0],
               [8.5, 40.0, 80.0, 0.5, 0.00055, 0.0023, 130.0],
               [7.7, 75.0, 160.0, 0.2, 0.00025, 0.0015, 320.0]])
UB = np.array([[900.0, 50.0, 75.0, 1.6, 0.2, 0.95, 70.0],
               [850.0, 80.0, 160.0, 1.3, 0.055, 0.23, 220.0],
               [770.0, 120.0, 255.0, 0.7, 0.025, 0.15, 450.0]])
LOGK = (0, 4, 5)          # khyd, kis, ksec are calibrated on a log scale (Eq. 28)
DECREASING = (0, 3, 4, 5)  # khyd, alpha, kis, ksec decrease from Hyd 1 to Hyd 3 (Eq. 1, 29)
INCREASING = (1, 6)        # emin, esec increase (Eq. 1, 30); emax via de (Eq. 31)
DE_LB = LB[:, 2] - LB[:, 1]   # Eq. (32): [5, 40, 85] mm
DE_UB = UB[:, 2] - UB[:, 1]   #           [25, 80, 135] mm

# ---------------------------------------------------------------- fixed model settings (main_exe.R)
F_MELT = 4.0        # degree-day melt factor, mm / (degC day)
T_F = 0.5           # snow-melt threshold temperature, degC
K_B = 0.00043       # baseflow storage coefficient, 1/day (Bittner et al. 2018; not calibrated)
L_HYD = np.array([1600.0, 900.0, 960.0])   # mean distance hydrotope -> spring, m
AREA = 2.5e6                                # recharge area, m^2
SHARE = np.array([0.13, 0.56, 0.27])        # area shares of Hyd 1-3 (quarry: 0.04)
A_HYD = AREA*SHARE
E_B0 = 2900.0       # initial baseflow storage, mm
DT = 1.0            # day
CAL = (1826, 2922)  # 0-based slice of days for 2006-01-01 .. 2008-12-31 (R: cal_p 1827:2922)
NOISE = 0.05        # 5 % Gaussian noise: Gamma_ii = (0.05 d_i)^2 (Sec. 4)


def _load_inputs():
    F = np.loadtxt(os.path.join(HERE, 'forcing.csv'), delimiter=',')
    M = np.loadtxt(os.path.join(HERE, 'monthly_mean_temp.csv'), delimiter=',')
    return F, M


def _approx(xk, yk, xi):
    return np.interp(xi, xk, yk)


def forcing():
    """Source terms S_i (mm/day) for Hyd 1-3 and observed discharge (l/s), as in main_exe.R.

    Returns S of shape (5844,) (all three calibrated hydrotopes share it: they all
    have interception) and q_obs of shape (5844,).
    """
    F, M = _load_inputs()
    precip, q_m3d, temp = F[:, 1], F[:, 2], F[:, 3]
    n = len(precip)
    # --- Thornthwaite (1948) potential ET, reproducing the R code exactly
    mean_t = M[:, 2]
    with np.errstate(invalid='ignore'):
        hi = (mean_t/5)**1.514
    hi[np.isnan(hi)] = 0
    # R: matrix(heat_index, nrow = 16) fills column-major, rowSums sums every 16th month
    hi = hi.reshape(12, 16).T.sum(axis=1)
    hi = np.repeat(hi, 12)
    a = 0.000000675*hi**3 - 0.000071*hi**2 + 0.01792*hi + 0.49239
    with np.errstate(invalid='ignore', divide='ignore'):
        pe = 16*((10*mean_t)/hi)**a
    pe[np.isnan(pe)] = 0
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    mdays2 = mdays.copy()
    mdays2[1] = 29
    leap = [False, False, False, True]*4                    # 2001 ... 2016
    month_len = np.concatenate([mdays2 if lp else mdays for lp in leap])
    pe = pe/month_len
    knots = np.cumsum(np.concatenate([[15], month_len]))[:192]   # mid-month day numbers
    days_i = np.arange(knots.min(), knots.max() + 1)
    pe_daily = _approx(knots, pe, days_i)
    et = np.zeros(n)
    et[days_i - 1] = pe_daily                                # left_join on dates; NA -> 0
    # --- interception (DVWK 1996, beeches), capped at 5 mm/day
    i_loss = np.concatenate([np.tile([0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.16, 0.15,
                                      0.14, 0.13, 0.12], 16), [0.11]])
    knots_i = np.cumsum(np.concatenate([[1], month_len]))     # first day of each month
    i_daily = _approx(knots_i, i_loss, np.arange(knots_i.min(), knots_i.max() + 1))[:n]
    i_daily = np.minimum(i_daily*precip, 5.0)
    s_a = precip - i_daily
    # --- degree-day snow model (same loops as the R code, incl. its indexing)
    m_rate = np.where(temp <= T_F, 0.0, F_MELT*(temp - T_F))
    s_s = 0.0
    s_tot = np.empty(n)
    for i in range(n):
        if temp[i] < T_F:
            melt = 0.0
            s_s = s_s + (s_a[i + 1] if i + 1 < n else np.nan)   # R adds the next day's input
            s_tot[i] = 0.0
            continue
        if s_s >= m_rate[i]:
            melt = m_rate[i]
            s_s = s_s - m_rate[i]
        else:
            melt = s_s
            s_s = 0.0
        s_tot[i] = melt + s_a[i] if melt > 0 else s_a[i]
    S = s_tot - et
    q_obs = q_m3d*1000/86400
    return S, q_obs


_CACHE = {}


def _forcing_cached():
    if 'S' not in _CACHE:
        _CACHE['S'], _CACHE['q_obs'] = forcing()
    return _CACHE['S'], _CACHE['q_obs']


def observed():
    """Observed spring discharge d (l/s) in the calibration window, shape (1096,)."""
    return _forcing_cached()[1][CAL[0]:CAL[1]].copy()


# ---------------------------------------------------------------- transform (Eqs. 28-33)
def _interim_bounds():
    lo = np.zeros((3, 7))
    hi = np.ones((3, 7))
    lo[0], hi[0] = LB[0], UB[0]
    lo[0, LOGK], hi[0, LOGK] = np.log(LB[0, LOGK]), np.log(UB[0, LOGK])
    lo[:, 2], hi[:, 2] = DE_LB, DE_UB
    return lo.ravel(), hi.ravel()


INTERIM_LB, INTERIM_UB = _interim_bounds()


def to_physical(x, jac=False):
    """Map x in [-1, 1]^21 (Eq. 33 order) to the 21 Table 1 parameters.

    x has shape (..., 21). With jac=True also returns dp/dx of shape (..., 21, 21)
    (a.e.; the min/max in Eqs. 29-30 make the map piecewise smooth).
    """
    x = np.asarray(x, float)
    shp = x.shape[:-1]
    x2 = x.reshape(-1, 21)
    B = x2.shape[0]
    u = (x2 + 1)/2
    v = (INTERIM_LB + u*(INTERIM_UB - INTERIM_LB)).reshape(B, 3, 7)   # interim parameters
    dv = ((INTERIM_UB - INTERIM_LB)/2).reshape(3, 7)                   # dv/dx (diagonal)
    u = u.reshape(B, 3, 7)
    lbq, ubq = LB.copy(), UB.copy()
    lbq[:, LOGK], ubq[:, LOGK] = np.log(LB[:, LOGK]), np.log(UB[:, LOGK])
    q = np.empty((B, 3, 7))           # physical (log for k) values
    J = np.zeros((B, 3, 7, 21))       # dq/dx
    eye = np.eye(21).reshape(3, 7, 21)
    for j in DECREASING + INCREASING:
        q[:, 0, j] = v[:, 0, j]
        J[:, 0, j] = dv[0, j]*eye[0, j]
        for i in (1, 2):
            prev, dprev = q[:, i - 1, j], J[:, i - 1, j]
            if j in DECREASING:      # Eq. 29
                cap = np.minimum(ubq[i, j], prev)
                q[:, i, j] = lbq[i, j] + u[:, i, j]*(cap - lbq[i, j])
                dcap = (prev < ubq[i, j])[:, None]*dprev
                J[:, i, j] = (cap - lbq[i, j])[:, None]*0.5*eye[i, j] + u[:, i, j, None]*dcap
            else:                    # Eq. 30
                flo = np.maximum(prev, lbq[i, j])
                q[:, i, j] = flo + u[:, i, j]*(ubq[i, j] - flo)
                dflo = (prev > lbq[i, j])[:, None]*dprev
                J[:, i, j] = dflo*(1 - u[:, i, j, None]) + (ubq[i, j] - flo)[:, None]*0.5*eye[i, j]
    # Eq. 31: emax_i = emin_i + de_i
    de = np.empty((B, 3))
    de[:, 0] = v[:, 0, 2]
    de[:, 1:] = DE_LB[1:] + u[:, 1:, 2]*(DE_UB[1:] - DE_LB[1:])
    q[:, :, 2] = q[:, :, 1] + de
    for i in range(3):
        J[:, i, 2] = J[:, i, 1] + (DE_UB[i] - DE_LB[i])/2*eye[i, 2]
    p = q.copy()
    p[:, :, LOGK] = np.exp(q[:, :, LOGK])
    J[:, :, LOGK] *= p[:, :, LOGK, None]
    p = p.reshape(shp + (21,))
    if jac:
        return p, J.reshape(shp + (21, 21))
    return p


def to_calibration(p):
    """Inverse of to_physical: Table 1 parameters (..., 21) -> x in [-1, 1]^21."""
    p = np.asarray(p, float)
    shp = p.shape[:-1]
    q = p.reshape(-1, 3, 7).copy()
    q[:, :, LOGK] = np.log(q[:, :, LOGK])
    lbq, ubq = LB.copy(), UB.copy()
    lbq[:, LOGK], ubq[:, LOGK] = np.log(LB[:, LOGK]), np.log(UB[:, LOGK])
    u = np.empty_like(q)
    for j in DECREASING + INCREASING:
        u[:, 0, j] = (q[:, 0, j] - lbq[0, j])/(ubq[0, j] - lbq[0, j])
        for i in (1, 2):
            prev = q[:, i - 1, j]
            if j in DECREASING:
                cap = np.minimum(ubq[i, j], prev)
                u[:, i, j] = (q[:, i, j] - lbq[i, j])/(cap - lbq[i, j])
            else:
                flo = np.maximum(prev, lbq[i, j])
                u[:, i, j] = (q[:, i, j] - flo)/(ubq[i, j] - flo)
    de = q[:, :, 2] - q[:, :, 1]
    u[:, :, 2] = (de - DE_LB)/(DE_UB - DE_LB)
    return (2*u - 1).reshape(shp + (21,))


# ---------------------------------------------------------------- the model
def _check(p):
    P = np.atleast_2d(np.asarray(p, float))
    if P.shape[-1] != 21 or not np.all(np.isfinite(P)):
        raise ValueError('need finite parameters of shape (..., 21)')
    H = P.reshape(-1, 3, 7)
    if np.any(H[:, :, [0, 4, 5]] <= 0):
        raise ValueError('khyd, kis, ksec must be positive')
    if np.any(H[:, :, 2] <= H[:, :, 1]):
        raise ValueError('emax must exceed emin')
    if np.any(H[:, :, 3] <= 0):
        raise ValueError('alpha must be positive')
    return P


def simulate(p, tangent=False, full=False, k_b=K_B):
    """Run LuKARS for a batch of Table 1 parameter vectors p (B, 21).

    Returns Q_tot (B, 1096) in l/s over the calibration window (all 5844 days
    if full=True). With tangent=True also returns dQ/dp of shape (B, n_t, 21),
    the exact derivative of the time-stepping map on each branch (forward mode).
    k_b is the held baseflow coefficient (1/day).
    """
    P = _check(p)
    B = P.shape[0]
    H = P.reshape(B, 3, 7)
    khyd, emin, emax, alpha, kis, ksec, esec = (H[:, :, k] for k in range(7))
    ke = khyd/L_HYD
    S, _ = _forcing_cached()
    n = len(S)
    keep = slice(0, n + 1) if full else slice(CAL[0], CAL[1])
    E = np.zeros((B, 3))
    Qe = np.zeros((B, 3))
    Qis = np.zeros((B, 3))
    Qsec = np.zeros((B, 3))
    Eb = np.full(B, E_B0)
    Qb = np.zeros(B)
    Qtot = np.empty((B, n + 1))
    Qtot[:, 0] = 35*86400
    D = emax - emin
    if tangent:
        # tangents w.r.t. the 7 local parameters of each hydrotope; baseflow w.r.t. all 21
        dE = np.zeros((B, 3, 7))
        dQe = np.zeros((B, 3, 7))
        dQis = np.zeros((B, 3, 7))
        dQsec = np.zeros((B, 3, 7))
        dEb = np.zeros((B, 3, 7))
        dQb = np.zeros((B, 3, 7))
        dQtot = np.zeros((B, n + 1, 3, 7))
        e = np.eye(7)
        dD = (e[2] - e[1])[None, None, :]
    for i in range(n):
        cand = E + (S[i] - (Qsec + Qis + Qe)/A_HYD)*DT
        pos = cand >= 0
        Enew = np.where(pos, cand, 0.0)
        Ebnew = Eb + ((Qis.sum(axis=1) - Qb)/AREA)*DT
        on_sec = Enew >= esec
        Qsec_n = np.where(on_sec, ksec*A_HYD*(Enew - esec), 0.0)
        Qis_n = A_HYD*kis*Enew
        Qb_n = AREA*k_b*Ebnew
        act = ((Enew >= emin) & (Qe > 0)) | ((Enew >= emax) & (Qe <= 0))
        r = np.where(act, (Enew - emin)/D, 0.0)
        ra = np.where(act, r**alpha, 0.0)
        Qe_n = ra*ke*A_HYD
        if tangent:
            dEnew = np.where(pos[:, :, None], dE - (dQsec + dQis + dQe)/A_HYD[None, :, None]*DT, 0.0)
            dEbnew = dEb + (dQis*DT - dQb*DT)/AREA
            dQsec_n = np.where(on_sec[:, :, None],
                               A_HYD[None, :, None]*((Enew - esec)[:, :, None]*e[5]
                                                     + ksec[:, :, None]*(dEnew - e[6])), 0.0)
            dQis_n = A_HYD[None, :, None]*(Enew[:, :, None]*e[4] + kis[:, :, None]*dEnew)
            dQb_n = AREA*k_b*dEbnew
            good = act & (r > 0)
            rs = np.where(good, r, 1.0)
            dr = ((dEnew - e[1]) - rs[:, :, None]*dD)/D[:, :, None]
            dra = ra[:, :, None]*(np.log(rs)[:, :, None]*e[3]
                                  + (alpha/rs)[:, :, None]*dr)
            dra = np.where(good[:, :, None], dra, 0.0)
            dQe_n = (A_HYD/L_HYD)[None, :, None]*(ra[:, :, None]*e[0] + khyd[:, :, None]*dra)
            dE, dEb, dQsec, dQis, dQb, dQe = dEnew, dEbnew, dQsec_n, dQis_n, dQb_n, dQe_n
            dQtot[:, i + 1] = dQe + dQb
        E, Eb, Qsec, Qis, Qb, Qe = Enew, Ebnew, Qsec_n, Qis_n, Qb_n, Qe_n
        Qtot[:, i + 1] = Qe.sum(axis=1) + Qb
    Q = Qtot[:, keep]/86400
    if not np.all(np.isfinite(Q)):
        raise FloatingPointError('non-finite discharge')
    if not tangent:
        return Q
    # dQtot/dp: quickflow of hydrotope h depends on its own parameters only; Qb on all
    dQ = dQtot[:, keep].reshape(B, -1, 21)/86400
    return Q, dQ


def discharge(p):
    """Simulated spring discharge (l/s), 2006-01-01 .. 2008-12-31, shape (B, 1096)."""
    return simulate(p)


def misfit(x, noise=NOISE, k_b=K_B):
    """Bayesian data misfit f_d(x) (Eq. 20, Sec. 4) for x in [-1, 1]^21, shape (B,)."""
    x = np.atleast_2d(x)
    d = observed()
    G = simulate(to_physical(x), k_b=k_b)
    return 0.5*np.sum(((d - G)/(noise*d))**2, axis=1)


def misfit_and_grad(x):
    """f_d(x) (B,) and its gradient w.r.t. x (B, 21), exact on each branch of the model."""
    x = np.atleast_2d(x)
    d = observed()
    p, J = to_physical(x, jac=True)
    G, dG = simulate(p, tangent=True)
    w = (d - G)/(NOISE*d)**2
    f = 0.5*np.sum((d - G)**2/(NOISE*d)**2, axis=1)
    grad_p = -np.einsum('bt,btk->bk', w, dG)
    grad_x = np.einsum('bk,bkj->bj', grad_p, J)
    return f, grad_x


#: a central point: x = 0
NOMINAL_X = np.zeros(21)
