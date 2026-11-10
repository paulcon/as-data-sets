"""HBV-SASK daily rainfall-runoff model of the Oldman River basin, with exact forward sensitivities.

A numpy reimplementation, written for this data set, of the HBV-SASK model described in
Razavi et al. (2019), VARS-TOOL, Environmental Modelling & Software 112:95-107,
https://doi.org/10.1016/j.envsoft.2018.10.005 (Fig. 9 and Sect. 9, pp. 104-105; parameters in
Table 2, p. 105). No code of the VARS-TOOL implementation is used here.

One day t (all water in mm over the basin; states are values at the start of the day):

    P = PM * P_t
    snow (T_t < TT):  SWE += P,  ponding = 0
    rain (T_t >= TT): MS = min(C0 (T_t - TT), SWE),  SWE -= MS,  ponding = P + MS
    PET = max(0, (1 + ETF (T_t - Tmth)) PETmth)          Tmth, PETmth: long-term means of the month
    AET = min(SMS, PET min(SMS / (LP FC), 1))
    R   = ponding min(SMS / FC, 1)^beta
    Q1  = min(K1 S1^alpha, S1),   Q2 = K2 S2
    SMS += ponding - R - AET,  S1 += FRAC R - Q1,  S2 += (1 - FRAC) R - Q2
    Q1route_t = sum_k w_k(UBAS) Q1_{t-k+1}   (triangular unit hydrograph of base UBAS days)
    Q_t (m3/s) = (Q1route_t + Q2) * area_km2 / 86.4

The equations of Fig. 9 are written there as AET = PET (SM/LP), R = (RF + MS)(SM/FC)^beta,
Q1 = K1 S1^alpha, Q2 = K2 S2; the caps (the min() above), the order of the updates within a day,
the use of start-of-day states and the unit-hydrograph weights follow the behaviour of the
VARS-TOOL Python implementation (run as an outside tool to check this port; DATA.md).
w_k is the area of the symmetric triangle on [0, UBAS] (unit area) over day k.

The run starts on 1979-01-01 from the initial states in oldman_basin.csv. The quantities of
interest are computed over the window 1982-01-01 to 1996-12-31 (the window of the VARS-TOOL
HBV-SASK tutorial; the first three years are spin-up):

    NSE    Nash-Sutcliffe efficiency of simulated vs observed daily streamflow (-)
    Qmean  mean simulated streamflow (m3/s)
    Qpeak  mean over the 15 calendar years of the annual maximum daily simulated flow (m3/s)

Gradients are forward (tangent) sensitivities of the discrete daily recursion with respect to
the 12 physical parameters. The recursion is piecewise smooth: min() caps give kinks, and the
rain/snow switch T_t < TT makes every output jump when TT crosses the temperature of a wet day
(temperatures are recorded to 0.01 degC). The tangent is the exact derivative wherever it
exists; it does not contain those jumps (see DATA.md).

Only numpy is imported. Files are read relative to this file.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = ['TT', 'C0', 'ETF', 'LP', 'FC', 'beta', 'FRAC', 'K1', 'alpha', 'K2', 'UBAS', 'PM']
UNITS = ['degC', 'mm/degC/day', '1/degC', '-', 'mm', '-', '-', '1/day', '-', '1/day', 'day', '-']
# Table 2, p. 105, Razavi et al. (2019)
LB = np.array([-4.0, 0.0, 0.0, 0.0, 50.0, 1.0, 0.1, 0.05, 1.0, 0.0, 1.0, 0.5])
UB = np.array([4.0, 10.0, 1.0, 1.0, 500.0, 3.0, 0.9, 1.0, 3.0, 0.05, 3.0, 2.0])
# parameter values of the VARS-TOOL tutorial (pars.inp); not in the paper
TUTORIAL = np.array([0.0, 1.0, 0.1, 0.3, 250.0, 2.0, 0.7, 0.05, 1.5, 0.01, 1.0, 1.0])
QOI_NAMES = ['NSE', 'Qmean', 'Qpeak']
QOI_UNITS = ['-', 'm3/s', 'm3/s']
WINDOW = ((1982, 1, 1), (1996, 12, 31))

(TT, C0, ETF, LP, FC, BETA, FRAC, K1, ALPHA, K2, UBAS, PM) = range(12)


def _load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', ndmin=2)


_DAILY = _load('oldman_daily.csv')
_MONTHLY = _load('oldman_monthly.csv')
_BASIN = _load('oldman_basin.csv')[0]

YEAR, MONTH, DAY = (_DAILY[:, i].astype(int) for i in range(3))
P_OBS, T_OBS, Q_OBS = _DAILY[:, 3], _DAILY[:, 4], _DAILY[:, 5]
AREA_KM2 = float(_BASIN[0])
STATE0 = _BASIN[1:5].copy()       # SWE, SMS, S1, S2 (mm)
TMTH = _MONTHLY[MONTH - 1, 1]
PETMTH = _MONTHLY[MONTH - 1, 2]
TO_CMS = AREA_KM2 / 86.4           # mm/day over the basin -> m3/s


def _day_index(ymd):
    y, m, d = ymd
    return int(np.nonzero((YEAR == y) & (MONTH == m) & (DAY == d))[0][0])


I0, I1 = _day_index(WINDOW[0]), _day_index(WINDOW[1]) + 1
NDAYS = len(P_OBS)


def uh_weights(ubas, grad=False):
    """Weights w_1..w_3 of the triangular unit hydrograph with base ubas in [1, 3] (shape (M, 3)),
    and their derivatives with respect to ubas when grad=True."""
    u = np.asarray(ubas, dtype=float)[:, None]
    x = np.array([0.0, 1.0, 2.0, 3.0])[None, :]
    a = np.minimum(x, u)                      # F(x) = F(min(x, u))
    left = a <= u / 2
    F = np.where(left, 2 * a ** 2 / u ** 2, 1 - 2 * (u - a) ** 2 / u ** 2)
    w = np.diff(F, axis=1)
    if not grad:
        return w
    inside = x < u                            # for x >= u, F = 1 and dF/du = 0
    dF = np.where(left, -4 * a ** 2 / u ** 3, -4 * (u - a) * a / u ** 3)
    dF = np.where(inside, dF, 0.0)
    return w, np.diff(dF, axis=1)


def _check(theta):
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    if theta.shape[1] != 12:
        raise ValueError('theta must have 12 columns: ' + ', '.join(NAMES))
    if not np.all(np.isfinite(theta)):
        raise ValueError('theta has non-finite entries')
    lo = np.array([-np.inf, 0, 0, 0, 1e-9, 0, 0, 0, 0, 0, 1.0, 0])
    hi = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, 1, 3.0, np.inf])
    bad = (theta < lo) | (theta > hi)
    if bad.any():
        j = int(np.nonzero(bad.any(axis=0))[0][0])
        raise ValueError(f'{NAMES[j]} outside the range the model accepts [{lo[j]}, {hi[j]}]')
    return theta


def simulate(theta):
    """Daily simulated streamflow (m3/s), shape (M, NDAYS), for parameter rows theta (M, 12)."""
    th = _check(theta)
    q, _ = _run(th, grad=False, series=True)
    return q


def qoi(theta, grad=False):
    """QoIs (M, 3) = NSE, Qmean, Qpeak; with grad=True also their gradients (M, 3, 12) with
    respect to the physical parameters."""
    th = _check(theta)
    return _run(th, grad=grad, series=False)


def _run(th, grad, series):
    M = th.shape[0]
    p = [th[:, j] for j in range(12)]
    swe, sms, s1, s2 = (np.full(M, v) for v in STATE0)
    lpfc = p[LP] * p[FC]
    w, dw = uh_weights(p[UBAS], grad=True)
    q1h = np.zeros((M, 3))                    # Q1 of days t, t-1, t-2
    qobs_w = Q_OBS[I0:I1]
    denom = np.sum((qobs_w - qobs_w.mean()) ** 2)
    sse = np.zeros(M)
    qsum = np.zeros(M)
    years = YEAR[I0:I1]
    ny = len(np.unique(years))
    peak_sum = np.zeros(M)
    cur_peak = np.full(M, -np.inf)
    qout = np.zeros((M, NDAYS)) if series else None
    if grad:
        E = np.eye(12)
        dswe, dsms, ds1, ds2 = (np.zeros((M, 12)) for _ in range(4))
        dq1h = np.zeros((M, 3, 12))
        dlpfc = np.zeros((M, 12))
        dlpfc[:, LP] = p[FC]
        dlpfc[:, FC] = p[LP]
        dsse = np.zeros((M, 12))
        dqsum = np.zeros((M, 12))
        dpeak_sum = np.zeros((M, 12))
        dcur_peak = np.zeros((M, 12))
    for t in range(NDAYS):
        Pt, Tt = P_OBS[t], T_OBS[t]
        # precipitation and snow module
        P = p[PM] * Pt
        snow = Tt < p[TT]
        mpot = p[C0] * (Tt - p[TT])
        capm = mpot > swe
        melt = np.where(snow, 0.0, np.where(capm, swe, mpot))
        pond = np.where(snow, 0.0, P + melt)
        if grad:
            dP = Pt * E[PM]
            dmpot = np.zeros((M, 12))
            dmpot[:, C0] = Tt - p[TT]
            dmpot[:, TT] = -p[C0]
            dmelt = np.where(snow[:, None], 0.0, np.where(capm[:, None], dswe, dmpot))
            dpond = np.where(snow[:, None], 0.0, dP[None, :] + dmelt)
            dswe = dswe + np.where(snow[:, None], dP[None, :], -dmelt)
        swe = swe + np.where(snow, P, -melt)
        # evapotranspiration module (start-of-day soil moisture)
        fac = 1 + p[ETF] * (Tt - TMTH[t])
        pos = fac > 0
        pet = np.where(pos, fac * PETMTH[t], 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_lp = np.where(sms < lpfc, sms / lpfc, 1.0)
        lim = sms < lpfc
        aet0 = pet * ratio_lp
        capa = aet0 > sms
        aet = np.where(capa, sms, aet0)
        if grad:
            dpet = np.zeros((M, 12))
            dpet[:, ETF] = np.where(pos, (Tt - TMTH[t]) * PETMTH[t], 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                dratio = np.where(lim[:, None], dsms / lpfc[:, None] - (sms / lpfc ** 2)[:, None] * dlpfc, 0.0)
            daet0 = dpet * ratio_lp[:, None] + pet[:, None] * dratio
            daet = np.where(capa[:, None], dsms, daet0)
        # soil and reservoirs
        rfc = np.minimum(sms / p[FC], 1.0)
        rb = rfc ** p[BETA]
        R = pond * rb
        q1p = p[K1] * s1 ** p[ALPHA]
        capq = q1p > s1
        q1 = np.where(capq, s1, q1p)
        q2 = p[K2] * s2
        if grad:
            drfc = np.where((sms < p[FC])[:, None], dsms / p[FC][:, None] - (sms / p[FC] ** 2)[:, None] * E[FC], 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                lnr = np.where(rfc > 0, np.log(rfc), 0.0)
                drb = np.where((rfc > 0)[:, None], (p[BETA] * rfc ** (p[BETA] - 1))[:, None] * drfc, 0.0)
                lns1 = np.where(s1 > 0, np.log(s1), 0.0)
            drb = drb + (rb * lnr)[:, None] * E[BETA]
            dR = dpond * rb[:, None] + pond[:, None] * drb
            dq1p = (s1 ** p[ALPHA])[:, None] * E[K1] + (p[K1] * p[ALPHA] * s1 ** (p[ALPHA] - 1))[:, None] * ds1 \
                + (q1p * lns1)[:, None] * E[ALPHA]
            dq1 = np.where(capq[:, None], ds1, dq1p)
            dq2 = s2[:, None] * E[K2] + p[K2][:, None] * ds2
            dsms = dsms + dpond - dR - daet
            ds1 = ds1 + p[FRAC][:, None] * dR + R[:, None] * E[FRAC] - dq1
            ds2 = ds2 + (1 - p[FRAC])[:, None] * dR - R[:, None] * E[FRAC] - dq2
        sms = sms + pond - R - aet
        s1 = s1 + p[FRAC] * R - q1
        s2 = s2 + (1 - p[FRAC]) * R - q2
        # routing of Q1 by the triangular unit hydrograph
        q1h = np.column_stack([q1, q1h[:, 0], q1h[:, 1]])
        qs = TO_CMS * (np.sum(w * q1h, axis=1) + q2)
        if grad:
            dq1h = np.concatenate([dq1[:, None, :], dq1h[:, :2, :]], axis=1)
            dqs = TO_CMS * (np.einsum('mk,mkj->mj', w, dq1h) + dq2)
            dqs[:, UBAS] += TO_CMS * np.sum(dw * q1h, axis=1)
        if series:
            qout[:, t] = qs
        if I0 <= t < I1:
            e = qs - Q_OBS[t]
            sse += e * e
            qsum += qs
            new = qs > cur_peak
            cur_peak = np.where(new, qs, cur_peak)
            if grad:
                dsse += 2 * e[:, None] * dqs
                dqsum += dqs
                dcur_peak = np.where(new[:, None], dqs, dcur_peak)
            if t + 1 == I1 or YEAR[t + 1] != YEAR[t]:
                peak_sum += cur_peak
                cur_peak = np.full(M, -np.inf)
                if grad:
                    dpeak_sum += dcur_peak
                    dcur_peak = np.zeros((M, 12))
    nw = I1 - I0
    f = np.column_stack([1 - sse / denom, qsum / nw, peak_sum / ny])
    if series:
        return qout, f
    if not grad:
        return f
    G = np.stack([-dsse / denom, dqsum / nw, dpeak_sum / ny], axis=1)
    return f, G
