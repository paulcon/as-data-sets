"""HyMod rainfall-runoff model, as run by the SAFE toolbox (Pianosi, Sarrazin and
Wagener, Environ. Model. Softw. 70:80-85, 2015), with forward-mode derivatives.

HyMod (Boyle 2001; Wagener et al. 2001) is a daily conceptual model with five
parameters:

    Sm    maximum soil moisture storage, mm
    beta  shape exponent of the soil-moisture (excess rainfall) curve, -
    alfa  fraction of excess rainfall routed to the quick-flow cascade, -
    Rs    slow (groundwater) reservoir outflow coefficient, 1/day
    Rf    outflow coefficient of each of the three quick reservoirs, 1/day

and five states, all in mm and all zero at the start: soil moisture s, slow
store L, and quick stores f1, f2, f3. One day t, with rainfall r and potential
evapotranspiration e (mm/day), is

    u   = s / Sm
    F   = 1 - (1 - u)**beta                   fraction of rain that is excess
    s1  = clip(s + r - F r, 0, Sm)            store after rain; overflow is excess
    Pe  = s + r - s1                          effective rainfall (excess + overflow)
    Ea  = s1 - s'                             with s' = clip(s1 - min(u, 1) e, 0, Sm)
    Qs  = Rs L,      L'  = L + (1 - alfa) Pe - Qs
    Qf  = Rf f3,     f1' = f1 + alfa Pe - Rf f1,
                     f2' = f2 + Rf f1 - Rf f2,  f3' = f3 + Rf f2 - Rf f3
    Q   = Qs + Qf                             simulated flow, mm/day

These are the update rules of SAFE's `hymod_sim` (soil routine applied before
evaporation, evaporation scaled by min(|s/Sm|, 1) at the start of the day, the
soil clip before evaporation, explicit daily steps, outflows from start-of-day
storages, Sm floored at machine epsilon). The rules were taken from reading
SAFE's code (SAFE-python `HyMod.hymod_sim`, SAFE-matlab `hymod_sim.m`; both
GPL-3.0); they are not in a published model description. This module was
written fresh in numpy: no SAFE code or comments are copied. It agrees with
SAFEpython 0.2.0 to rounding (see DATA.md, "Accuracy").

The quantities of interest are computed over days WARMUP+1 ... 365 of the
forcing (the settings of SAFE-matlab's `workflow_eet_hymod.m`, the script
excerpted in the paper's Fig. 4: `LeafCatch(1:365,:)`, `warmup = 30`, stores
empty at day 1): the Nash-Sutcliffe efficiency of
simulated vs observed flow (SAFE `hymod_nse`), the mean simulated flow, and the
peak simulated flow (SAFE `hymod_max`).

The model is continuous and piecewise smooth in the parameters (the clips are
kinks). `qoi_and_grad` propagates exact derivatives of each smooth piece
(forward mode, chain rule through the daily recursion); they are the gradient
almost everywhere.

Only numpy is imported.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: Parameter names, in column order (Fig. 4 of the paper, `X_labels`).
NAMES = ('Sm', 'beta', 'alfa', 'Rs', 'Rf')
UNITS = ('mm', '-', '-', '1/day', '1/day')

#: Parameter ranges, "from literature" (Fig. 4 of the paper, `xmin`, `xmax`).
LOWER = np.array([0., 0., 0., 0., .1])
UPPER = np.array([400., 2., 1., .1, 1.])

#: The paper gives no nominal point; the range midpoints are used where one is needed.
NOMINAL = (LOWER + UPPER) / 2

#: Warm-up days excluded from the QoIs (`warmup = 30` in SAFE-matlab's workflow_eet_hymod.m).
WARMUP = 30

#: QoI names and units.
QOI_NAMES = ('NSE', 'Qmean', 'Qmax')
QOI_UNITS = ('-', 'mm/day', 'mm/day')

_EPS = np.finfo(float).eps


def load_forcing(path=None):
    """Rainfall, potential evapotranspiration and observed flow (mm/day), 365 days each."""
    path = path or os.path.join(HERE, 'forcing.csv')
    d = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    return d[:, 1], d[:, 2], d[:, 3]


def simulate(P, rain, pet, grad=False):
    """Simulated daily flow for each row of physical parameters P, shape (N, 5).

    Returns Q, shape (N, T); with grad=True also dQ/dP, shape (N, T, 5).
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.shape[1] != 5:
        raise ValueError('P must have 5 columns (Sm, beta, alfa, Rs, Rf)')
    if not np.all(np.isfinite(P)):
        raise ValueError('non-finite parameter')
    N, T = len(P), len(rain)
    Sm = np.maximum(P[:, 0], _EPS)
    beta, alfa, Rs, Rf = P[:, 1], P[:, 2], P[:, 3], P[:, 4]

    s, L, f1, f2, f3 = (np.zeros(N) for _ in range(5))
    Q = np.empty((N, T))
    if grad:
        E = np.eye(5)
        dSm = np.outer(P[:, 0] > _EPS, E[0])          # d Sm / dP, (N, 5)
        eb, ea, es, ef = E[1], E[2], E[3], E[4]
        ds, dL, d1, d2, d3 = (np.zeros((N, 5)) for _ in range(5))
        dQ = np.empty((N, T, 5))

    for t in range(T):
        r, e = rain[t], pet[t]
        u = s / Sm
        full = u >= 1.
        om = np.where(full, 1., 1. - u)                # 1 - u, guarded where the store is full
        g = np.where(full, 0., om ** beta)             # 1 - F
        a = s + r * g                                  # s + r - F r
        over = a > Sm
        s1 = np.where(over, Sm, a)                     # a >= 0 always
        pe = s + r - s1
        W = np.minimum(u, 1.)
        b = s1 - W * e
        dry = b < 0.
        s_new = np.where(dry, 0., b)                   # b <= Sm always
        qs, qf = Rs * L, Rf * f3
        Q[:, t] = qs + qf

        if grad:
            du = ds / Sm[:, None] - (s / Sm ** 2)[:, None] * dSm
            lg = np.where(full, 0., np.log(np.where(full, 1., om)))
            with np.errstate(divide='ignore', invalid='ignore'):
                dg = np.where(full[:, None], 0.,
                              g[:, None] * (-(beta / om)[:, None] * du + lg[:, None] * eb))
            da = ds + r * dg
            ds1 = np.where(over[:, None], dSm, da)
            dpe = ds - ds1
            dW = np.where(full[:, None], 0., du)
            ds_new = np.where(dry[:, None], 0., ds1 - e * dW)
            dqs = Rs[:, None] * dL + L[:, None] * es
            dqf = Rf[:, None] * d3 + f3[:, None] * ef
            dQ[:, t] = dqs + dqf
            dL_new = dL + (1 - alfa)[:, None] * dpe - pe[:, None] * ea - dqs
            d1_new = d1 + alfa[:, None] * dpe + pe[:, None] * ea - Rf[:, None] * d1 - f1[:, None] * ef
            d2_new = d2 + Rf[:, None] * (d1 - d2) + (f1 - f2)[:, None] * ef
            d3_new = d3 + Rf[:, None] * (d2 - d3) + (f2 - f3)[:, None] * ef
            ds, dL, d1, d2, d3 = ds_new, dL_new, d1_new, d2_new, d3_new

        L_new = L + (1 - alfa) * pe - qs
        f1_new = f1 + alfa * pe - Rf * f1
        f2_new = f2 + Rf * (f1 - f2)
        f3_new = f3 + Rf * (f2 - f3)
        s, L, f1, f2, f3 = s_new, L_new, f1_new, f2_new, f3_new

    if not np.all(np.isfinite(Q)):
        raise FloatingPointError('non-finite simulated flow')
    return (Q, dQ) if grad else Q


def qoi(P, forcing=None, warmup=WARMUP):
    """(NSE, mean flow, peak flow) over days warmup+1..T for each row of P, shape (N, 3).

    `forcing` is (rain, pet, flow), default `load_forcing()`; T is its length.
    """
    return qoi_and_grad(P, forcing, grad=False, warmup=warmup)


def qoi_and_grad(P, forcing=None, grad=True, warmup=WARMUP):
    """QoIs, shape (N, 3), and (with grad=True) their gradients w.r.t. P, shape (N, 3, 5)."""
    rain, pet, flow = forcing if forcing is not None else load_forcing()
    if not 0 <= warmup < len(rain) - 1:
        raise ValueError('warmup must be in [0, T - 1)')
    out = simulate(P, rain, pet, grad=grad)
    Q = (out[0] if grad else out)[:, warmup:]
    obs = flow[warmup:]
    den = np.sum((obs - obs.mean()) ** 2)
    res = Q - obs
    k = np.argmax(Q, axis=1)
    rows = np.arange(len(Q))
    f = np.column_stack([1 - np.sum(res ** 2, axis=1) / den, Q.mean(axis=1), Q[rows, k]])
    if not grad:
        return f
    dQ = out[1][:, warmup:]
    G = np.stack([-2 * np.einsum('nt,ntj->nj', res, dQ) / den,
                  dQ.mean(axis=1),
                  dQ[rows, k]], axis=1)
    return f, G
