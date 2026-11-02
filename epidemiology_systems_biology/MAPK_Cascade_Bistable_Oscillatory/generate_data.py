"""Write the nine CSVs of this data set from MAPK_model.py.

    python generate_data.py                          # all three studies, 512 samples each, writes here
    python generate_data.py --study bistable_log     # one study
    python generate_data.py --out DIR --rows 0:16 --check   # a subset in DIR, compared with the files here
    python generate_data.py --fd-check 5             # gradients vs. central differences, 5 rows per study

Three studies. The first two are the designs of the Sobol' analysis of Linden, Kramer and
Rangamani (2022), Fig. 4, in which every free parameter is uniform on its range; the third
is this data set's own choice, not the paper's.

- bistable (the notebook's headline): the seven free parameters (k1..k6, alpha) uniform on
  the ranges of the authors' code (k1: [0, 0.1]; the paper's S2 Table gives [0, 0.05]), the
  other seven at the bistable nominal values. Outputs x2 and x3 at t = 1800 s (30 min) from
  the paper's high- and low-steady-state initial conditions; gradients by forward sensitivities.
- oscillatory: the same parameters (k5 >= 1e-5), the other seven at the oscillatory
  nominal values. Outputs the time-mean of x3 over 980 <= t <= 5400 s (gradient by
  forward sensitivities) and the limit-cycle amplitude and period of x3 (no gradients;
  0 = no limit cycle) with a 0/1 flag.
- bistable_log (secondary): as `bistable`, but ln(theta) uniform on [ln(U/100), ln(U)], U = the
  upper bounds of the bistable ranges, so the box lies inside the paper's box and contains
  the bistable nominal point. The paper uses uniform inputs throughout; the log-uniform
  design is ours. Inputs and gradients are stored in physical units, like the other studies.

Design: a scrambled Sobol' sequence (scipy.stats.qmc.Sobol, scramble=True) of 2^9 = 512
points per study, seeds 570, 571 and 572, mapped linearly to the ranges (to the log
ranges for bistable_log). `--rows a:b` evaluates only those rows of the same design.
One process; the run time is in DATA.md ("Regenerating").

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MAPK_model as mm  # noqa: E402

M = 512
SEEDS = {'bistable': 570, 'oscillatory': 571, 'bistable_log': 572}
LOG_SPAN = 100.0
UPPER_LOG = mm.UPPER_BISTABLE.copy()      # physical bounds of the bistable_log study:
LOWER_LOG = UPPER_LOG/LOG_SPAN            # [U/100, U], U = the paper's upper bounds

B_OUT = ('x2_high', 'x3_high', 'x2_low', 'x3_low')
O_OUT = ('mean_x3', 'amplitude', 'period', 'oscillates')
FILES = ('inputs', 'outputs', 'gradients')


def design(study, n=M):
    u = qmc.Sobol(d=7, scramble=True, seed=SEEDS[study]).random(n)
    if study == 'bistable':
        return mm.LOWER_BISTABLE + (mm.UPPER_BISTABLE - mm.LOWER_BISTABLE)*u
    if study == 'oscillatory':
        return mm.LOWER_OSCILLATORY + (mm.UPPER_OSCILLATORY - mm.LOWER_OSCILLATORY)*u
    lo, hi = np.log(LOWER_LOG), np.log(UPPER_LOG)
    return np.exp(lo + (hi - lo)*u)


def run_bistable(X, tag='bistable', log=True):
    F = np.empty((len(X), 4))
    G = np.empty((len(X), 4, 7))
    t0 = time.time()
    for i, x in enumerate(X):
        qh, gh = mm.bistable(x, mm.X0_HIGH)
        ql, gl = mm.bistable(x, mm.X0_LOW)
        F[i] = np.r_[qh, ql]
        G[i] = np.vstack([gh, gl])
        if log and (i + 1) % 64 == 0:
            print(f'  {tag} {i + 1}/{len(X)}  {time.time() - t0:.0f} s', flush=True)
    return F, G.reshape(len(X), -1)


def run_osc(X, log=True):
    F = np.empty((len(X), 4))
    G = np.empty((len(X), 7))
    t0 = time.time()
    for i, x in enumerate(X):
        o = mm.oscillatory(x)
        F[i] = [o['mean_x3'], o['amplitude'], o['period'], float(o['oscillates'])]
        G[i] = o['grad_mean_x3']
        if log and (i + 1) % 64 == 0:
            print(f'  oscillatory {i + 1}/{len(X)}  {time.time() - t0:.0f} s', flush=True)
    return F, G


def _save(path, A, desc, cols):
    np.savetxt(path, A, fmt='%.15g', delimiter=',', header=desc + '\n' + ','.join(cols))


def write(out, study, X, F, G):
    names = list(mm.NAMES)
    units = 'rate constants k1, k2, k4, k6 in 1/s, k3 and k5 in 1/(nM s), alpha dimensionless'
    if study == 'oscillatory':
        rng = 'uniform on the oscillatory ranges of the authors\' code (k5 >= 1e-5)'
        odesc = ('x0 = [10, 80, 80]; mean of x3 (nM) on the 20 s grid over 980 <= t <= 5400 s; '
                 'limit-cycle amplitude of x3 (nM) and period (min), 0 = no limit cycle; oscillates (0/1)')
        ocols, gcols = list(O_OUT), [f'dmean_x3/d{n}' for n in names]
        gdesc = 'gradient of mean_x3 w.r.t. the 7 physical inputs (forward sensitivities)'
    else:
        rng = ('uniform on the bistable ranges of the authors\' code' if study == 'bistable' else
               'ln(theta) uniform on [ln(U/100), ln(U)], U = upper bounds of the authors\' bistable ranges')
        odesc = ('x2 and x3 (nM) at t = 1800 s from the high-steady-state (x0 = [0.1245, 2.4870, 31.2623]) '
                 'and low-steady-state (x0 = [0.0015, 3.6678, 28.7307]) initial conditions')
        ocols, gcols = list(B_OUT), [f'd{o}/d{n}' for o in B_OUT for n in names]
        gdesc = ('gradients of the 4 outputs w.r.t. the 7 physical inputs (forward sensitivities); '
                 'column 7*i + j is d(output i)/d(input j)')
    _save(os.path.join(out, f'{study}_inputs.csv'), X,
          f'{len(X)} scrambled Sobol points (seed {SEEDS[study]}) of the 7 free MAPK parameters, {rng}; {units}',
          names)
    _save(os.path.join(out, f'{study}_outputs.csv'), F, odesc, ocols)
    _save(os.path.join(out, f'{study}_gradients.csv'), G, gdesc, gcols)


def _load(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def check(out, study, rows):
    for f in FILES:
        nm = f'{study}_{f}'
        new = _load(os.path.join(out, nm + '.csv'))
        old = _load(os.path.join(HERE, nm + '.csv'))[rows]
        scale = np.maximum(np.abs(old).max(axis=1, keepdims=True), 1e-300)
        print(f'{nm:24s} max relative difference {np.max(np.abs(new - old)/scale):.2e}')


def _fd(q, x, rel=1e-5):
    """Central differences of q at x with step rel*x[j]; a forward difference with step
    rel*1e-3 where x[j] = 0 (the parameters must stay nonnegative)."""
    g = np.empty(len(x))
    for j in range(len(x)):
        a, b = x.copy(), x.copy()
        if x[j] > 0:
            h = rel*x[j]
            a[j] += h
            b[j] -= h
            g[j] = (q(a) - q(b))/(2*h)
        else:
            h = rel*1e-3
            a[j] += h
            g[j] = (q(a) - q(x))/h
    return g


def fd_check(n):
    """Central differences of state-only solves at rtol = atol = 1e-12 vs. the stored gradients."""
    tol = 1e-12
    tb = [0.0, mm.T_END_BISTABLE]
    for study in ('bistable', 'bistable_log'):
        X = _load(os.path.join(HERE, f'{study}_inputs.csv'))
        G = _load(os.path.join(HERE, f'{study}_gradients.csv')).reshape(-1, 4, 7)
        rows = np.linspace(0, len(X) - 1, n).astype(int)
        errs = []
        for i in rows:
            for k, x0 in ((1, mm.X0_HIGH), (3, mm.X0_LOW)):
                q = lambda th: mm.solve(mm._full(th, mm.NOMINAL_BISTABLE), x0, tb, False, tol, tol)[0][2, -1]  # noqa
                g = _fd(q, X[i])
                e = np.linalg.norm(g - G[i, k])/np.linalg.norm(g)
                errs.append(e)
                print(f'{study} row {i:3d} {B_OUT[k]:8s} relative error {e:.1e}')
        print(f'{study} max {max(errs):.1e}')
    X = _load(os.path.join(HERE, 'oscillatory_inputs.csv'))
    G = _load(os.path.join(HERE, 'oscillatory_gradients.csv'))
    rows = np.linspace(0, len(X) - 1, n).astype(int)
    errs = []
    t = mm.osc_times()
    for i in rows:
        q = lambda th: mm.solve(mm._full(th, mm.NOMINAL_OSCILLATORY), mm.X0_OSC, t, False, tol, tol)[0][2, mm.START_OSC:].mean()  # noqa
        g = _fd(q, X[i])
        e = np.linalg.norm(g - G[i])/np.linalg.norm(g)
        errs.append(e)
        print(f'oscillatory row {i:3d} mean_x3  relative error {e:.1e}')
    print(f'oscillatory max {max(errs):.1e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--study', default='all', choices=['all'] + list(SEEDS))
    ap.add_argument('--rows', default=None, help='a:b, a slice of the design')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--fd-check', type=int, default=0)
    a = ap.parse_args()
    if os.path.abspath(a.out) == HERE and (a.check or a.rows):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    if a.fd_check:
        fd_check(a.fd_check)
        return
    rows = slice(*[int(v) for v in a.rows.split(':')]) if a.rows else slice(None)
    os.makedirs(a.out, exist_ok=True)
    for study in (SEEDS if a.study == 'all' else [a.study]):
        X = design(study)[rows]
        t0 = time.time()
        F, G = run_osc(X) if study == 'oscillatory' else run_bistable(X, study)
        print(f'{study}: {time.time() - t0:.0f} s', flush=True)
        write(a.out, study, X, F, G)
        if a.check:
            check(a.out, study, rows)


if __name__ == '__main__':
    main()
