"""Write the data files of Karst_Aquifer_LuKARS_Calibration from lukars_model.py.

    python generate_data.py                 # all data files (about 3 min)
    python generate_data.py --check         # recompute and compare with the files here
    python generate_data.py --forcing DIR   # (once) convert the authors' forcing files

Design (Teixeira Parente et al. 2019, Sec. 4): x ~ U[-1, 1]^21 on the normalized
calibration parameters of Eq. (33), N = 1000 Monte Carlo samples drawn with
numpy's default_rng(SEED). For each sample: the 21 physical parameters (Table 1
units), the Bayesian data misfit f_d (5 % Gaussian noise), its exact gradient
with respect to x (forward-mode tangent of the time stepping, chained through the
Eq. 28-33 transform), and the simulated daily discharge 2006-2008. A second,
independent Monte Carlo set of NVAL = 5000 misfits (seed SEED + 1) is written
for validating surrogates. Three sets of central finite-difference gradients
are written for comparison with the paper only (the paper states "central
finite differences" and no step):
  gradients_fd.csv              untruncated parameters, h = 1e-4 in x (the step
                                in the authors' 2020 HydroShare model.py);
  gradients_fd_rounded.csv      h = 1e-5, physical parameters rounded to 6
                                decimals before each run, as the '%f' format in
                                the authors' model.py does when it passes them
                                to Rscript;
  gradients_fd_rounded_lb70.csv the same with the e_min,3 lower bound of the
                                authors' settings file (70 mm) in the transform
                                instead of Table 1's 75 mm.

`--forcing DIR` converts kerschbaum.txt, snow_waidhofen.csv and monthly_mean_temp.csv
from the HydroShare resource (DIR = .../as_dwt/examples/as_wavelets/scens/scen_main)
to forcing.csv and monthly_mean_temp.csv. The numbers are copied as text, so the
conversion is lossless; dates become day numbers (day 1 = 2001-01-01).

Only numpy and the standard library are imported.
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lukars_model as lm  # noqa: E402

SEED = 2019
N = 1000
NVAL = 5000
BATCH = 250
H_FD = 1e-4   # central-difference step in x used by the authors' 2020 model.py (HydroShare code)
H_FD_ROUNDED = 1e-5   # step for the '%f'-rounding emulation (reproduces the paper's Fig. 2 best)
EMIN3_LB_AUTHORS = 70.0   # e_min,3 lower bound in the authors' settings file (Table 1: 75)


def convert_forcing(src, out):
    rows = [ln.rstrip('\n').split(';') for ln in open(os.path.join(src, 'kerschbaum.txt'))]
    assert rows[0] == ['"date"', '"precipitation_mm_d"', '"discharge_m3_d"'], rows[0]
    snow = [ln.rstrip('\n').split(';') for ln in open(os.path.join(src, 'snow_waidhofen.csv'))]
    assert snow[0][0] == 'date' and snow[0][-1] == 'temp_mitterlug', snow[0]
    rows, snow = rows[1:], snow[1:]
    d0 = dt.date(2001, 1, 1)
    lines = []
    for k, (r, s) in enumerate(zip(rows, snow)):
        day = dt.date.fromisoformat(r[0])
        assert day == d0 + dt.timedelta(days=k)
        assert dt.datetime.strptime(s[0], '%d.%m.%Y').date() == day
        lines.append(f'{k + 1},{r[1]},{r[2]},{s[-1]}')
    assert len(lines) == 5844 and len(snow) == 5844
    with open(os.path.join(out, 'forcing.csv'), 'w') as fh:
        fh.write('# Kerschbaum spring daily forcing and discharge 2001-01-01..2016-12-31 (day 1 = '
                 '2001-01-01); from HydroShare 4901a0d654334c259f4ff9b49dc0a74e (kerschbaum.txt, '
                 'snow_waidhofen.csv), CC BY 4.0, values copied as text\n')
        fh.write('# day,precipitation (mm/day),spring discharge (m^3/day),air temperature at '
                 'Mitterlug (degC)\n')
        fh.write('\n'.join(lines) + '\n')
    mon = [ln.rstrip('\n').split(';') for ln in open(os.path.join(src, 'monthly_mean_temp.csv'))]
    assert mon[0] == ['month', 'year', 'temperature'], mon[0]
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December']
    out_lines = []
    for k, m in enumerate(mon[1:]):
        assert m[0] == months[k % 12] and int(m[1]) == 2001 + k//12
        out_lines.append(f'{m[1]},{months.index(m[0]) + 1},{m[2]}')
    with open(os.path.join(out, 'monthly_mean_temp.csv'), 'w') as fh:
        fh.write('# monthly mean air temperature 2001-2016 used for Thornthwaite ET; from HydroShare '
                 '4901a0d654334c259f4ff9b49dc0a74e (monthly_mean_temp.csv), CC BY 4.0\n')
        fh.write('# year,month (1-12),mean air temperature (degC)\n')
        fh.write('\n'.join(out_lines) + '\n')
    # lossless check
    F = np.loadtxt(os.path.join(out, 'forcing.csv'), delimiter=',')
    assert np.array_equal(F[:, 1], [float(r[1]) for r in rows])
    assert np.array_equal(F[:, 2], [float(r[2]) for r in rows])
    assert np.array_equal(F[:, 3], [float(s[-1]) for s in snow])
    print('forcing converted:', F.shape)


def compute(n, seed, grads=True):
    X = np.random.default_rng(seed).uniform(-1, 1, size=(n, 21))
    f = np.empty(n)
    G = np.empty((n, 21)) if grads else None
    Q = np.empty((n, 1096)) if grads else None
    for s in range(0, n, BATCH):
        sl = slice(s, min(n, s + BATCH))
        if grads:
            f[sl], G[sl] = lm.misfit_and_grad(X[sl])
            Q[sl] = lm.discharge(lm.to_physical(X[sl]))
        else:
            f[sl] = lm.misfit(X[sl])
    return X, f, G, Q


def round_like_percent_f(P):
    """Write each parameter with Python's '%f' (6 decimals) and read it back.

    This is what the authors' model.py (HydroShare, as_dwt/examples/as_wavelets,
    line 35) does when it passes the physical parameters to Rscript, whose
    main_exe.R reads them with as.numeric (line 3).
    """
    P = np.asarray(P, float)
    return np.array([float('%f' % v) for v in P.ravel()]).reshape(P.shape)


def misfit_rounded(X, emin3_lb=None):
    """f_d(x) with the physical parameters rounded as by '%f' before the run.

    emin3_lb replaces the e_min,3 lower bound of the transform (Table 1: 75 mm;
    the authors' settings file: 70 mm). lukars_model is restored afterwards.
    """
    old = lm.LB[2, 1]
    if emin3_lb is not None:
        lm.LB[2, 1] = emin3_lb
    try:
        P = lm.to_physical(np.atleast_2d(X))
    finally:
        lm.LB[2, 1] = old
    d = lm.observed()
    Q = lm.simulate(round_like_percent_f(P))
    return 0.5*np.sum(((d - Q)/(lm.NOISE*d))**2, axis=1)


def fd_gradients(X, h=H_FD, rounded=False, emin3_lb=None):
    """Central finite differences of f_d in x, 42 runs per sample.

    rounded=False: untruncated parameters (lukars_model.misfit). rounded=True:
    parameters rounded to 6 decimals before each run (misfit_rounded), as in the
    authors' interface. Not clipped at the box: x +- h may leave [-1, 1] by h,
    where the transform extends linearly.
    """
    fun = (lambda Z: misfit_rounded(Z, emin3_lb)) if rounded else lm.misfit
    X = np.atleast_2d(X)
    out = np.empty_like(X)
    E = h*np.eye(21)
    for s in range(0, len(X), 50):
        Xs = X[s:s + 50]
        Xp = np.concatenate([Xs[:, None, :] + E, Xs[:, None, :] - E], 1).reshape(-1, 21)
        fp = fun(Xp).reshape(len(Xs), 2, 21)
        out[s:s + 50] = (fp[:, 0] - fp[:, 1])/(2*h)
    return out


DESC = ('LuKARS (lukars_model.py, numpy port of the authors\' main_exe.R), Kerschbaum spring; '
        'x ~ U[-1,1]^21 on the Eq. 33 calibration parameters, Monte Carlo with '
        'numpy default_rng(%d); written by generate_data.py')


def write(out):
    t0 = time.time()
    X, f, G, Q = compute(N, SEED)
    t1 = time.time()
    Xv, fv, _, _ = compute(NVAL, SEED + 1, grads=False)
    t2 = time.time()
    Gfd = fd_gradients(X)
    t3 = time.time()
    Gr75 = fd_gradients(X, H_FD_ROUNDED, rounded=True)
    Gr70 = fd_gradients(X, H_FD_ROUNDED, rounded=True, emin3_lb=EMIN3_LB_AUTHORS)
    t4 = time.time()
    P, Pv = lm.to_physical(X), lm.to_physical(Xv)
    names = ','.join(f'{nm} ({lm.UNITS[nm.rsplit("_", 1)[0]]})' for nm in lm.NAMES)
    np.savetxt(os.path.join(out, 'inputs.csv'), P, fmt='%.17g', delimiter=',',
               header='%d samples of the 21 physical LuKARS parameters (paper Table 1 order and '
                      'units). ' % N + DESC % SEED + '\n' + names)
    np.savetxt(os.path.join(out, 'outputs.csv'), f[:, None], fmt='%.17g', delimiter=',',
               header='Bayesian data misfit f_d = 1/2 sum((d - G)/(0.05 d))^2 over 2006-01-01..'
                      '2008-12-31 (1096 days) for the rows of inputs.csv. ' + DESC % SEED
                      + '\nf_d (dimensionless)')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='df_d/dx_j, x = normalized calibration parameters of Eq. 33 (x = '
                      'lukars_model.to_calibration(inputs)); exact forward-mode derivative of the '
                      'model on each branch. ' + DESC % SEED + '\n'
                      + ','.join(f'df/d{c}' for c in lm.CAL_NAMES))
    np.savez_compressed(os.path.join(out, 'discharge.npz'), discharge=Q.astype(np.float32),
                        observed=lm.observed(),
                        days=np.arange(lm.CAL[0] + 1, lm.CAL[1] + 1),
                        description=np.array('Simulated spring discharge (l/s, stored as float32), rows = rows of '
                                             'inputs.csv, columns = days 1827..2922 (2006-01-01..'
                                             '2008-12-31, day 1 = 2001-01-01); observed = measured '
                                             'discharge (l/s) on the same days'))
    np.savetxt(os.path.join(out, 'validation.csv'), np.hstack([Pv, fv[:, None]]), fmt='%.17g',
               delimiter=',',
               header='%d independent samples (physical parameters, Table 1 order) and the misfit '
                      'f_d, for surrogate validation. ' % NVAL + DESC % (SEED + 1) + '\n'
               + names + ',f_d (dimensionless)')
    np.savetxt(os.path.join(out, 'gradients_fd.csv'), Gfd, fmt='%.17g', delimiter=',',
               header='df_d/dx_j by central finite differences of the untruncated model with '
                      "step h = 1e-4 in x (the step in the authors' 2020 model.py), same rows as "
                      'inputs.csv; for comparison with the paper only (see DATA.md). '
                      + DESC % SEED + '\n' + ','.join(f'df/d{c}' for c in lm.CAL_NAMES))
    for G_, fn, lbtxt in ((Gr75, 'gradients_fd_rounded.csv', 'Table 1 bounds (e_min,3 >= 75 mm)'),
                          (Gr70, 'gradients_fd_rounded_lb70.csv',
                           "the authors' settings-file bound e_min,3 >= 70 mm in the transform")):
        np.savetxt(os.path.join(out, fn), G_, fmt='%.17g', delimiter=',',
                   header="df_d/dx_j by central finite differences with step h = 1e-5 in x, with the "
                          "physical parameters rounded to 6 decimals before each run (emulating the "
                          "'%%f' format in the authors' model.py), %s; same x as inputs.csv; for "
                          "comparison with the paper only, not gradients of the model (see DATA.md). "
                          % lbtxt + DESC % SEED + '\n' + ','.join(f'df/d{c}' for c in lm.CAL_NAMES))
    print(f'gradient set {t1 - t0:.1f} s, validation set {t2 - t1:.1f} s, '
          f'FD gradients {t3 - t2:.1f} s, rounded FD gradients {t4 - t3:.1f} s, total {t4 - t0:.1f} s')
    return X, f, G, Q, fv


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def check():
    t0 = time.time()
    X, f, G, Q = compute(N, SEED)
    _, fv, _, _ = compute(NVAL, SEED + 1, grads=False)
    P = read_csv(os.path.join(HERE, 'inputs.csv'))
    rel = lambda a, b: np.max(np.abs(a - b)/np.maximum(np.abs(b), 1e-300))  # noqa: E731
    print('inputs  max rel diff', rel(lm.to_physical(X), P))
    print('x round trip max abs', np.max(np.abs(lm.to_calibration(P) - X)))
    print('outputs max rel diff', rel(f, read_csv(os.path.join(HERE, 'outputs.csv'))[:, 0]))
    Gs = read_csv(os.path.join(HERE, 'gradients.csv'))
    print('gradients max rel diff (row norms)',
          np.max(np.linalg.norm(G - Gs, axis=1)/np.linalg.norm(Gs, axis=1)))
    Qs = np.load(os.path.join(HERE, 'discharge.npz'))['discharge']
    print('discharge max rel diff', rel(Q.astype(np.float32), Qs))
    print('validation max rel diff', rel(fv, read_csv(os.path.join(HERE, 'validation.csv'))[:, -1]))
    Gf = read_csv(os.path.join(HERE, 'gradients_fd.csv'))
    print('FD gradients max abs diff / max abs', np.max(np.abs(fd_gradients(X) - Gf))/np.max(np.abs(Gf)))
    for fn, lb in (('gradients_fd_rounded.csv', None), ('gradients_fd_rounded_lb70.csv', EMIN3_LB_AUTHORS)):
        Gf = read_csv(os.path.join(HERE, fn))
        Gc = fd_gradients(X, H_FD_ROUNDED, rounded=True, emin3_lb=lb)
        print(fn, 'max abs diff / max abs', np.max(np.abs(Gc - Gf))/np.max(np.abs(Gf)))
    print(f'{time.time() - t0:.1f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--forcing', help='HydroShare scen_main directory to convert')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    if a.forcing:
        convert_forcing(a.forcing, a.out)
    elif a.check:
        check()
    else:
        write(a.out)
