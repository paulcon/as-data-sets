"""Print the accuracy numbers of DATA.md that the notebook and generate_data.py do not print.

    python check_accuracy.py --pygem DIR      # DIR = PyGEM v0.1 source (git clone --branch v0.1 https://github.com/drounce/PyGEM)
    python check_accuracy.py                   # checks 2-4 only; check 1 needs PyGEM v0.1 and pandas

Check 1: pygem_mb_model against PyGEM v0.1 itself (pygemfxns_massbalance.runmassbalance at commit
         4d710538dae04fc178a383cafaae04198e3e88dd, MIT), fed the same glacier bins and ERA5 series,
         at 12 points (moving and constant area, 3 and 5 inputs).
Check 2: complex-step gradient against central differences at the nominal point, several steps.
Check 3: the size of the jumps of B along a fine T_bias scan (branch changes).
Check 4: B at the three parameter sets of Rounce et al. (2020, p. 179) that match the observation of
         RGI60-13.26360, evaluated here for Hintereisferner.
Check 5: the complex step gives the derivative of the branch the model takes, which can differ from
         the slope of B across the small jumps. Scans of 481 points along k_p, T_snow and d_prec over
         their boxes (other inputs nominal): points where the branch derivative departs by more than
         5 % from the running median of 11 points, the most extreme one, and the integral of the branch
         derivative against the change of B over the scan.
Check 6: all 1000 rows of study m3: the shipped complex-step gradients against a smoothed central
         difference (step 0.005 x box width, which spans many jumps and follows the response slope);
         rows where they differ by more than 10 % and 50 % (box-normalized 2-norm); and the active
         subspace (plain numpy, box-normalized) from either set of gradients.
Check 7 (with --pygem or --priors): the regional priors of PyGEM v0.1's pygem_input.py against the
         box (k_p gamma means and 97.5 % quantiles; T_bias mean + 2 sd). The file is parsed as data
         (ast.literal_eval), not imported. Needs scipy.

    python check_accuracy.py --priors DIR      # DIR = PyGEM v0.1 source; checks 2-7, without check 1
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pygem_mb_model as pm  # noqa: E402

PYGEM_COMMIT = '4d710538dae04fc178a383cafaae04198e3e88dd'
POINTS = [  # k_p, T_bias, f_snow, d_prec, T_snow, area_constant
    (1.0, 0.0, 4.1, 1e-4, 1.0, False),
    (1.0, -2.0, 4.1, 1e-4, 1.0, False),
    (3.1, 0.0, 2.6, 1e-4, 1.0, False),
    (7.4, 0.0, 5.6, 1e-4, 1.0, False),
    (0.5, 2.0, 7.1, 1e-4, 1.0, False),
    (3.5, -2.0, 1.1, 1e-4, 1.0, False),
    (2.0, 1.0, 4.0, 1e-4, 1.0, False),
    (5.0, -3.0, 2.0, 1e-4, 1.0, False),
    (1.7, 0.4, 3.3, 0.0, 0.2, False),
    (2.6, -1.1, 5.5, 2e-4, 1.9, False),
    (1.0, 0.0, 4.1, 1e-4, 1.0, True),
    (3.5, -2.0, 1.1, 1e-4, 1.0, True),
]


def pygem_v01(path):
    try:
        import pandas as pd
    except ImportError:
        sys.exit('check 1 needs pandas (pip install pandas==3.1.0) and PyGEM v0.1: '
                 'git clone --branch v0.1 https://github.com/drounce/PyGEM DIR; then rerun with --pygem DIR.')
    if not os.path.exists(os.path.join(path, 'pygemfxns_massbalance.py')):
        sys.exit(f'no pygemfxns_massbalance.py in {path}: git clone --branch v0.1 https://github.com/drounce/PyGEM DIR '
                 f'(commit {PYGEM_COMMIT[:8]}) and rerun with --pygem DIR.')
    sys.path.insert(0, os.path.abspath(path))
    read_csv = pd.read_csv
    # pygem_input.py reads a High Mountain Asia region table at import; it is not used by runmassbalance
    pd.read_csv = lambda *a, **k: pd.DataFrame({'RGIId': [], 'bolch_name': []})
    try:
        import pygem_input  # noqa: F401
    finally:
        pd.read_csv = read_csv
    import pygemfxns_massbalance as mbv

    def B(kp, tb, fs, pg, ts, const):
        mp = [pm.LAPSE_RATE, pm.LAPSE_RATE, kp, pg, fs / 1000, fs / 1000 / pm.DDF_ICE_RATIO, ts, tb]
        table = pd.Series({'RGIId': 'RGI60-11.00897', 'TermType': 0, 'Zmed': pm.Z_MED})
        nt = len(pm.T_GCM)
        dates = pd.DataFrame({'year': pm.YEARS, 'month': pm.MONTHS, 'daysinmonth': pm.DAYS})
        out = mbv.runmassbalance(mp, table, pm.AREA0.copy(), pm.THICK0.copy(), pm.AREA0.copy(), pm.Z_BINS.copy(),
                                 pm.T_GCM.copy(), np.zeros(nt), pm.P_GCM.copy(), pm.Z_GCM,
                                 np.full(nt, pm.LAPSE_RATE), np.full(nt, pm.LAPSE_RATE), dates,
                                 option_areaconstant=int(const))
        area = out[17][:-1].repeat(12)
        # mass balance as in run_calibration.py: total mass change / initial area / years
        return (out[13] / 1000 * area).sum() / area[0] * 1000 / pm.NYEARS, out[17][-1]
    return B


BOX = {'k_p': (0.5, 3.5), 'T_bias': (-2.0, 2.0), 'f_snow': (1.1, 7.1), 'd_prec': (0.0, 2e-4), 'T_snow': (0.0, 2.0)}


def branch_derivative(x, j):
    """dB/dx_j by the complex step at x (5 physical inputs)."""
    h = 1e-30 * max(abs(x[j]), 1e-4)
    xc = [complex(v) for v in x]
    xc[j] += 1j * h
    return float(np.imag(pm.simulate(*xc)['B'])) / h


def branch_checks():
    print('Check 5: branch derivative (complex step) against the slope of B along scans of 481 points')
    x0 = [pm.NOMINAL[n] for n in pm.NAMES5]
    for j, n in [(0, 'k_p'), (4, 'T_snow'), (3, 'd_prec')]:
        t = np.linspace(*BOX[n], 481)
        B = np.empty(len(t))
        d = np.empty(len(t))
        for i, v in enumerate(t):
            x = list(x0)
            x[j] = v
            B[i] = pm.mass_balance(x)
            d[i] = branch_derivative(x, j)
        med = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(d, 5, mode='edge'), 11), axis=1)
        dev = np.abs(d - med) / np.abs(med)
        k = int(np.argmax(np.abs(d - med)))
        integral = np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(t))
        print(f'  {n}: points departing by more than 5 % from the running median: {(dev > 0.05).sum()} of {len(t)} '
              f'({100 * np.mean(dev > 0.05):.1f} %); most extreme at {n} = {t[k]:.6g}: branch derivative {d[k]:+.4g}, '
              f'running median {med[k]:+.4g}; integral of dB/d{n} over the scan {integral:.5g} against '
              f'B(end) - B(start) = {B[-1] - B[0]:.5g} ({100 * (integral / (B[-1] - B[0]) - 1):+.1f} %)')
        if n == 'k_p':
            # the extreme point by central differences with small steps: the model's branch, not a coding error
            x = list(x0)
            x[0] = t[k]
            fd = []
            for h in (1e-9, 1e-8, 1e-7):
                xp, xm = list(x), list(x)
                xp[0] += h
                xm[0] -= h
                fd.append((pm.mass_balance(xp) - pm.mass_balance(xm)) / (2 * h))
            print(f'    central differences there with steps 1e-9, 1e-8, 1e-7: ' + ', '.join(f'{v:+.4g}' for v in fd))


def as_numpy(G, half):
    Gn = G * half
    lam, W = np.linalg.eigh(Gn.T @ Gn / len(Gn))
    lam, W = lam[::-1], W[:, ::-1]
    return lam, W


def smoothed_gradient_check():
    print('Check 6: shipped complex-step gradients (m3) against smoothed central differences, step 0.005 x box width')
    X = np.loadtxt(os.path.join(HERE, 'inputs_m3.csv'), delimiter=',')
    G = np.loadtxt(os.path.join(HERE, 'gradients_m3.csv'), delimiter=',')
    lo = np.array([BOX[n][0] for n in pm.NAMES3])
    hi = np.array([BOX[n][1] for n in pm.NAMES3])
    half = (hi - lo) / 2
    H = 0.005 * (hi - lo)
    Gs = np.empty_like(G)
    for i, x in enumerate(X):
        for j in range(3):
            xp, xm = x.copy(), x.copy()
            xp[j] += H[j]
            xm[j] -= H[j]
            Gs[i, j] = (pm.mass_balance(xp) - pm.mass_balance(xm)) / (2 * H[j])
    a, b = G * half, Gs * half
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    cos = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    print(f'  relative difference (box-normalized 2-norm): median {100 * np.median(rel):.2f} %, rows above 10 %: '
          f'{(rel > 0.1).sum()}, above 50 %: {(rel > 0.5).sum()} of {len(X)}; smallest direction cosine {cos.min():.4f}')
    lam1, W1 = as_numpy(G, half)
    lam2, W2 = as_numpy(Gs, half)
    w1, w2 = W1[:, 0] * np.sign(W1[0, 0]), W2[:, 0] * np.sign(W2[0, 0])
    print(f'  active subspace, complex step: lambda1/lambda2 = {lam1[0] / lam1[1]:.2f}, w1 = {np.array2string(w1, precision=4)}; '
          f'smoothed differences: lambda1/lambda2 = {lam2[0] / lam2[1]:.2f}, w1 = {np.array2string(w2, precision=4)}; '
          f'largest |difference| in w1 {np.abs(w1 - w2).max():.1e}')


def prior_check(path):
    import ast
    try:
        from scipy import stats
    except ImportError:
        sys.exit('check 7 needs scipy (pip install scipy) and rerun.')
    src = open(os.path.join(path, 'pygem_input.py'), encoding='utf-8-sig').read()   # the file starts with a BOM
    tree = ast.parse(src)
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ('precfactor_gamma_region_dict', 'tempchange_norm_region_dict'):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    kp = np.array(list(vals['precfactor_gamma_region_dict'].values()))     # alpha, beta (rate)
    tb = np.array(list(vals['tempchange_norm_region_dict'].values()))      # mu, sigma
    mean = kp[:, 0] / kp[:, 1]
    q975 = stats.gamma.ppf(0.975, kp[:, 0], scale=1 / kp[:, 1])
    above = 1 - stats.gamma.cdf(BOX['k_p'][1], kp[:, 0], scale=1 / kp[:, 1])
    print(f'Check 7: {len(kp)} regional priors in pygem_input.py. k_p gamma means {mean.min():.3f}-{mean.max():.3f}; '
          f'97.5 % quantiles {q975.min():.2f}-{q975.max():.2f} (median {np.median(q975):.2f}); prior mass above the box '
          f'end k_p = {BOX["k_p"][1]}: median {100 * np.median(above):.1f} %, largest {100 * above.max():.1f} %')
    tin = stats.norm.cdf(2, tb[:, 0], tb[:, 1]) - stats.norm.cdf(-2, tb[:, 0], tb[:, 1])
    print(f'  T_bias normal: means {tb[:, 0].min():+.2f} to {tb[:, 0].max():+.2f} degC, sds {tb[:, 1].min():.2f}-'
          f'{tb[:, 1].max():.2f} (median {np.median(tb[:, 1]):.2f}); mean + 2 sd up to {(tb[:, 0] + 2 * tb[:, 1]).max():.2f} '
          f'degC; prior mass inside [-2, 2] degC: median {100 * np.median(tin):.1f} %, smallest {100 * tin.min():.1f} %')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pygem', help='directory with the PyGEM v0.1 source (check 1)')
    ap.add_argument('--priors', help='directory with the PyGEM v0.1 source, read as data for check 7 only')
    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.pygem:
        B01 = pygem_v01(args.pygem)
        print('Check 1: port vs PyGEM v0.1 runmassbalance (B in m w.e. a-1; area at the end of 2017 in km2)')
        worst = 0.0
        for p in POINTS:
            b01, a01 = B01(*p)
            r = pm.simulate(*p[:5], area_constant=p[5])
            b, a = float(np.real(r['B'])), float(np.real(r['area_annual'][:, -1].sum()))
            worst = max(worst, abs(b - b01))
            print(f'  {p}: PyGEM v0.1 {b01:+.12f}, port {b:+.12f}, |diff| {abs(b - b01):.1e}; '
                  f'area {a01:.6f} vs {a:.6f}')
        print(f'  largest |B_port - B_PyGEM| = {worst:.1e} m w.e. a-1 over {len(POINTS)} points')
    else:
        print('Check 1 skipped (needs --pygem DIR with PyGEM v0.1).')

    print('Check 2: complex-step gradient vs central differences at the nominal point (1, 0, 4.1, 1e-4, 1)')
    x0 = np.array([1.0, 0.0, 4.1, 1e-4, 1.0])
    _, g = pm.mass_balance(x0, grad=True)
    widths = np.array([3.0, 4.0, 6.0, 2e-4, 2.0])      # the box widths of generate_data.py
    for rel in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fd = np.empty(5)
        for j in range(5):
            h = rel * widths[j]
            xp, xm = x0.copy(), x0.copy()
            xp[j] += h
            xm[j] -= h
            fd[j] = (pm.mass_balance(xp) - pm.mass_balance(xm)) / (2 * h)
        err = np.abs(fd - g) / np.abs(g)
        print(f'  step {rel:g} x box width: relative error per input {np.array2string(err, precision=1)}')
    print(f'  complex-step gradient: {np.array2string(g, precision=6)}')

    print('Check 3: jumps of B along T_bias in [-0.5, 0.5] degC (2001 points; k_p 1.5, f_snow 4.1)')
    T = np.linspace(-0.5, 0.5, 2001)
    f = np.array([pm.mass_balance((1.5, t, 4.1)) for t in T])
    d = np.diff(f)
    # a step's departure from the median of the 11 steps around it (the smooth trend)
    pad = np.pad(d, 5, mode='edge')
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(pad, 11), axis=1)
    jump = np.abs(d - smooth)
    print(f'  B range {f.min():.4f} to {f.max():.4f}; largest departure of a step from the local slope '
          f'{jump.max():.1e} m w.e. a-1 (median {np.median(jump):.1e}); steps with departure > 1e-5: {(jump > 1e-5).sum()}')

    print('Check 4: the three parameter sets of p. 179 (T_bias, k_p, f_snow) evaluated for Hintereisferner')
    for tb, kp, fs in [(-2.0, 1.0, 4.1), (0.0, 3.1, 2.6), (0.0, 7.4, 5.6)]:
        print(f'  T_bias {tb:+.0f} degC, k_p {kp}, f_snow {fs}: B = {pm.mass_balance((kp, tb, fs)):+.4f} m w.e. a-1')
    branch_checks()
    smoothed_gradient_check()
    if args.pygem or args.priors:
        prior_check(args.pygem or args.priors)
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
