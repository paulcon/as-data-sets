"""Accuracy checks quoted in DATA.md that the notebook and generate_data.py do not print.

    python check_accuracy.py                       # model-only checks (1, 2, 5)
    python check_accuracy.py --vars-tool DIR       # also checks 3 and 4 against VARS-TOOL

1. TT jumps. On a grid of 2001 values of TT over +-0.25 degC around 8 uniform samples, the
   change of each output is split into the integral of the tangent derivative and the rest,
   which is the sum of the jumps at the rain/snow switch.
2. Step of the TT difference. At 200 uniform samples, the central differences in TT with
   steps 0.025, 0.05, 0.2 and 0.4 degC are compared with the step 0.1 degC used in
   gradients.csv, relative to the 2-norm of the gradient in [-1, 1] coordinates.
3. Port vs stored VARS-TOOL runs. DIR is the `tutorials` folder of a clone of
   https://github.com/vars-tool/vars-tool (commit df72ad2a was used). The 400 NSE values in
   "9b - sample_HBV.csv" (the VARS-TOOL tutorial's uniform random sample, NSE over
   1982-01-01..1996-12-31) are recomputed with hbv_sask_model.
4. Port vs the VARS-TOOL Python model, daily series. Needs pandas, scipy, matplotlib and
   IPython (VARS-TOOL's tutorials/hbv.py imports them); the original is imported from DIR and
   run as an outside tool at the tutorial point and 4 uniform samples, and the daily Q (m3/s)
   of 1979-2008 is compared. pandas 3 removed the read_csv keyword `delim_whitespace` that
   hbv.py uses; this script maps it to sep=r'\\s+' for that import only.

5. Flow floor. VARS-TOOL's hbv.py sets simulated flows below 1e-5 m3/s to 1e-5; the port has no
   such floor. At the 2000 shipped samples, the number of samples with a daily flow below 1e-5
   m3/s (1979-2008) is printed, and the three outputs are recomputed from the daily series with
   and without the floor.

Nothing is written. Only numpy is needed for checks 1, 2, 3 and 5.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hbv_sask_model as hm  # noqa: E402

SEED = 689


def tt_jumps():
    rng = np.random.default_rng(SEED + 1)
    X = hm.LB + (hm.UB - hm.LB) * rng.random((8, 12))
    share = np.zeros((8, 3))
    for i, x in enumerate(X):
        tts = np.linspace(x[0] - 0.25, x[0] + 0.25, 2001)
        Xg = np.repeat(x[None], len(tts), 0)
        Xg[:, 0] = tts
        f, G = hm.qoi(Xg, grad=True)
        for q in range(3):
            total = f[-1, q] - f[0, q]
            smooth = np.trapezoid(G[:, q, 0], tts) if hasattr(np, 'trapezoid') else np.trapz(G[:, q, 0], tts)
            share[i, q] = abs(total - smooth) / (abs(total - smooth) + abs(smooth))
    print('1. TT jumps: share |jumps| / (|jumps| + |tangent integral|) of the change over +-0.25 degC, 8 samples')
    for q in range(3):
        print(f'   {hm.QOI_NAMES[q]:6s} median {np.median(share[:, q]):.2f}, range [{share[:, q].min():.2f}, '
              f'{share[:, q].max():.2f}]')


def tt_step():
    rng = np.random.default_rng(SEED + 2)
    X = hm.LB + (hm.UB - hm.LB) * rng.random((200, 12))
    f, G = hm.qoi(X, grad=True)
    s = (hm.UB - hm.LB) / 2
    res = {}
    for h in (0.025, 0.05, 0.1, 0.2, 0.4):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, 0] += h
        Xm[:, 0] -= h
        res[h] = (hm.qoi(Xp) - hm.qoi(Xm)) / (2 * h)
    print('2. TT difference step: |FD_h - FD_0.1| * 4 / ||grad||_2 (normalized coordinates), 200 samples; '
          'and the TT share |dTT| * 4 / ||grad|| for the tangent and for FD_0.1')
    for q in range(3):
        Gq = G[:, q].copy()
        Gq[:, 0] = res[0.1][:, q]
        norm = np.linalg.norm(Gq * s, axis=1)
        parts = []
        for h in (0.025, 0.05, 0.2, 0.4):
            d = np.abs(res[h][:, q] - res[0.1][:, q]) * 4 / norm
            parts.append(f'h={h}: median {np.median(d):.3f}, 90% {np.quantile(d, 0.9):.3f}')
        print(f'   {hm.QOI_NAMES[q]:6s} ' + '; '.join(parts))
        print(f'          TT share: tangent median {np.median(np.abs(G[:, q, 0]) * 4 / norm):.3f}, '
              f'FD_0.1 median {np.median(np.abs(res[0.1][:, q]) * 4 / norm):.3f}; '
              f'corr(tangent, FD_0.1) {np.corrcoef(G[:, q, 0], res[0.1][:, q])[0, 1]:.2f}')


def stored_9b(tut):
    path = os.path.join(tut, '9b - sample_HBV.csv')
    A = np.loadtxt(path, delimiter=',', skiprows=1)
    f = hm.qoi(A[:, :12])[:, 0]
    d = np.abs(f - A[:, 12])
    print(f'3. port vs "9b - sample_HBV.csv" ({len(A)} rows, NSE in [{A[:, 12].min():.2f}, {A[:, 12].max():.4f}]): '
          f'max |difference| {d.max():.1e}, max relative {np.max(d / np.abs(A[:, 12])):.1e}')


def original_series(tut):
    try:
        import pandas as pd
    except ImportError:
        sys.exit('check 4 needs pandas, scipy, matplotlib and IPython (pip install pandas scipy matplotlib ipython); '
                 'rerun with --vars-tool DIR')
    rc = pd.read_csv

    def read_csv(*a, **k):
        if k.pop('delim_whitespace', False):
            k['sep'] = r'\s+'
        return rc(*a, **k)

    pd.read_csv = read_csv
    cwd = os.getcwd()
    sys.path.insert(0, tut)
    os.chdir(tut)
    try:
        import hbv
        rng = np.random.default_rng(SEED + 3)
        X = np.vstack([hm.TUTORIAL, hm.LB + (hm.UB - hm.LB) * rng.random((4, 12))])
        q = hm.simulate(X)
        worst = 0.0
        for i, x in enumerate(X):
            flux, state, forcing = hbv.HBV_SASK('Oldman Basin', dict(zip(hm.NAMES, (float(v) for v in x))))
            qo = flux['Q_cms'].values
            worst = max(worst, float(np.max(np.abs(qo - q[i])) / np.max(np.abs(qo))))
        print(f'4. port vs VARS-TOOL hbv.py, daily Q 1979-2008 ({hm.NDAYS} days) at the tutorial point and 4 samples: '
              f'max |difference| / max Q {worst:.1e} (pandas {pd.__version__})')
    finally:
        os.chdir(cwd)
        pd.read_csv = rc


def flow_floor():
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    q = hm.simulate(X)
    low = q < 1e-5

    def outputs(qq):
        w = qq[:, hm.I0:hm.I1]
        obs = hm.Q_OBS[hm.I0:hm.I1]
        nse = 1 - np.sum((w - obs) ** 2, axis=1) / np.sum((obs - obs.mean()) ** 2)
        years = hm.YEAR[hm.I0:hm.I1]
        peak = np.mean([w[:, years == y].max(axis=1) for y in np.unique(years)], axis=0)
        return np.column_stack([nse, w.mean(axis=1), peak])

    f0, f1 = outputs(q), outputs(np.maximum(q, 1e-5))
    ref = hm.qoi(X)
    print(f'5. flow floor 1e-5 m3/s: {int(low.any(axis=1).sum())} of {len(X)} samples have a daily flow below it '
          f'({int(low[:, hm.I0:hm.I1].any(axis=1).sum())} inside 1982-1996; min daily flow {q.min():.1e} m3/s); '
          f'series vs qoi() max |diff| {np.abs(f0 - ref).max():.1e}; with the floor, max |change| NSE '
          f'{np.abs(f1 - f0)[:, 0].max():.1e}, Qmean {np.abs(f1 - f0)[:, 1].max():.1e} m3/s, '
          f'Qpeak {np.abs(f1 - f0)[:, 2].max():.1e} m3/s')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--vars-tool', metavar='DIR', help='the tutorials folder of a VARS-TOOL clone (checks 3, 4)')
    args = ap.parse_args()
    tt_jumps()
    tt_step()
    flow_floor()
    if not args.vars_tool:
        sys.exit('checks 3 and 4 need VARS-TOOL: git clone https://github.com/vars-tool/vars-tool and rerun with '
                 '--vars-tool vars-tool/tutorials')
    stored_9b(args.vars_tool)
    original_series(args.vars_tool)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, NotADirectoryError) as err:
        sys.exit(f'file not found: {err.filename}\n'
                 '--vars-tool takes the tutorials folder of a clone of github.com/vars-tool/vars-tool.')
