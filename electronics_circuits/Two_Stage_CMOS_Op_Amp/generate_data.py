"""Write the two sample sets of Two_Stage_CMOS_Op_Amp.

Runs ngspice (through opamp_spice.py) at designs drawn independently and log-uniformly on
the AutoCkt design box (Monte Carlo), and at the 14 points of a central-difference stencil
around each design, with steps x_j * exp(+-H) (H in ln x_j).

- set "box" (inputs.csv, outputs.csv, gradients.csv): M = 1000 draws, seed SEED.
- set "gain200" (gain200_inputs.csv, gain200_outputs.csv, gain200_gradients.csv): the
  first M draws, from seed SEED_GAIN200, whose gain is at least GAIN_MIN = 200 V/V (the
  paper's lower gain target), i.e. the log-uniform distribution conditioned on gain >= 200;
  rejection sampling, and the number of draws it took is printed and written in the header.

    python generate_data.py                       # both sets, 2 worker processes
    python generate_data.py --set box --rows 0:10 --out /tmp/oa --check   # a subset, compared
    python generate_data.py --fd-check 0:5        # gradient step study (H vs 10H and H/10)

Needs numpy, scipy and the ngspice program on PATH (or NGSPICE=/path/to/ngspice).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opamp_spice as op  # noqa: E402

SEED = 2020          # set "box"
SEED_GAIN200 = 2021  # set "gain200"
GAIN_MIN = 200.0     # V/V, Sec. III.B lower gain target
M = 1000             # samples per set
H = 1e-4             # central-difference step in ln x_j
m = len(op.NAMES)
nq = len(op.QOI_NAMES)


def draw_inputs(seed=SEED, n=M):
    """Independent log-uniform samples on [LOWER, UPPER] (physical units)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log(op.LOWER), np.log(op.UPPER)
    return np.exp(lo + (hi - lo) * rng.random((n, m)))


def sample(x, h=H):
    """QoIs at x and their central-difference gradients w.r.t. the physical inputs.

    Returns (f, G, note): f (4,), G (4, 7); ugbw and pm (and their gradients) are NaN where
    the unity-gain measurement is undefined at x or at any stencil point; note says why.
    """
    f = op.qoi_vector(x, strict=False)
    G = np.full((nq, m), np.nan)
    plus, minus = np.empty((m, nq)), np.empty((m, nq))
    for j in range(m):
        xp, xm = x.copy(), x.copy()
        xp[j] *= np.exp(h)
        xm[j] *= np.exp(-h)
        plus[j] = op.qoi_vector(xp, strict=False)
        minus[j] = op.qoi_vector(xm, strict=False)
    dlog = (plus - minus) / (2 * h)              # (m, nq): d q / d ln x_j
    G = (dlog / x[:, None]).T                    # (nq, m): d q / d x_j
    note = ''
    if np.isnan(f[2]):
        note = 'ugbw/pm undefined at the sample'
    elif np.isnan(G[2]).any():
        note = 'ugbw/pm undefined at a stencil point'
        G[1:3] = np.nan
    return f, G, note


def _gain(x):
    return op.simulate(x, strict=False)['gain']


def draw_gain200(pool, n=M, batch=2000):
    """First n log-uniform draws (seed SEED_GAIN200) with gain >= GAIN_MIN; also the draw count."""
    rng = np.random.default_rng(SEED_GAIN200)
    lo, hi = np.log(op.LOWER), np.log(op.UPPER)
    kept, draws = [], 0
    while len(kept) < n:
        Xb = np.exp(lo + (hi - lo) * rng.random((batch, m)))
        gains = pool.map(_gain, list(Xb), chunksize=50)
        for x, g in zip(Xb, gains):
            draws += 1
            if g >= GAIN_MIN:
                kept.append(x)
                if len(kept) == n:
                    break
    return np.array(kept), draws


def _work(args):
    i, x = args
    f, G, note = sample(x)
    return i, f, G, note


def read_csv(path):
    return np.loadtxt(path, delimiter=',', ndmin=2)


def write_csv(path, A, description, columns):
    header = description + '\n' + ','.join(columns)
    np.savetxt(path, A, delimiter=',', fmt='%.16e', header=header, comments='# ')


INPUT_COLS = [f'{n} [{u}]' for n, u in zip(op.NAMES, op.UNITS)]
OUTPUT_COLS = [f'{n} [{u}]' for n, u in zip(op.QOI_NAMES, op.QOI_UNITS)]
GRAD_COLS = [f'd{q}/d{n} [{qu}/{u}]' for q, qu in zip(op.QOI_NAMES, op.QOI_UNITS)
             for n, u in zip(op.NAMES, op.UNITS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=('box', 'gain200', 'all'), default='all')
    ap.add_argument('--rows', default=None, help='a:b subset of the samples')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--check', action='store_true', help='compare with the shipped files')
    ap.add_argument('--fd-check', default=None, help='a:b rows: compare steps H, 10H, H/10')
    a = ap.parse_args()
    if a.check and os.path.abspath(a.out) == HERE:
        ap.error('--check needs --out elsewhere, so that the shipped files are not overwritten')
    if a.fd_check:   # on the box draws, or with --set gain200 on the shipped gain200 inputs
        X = read_csv(os.path.join(HERE, 'gain200_inputs.csv')) if a.set == 'gain200' else draw_inputs()
        r0, r1 = (int(s) for s in a.fd_check.split(':'))
        print(f'{op.ngspice_version()}; central differences in ln x, rel. error of each gradient '
              f'vector (norm over the 7 inputs) against step H = {H:g}')
        for i in range(r0, r1):
            _, G0, _ = sample(X[i], H)
            for h in (10 * H, H / 10):
                _, G1, _ = sample(X[i], h)
                err = np.linalg.norm((G1 - G0) * X[i], axis=1) / np.linalg.norm(G0 * X[i], axis=1)
                print(f'row {i:4d} step {h:7.0e}: ' +
                      ', '.join(f'{q} {e:.1e}' for q, e in zip(op.QOI_NAMES, err)))
        return
    t0 = time.perf_counter()
    for name in (('box', 'gain200') if a.set == 'all' else (a.set,)):
        run_set(name, a)
    print(f'total {time.perf_counter() - t0:.0f} s')


def run_set(name, a):
    pool = Pool(a.workers)
    if name == 'box':
        X = draw_inputs()
        prefix, desc_set = '', f'{M} log-uniform Monte Carlo samples on the design box, seed {SEED}'
    else:
        prefix = 'gain200_'
        t0 = time.perf_counter()
        X, draws = draw_gain200(pool)
        desc_set = (f'the first {M} of {draws} log-uniform Monte Carlo draws (seed {SEED_GAIN200}) with '
                    f'gain >= {GAIN_MIN:g} V/V')
        print(f'gain200: {draws} draws for {M} with gain >= {GAIN_MIN:g} V/V '
              f'(acceptance {M / draws:.4f}); {time.perf_counter() - t0:.0f} s')
    rows = list(range(M) if a.rows is None else range(*(int(s) for s in a.rows.split(':'))))
    t0 = time.perf_counter()
    F, G = np.empty((len(rows), nq)), np.empty((len(rows), nq * m))
    notes = {}
    for k, (i, f, g, note) in enumerate(pool.imap(_work, [(i, X[i]) for i in rows], chunksize=4)):
        F[k], G[k] = f, g.ravel()
        if note:
            notes[i] = note
    pool.close()
    dt = time.perf_counter() - t0
    print(f'set {name}: {len(rows)} samples x {2 * m + 1} ngspice runs in {dt:.0f} s on {a.workers} workers '
          f'({op.ngspice_version()})')
    print(f'ugbw/pm undefined at the sample: {sum(v.endswith("sample") for v in notes.values())}; '
          f'defined there but not at a stencil point (gradient NaN): '
          f'{sum(v.endswith("point") for v in notes.values())}')

    os.makedirs(a.out, exist_ok=True)
    desc = f'Two-stage CMOS op-amp (Settaluri et al., DATE 2020), ngspice; {desc_set}'
    if a.rows is not None:
        desc += f'; rows {a.rows}'
    write_csv(os.path.join(a.out, prefix + 'inputs.csv'), X[rows], desc + '; inputs', INPUT_COLS)
    write_csv(os.path.join(a.out, prefix + 'outputs.csv'), F,
              desc + '; outputs (nan: ugbw and pm undefined, gain <= 1 V/V)', OUTPUT_COLS)
    write_csv(os.path.join(a.out, prefix + 'gradients.csv'), G,
              desc + f'; central differences, step {H:g} in ln x; column 7k+j = dq_k/dx_j', GRAD_COLS)

    if a.check:
        F0 = read_csv(os.path.join(HERE, prefix + 'outputs.csv'))[rows]
        G0 = read_csv(os.path.join(HERE, prefix + 'gradients.csv'))[rows]
        same_nan = np.array_equal(np.isnan(F0), np.isnan(F)) and np.array_equal(np.isnan(G0), np.isnan(G))
        rf = np.nanmax(np.abs(F - F0) / np.abs(F0))
        rg = np.nanmax(np.abs(G - G0) / np.maximum(np.abs(G0), 1e-300))
        print(f'check against the shipped files: same NaN pattern {same_nan}; '
              f'max relative difference outputs {rf:.1e}, gradients {rg:.1e}')


if __name__ == '__main__':
    main()
