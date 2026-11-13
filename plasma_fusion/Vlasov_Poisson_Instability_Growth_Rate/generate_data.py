"""Regenerate the data files of Vlasov_Poisson_Instability_Growth_Rate from plasma_dispersion_model.py.

    python generate_data.py                      # all studies, 2 processes, writes here
    python generate_data.py --check              # all studies, in a temporary directory, compared with the files here
    python generate_data.py --out /tmp/vp --check --studies global
    python generate_data.py --rows 50 --out /tmp/vp --check     # first 50 rows of each study

Studies (inputs uniform on a box; paper Sec. IV, Eq. (10)):

  global        Bi-Maxwellian, the paper's global box (p. 29), M = 1000          -> global_bimaxwellian.csv
  double_beam   Bi-Maxwellian, Table II Double Beam baseline +- 1, 5, 10, 15, 20, 25 %  -> double_beam.csv
  bump_on_tail  Bi-Maxwellian, Table II Bump-on-Tail baseline +- 1, 5, 10, 15, 25 %    -> bump_on_tail.csv
  bump_on_tail_s025
                Bump-on-Tail as the paper's numbers imply it was run: sigma1^2 = sigma2^2
                = 0.25 and mu1 held at 0 (see DATA.md), +- 1, 5, 10, 15, 25 %  -> bump_on_tail_s025.csv
  two_stream    Two-Stream, Table I baseline +- 1, 5, 10, 15, 20, 25, 50 % (m = 3)  -> two_stream.csv

512 samples per percentage study (the paper's N). For a zero baseline the range is
+- pct/100 in absolute value (paper p. 11), except in bump_on_tail_s025, where mu1 is held.

Each row: [pct,] inputs (physical), gamma, omega_r, dgamma/dinputs (physical, implicit
function theorem), gap, where gap = gamma - k Im(u) of the next root found (how far the most
unstable root is from the next one; small values mark near-crossings of root branches).

Random numbers: numpy default_rng([SEED, study index, pct]); SEED = 2024. Samples are
independent, so --workers only changes the run time. A sample whose root cannot be
certified raises RootError and stops the run (none did in the shipped run).

--check without --out writes to a temporary directory and removes it. It prints the largest
differences from the shipped files and fails (exit status 1) only on structural mismatches (shapes,
NaN pattern, input columns (pct and the inputs) differing by more than INPUT_RTOL).

Only numpy, scipy and the standard library are imported.
"""
import argparse
import atexit
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plasma_dispersion_model as pm  # noqa: E402

SEED = 2024
N_PCT = 512
N_GLOBAL = 1000
INPUT_RTOL = 1e-8  # inputs are deterministic draws: a larger difference means a different seed or range

NOMINAL_BOT_S025 = np.array([0.5, 0.0, 4.0, 0.25, 0.25, 0.8])

# name, file, kind, list of (pct, lower, upper); pct = 0 for the global box
STUDIES = {
    'global': ('global_bimaxwellian.csv', 'bm', [(0, pm.GLOBAL_LOWER, pm.GLOBAL_UPPER)]),
    'double_beam': ('double_beam.csv', 'bm',
                    [(p,) + pm.percent_box(pm.NOMINAL_DB, p) for p in (1, 5, 10, 15, 20, 25)]),
    'bump_on_tail': ('bump_on_tail.csv', 'bm',
                     [(p,) + pm.percent_box(pm.NOMINAL_BOT, p) for p in (1, 5, 10, 15, 25)]),
    'bump_on_tail_s025': ('bump_on_tail_s025.csv', 'bm',
                          [(p,) + tuple(np.where(NOMINAL_BOT_S025 == 0, 0.0, b)
                                        for b in pm.percent_box(NOMINAL_BOT_S025, p))
                           for p in (1, 5, 10, 15, 25)]),
    'two_stream': ('two_stream.csv', 'ts',
                   [(p,) + pm.percent_box(pm.NOMINAL_TS, p) for p in (1, 5, 10, 15, 20, 25, 50)]),
}
STUDY_INDEX = {s: i for i, s in enumerate(STUDIES)}

DESCRIPTION = {
    'global': 'Bi-Maxwellian growth rate, global box of Terrab & Pankavich (p. 29); M = 1000 '
              'uniform samples (seed 2024)',
    'double_beam': 'Bi-Maxwellian Double Beam (Table II baseline), uniform +- pct %; 512 '
                   'samples per pct (seed 2024)',
    'bump_on_tail': 'Bi-Maxwellian Bump-on-Tail (Table II baseline, sigma^2 = 0.5), uniform '
                    '+- pct %; 512 samples per pct (seed 2024)',
    'bump_on_tail_s025': 'Bump-on-Tail with sigma1^2 = sigma2^2 = 0.25 and mu1 held at 0 '
                         '(the setup the paper\'s numbers imply), uniform +- pct %; 512 '
                         'samples per pct (seed 2024)',
    'two_stream': 'Two-Stream growth rate (Table I baseline), uniform +- pct %; 512 samples '
                  'per pct (seed 2024)',
}


def draw(study, pct, lower, upper):
    n = N_GLOBAL if study == 'global' else N_PCT
    rng = np.random.default_rng([SEED, STUDY_INDEX[study], pct])
    return lower + (upper - lower)*rng.random((n, len(lower)))


def _one(args):
    x, kind = args
    g, w, G, info = pm.growth_rate_and_grad(x, kind, return_info=True)
    return g, w, G, x[0]*info['gap'], info['refined']


def columns(study):
    kind = STUDIES[study][1]
    names = list(pm.NAMES[kind])
    cols = ([] if study == 'global' else ['pct (%)']) + [f'{n} (-)' for n in names]
    cols += ['gamma (-)', 'omega_r (-)'] + [f'dgamma/d{n} (-)' for n in names] + ['gap (-)']
    return cols


def run_study(study, pool, rows=None):
    fname, kind, boxes = STUDIES[study]
    out, nref = [], 0
    for pct, lo, hi in boxes:
        X = draw(study, pct, lo, hi)
        if rows is not None:
            X = X[:rows]
        res = pool.map(_one, [(x, kind) for x in X], chunksize=16)
        for x, (g, w, G, gap, ref) in zip(X, res):
            nref += ref
            row = ([] if study == 'global' else [pct]) + list(x) + [g, w] + list(G) + [gap]
            out.append(row)
    return np.array(out), nref


def write(path, study, A):
    hdr = DESCRIPTION[study] + '; written by generate_data.py\n' + ','.join(columns(study))
    np.savetxt(path, A, fmt='%.17g', delimiter=',', header=hdr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--studies', nargs='*', default=list(STUDIES))
    ap.add_argument('--rows', type=int, default=None, help='first rows of each box only')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--check', action='store_true', help='compare with the files next to this script')
    a = ap.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            a.out = HERE
    if os.path.abspath(a.out) == HERE and (a.check or a.rows is not None):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    ok, diffs = True, []
    with mp.Pool(a.workers) as pool:
        for s in a.studies:
            t = time.perf_counter()
            A, nref = run_study(s, pool, a.rows)
            path = os.path.join(a.out, STUDIES[s][0])
            write(path, s, A)
            print(f'{s}: {A.shape[0]} samples, {nref} needed the fine rescan, 0 failed; '
                  f'{time.perf_counter() - t:.1f} s -> {path}')
            if a.check:
                same, d = compare(A, s, a.rows)
                ok = same and ok
                if d is not None:
                    diffs.append((d, STUDIES[s][0]))
    print(f'total {time.perf_counter() - t0:.1f} s')
    if a.check:
        if diffs:
            d, n = max(diffs, key=lambda t: t[0])
            print(f'largest difference from the shipped files: {d:.1e} (relative, {n})')
        if not ok:
            sys.exit(1)


def compare(A, study, rows=None, ship=HERE):
    """Compare the regenerated rows A of study with the file in ship.

    Returns (False if there is a structural mismatch, the largest elementwise relative difference or None).
    """
    name = STUDIES[study][0]
    ref = np.loadtxt(os.path.join(ship, name), delimiter=',', ndmin=2)
    B = ref if rows is None else _first_rows(ref, study, rows)
    if A.shape != B.shape or not np.array_equal(np.isnan(A), np.isnan(B)):
        print(f'  FAIL: {name}: shape {A.shape} or NaN pattern differs from the shipped file {B.shape}')
        return False, None
    d = np.abs(A - B)/(1e-12 + np.abs(B))
    nin = (study != 'global') + len(pm.NAMES[STUDIES[study][1]])      # pct and the inputs
    din, dout = float(np.nanmax(d[:, :nin])), float(np.nanmax(d[:, nin:]))
    print(f'  largest elementwise relative difference from the shipped file: inputs {din:.1e}, outputs {dout:.1e}')
    if din > INPUT_RTOL:
        print(f'  FAIL: {name}: the input columns differ from the shipped ones by more than {INPUT_RTOL:g} (relative)')
    return din <= INPUT_RTOL, max(din, dout)


def _first_rows(ref, study, rows):
    if study == 'global':
        return ref[:rows]
    keep = []
    for p in np.unique(ref[:, 0]):
        idx = np.flatnonzero(ref[:, 0] == p)[:rows]
        keep.extend(idx)
    return ref[np.sort(keep)]


if __name__ == '__main__':
    main()
