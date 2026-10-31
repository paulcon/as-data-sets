"""Build this data set's CSV files from the SU2 output posted by Grey and Constantine.

Z. J. Grey and P. G. Constantine, "Active Subspaces of Airfoil Shape Parameterizations",
AIAA Journal 56(5):2003-2017, 2018, doi:10.2514/1.J056054, ran SU2 4.2.0 (2-D Euler,
Mach 0.8, angle of attack 0) on NACA 0012-like airfoils drawn from two shape
parameterizations: PARSEC (m = 11) and CST (m = 10). The authors posted the SU2 output in
https://github.com/zgrey/ASAP as three zip files, 500_Samples.zip, 1000_Samples.zip and
5000_Samples.zip. Each holds CST_forces.zip and PARSEC_forces.zip, and those hold one SU2
forces_breakdown file per run, forces/airfoil<i>.su2.dat.

The design inputs are NOT in the zips. The authors' script mixsamples.m loads them from
CST_designs*.mat and PARSEC_designs*.mat files that were never posted. This script
regenerates them from the seeds recorded in the authors' sampling script
(AUTO/parsec_samples.m: "rng(43); % Used for 500 Sample Designs", rng(44) for 1000,
rng(45) for 5000):

    MATLAB  rng(seed); X = 2*rand(N, m) - 1;

MATLAB's default generator is the Mersenne Twister seeded with init_genrand(seed), and
rand returns 53-bit doubles; numpy.random.RandomState(seed).random_sample does the same,
so the column-major MATLAB matrix is RandomState(seed).random_sample(N*m).reshape(m, N).T.
The check that this is right is in the data: with these seeds a linear fit of CST lift in
the regenerated inputs has R^2 = 0.9987 in each of the three sets, against at most 0.11
for any other seed from 0 to 1999 (see DATA.md). The generator itself is checked against
the authors' own files: the expression above reproduces, bit for bit, the design matrices
X of the three .mat files of the authors' later study that the repository does hold
(AUTO/designs/FINE/PARSEC_designs_m9_N1000_pm20pct.mat with seed 44,
AUTO/designs/FINE/designs_m10_N1000_pm20pct.mat and AUTO/designs/COARSE/designs_m10_N500_pm20pct.mat
with seed 47); `--verify --mat DIR` repeats this. (MATLAB maps rng(0) to init_genrand(5489);
the seeds used here are nonzero, so the plain seed applies.)

CST: every design was run, so run i is row i of X (N = 500, 1000, 5000).
PARSEC: the authors' script drops designs whose surfaces cross ("feasible" check) before
meshing and numbers the remaining runs 1..N-NF. The dropped rows are not recorded. They
were found by aligning the SU2 values to the regenerated rows (dynamic programming over
the number of rows dropped so far, under a least-squares quadratic; `--verify` repeats
it) and are listed in DROPPED below. All 28 lie in the corner x5 > 0.85, x6 < -0.85
(trailing-edge offset high, trailing-edge thickness low) in the normalized inputs, which
holds 52 of the 6500 drawn rows. The outputs alone fix 18 of the 28 rows; for the other 10
a row one or two places away fits almost as well, and the corner pattern decides (`--verify`
prints these near-ties). Moving all 10 changes the design row of 10 runs and the
eigenvectors by less than 5e-5 in |w.w| (DATA.md).

The PARSEC columns are reordered to the paper's Table 1 numbering exactly as mixsamples.m
does ("Change Indices to Match Paper").

Outputs (two '#' lines, then numbers; np.loadtxt(path, delimiter=',') reads them):

    cst_inputs.csv      6500 x 10  normalized CST design variables x1..x10, in [-1, 1]
    cst_outputs.csv     6500 x 5   sample_set, run, design_row, CL, CD
    parsec_inputs.csv   6472 x 11  normalized PARSEC design variables x1..x11 (Table 1 order)
    parsec_outputs.csv  6472 x 5   sample_set, run, design_row, CL, CD

Rows are in the order of mixsamples.m: the 500-sample set, then 1000, then 5000, each in
run order. Inputs are written with Python's repr (the shortest decimal that reads back as
the same double). CL and CD are the "Total CL:" and "Total CD:" values of each
forces_breakdown file, which SU2 prints to 6 significant digits; they are written as those
same decimals, so reading them back gives float(text) exactly.

Usage:
    python convert_data.py --zips DIR            # DIR holds the three ASAP zip files; writes the CSVs
    python convert_data.py --zips DIR --check    # rebuilds in memory and compares with the shipped CSVs
    python convert_data.py --zips DIR --verify   # also repeats the PARSEC alignment search
"""

import argparse
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_REPO = 'https://github.com/zgrey/ASAP'
SOURCE_COMMIT = '79ae7ed62f282c7ee0bebf508af55530f69579e7'      # 2019-03-13, master
SOURCE_SHA256 = {
    '500_Samples.zip': 'ea90e7eed0c1761a4ecf070beb68911f3eae02f2f628bba397080cb1f3c99aa0',
    '1000_Samples.zip': '55271fbec2c171f8f6895e81e0577be0ee7af525d7c868f002a179bd5444469d',
    '5000_Samples.zip': '639f37a3394b5be8cfaa2cc9f927767eda1a82af212c3d5e94ce94807b926239',
}
SETS = ((500, 43), (1000, 44), (5000, 45))      # (designs drawn, MATLAB rng seed)

# 0-based rows of each regenerated PARSEC design matrix that have no SU2 run
DROPPED = {500: (106, 258, 312), 1000: (9, 257),
           5000: (376, 528, 574, 778, 822, 865, 1086, 1107, 1351, 1972, 2035, 2576, 2650,
                  2906, 3489, 3597, 3723, 4356, 4413, 4666, 4711, 4741, 4833)}

# PARSEC: column j of the paper's numbering is column PARSEC_CODE_COLUMN[j] (1-based) of the
# design matrix of parsec_samples.m / parsec.m (rle, Xup, Yup, YXXup, Xlo, Ylo, YXXlo,
# yte, dyte, alpha_te, beta_te); from mixsamples.m.
PARSEC_CODE_COLUMN = (2, 5, 3, 6, 8, 9, 10, 11, 4, 7, 1)

PARSEC_NAMES = ('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11')
PARSEC_DESCRIPTIONS = (                                   # paper Table 1, p. 2005
    'upper-surface domain coordinate', 'lower-surface domain coordinate',
    'upper-surface shape coordinate', 'lower-surface shape coordinate',
    'trailing-edge shape coordinate', 'trailing-edge half-thickness', 'wedge angle',
    'wedge half-angle', 'upper-surface inflection', 'lower-surface inflection',
    'leading-edge radius')
# paper Table 2, p. 2008, as printed ("Lower", "Upper"); x7 and x9 are printed with
# Lower > Upper. The map from the normalized values to these is not recoverable (DATA.md).
PARSEC_TABLE_LOWER = (0.242, 0.242, 0.048, -0.072, -0.004, 0.008, -2.223, 7.40, -0.400, 0.400, 0.012)
PARSEC_TABLE_UPPER = (0.363, 0.363, 0.072, -0.048, 0.004, 0.012, -3.335, 11.10, -0.600, 0.600, 0.018)

CST_NAMES = tuple(f'x{j}' for j in range(1, 11))
# paper Table 3, p. 2010
CST_TABLE_LOWER = (0.12, 0.8, 0.8, 0.8, 0.8, -0.18, -1.2, -1.2, -1.2, -1.2)
CST_TABLE_UPPER = (0.18, 1.2, 1.2, 1.2, 1.2, -0.12, -0.8, -0.8, -0.8, -0.8)

LOWER = {'cst': -np.ones(10), 'parsec': -np.ones(11)}      # the stored inputs are normalized
UPPER = {'cst': np.ones(10), 'parsec': np.ones(11)}

# APPROXIMATE physical values (DATA.md, "Physical values"). The authors' sampling scripts map
# X in [-1, 1] to physical = nominal * (1 + PCT * X), i.e. X = -1 -> 0.8 * nominal
# (lb0 = (1 - pct) * dv; confirmed bit for bit in their m = 9 PARSEC .mat). Nominals are the
# midpoints of Tables 2 and 3, so they carry the tables' 2-3 digit rounding. PARSEC x5 has
# nominal 0 and is mapped to its printed range, 0.004 * X. The orientation of PARSEC x4 and
# CST x6 (X = +1 -> larger magnitude) follows from the data (DATA.md); the 2016 code is not
# posted, and the meaning of CST x2-x5, x7-x10 (no effect in the data) is unknown.
PCT = 0.2
PARSEC_NOMINAL = (0.3025, 0.3025, 0.06, -0.06, 0.0, 0.01, -2.779, 9.25, -0.5, 0.5, 0.015)
PARSEC_UNITS = ('chord', 'chord', 'chord', 'chord', 'chord', 'chord', 'deg', 'deg', '1/chord', '1/chord', 'chord')
CST_NOMINAL = (0.15, 1.0, 1.0, 1.0, 1.0, -0.15, -1.0, -1.0, -1.0, -1.0)


def physical_approx(kind, X):
    """APPROXIMATE physical values of normalized inputs X (n x m); see the note above and DATA.md."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nom = np.array(PARSEC_NOMINAL if kind == 'parsec' else CST_NOMINAL)
    P = nom * (1 + PCT * X)
    if kind == 'parsec':
        P[:, 4] = 0.004 * X[:, 4]                 # x5: nominal 0, printed range [-0.004, 0.004]
    return P

CREDIT = ('SU2 4.2.0 Euler runs (Mach 0.8, AoA 0) of Grey and Constantine, AIAA J. 56(5) 2018, '
          'doi:10.2514/1.J056054, from github.com/zgrey/ASAP @' + SOURCE_COMMIT[:7])

_NUM = r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)'
_CHECKS = {   # every forces file must state these flow conditions
    'Mach number (non-dim): 0.8': r'Mach number \(non-dim\): 0\.8\s',
    'free-stream velocity (265.057, 0) m/s, so AoA = 0': r'Free-stream velocity: \(265\.057, 0\) m/s',
    'free-stream pressure 101325 Pa': r'Free-stream static pressure: 101325 Pa',
    'free-stream temperature 273.15 K': r'Free-stream temperature: 273\.15 K',
    'SU2 release 4.2.0': r'Release 4\.2\.0',
    'inviscid (Euler)': r'Inviscid flow',
}


def matlab_rand(seed, N, m):
    """MATLAB `rng(seed); rand(N, m)` (Mersenne Twister, column-major fill)."""
    return np.random.RandomState(seed).random_sample(N * m).reshape(m, N).T


def designs(kind, N, seed):
    """Normalized design matrix 2*rand(N, m) - 1 of the authors' sampling scripts."""
    m = 10 if kind == 'cst' else 11
    return 2 * matlab_rand(seed, N, m) - 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_forces(outer_zip, kind):
    """{run: (CL text, CD text)} from <kind>_forces.zip inside one ASAP zip."""
    inner_name = {'cst': 'CST_forces.zip', 'parsec': 'PARSEC_forces.zip'}[kind]
    with zipfile.ZipFile(outer_zip) as zo:
        inner = io.BytesIO(zo.read(inner_name))
    out = {}
    with zipfile.ZipFile(inner) as zi:
        for name in zi.namelist():
            if name.endswith('/'):
                continue
            m = re.fullmatch(r'forces/airfoil(\d+)\.su2\.dat', name)
            if not m:
                raise ValueError(f'{outer_zip.name}/{inner_name}: unexpected member {name}')
            text = zi.read(name).decode('latin-1')
            for what, pat in _CHECKS.items():
                if not re.search(pat, text):
                    raise ValueError(f'{name}: expected {what}')
            cl = re.search(r'^Total CL:\s*' + _NUM + r'\s*\|', text, re.M)
            cd = re.search(r'^Total CD:\s*' + _NUM + r'\s*\|', text, re.M)
            fr = re.search(r'^Total CD:.*Friction Component \(\s*0%\):\s*0\s*$', text, re.M)
            if not (cl and cd and fr):
                raise ValueError(f'{name}: no Total CL / Total CD line, or nonzero friction')
            run = int(m.group(1))
            if run in out:
                raise ValueError(f'{name}: run {run} twice')
            out[run] = (cl.group(1), cd.group(1))
    return out


def build(zipdir):
    """Return {kind: (X, ids, CL_text, CD_text)} with rows in mixsamples.m order."""
    zipdir = Path(zipdir)
    data = {}
    for kind in ('cst', 'parsec'):
        Xs, ids, cls, cds = [], [], [], []
        for N, seed in SETS:
            path = zipdir / f'{N}_Samples.zip'
            got = sha256(path)
            if got != SOURCE_SHA256[path.name]:
                raise SystemExit(f'{path}: SHA-256 {got} differs from {SOURCE_SHA256[path.name]}')
            forces = read_forces(path, kind)
            X = designs(kind, N, seed)
            rows = np.arange(N)
            if kind == 'parsec':
                rows = np.setdiff1d(rows, DROPPED[N])
            k = len(rows)
            if sorted(forces) != list(range(1, k + 1)):
                raise ValueError(f'{path.name} {kind}: runs are not 1..{k}')
            Xk = X[rows]
            if kind == 'parsec':
                Xk = Xk[:, np.array(PARSEC_CODE_COLUMN) - 1]
            Xs.append(Xk)
            ids.append(np.column_stack([np.full(k, N), np.arange(1, k + 1), rows + 1]))
            cls += [forces[r][0] for r in range(1, k + 1)]
            cds += [forces[r][1] for r in range(1, k + 1)]
        data[kind] = (np.vstack(Xs), np.vstack(ids), cls, cds)
    return data


def texts(data):
    """{filename: file text}."""
    out = {}
    for kind, names in (('cst', CST_NAMES), ('parsec', PARSEC_NAMES)):
        X, ids, cls, cds = data[kind]
        label = kind.upper()
        lines = [f'# normalized {label} design variables in [-1,1] (the authors\' 2*rand-1, regenerated '
                 f'from their MATLAB seeds; paper numbering); {CREDIT}',
                 '# ' + ','.join(f'{n} (-)' for n in names)]
        lines += [','.join(repr(float(v)) for v in row) for row in X]
        out[f'{kind}_inputs.csv'] = '\n'.join(lines) + '\n'
        lines = [f'# {label} runs: sample set (500/1000/5000 zip), run number i of forces/airfoil<i>.su2.dat, '
                 f'1-based row of the regenerated design matrix, lift and drag coefficients; {CREDIT}',
                 '# sample_set (-),run (-),design_row (-),CL (-),CD (-)']
        lines += [f'{a},{b},{c},{cl},{cd}' for (a, b, c), cl, cd in zip(ids.astype(int), cls, cds)]
        out[f'{kind}_outputs.csv'] = '\n'.join(lines) + '\n'
    return out


def check_texts(data, tx):
    """Each CSV text reads back with np.loadtxt equal, bit for bit, to the values written."""
    for kind in ('cst', 'parsec'):
        X, ids, cls, cds = data[kind]
        Xr = np.loadtxt(io.StringIO(tx[f'{kind}_inputs.csv']), delimiter=',')
        Or = np.loadtxt(io.StringIO(tx[f'{kind}_outputs.csv']), delimiter=',')
        ref = np.column_stack([ids.astype(float), [float(s) for s in cls], [float(s) for s in cds]])
        assert Xr.shape == X.shape and np.array_equal(Xr.view(np.uint64), X.view(np.uint64)), kind
        assert Or.shape == ref.shape and np.array_equal(Or.view(np.uint64), ref.view(np.uint64)), kind


def _quad_basis(X):
    N, m = X.shape
    i, j = np.triu_indices(m)
    return np.column_stack([np.ones(N), X, X[:, i] * X[:, j]])


def _align(X, y, coef, n_drop):
    """Rows of X matching the k = len(y) runs, with n_drop rows skipped, by dynamic programming."""
    k, S = len(y), n_drop + 1
    sd = y.std(axis=0)
    err = np.stack([(((y - _quad_basis(X[s:s + k]) @ coef) / sd) ** 2).sum(axis=1) for s in range(S)], axis=1)
    D, B = err[0].copy(), np.zeros((k, S), dtype=int)
    for i in range(1, k):
        best = np.zeros(S, dtype=int)
        for s in range(1, S):
            best[s] = s if D[s] < D[best[s - 1]] else best[s - 1]
        B[i] = best
        D = D[best] + err[i]
    path = [n_drop]
    for i in range(k - 1, 0, -1):
        path.append(B[i, path[-1]])
    rows = np.arange(k) + np.array(path[::-1])
    return tuple(int(r) for r in np.setdiff1d(np.arange(k + n_drop), rows))


def verify_parsec_alignment(zipdir):
    """Repeat the search for the dropped PARSEC rows; return {N: (dropped, R^2 of CL, CD)}."""
    zipdir = Path(zipdir)
    res, coef = {}, None
    for N, seed in ((5000, 45), (1000, 44), (500, 43)):
        f = read_forces(zipdir / f'{N}_Samples.zip', 'parsec')
        y = np.array([[float(f[r][0]), float(f[r][1])] for r in range(1, len(f) + 1)])
        X = designs('parsec', N, seed)
        n_drop = N - len(y)
        if coef is None:          # start: linear fit on the first 100 runs of the 5000 set
            A = np.column_stack([np.ones(100), X[:100]])
            lin = np.linalg.lstsq(A, y[:100], rcond=None)[0]
            coef = np.zeros((_quad_basis(X[:1]).shape[1], 2))
            coef[:12] = lin
        c = coef
        for _ in range(3):
            dropped = _align(X, y, c, n_drop)
            Xk = X[np.setdiff1d(np.arange(N), dropped)]
            c = np.linalg.lstsq(_quad_basis(Xk), y, rcond=None)[0]
        r = y - _quad_basis(Xk) @ c
        r2 = 1 - (r ** 2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        if N == 5000:
            coef = c
        res[N] = (dropped, r2)
        corner = (X[list(dropped), 7] > 0.85) & (X[list(dropped), 8] < -0.85)   # code columns yte, dyte
        print(f'PARSEC {N}: dropped rows {list(dropped)}; quadratic R^2 CL {r2[0]:.4f}, CD {r2[1]:.4f}; '
              f'all in the corner x5 > 0.85, x6 < -0.85: {bool(corner.all())}; '
              f'same as DROPPED: {dropped == DROPPED[N]}')
        # near-ties: moving one drop by up to 8 rows raises the sum of squared standardized
        # residuals (own-set quadratic) by less than 2
        sd, keep = y.std(axis=0), np.setdiff1d(np.arange(N), dropped)
        cost = lambda rows: (((y - _quad_basis(X[rows]) @ c) / sd) ** 2).sum()
        base, in_corner = cost(keep), (X[:, 7] > 0.85) & (X[:, 8] < -0.85)
        for d in dropped:
            near = []
            for t in range(-8, 9):
                dd = d + t
                if t == 0 or not 0 <= dd < N or dd in dropped:
                    continue
                dc = cost(np.setdiff1d(np.arange(N), sorted(set(dropped) - {d} | {dd}))) - base
                if dc < 2:
                    near.append(f'{dd} (+{dc:.2f}, corner {bool(in_corner[dd])})')
            if near:
                print(f'    near-tie for row {d} (corner {bool(in_corner[d])}): ' + ', '.join(near))
    return res


def verify_mat(matdir):
    """Check matlab_rand against the authors' later-study design files (needs scipy).

    The files are tracked in zgrey/ASAP under AUTO/designs/ (outside a sparse checkout of the
    zips: `git -C ASAP checkout HEAD -- AUTO/designs`). Returns True if every X matches bit for bit.
    """
    import scipy.io
    ok = True
    for name, seed in (('PARSEC_designs_m9_N1000_pm20pct.mat', 44), ('designs_m10_N1000_pm20pct.mat', 47),
                       ('designs_m10_N500_pm20pct.mat', 47)):
        hits = sorted(Path(matdir).rglob(name))
        if not hits:
            print(f'{name}: not found under {matdir}; skipped')
            continue
        d = scipy.io.loadmat(hits[0])
        X, IP = d['X'], d['IP'].ravel().astype(bool)
        Xr = (2 * matlab_rand(seed, IP.size, X.shape[1]) - 1)[IP]
        lb0, ub0 = d['lb0'].ravel(), d['ub0'].ravel()
        same = Xr.shape == X.shape and np.array_equal(Xr, X)
        rule = np.array_equal(lb0 + (ub0 - lb0) * 0.5 * (X + 1), d['X0'])
        ok &= same
        print(f'{name}: X == 2*rand-1 regenerated with seed {seed}, bit for bit: {same}; '
              f'X0 == lb0 + (ub0 - lb0)(X + 1)/2 exactly: {rule}')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--zips', required=True, help='directory with 500/1000/5000_Samples.zip from zgrey/ASAP')
    ap.add_argument('--check', action='store_true', help='compare with the shipped CSVs instead of writing')
    ap.add_argument('--verify', action='store_true', help='also repeat the PARSEC alignment search')
    ap.add_argument('--mat', help='with --verify: directory holding the authors\' later-study design .mat files '
                                  '(zgrey/ASAP AUTO/designs); checks the MATLAB generator against them')
    ap.add_argument('--out', default=str(HERE), help='output directory (default: this directory)')
    a = ap.parse_args(argv)
    data = build(a.zips)
    tx = texts(data)
    check_texts(data, tx)
    for kind in ('cst', 'parsec'):
        X, ids, cls, cds = data[kind]
        print(f'{kind}: {X.shape[0]} runs x {X.shape[1]} inputs; per set ' +
              ', '.join(f'{N}: {(ids[:, 0] == N).sum()}' for N, _ in SETS))
    if a.verify:
        res = verify_parsec_alignment(a.zips)
        if any(res[N][0] != DROPPED[N] for N in DROPPED):
            raise SystemExit('alignment search disagrees with DROPPED')
        if a.mat and not verify_mat(a.mat):
            raise SystemExit('the regenerated design matrices differ from the authors\' .mat files')
    out = Path(a.out)
    if a.check:
        bad = [n for n, t in tx.items() if (out / n).read_text() != t]
        print('identical to the shipped CSVs' if not bad else f'DIFFERENT: {bad}')
        return 1 if bad else 0
    for n, t in tx.items():
        (out / n).write_text(t)
        print(f'wrote {out / n} ({len(t.encode()) / 1e6:.2f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
