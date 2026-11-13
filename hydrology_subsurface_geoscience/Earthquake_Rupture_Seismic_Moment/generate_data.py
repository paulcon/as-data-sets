"""Write design_inputs.csv (numpy, scipy) and, with --rerun, fdfault_reruns.csv (needs fdfault).

design_inputs.csv needs only numpy and scipy. fdfault_reruns.csv needs the outside code fdfault
v1.0.1 (its Python package and compiled binary, run through mpiexec); without it --rerun exits
with status 3 and says what to install. The shipped reruns stay the record: fdfault is rerun only
if a problem is found in them.

design_inputs.csv: the paper's two Latin hypercube designs at full precision. fabmogp draws
them with mogp_emulator 0.4.0 after np.random.seed(seed), seed 4742322 for the 20 training
runs and 63684 for the 10 validation runs (fabmogp_paper/manuscript/Makefile). They are
redrawn here (fdfault_driver.lhs_mogp) and matched, row by row, to paper_tables.csv: every
input must round to the printed value, or the script stops.

fdfault_reruns.csv: selected rows of paper_tables.csv rerun with the compiled fdfault
simulator, giving the integrated final slip to full precision and the run time.

    python generate_data.py                      # design_inputs.csv (under a second)
    python generate_data.py --check              # redraw into a temporary directory, compare with the shipped
                                                 # design_inputs.csv

--check prints the largest differences from the shipped files and fails only on structural mismatches
(shape, the table and row columns, inputs differing by more than INPUT_RTOL).
    python generate_data.py --rerun 1,11,13,21,29 --fdfault-dir /path/to/fdfault
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdfault_driver as fp  # noqa: E402

INSTALL = ('fdfault v1.0.1 is needed for --rerun and was not found ({}). Install it as DATA.md '
           '("Regenerating") describes: Open MPI, then git clone https://github.com/edaub/fdfault, '
           'git checkout v1.0.1, make in fdfault/src, pip install ./fdfault/python; then rerun this '
           'script with --fdfault-dir (or FDFAULT_DIR) set to the fdfault directory that holds the '
           'compiled fdfault binary.')

SEEDS = {2: (4742322, 20), 3: (63684, 10)}     # table -> (seed, size)
INPUT_RTOL = 1e-8   # inputs are deterministic draws: a larger difference means a different seed or range
COLS = 'normal stress sigma_yy (MPa), shear/normal stress ratio (-), normal stress ratio sigma_xx/sigma_yy (-)'


def read_csv(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', ndmin=2)


def design_inputs():
    """Full-precision inputs in the row order of paper_tables.csv, and the LHS row of each."""
    tab = read_csv('paper_tables.csv')
    X = np.empty((len(tab), 3))
    lhs_row = np.empty(len(tab), dtype=int)
    for t, (seed, n) in SEEDS.items():
        P = fp.lhs_mogp(n, seed)
        rows = np.flatnonzero(tab[:, 0] == t)
        assert len(rows) == n
        printed = tab[rows, 2:5]
        for i, r in enumerate(rows):
            match = np.flatnonzero(np.all(np.abs(np.round(P * [100, 1000, 1000]) / [100, 1000, 1000]
                                                 - printed[i]) < 1e-9, axis=1))
            if len(match) != 1:
                raise RuntimeError(f'table {t} row {i + 1}: {len(match)} design points round to {printed[i]}')
            X[r] = P[match[0]]
            lhs_row[r] = match[0]
    return tab, X, lhs_row


def write_design(path):
    tab, X, lhs_row = design_inputs()
    with open(path, 'w') as f:
        f.write('# Daub et al. (2021) Latin hypercube designs redrawn at full precision (mogp_emulator 0.4.0 algorithm; '
                'seed 4742322 = Table 2, 63684 = Table 3); rows in the order of paper_tables.csv\n')
        f.write(f'# table, row in table, row in the LHS draw (0-based), {COLS}\n')
        for k in range(len(tab)):
            f.write(f'{int(tab[k, 0])},{int(tab[k, 1])},{lhs_row[k]},' + ','.join(repr(float(v)) for v in X[k]) + '\n')
    return X


def compare(path, ship=HERE):
    """Compare a redrawn design_inputs.csv with the one in ship; False on a structural mismatch."""
    new = np.loadtxt(path, delimiter=',', ndmin=2)
    old = np.loadtxt(os.path.join(ship, 'design_inputs.csv'), delimiter=',', ndmin=2)
    if new.shape != old.shape:
        print(f'shape {new.shape} differs from the shipped {old.shape}')
        return False
    same_int = np.array_equal(new[:, :3], old[:, :3])
    d = float(np.max(np.abs(new[:, 3:] - old[:, 3:]) / np.abs(old[:, 3:]).max(axis=0)))
    print(f'table and row columns identical: {same_int}; inputs: largest difference / column maximum {d:.1e}')
    if not same_int:
        print('FAIL: the table and row columns differ from the shipped file')
    if d > INPUT_RTOL:
        print(f'FAIL: the inputs differ from the shipped ones by more than {INPUT_RTOL:g} (relative)')
    print(f'largest difference from the shipped files: {d:.1e} (relative, design_inputs.csv)')
    return bool(same_int and d <= INPUT_RTOL)


def require_fdfault(fdfault_dir):
    """Exit with status 3 and an install message if fdfault cannot be run."""
    import shutil
    try:
        import fdfault  # noqa: F401
        import fdfault.analysis  # noqa: F401
    except ImportError:
        print(INSTALL.format('no Python package fdfault'), file=sys.stderr)
        sys.exit(3)
    if not fdfault_dir or not os.path.isfile(os.path.join(fdfault_dir, 'fdfault')):
        print(INSTALL.format(f'no fdfault binary in --fdfault-dir {fdfault_dir!r}'), file=sys.stderr)
        sys.exit(3)
    if shutil.which('mpiexec') is None:
        print(INSTALL.format('no mpiexec on the PATH'), file=sys.stderr)
        sys.exit(3)


def rerun(rows, fdfault_dir, out):
    require_fdfault(fdfault_dir)
    tab, X, _ = design_inputs()
    work = tempfile.mkdtemp(prefix='fdfault_')
    lines = []
    for r in rows:          # 1-based rows of paper_tables.csv
        t0 = time.perf_counter()
        m, x, slip = fp.run(X[r - 1], work, fdfault_dir, n_proc=1, name=f'row{r}')
        dt = time.perf_counter() - t0
        print(f'row {r:2d} (table {int(tab[r - 1, 0])}, row {int(tab[r - 1, 1])}): moment {m:.6f} m km, '
              f'paper {tab[r - 1, 5]:.2f}, {dt:.0f} s', flush=True)
        lines.append(f'{r},' + ','.join(repr(float(v)) for v in X[r - 1]) + f',{m!r},{float(slip.max())!r},{dt:.1f}\n')
    with open(out, 'w') as f:
        f.write('# fdfault v1.0.1 reruns of rows of paper_tables.csv (inputs from design_inputs.csv); one MPI process; '
                'Simpson integral of the final slip over 401 fault nodes\n')
        f.write(f'# row of paper_tables.csv (1-based), {COLS}, integrated final slip (m km), '
                'maximum final slip (m), wall time (s)\n')
        f.writelines(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--rerun', default=None, help='comma-separated 1-based rows of paper_tables.csv')
    ap.add_argument('--fdfault-dir', default=os.environ.get('FDFAULT_DIR'))
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    a = ap.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            a.out = HERE
    elif a.check and os.path.realpath(a.out) == os.path.realpath(HERE):
        ap.error('--check needs --out DIR outside this directory, so that the shipped file is not overwritten')
    if a.rerun:
        rerun([int(s) for s in a.rerun.split(',')], a.fdfault_dir, os.path.join(a.out, 'fdfault_reruns.csv'))
    else:
        path = os.path.join(a.out, 'design_inputs.csv')
        X = write_design(path)
        if a.check:
            if not compare(path):
                sys.exit(1)
        else:
            print(f'wrote {path}: {X.shape}')
