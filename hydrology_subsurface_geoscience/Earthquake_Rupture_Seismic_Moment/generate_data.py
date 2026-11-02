"""Write design_inputs.csv (numpy, scipy) and, with --rerun, fdfault_reruns.csv (needs fdfault).

design_inputs.csv: the paper's two Latin hypercube designs at full precision. fabmogp draws
them with mogp_emulator 0.4.0 after np.random.seed(seed), seed 4742322 for the 20 training
runs and 63684 for the 10 validation runs (fabmogp_paper/manuscript/Makefile). They are
redrawn here (fdfault_problem.lhs_mogp) and matched, row by row, to paper_tables.csv: every
input must round to the printed value, or the script stops.

fdfault_reruns.csv: selected rows of paper_tables.csv rerun with the compiled fdfault
simulator, giving the integrated final slip to full precision and the run time.

    python generate_data.py                      # design_inputs.csv (under a second)
    python generate_data.py --check              # compare with the shipped design_inputs.csv
    python generate_data.py --rerun 1,11,13,21,29 --fdfault-dir /path/to/fdfault
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdfault_problem as fp  # noqa: E402

SEEDS = {2: (4742322, 20), 3: (63684, 10)}     # table -> (seed, size)
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


def rerun(rows, fdfault_dir, out):
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
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    if a.rerun:
        rerun([int(s) for s in a.rerun.split(',')], a.fdfault_dir, os.path.join(a.out, 'fdfault_reruns.csv'))
    else:
        path = os.path.join(a.out if not a.check else tempfile.mkdtemp(), 'design_inputs.csv')
        X = write_design(path)
        if a.check:
            old = read_csv('design_inputs.csv')
            print(f'max |difference| from the shipped design_inputs.csv: {np.abs(old[:, 3:] - X).max():.1e}')
        else:
            print(f'wrote {path}: {X.shape}')
