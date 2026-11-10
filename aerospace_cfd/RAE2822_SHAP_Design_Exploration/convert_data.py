"""Convert the authors' MATLAB file of the viscous RAE 2822 case of Palar et al. (2024) into CSV files.

Stored data: the values were computed by the authors with SU2 (RANS, Spalart-Allmaras) and posted in
their repository https://github.com/optimuspram/SHAPMODE (commit 159f915d57f50d6a70d1e3147e140ddbdcd107e6,
file ``SHAP_MODE/test_problems_and_data/Airfoil_Viscous_data.mat``). This script only changes the file
layout: it checks the file's SHA-256, reads the arrays ``X`` (98 x 20) and ``ff`` (98 x 2) with
``scipy.io.loadmat`` and writes every double as the shortest decimal that reads back as the same double
(``repr``), after two ``#`` header lines (a description; column names with units), so
``np.loadtxt(path, delimiter=',')`` reads the files. The conversion is lossless.

    python convert_data.py <folder with Airfoil_Viscous_data.mat> --out DIR          # write the two CSVs into DIR
    python convert_data.py <folder with Airfoil_Viscous_data.mat> --out DIR --check  # build in DIR, compare with the shipped files

``--out`` is required, so nothing is written into this directory unless it is named; with ``--check``
DIR must lie outside this directory. The comparison reads both files with numpy and requires
|difference| <= 1e-15 * |value| for every entry (it prints the largest difference found).
Needs numpy and scipy.
"""

import hashlib
import os
import sys

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = 'Airfoil_Viscous_data.mat'
SHA256 = '984026f1f7f5a7c24fa8df169181f752bc04081bd0691e611e18d2d6dd8ae56a'
RTOL = 1e-15

IN_NAMES = [f'c{i} [chord]' for i in range(1, 21)]
OUT_NAMES = ['Cd [1]', 'abs_Cm [1]']
FILES = {
    'rae2822_hh_inputs.csv': (
        'X', IN_NAMES,
        'Palar et al. 2024 (SMO, doi:10.1007/s00158-024-03769-z) viscous RAE 2822 case: 20 Hicks-Henne bump '
        'amplitudes c1-c10 (upper surface) and c11-c20 (lower surface) of the 98 SU2 RANS runs, each in '
        '[-0.003, 0.003] (fraction of chord); array X of Airfoil_Viscous_data.mat, values unchanged'),
    'rae2822_hh_outputs.csv': (
        'ff', OUT_NAMES,
        'Palar et al. 2024 viscous RAE 2822 case: drag coefficient Cd and absolute pitching moment |Cm| about '
        'the quarter chord at Mach 0.729 and fixed Cl = 0.723; row i matches row i of rae2822_hh_inputs.csv; '
        'array ff of Airfoil_Viscous_data.mat, values unchanged'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(src):
    path = os.path.join(src, MAT)
    got = sha256(path)
    if got != SHA256:
        raise ValueError(f'{MAT}: SHA-256 {got} differs from the recorded {SHA256}')
    d = scipy.io.loadmat(path)
    out = {}
    for name, (key, cols, desc) in FILES.items():
        A = np.asarray(d[key], dtype=np.float64)
        if A.shape != (98, len(cols)) or not np.all(np.isfinite(A)):
            raise ValueError(f'{key}: unexpected shape {A.shape} or non-finite values')
        rows = [','.join(repr(float(v)) for v in r) for r in A]
        for r, txt in zip(A, rows):  # lossless: the text reads back as the same doubles
            if not np.array_equal(np.array([float(t) for t in txt.split(',')]), r):
                raise ValueError(f'{key}: a value does not read back unchanged')
        out[name] = f'# {desc}\n# ' + ','.join(cols) + '\n' + '\n'.join(rows) + '\n'
    return out


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if '--out' not in argv or len(args) < 2:
        print(__doc__)
        return 2
    src = args[0]
    out = argv[argv.index('--out') + 1]
    check = '--check' in argv
    if check and inside(out, HERE):
        print('--check needs --out DIR outside the data-set directory')
        return 2
    os.makedirs(out, exist_ok=True)
    texts = build(src)
    for name, text in texts.items():
        with open(os.path.join(out, name), 'w') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    if not check:
        return 0
    bad = 0
    for name in texts:
        a = np.loadtxt(os.path.join(out, name), delimiter=',')
        b = np.loadtxt(os.path.join(HERE, name), delimiter=',')
        ok = a.shape == b.shape and np.all(np.abs(a - b) <= RTOL * np.abs(b))
        diff = np.max(np.abs(a - b)) if a.shape == b.shape else float('nan')
        print(f'{name}: shape {b.shape}, max |difference| {diff:.1e}: {"agrees" if ok else "DIFFERENT"}')
        bad += not ok
    return int(bad > 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
