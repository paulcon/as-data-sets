"""Recompute the numbers of DATA.md (Stagnation_Temperature_Probe_Design) that come from the
authors' fitted subspaces, and the effect of the misprinted Eq. 4.

    python check_accuracy.py --src DIR        DIR holds Yp_subspace.joblib and Rr_subspace.joblib
    python check_accuracy.py --download DIR   fetch them first from the pinned commit into DIR
    python check_accuracy.py --eq4-only       only check 1 (needs no outside file)

The two files are in the authors' companion repository, github.com/ascillitoe/probe-subspaces,
commit 072a0b0ea992afb9cbdf1d5862adfb1fe8f7a452, Example_Notebooks/ (URLs and SHA-256 below); they
are not shipped here. They are joblib pickles of equadratures Subspaces objects. This script never
unpickles them (unpickling runs code from the file): it checks their SHA-256 and reads two float64
arrays from the raw bytes, `sample_points` (128 x 7) and `_subspace` (7 x 7), locating each by its
key name and the shape written before it. Run it from any directory; it reads the shipped CSVs and
convert_data.py next to this file. Each line prints the recomputed value; DATA.md quotes them.

Checks:
 1. Eq. 4 as printed, x = (2 x_hat - (x_U - x_L)) / (x_U - x_L), applied to the Table 1 bounds: the
    range it gives for each parameter (it should be [-1, 1]); Eq. 4 as intended maps them to [-1, 1].
 2. The files: SHA-256; sample_points equal to the archive's X (inputs_normalized.csv), bit for bit;
    |W^T W - I| (Frobenius) of each 7 x 7 matrix.
 3. Loss: the first column of the authors' Yp matrix in archive order and, under the column map of
    convert_data.py, in Table 1 order; its cosine with Eq. 12 and largest difference, also without
    the hole-angle sign change; the range of its active variable over the 128 designs (Fig. 7) and
    the baseline's value.
 4. Recovery: the code's direction vectors v_a (design 58 to 112) and v_b (design 115 to 26) in the
    plane of the first two columns of the Rr matrix (dim_red.py lines 240-248), the vectors v_a U_R
    and v_b U_R with the code's hole-angle flip (lines 294, 319), in Table 1 order (Fig. 11); the
    columns in the paper's order (the code stores them the other way round) and v_a, v_b in that
    order; the active-variable box over the 128 designs and the baseline (Fig. 10(a)).
 5. The 6-decimal constants UY_authors and UR_authors of the notebook against the values read here.

Deterministic. Imports numpy and the standard library (urllib only for --download).
"""
import argparse
import hashlib
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_data as cd  # noqa: E402

COMMIT = '072a0b0ea992afb9cbdf1d5862adfb1fe8f7a452'
BASE = f'https://raw.githubusercontent.com/ascillitoe/probe-subspaces/{COMMIT}/Example_Notebooks/'
FILES = {'Yp_subspace.joblib': 'a5e8930d80c2e34943572c8bf150f768cd8465b6665c5bbcabc021081f453c5f',
         'Rr_subspace.joblib': '92c777f38b00e904a6bf1ce77beaff9eaa6b01f95f685f554c05c7caf25c3ca4'}
EQ12 = np.array([0.05, 0.01, -0.12, -0.02, 0.98, 0.02, 0.17])          # U_Y, Table 1 order (p. 6)
# the notebook's constants (cell that estimates the subspaces), Table 1 order, paper's sign convention
UY_AUTHORS = np.array([0.045220, 0.012342, -0.122653, -0.018035, 0.976544, 0.015808, 0.168954])
UR_AUTHORS = np.array([[0.337052, 0.002553, -0.108296, 0.003266, 0.013707, -0.230720, 0.906218],
                       [0.049804, 0.046869, 0.197496, -0.020082, 0.511531, -0.806805, -0.208128]]).T


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_array(data, key):
    """The float64 array stored under `key` in a joblib pickle, read from the raw bytes."""
    if data.count(b'\x8c\x02f8') != 1 or data.count(b'\x8c\x01<') != 1:
        sys.exit('unexpected dtype declarations: not a single little-endian float64 type')
    name = b'\x8c' + bytes([len(key)]) + key.encode()
    pos = data.find(name)
    if pos < 0 or data.count(name) != 1:
        sys.exit(f'key {key} not found once')
    start = data.find(b'\x88ub', pos) + 3                     # NEWTRUE SETITEMS BUILD, then the bytes
    shape_m = re.search(rb'((?:K.|M..|J....)+)\x86', data[pos:start], flags=re.S)
    toks, shape, i = shape_m.group(1), [], 0
    while i < len(toks):
        n = {ord('K'): 1, ord('M'): 2, ord('J'): 4}[toks[i]]
        shape.append(int.from_bytes(toks[i + 1:i + 1 + n], 'little'))
        i += 1 + n
    size = int(np.prod(shape)) * 8
    return np.frombuffer(data[start:start + size], dtype='<f8').reshape(shape), start + size


def to_table1(v):
    """A vector in archive order -> Table 1 order with the paper's hole-angle sign."""
    out = np.empty(7)
    for j, (k, s) in enumerate(zip(cd.COLUMN_MAP, cd.SIGN)):
        out[k] = s * v[j]
    return out


def unit(v):
    return v / np.linalg.norm(v)


def fmt(v, p=4):
    return '[' + ', '.join(f'{x:.{p}f}' for x in v) + ']'


def check_eq4():
    w = cd.UPPER - cd.LOWER
    lo, hi = (2 * cd.LOWER - w) / w, (2 * cd.UPPER - w) / w
    print('1. Eq. 4 as printed, applied to the Table 1 bounds (x_L -> x_U):')
    for n, a, b in zip(cd.NAMES, lo, hi):
        print(f'   {n:14s} [{a:.2f}, {b:.2f}]')
    print(f'   Eq. 4 as intended (convert_data.to_normalized) maps every bound to -1 and +1: '
          f'{np.allclose(cd.to_normalized(cd.LOWER), -1) and np.allclose(cd.to_normalized(cd.UPPER), 1)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', help='directory holding Yp_subspace.joblib and Rr_subspace.joblib')
    ap.add_argument('--download', metavar='DIR', help='fetch the two files from the pinned commit into DIR')
    ap.add_argument('--eq4-only', action='store_true')
    a = ap.parse_args()
    check_eq4()
    if a.eq4_only:
        return
    src = a.download or a.src
    if a.download:
        import urllib.request
        os.makedirs(a.download, exist_ok=True)
        for f in FILES:
            urllib.request.urlretrieve(BASE + f, os.path.join(a.download, f))
    missing = [f for f in FILES if not src or not os.path.isfile(os.path.join(src, f))]
    if missing:
        print('checks 2-5 need the authors\' files ' + ', '.join(missing) + ' (not shipped). Download them '
              '(' + ', '.join(BASE + f for f in FILES) + '; SHA-256 ' + ', '.join(FILES.values()) +
              ') into a directory and rerun check_accuracy.py with --src DIR, or rerun with --download DIR.', file=sys.stderr)
        sys.exit(3)
    raw, W = {}, {}
    X = np.loadtxt(os.path.join(HERE, 'inputs_normalized.csv'), delimiter=',')
    print('2. the authors\' files:')
    for f, want in FILES.items():
        path = os.path.join(src, f)
        got = sha256(path)
        if got != want:
            sys.exit(f'{f}: SHA-256 {got} is not the expected {want}')
        with open(path, 'rb') as fh:
            raw[f] = fh.read()
        S, _ = read_array(raw[f], 'sample_points')
        W[f], end = read_array(raw[f], '_subspace')
        print(f'   {f}: SHA-256 as expected; sample_points {S.shape} equal to the archive X bit for bit: '
              f'{np.array_equal(S.view(np.uint64), X.view(np.uint64))}; _subspace {W[f].shape}, ends '
              f'{len(raw[f]) - end} bytes before the end of the file; |W^T W - I| = '
              f'{np.linalg.norm(W[f].T @ W[f] - np.eye(7)):.1e}')

    z0 = cd.to_normalized(cd.NOMINAL)
    Z = cd.to_paper_normalized(X)
    uy_a = W['Yp_subspace.joblib'][:, 0]
    uy = to_table1(uy_a)
    uy *= np.sign(uy[4])
    no_flip = uy * np.array([1, 1, -1, 1, 1, 1, 1])
    print(f'3. loss, first column in archive order {fmt(uy_a)}')
    print(f'   in Table 1 order {fmt(uy)}; cosine with Eq. 12 {abs(unit(uy) @ unit(EQ12)):.5f}, largest |difference| '
          f'{np.abs(uy - EQ12).max():.4f}; without the hole-angle sign change: cosine '
          f'{abs(unit(no_flip) @ unit(EQ12)):.4f}')
    print(f'   active variable over the 128 designs [{(Z @ uy).min():.3f}, {(Z @ uy).max():.3f}]; baseline {z0 @ uy:.3f}')

    Wr = W['Rr_subspace.joblib'][:, :2]
    u2 = X @ Wr                                        # the code's u2 (dim_red.py line 61), archive order
    v1, v2 = unit(u2[112] - u2[58]), unit(u2[26] - u2[115])
    figs = []
    for v in (v1, v2):
        k = v[0] * Wr[:, 0] + v[1] * Wr[:, 1]
        k[-3] *= -1                                    # dim_red.py lines 294, 319
        figs.append(k[::-1])                           # archive order reversed = Table 1 order
    print(f'4. recovery: code v_a = {fmt(v1)}, v_b = {fmt(v2)} (code column order)')
    print(f'   v_a U_R (Fig. 11(a)), Table 1 order {fmt(figs[0], 3)}')
    print(f'   v_b U_R (Fig. 11(b)), Table 1 order {fmt(figs[1], 3)}')
    ur = np.column_stack([to_table1(Wr[:, 1]), to_table1(Wr[:, 0])])   # the paper's order
    ur *= np.sign(ur[6, 0]), np.sign(ur[4, 1])
    print(f'   in the paper\'s order (u_R1 = code column 2): v_a = {fmt(v1[::-1])}, v_b = {fmt(v2[::-1])}')
    yr, yb = Z @ ur, z0 @ ur
    print(f'   active variables over the 128 designs [{yr[:, 0].min():.2f}, {yr[:, 0].max():.2f}] x '
          f'[{yr[:, 1].min():.2f}, {yr[:, 1].max():.2f}]; baseline ({yb[0]:.3f}, {yb[1]:.3f})')

    d_y = np.abs(UY_AUTHORS - uy).max()
    d_r = np.abs(UR_AUTHORS - ur).max()
    print(f'5. the notebook\'s UY_authors and UR_authors against the values read here: largest |difference| '
          f'{d_y:.1e} and {d_r:.1e} (6-decimal rounding: at most 5e-07)')


if __name__ == '__main__':
    main()
