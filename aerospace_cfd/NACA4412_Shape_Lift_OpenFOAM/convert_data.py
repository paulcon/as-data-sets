"""Convert the stored files of Tezzele et al. (2020) into this data set's files.

Stored data: the 170 OpenFOAM runs (NACA 4412 airfoil, unsteady RANS, 30 s of flow) were made by the
authors; their shape parameters and lift-coefficient histories are in the GitHub repository
github.com/mtezzele/MORtech-2019-Special-Issue (commit ec05216d2c8e10c437503eaf2ff71a8f43d50e79,
2020-05-04; no licence stated). This script only changes the layout, losslessly:

* ``shape_inputs.csv``: the 10 shape weights of the 170 runs. ``c1..c5`` are ``parameters/parTop.npy``
  unchanged; ``d1..d5`` are ``parameters/parBot.npy`` with the sign flipped (the repository stores the
  lower-surface weights as negative offsets, the paper's Eq. 5 subtracts positive weights d_i). Negation
  is exact in floating point, and every value is written with Python's shortest round-trip repr, so
  ``np.loadtxt`` gives back the identical doubles. An 11th column flags the 70 runs the authors used for
  training (the first 70 rows, as in their ``dyas.py`` and ``predict.py``).
* ``lift_series.npz``: ``outputs/lift_170par_30000t.npy`` (170 x 30000 C_L values, one per time step
  of 1e-3 s). Every stored value has 7 significant decimal digits, so it is written as an integer
  mantissa times a power of ten: ``scale`` (int8) holds the power and ``mantissa_diff`` (int32) the
  differences of the mantissas along time. Decoding,

      lift = np.cumsum(z['mantissa_diff'], axis=1, dtype=np.int64) / 10.0 ** z['scale']

  gives back the stored doubles (the division of two exactly represented numbers is correctly rounded,
  so it returns the double nearest to the 7-digit decimal, which is the stored value). The script
  refuses to write if any value does not round-trip. The file is 2.8 MB instead of 41 MB.
  ``outputs/lift_170par_20000t.npy`` equals the first 20000 columns of the 30000-step file (checked
  here) and is not converted.

    python convert_data.py <repo folder> --out DIR            # write the two files into DIR
    python convert_data.py <repo folder> --out DIR --check    # write into DIR (outside this folder)
                                                              # and compare with the shipped files

``--out`` is required and must lie outside this folder, so the shipped files are never overwritten. The
``--check`` comparison passes when no decoded value differs from
the shipped one by more than 1e-12 (absolute; C_L values are of order 0.1 to 400) and it prints the
largest difference found. Imports numpy and the standard library only; the .npy files are read with
``allow_pickle=False``.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHA256 = {
    'parameters/parTop.npy': '177563dbba96736efd535fd9693ef1d5dac616bbb6eb75bd1d5e9872b0cac0cf',
    'parameters/parBot.npy': '3e7c0e90f4ac52b51ac06554ecc7a8aae4fc3db028301cc786854187c938721c',
    'outputs/lift_170par_30000t.npy': '24ca1ddfc49016455d5ff4edd262931bc227ea6b83d91ca8c2e3604798b96c3b',
    'outputs/lift_170par_20000t.npy': '0c52794f03ee7881e11fe14ac3064bb51fe4144994b2b2b9910142fcaa9f6e9a',
}
N_TRAIN = 70
TOL = 1e-12
NAMES = [f'c{i}' for i in range(1, 6)] + [f'd{i}' for i in range(1, 6)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(src):
    for rel, want in SHA256.items():
        got = sha256(os.path.join(src, rel))
        if got != want:
            raise ValueError(f'{rel}: SHA-256 {got} differs from the recorded {want}')
    top = np.load(os.path.join(src, 'parameters/parTop.npy'), allow_pickle=False)
    bot = np.load(os.path.join(src, 'parameters/parBot.npy'), allow_pickle=False)
    lift = np.load(os.path.join(src, 'outputs/lift_170par_30000t.npy'), allow_pickle=False)
    lift20 = np.load(os.path.join(src, 'outputs/lift_170par_20000t.npy'), allow_pickle=False)
    if top.shape != (170, 5) or bot.shape != (170, 5) or lift.shape != (170, 30000):
        raise ValueError(f'unexpected shapes {top.shape} {bot.shape} {lift.shape}')
    if not np.array_equal(lift20, lift[:, :20000]):
        raise ValueError('lift_170par_20000t.npy is not the first 20000 columns of the 30000-step file')
    return top, bot, lift


def inputs_text(top, bot):
    X = np.hstack([top, -bot])
    lines = ['# Tezzele et al. 2020 NACA 4412 shape study: 10 shape weights of the 170 OpenFOAM runs '
             '(c = parTop.npy; d = -parBot.npy, the sign of Eq. 5; values otherwise unchanged) and a flag '
             'for the 70 training runs of the paper',
             '# ' + ','.join(f'{n} [1]' for n in NAMES) + ',paper_training_run [1]']
    for i, row in enumerate(X):
        lines.append(','.join(repr(float(v)) for v in row) + f',{int(i < N_TRAIN)}')
    text = '\n'.join(lines) + '\n'
    back = np.array([[float(v) for v in ln.split(',')[:10]] for ln in lines[2:]])
    if not np.array_equal(back, X):
        raise ValueError('inputs do not round-trip')
    return text


def encode(lift):
    if not np.all(np.isfinite(lift)) or np.any(lift == 0):
        raise ValueError('non-finite or zero C_L value')
    scale = (6 - np.floor(np.log10(np.abs(lift)))).astype(np.int8)
    mant = np.round(lift * 10.0 ** scale).astype(np.int64)
    if np.abs(mant).max() >= 10 ** 7:
        raise ValueError('a value has more than 7 significant digits')
    diff = np.diff(mant, axis=1, prepend=0)
    if np.abs(diff).max() >= 2 ** 31:
        raise ValueError('mantissa difference does not fit int32')
    diff = diff.astype(np.int32)
    if not np.array_equal(decode(diff, scale), lift):
        raise ValueError('lift values do not round-trip through the 7-digit encoding')
    return diff, scale


def decode(diff, scale):
    return np.cumsum(diff, axis=1, dtype=np.int64) / 10.0 ** scale


def read_inputs(path):
    return np.loadtxt(path, delimiter=',')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src', help='folder of the authors\' repository (with parameters/ and outputs/)')
    ap.add_argument('--out', required=True, help='folder to write into')
    ap.add_argument('--check', action='store_true', help='compare the written files with the shipped ones')
    a = ap.parse_args(argv[1:])
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    if out == here or out.startswith(here + os.sep):
        print('--out must be a folder outside this data set folder, so that the shipped files are never '
              'overwritten; to replace them, copy the files from DIR after a passing --check', file=sys.stderr)
        return 2
    top, bot, lift = load(a.src)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'shape_inputs.csv'), 'w') as fh:
        fh.write(inputs_text(top, bot))
    diff, scale = encode(lift)
    np.savez_compressed(os.path.join(out, 'lift_series.npz'), mantissa_diff=diff, scale=scale)
    print(f'wrote {out}/shape_inputs.csv (170 x 11) and {out}/lift_series.npz (170 x 30000)')
    if not a.check:
        return 0
    bad = 0
    Xn, Xs = read_inputs(os.path.join(out, 'shape_inputs.csv')), read_inputs(os.path.join(HERE, 'shape_inputs.csv'))
    dx = np.abs(Xn - Xs).max()
    print(f'shape_inputs.csv: max |difference| {dx:.3g}; values differing: {int(np.sum(Xn != Xs))}')
    bad += dx > TOL
    with np.load(os.path.join(HERE, 'lift_series.npz')) as z:
        Ls = decode(z['mantissa_diff'], z['scale'])
    dl = np.abs(Ls - lift).max()
    print(f'lift_series.npz vs lift_170par_30000t.npy: max |difference| {dl:.3g}; '
          f'values differing: {int(np.sum(Ls != lift))}')
    bad += dl > TOL
    print('check passed (tolerance 1e-12)' if not bad else 'CHECK FAILED')
    return int(bad > 0)


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        sys.exit(f'file not found, or not of the expected kind: {err.filename}\n'
                 'The source is a clone of github.com/mtezzele/MORtech-2019-Special-Issue; see DATA.md, "Regenerating".')
