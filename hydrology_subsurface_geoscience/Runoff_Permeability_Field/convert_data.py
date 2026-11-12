"""Write inputs-ShortCorr.GW02.npz from the original inputs-ShortCorr.GW02.txt, and check that it is lossless.

    python convert_data.py --src DIR             # DIR holds inputs-ShortCorr.GW02.txt; writes the .npz here
    python convert_data.py --src DIR --check     # compares with the shipped .npz; writes nothing

The original text file (25,500,463 bytes) is archived on Zenodo, doi:10.5281/zenodo.24174085,
with the two runoff files. The script refuses a file whose SHA-256 differs from SHA256 below. It
stores the 10,000 x 100 values as float64 (`X`), then rebuilds the text from them, each value
written as %.18e and separated by single spaces, and checks that the rebuilt text equals the
original byte for byte. Without --src, or without the file, it exits with code 3 and says where to
get it. Imports numpy and the standard library.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'inputs-ShortCorr.GW02.txt'
OUT = 'inputs-ShortCorr.GW02.npz'
SHA256 = 'ac0c3abdbca4804898423f719718b5a8738ca612666ae0023b148f83147c6d58'
DOI = '10.5281/zenodo.24174085'


def rebuild(X):
    """The text file that `X` came from."""
    return b'\n'.join(b' '.join(b'%.18e' % v for v in row) for row in X) + b'\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', help=f'directory that holds {NAME} (download it from doi:{DOI})')
    ap.add_argument('--check', action='store_true', help=f'compare with the shipped {OUT}; write nothing')
    args = ap.parse_args()
    src = os.path.join(args.src, NAME) if args.src else None
    if src is None or not os.path.isfile(src):
        print(f'{NAME} not found{" in " + args.src if args.src else ""}. Download it from '
              f'https://doi.org/{DOI} and rerun with --src DIR.', file=sys.stderr)
        sys.exit(3)
    raw = open(src, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()
    if h != SHA256:
        sys.exit(f'{src}: SHA-256 {h} differs from the archived file ({SHA256})')
    X = np.loadtxt(src)
    same = rebuild(X) == raw
    print(f'{NAME}: {X.shape[0]} rows x {X.shape[1]} columns; text rebuilt from the array equals the '
          f'original byte for byte: {same}')
    if not same:
        sys.exit(1)
    if args.check:
        with np.load(os.path.join(HERE, OUT)) as z:
            ok = z['X'].dtype == X.dtype and np.array_equal(z['X'], X)
        print(f'{OUT}: identical to the conversion: {ok}')
        sys.exit(0 if ok else 1)
    np.savez_compressed(os.path.join(HERE, OUT), X=X)
    print(f'wrote {OUT} ({os.path.getsize(os.path.join(HERE, OUT)) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
