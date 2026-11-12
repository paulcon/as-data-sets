"""Write NRELwind_testing.npz from the original NRELwind_testing.txt, and check that it is lossless.

    python convert_data.py --src DIR             # DIR holds NRELwind_testing.txt; writes the .npz here
    python convert_data.py --src DIR --check     # compares with the shipped .npz; writes nothing

The original text file (15,074,183 bytes) is archived on Zenodo, doi:10.5281/zenodo.24174068.
The script refuses a file whose SHA-256 differs from SHA256 below. It stores the header line
verbatim (`header`, a string) and the 50,000 x 13 values as float64 (`data`), then rebuilds the
text from them, each value written as %.16e, and checks that the rebuilt text equals the original
byte for byte. Without --src, or without the file, it exits with code 3 and says where to get it.
Imports numpy and the standard library.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'NRELwind_testing.txt'
OUT = 'NRELwind_testing.npz'
SHA256 = '94e621651b9198cbffc877656a5f6343b030632767eeb72a1d161f5176e65dd2'
DOI = '10.5281/zenodo.24174068'


def rebuild(header, data):
    """The text file that `header` and `data` came from."""
    rows = (b','.join(b'%.16e' % v for v in row) for row in data)
    return header.encode() + b'\n' + b'\n'.join(rows) + b'\n'


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
    first = raw.index(b'\n')
    header = raw[:first].decode()
    data = np.loadtxt(src, delimiter=',', skiprows=1)
    same = rebuild(header, data) == raw
    print(f'{NAME}: {data.shape[0]} rows x {data.shape[1]} columns; text rebuilt from the arrays '
          f'equals the original byte for byte: {same}')
    if not same:
        sys.exit(1)
    if args.check:
        with np.load(os.path.join(HERE, OUT)) as z:
            ok = str(z['header']) == header and z['data'].dtype == data.dtype and np.array_equal(z['data'], data)
        print(f'{OUT}: header and data identical to the conversion: {ok}')
        sys.exit(0 if ok else 1)
    np.savez_compressed(os.path.join(HERE, OUT), header=np.str_(header), data=data)
    print(f'wrote {OUT} ({os.path.getsize(os.path.join(HERE, OUT)) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
