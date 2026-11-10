"""Round the gradients of a float64 lv_samples.npz to float32 (Tumour_Competition_ODE).

    python reduce_gradients.py --in OLD.npz --out DIR

The shipped lv_samples.npz was first written with float64 gradients (30.6 MB). This script
rewrites such a file with grid4_grad and sub8_grad rounded to float32 (numpy's round-to-nearest)
and every other array (inputs, values, cell indices, labels) copied unchanged; the description
string is replaced by the one generate_data.py now writes. The result holds the same arrays that
generate_data.py writes directly, so a full rerun of generate_data.py is not needed.
It prints the largest relative rounding of a gradient row (2-norm) over the rows inside float32's
normal range, the number of rows below it (q_sec gradients at points where q_sec is 0 or 1 to
far below rounding; they become float32 subnormals or 0) and the file sizes.

--out must be a directory outside this one (not this directory or a subdirectory of it), so that
the shipped file is not overwritten.
Imports numpy and generate_data (for FILE, DESCRIPTION, GRAD_DTYPE). Deterministic.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_data as gd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--in', dest='inp', required=True, help='lv_samples.npz with float64 gradients')
    ap.add_argument('--out', required=True, help='output directory (outside this directory)')
    args = ap.parse_args()
    here, out = os.path.realpath(HERE), os.path.realpath(args.out)
    if os.path.commonpath([here, out]) == here:
        ap.error('--out must be a directory outside this one (not this directory or a subdirectory of it)')
    os.makedirs(out, exist_ok=True)
    D = np.load(args.inp)
    data = {}
    for k in D.files:
        a = D[k]
        if k.endswith('_grad'):
            r = a.astype(gd.GRAD_DTYPE)
            tiny = np.finfo(gd.GRAD_DTYPE).tiny               # smallest normal float32, about 1.2e-38
            for q, qn in enumerate(('q_pri', 'q_sec')):
                n = np.linalg.norm(a[:, q], axis=1)
                e = np.linalg.norm(r[:, q].astype(np.float64) - a[:, q], axis=1)
                ok = n >= tiny
                low = (n > 0) & ~ok
                print(f'{k} {qn}: {a.dtype} -> {r.dtype}; largest relative rounding per row {np.max(e[ok] / n[ok]):.1e} '
                      f'over the {int(ok.sum())} rows with norm >= {tiny:.1e}; {int(low.sum())} rows with 0 < norm < {tiny:.1e} '
                      f'(below the float32 normal range; {int((low & (np.linalg.norm(r[:, q].astype(np.float64), axis=1) == 0)).sum())} become 0); '
                      f'{int((n == 0).sum())} rows already 0')
            a = r
        elif k == 'description':
            a = np.array(gd.DESCRIPTION)
        data[k] = a
    path = os.path.join(out, gd.FILE)
    np.savez_compressed(path, **data)
    print(f'wrote {path}: {os.path.getsize(args.inp) / 1e6:.1f} MB -> {os.path.getsize(path) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
