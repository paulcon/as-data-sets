"""Convert the released JAG ICF files of Anirudh et al. (2020) into this data set's CSV files.

Stored data: the values were computed at Lawrence Livermore National Laboratory with the JAG
semi-analytic implosion simulator (not released) and distributed in the GitHub repository
llnl/macc as ``data/icf-jag-10k.tar.gz`` (MIT licence of that repository). This script only
changes the file format: it reads the members ``jag10K_params.npy`` and ``jag10K_0_scalars.npy``
straight from the tarball (no member is extracted to disk or executed; ``allow_pickle=False``),
checks the tarball's SHA-256, and writes each value as decimal text that converts back to the
stored binary value: inputs (float64) with 17 significant digits, scalars (float32) with 9.
The images member is not read.

    python convert_data.py <icf-jag-10k.tar.gz> --out DIR            # write the two CSVs into DIR
    python convert_data.py <icf-jag-10k.tar.gz> --out DIR --check    # ... and compare them with
                                                                     # the shipped CSVs and the .npy

DIR must lie outside this data set's directory, subdirectories included (the shipped files are never
overwritten and nothing is written inside the data set).
``--check`` passes when, for every value, |written - shipped| <= 1e-12 (inputs) and <= 1e-8
(scalars, which are float32 near [0, 1]), and the CSV values round to the stored binary values
(float64 inputs; scalars after casting to float32). Requires numpy.
"""

import argparse
import hashlib
import io
import os
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHA256 = '0bdfc984fa5c10c45014bed1cdd3fae4980b5de9f39344105a052cfaad6bd14f'
TOL = {'jag_icf_inputs.csv': 1e-12, 'jag_icf_scalars.csv': 1e-8}

FILES = [
    # (member, output file, column names, format, stored dtype, description)
    ('jag10K_params.npy', 'jag_icf_inputs.csv',
     [f'x{i} [units not given]' for i in range(1, 6)], '%.17g', np.float64,
     'JAG ICF ensemble (Anirudh et al. 2020, PNAS 117:9741; llnl/macc data/icf-jag-10k.tar.gz, '
     'member jag10K_params.npy): 10000 runs x 5 inputs as released (names and units not given; '
     'whether the values are physical or scaled is not stated)'),
    ('jag10K_0_scalars.npy', 'jag_icf_scalars.csv',
     [f's{j:02d} [units not given]' for j in range(15)], '%.9g', np.float32,
     'JAG ICF ensemble: 15 diagnostic scalars of the 10000 runs as released (member '
     'jag10K_0_scalars.npy; names not released; values in (0, 1), apparently scaled, '
     'the scaling not documented); row i matches '
     'row i of jag_icf_inputs.csv'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_members(tarball):
    """Return {member: array} for the two .npy members, read from the stream without extracting."""
    want = {f[0] for f in FILES}
    out = {}
    with tarfile.open(tarball, 'r:gz') as tf:
        for info in tf:
            if info.name in want:
                if not info.isfile():
                    raise ValueError(f'{info.name} is not a regular file')
                buf = io.BytesIO(tf.extractfile(info).read())
                out[info.name] = np.lib.format.read_array(buf, allow_pickle=False)
    missing = want - set(out)
    if missing:
        raise ValueError(f'members not found in the tarball: {sorted(missing)}')
    return out


def write(arrays, outdir):
    for member, name, cols, fmt, dtype, desc in FILES:
        a = arrays[member]
        if a.dtype != dtype or a.ndim != 2 or a.shape[0] != 10000 or a.shape[1] != len(cols):
            raise ValueError(f'{member}: unexpected dtype/shape {a.dtype} {a.shape}')
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{member}: non-finite values')
        np.savetxt(os.path.join(outdir, name), a.astype(np.float64), fmt=fmt, delimiter=',',
                   header=desc + '\n' + ','.join(cols), comments='# ')
        print(f'wrote {name}: {a.shape[0]} rows x {a.shape[1]} columns')


def check(arrays, outdir):
    bad = 0
    for member, name, cols, fmt, dtype, desc in FILES:
        new = np.loadtxt(os.path.join(outdir, name), delimiter=',')
        old = np.loadtxt(os.path.join(HERE, name), delimiter=',')
        src = arrays[member]
        d_ship = np.abs(new - old).max()
        d_src = np.abs(new - src.astype(np.float64)).max()
        roundtrip = int(np.count_nonzero(new.astype(dtype) != src))
        ok = d_ship <= TOL[name] and d_src <= TOL[name] and roundtrip == 0
        print(f'{name}: max |written - shipped| = {d_ship:.1e}, max |written - npy| = {d_src:.1e} '
              f'(tolerance {TOL[name]:.0e}); values not rounding to the stored {np.dtype(dtype).name}: '
              f'{roundtrip}  -> {"OK" if ok else "FAILED"}')
        bad += not ok
    return int(bad > 0)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('tarball', help='icf-jag-10k.tar.gz from github.com/llnl/macc (data/)')
    p.add_argument('--out', help='output directory (required; outside this data set directory)')
    p.add_argument('--check', action='store_true', help='compare with the shipped CSVs')
    a = p.parse_args(argv[1:])
    if not a.out:
        p.error('--out DIR is required (DIR outside this data set directory)')
    out, here = os.path.realpath(a.out), os.path.realpath(HERE)
    if out == here or out.startswith(here + os.sep):
        p.error('--out must be outside the data set directory (subdirectories included); '
                'the shipped files are not overwritten')
    os.makedirs(a.out, exist_ok=True)
    got = sha256(a.tarball)
    if got != SHA256:
        raise SystemExit(f'SHA-256 of {a.tarball} is {got}, not the recorded {SHA256}')
    arrays = read_members(a.tarball)
    write(arrays, a.out)
    return check(arrays, a.out) if a.check else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
