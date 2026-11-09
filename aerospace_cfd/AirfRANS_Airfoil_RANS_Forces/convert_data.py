"""Reduce the AirfRANS archive (Bonnet et al. 2022) to a table of per-case lift and drag coefficients.

Stored data: the flow fields were computed by the authors with OpenFOAM v2112 (simpleFoam, k-omega SST)
and published as ``Dataset.zip`` (ODbL v1.0; URL, size and SHA-256 in DATA.md). This script reads the
archive without extracting it as a whole: for each of the 1000 cases it copies the two files the force
routine needs (``*_internal.vtu`` and ``*_aerofoil.vtp``) into a temporary folder, calls the airfrans
library's own routine ``Simulation(root, name).force_coefficient(reference=True)``, deletes the copies,
and writes one row per case.

What is copied and what is computed:
  * The inputs (inlet velocity, angle of attack, NACA digits) are the numbers in each case's folder name,
    written character for character, and the split flags are read from the archive's ``manifest.json``.
  * The six coefficients (C_D, its pressure and viscous parts, C_L and its two parts) are computed by
    airfrans from the stored fields; they are written with ``repr`` (shortest round-trip float64).
    airfrans integrates the pressure of the internal field at the wall nodes and a wall shear stress
    from the vtk/pyvista point gradient of the velocity (DATA.md, "How the coefficients are computed").

Usage (the airfrans library and its dependencies pyvista and vtk must be installed; versions in DATA.md):

    python convert_data.py --zip Dataset.zip --out DIR             # all 1000 cases -> DIR
    python convert_data.py --zip Dataset.zip --out DIR --rows 25   # 25 evenly spaced cases only
    python convert_data.py --zip Dataset.zip --out DIR --check     # also compare DIR's table with the
                                                                   # shipped airfrans_forces.csv
    The archive path may also be given by the environment variable AIRFRANS_ZIP.

The run is resumable: rows are appended to DIR/airfrans_forces.partial.csv as they are computed, and an
interrupted run (or one stopped by ``--max-seconds S``) continues where it stopped. When every requested
case is done the script writes DIR/airfrans_forces.csv and DIR/airfrans_case_names.txt. DIR must lie
outside this data set's directory. To run the notebook from these files, copy the two files of a full run
(all 1000 cases) from DIR into this directory (DATA.md). ``--check`` compares the inputs as text (identical) and the coefficients
with the tolerance |a - b| <= 1e-9 + 1e-6 |b|. The SHA-256 of the archive is verified once per DIR
(about 1 min for 10 GB) and recorded in DIR/Dataset.zip.sha256.ok; ``--skip-sha256`` skips it.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

ZIP_SIZE = 10029067577
ZIP_SHA256 = '6b301d75dee77fc6c7de6e551c44332be4acc84e30b253e9c60cfa756f6c96db'
NAME_RE = re.compile(r'^airFoil2D_SST_(-?[0-9.]+)_(-?[0-9.]+)((?:_-?[0-9.]+){3,4})$')
SPLITS = ('full_train', 'scarce_train', 'reynolds_train', 'aoa_train')
COLUMNS = ['family [4 or 5]', 'U_inf [m/s]', 'alpha [deg]', 'M_or_L [1]', 'P [1]', 'Q [1]', 'XX [1]',
           'CD [1]', 'CD_p [1]', 'CD_v [1]', 'CL [1]', 'CL_p [1]', 'CL_v [1]'] + [f'{s} [1]' for s in SPLITS]
DESCRIPTION = ('AirfRANS (Bonnet et al. 2022) force coefficients of the 1000 OpenFOAM RANS cases, computed '
               'from Dataset.zip with airfrans Simulation.force_coefficient(reference=True); inputs copied from '
               'the case names (Q = nan for NACA 4-digit rows); split flags from manifest.json; one row per case, '
               'cases in sorted name order (names in airfrans_case_names.txt)')
TABLE = 'airfrans_forces.csv'
NAMES = 'airfrans_case_names.txt'
PARTIAL = 'airfrans_forces.partial.csv'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def verify_archive(path, out, skip):
    size = os.path.getsize(path)
    if size != ZIP_SIZE:
        raise SystemExit(f'{path}: {size} bytes, expected {ZIP_SIZE} (the published Dataset.zip)')
    if skip:
        return
    stamp = os.path.join(out, 'Dataset.zip.sha256.ok')
    key = f'{ZIP_SHA256} {size} {int(os.path.getmtime(path))}'
    if os.path.exists(stamp) and open(stamp).read().strip() == key:
        return
    print('verifying SHA-256 of the archive (about a minute) ...', flush=True)
    got = sha256(path)
    if got != ZIP_SHA256:
        raise SystemExit(f'{path}: SHA-256 {got} differs from the recorded {ZIP_SHA256}')
    with open(stamp, 'w') as fh:
        fh.write(key + '\n')


def case_inputs(name):
    """Family and input strings, exactly as written in the case name."""
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError(f'unexpected case name {name!r}')
    digits = m.group(3)[1:].split('_')
    if len(digits) == 3:                       # NACA 4-digit: M, P, XX
        return ['4', m.group(1), m.group(2), digits[0], digits[1], 'nan', digits[2]]
    return ['5', m.group(1), m.group(2)] + digits  # NACA 5-digit: L, P, Q, XX


def list_cases(zf):
    names = sorted({n.split('/')[1] for n in zf.namelist() if n.startswith('Dataset/airFoil2D_')})
    for n in names:
        case_inputs(n)  # validates every name
    manifest = json.loads(zf.read('Dataset/manifest.json'))
    sets = [set(manifest[s]) for s in SPLITS]
    flags = {n: ['1' if n in members else '0' for members in sets] for n in names}
    return names, flags


def coefficients(zf, name, scratch):
    """airfrans force coefficients of one case: CD, CD_p, CD_v, CL, CL_p, CL_v."""
    import airfrans
    folder = os.path.join(scratch, name)
    os.makedirs(folder, exist_ok=True)
    try:
        for part in ('internal.vtu', 'aerofoil.vtp'):
            member = f'Dataset/{name}/{name}_{part}'
            with zf.open(member) as src, open(os.path.join(folder, f'{name}_{part}'), 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
        (cd, cdp, cdv), (cl, clp, clv) = airfrans.Simulation(scratch, name).force_coefficient(reference=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return [repr(float(v)) for v in (cd, cdp, cdv, cl, clp, clv)]


def inside(path, folder):
    """True if path is folder or lies inside it (after resolving links)."""
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return os.path.commonpath([path, folder]) == folder


def read_rows(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) == 1 + len(COLUMNS):
                    rows[parts[0]] = parts[1:]
    return rows


def read_table(folder):
    with open(os.path.join(folder, NAMES)) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    with open(os.path.join(folder, TABLE)) as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
    return dict(zip(names, (ln.split(',') for ln in lines)))


def check(out):
    mine, shipped = read_table(out), read_table(HERE)
    worst, bad = 0.0, 0
    for name, row in mine.items():
        ref = shipped.get(name)
        if ref is None:
            print(f'{name}: not in the shipped table')
            bad += 1
            continue
        if row[:7] != ref[:7] or row[13:] != ref[13:]:
            print(f'{name}: inputs or split flags differ')
            bad += 1
        for a, b in zip(row[7:13], ref[7:13]):
            a, b = float(a), float(b)
            worst = max(worst, abs(a - b) / (1e-9 + 1e-6 * abs(b)))
            bad += abs(a - b) > 1e-9 + 1e-6 * abs(b)
    print(f'--check: {len(mine)} cases compared with the shipped {TABLE}; inputs and flags identical: '
          f'{bad == 0}; largest coefficient difference / tolerance = {worst:.3g} '
          f'(tolerance 1e-9 + 1e-6 |value|) -> {"PASS" if bad == 0 else "FAIL"}')
    return int(bad > 0)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--zip', default=os.environ.get('AIRFRANS_ZIP'), help='path of Dataset.zip (or $AIRFRANS_ZIP)')
    ap.add_argument('--out', required=True, help='output folder (not this data set\'s directory)')
    ap.add_argument('--rows', type=int, default=None, help='only N evenly spaced cases')
    ap.add_argument('--check', action='store_true', help='compare the result with the shipped table')
    ap.add_argument('--max-seconds', type=float, default=None, help='stop after this many seconds (resumable)')
    ap.add_argument('--scratch', default=None, help='folder for the temporary copies (default: system temp)')
    ap.add_argument('--skip-sha256', action='store_true')
    args = ap.parse_args(argv)
    if not args.zip:
        ap.error('give --zip PATH or set AIRFRANS_ZIP')
    out = os.path.abspath(args.out)
    if inside(out, HERE):
        ap.error('--out must be a folder outside this data set\'s directory (not the directory or a folder in it)')
    if args.check and not os.path.exists(os.path.join(HERE, TABLE)):
        ap.error(f'--check compares with {TABLE} in this data set\'s directory, which is not there '
                 '(DATA.md, "Running the notebook from the archive")')
    try:
        import airfrans  # noqa: F401
    except ImportError:
        raise SystemExit('convert_data.py needs the airfrans library (tested with airfrans 0.1.5.1, '
                         'pyvista 0.49.2, vtk 9.7.2): pip install airfrans==0.1.5.1, then rerun.')
    os.makedirs(out, exist_ok=True)
    verify_archive(args.zip, out, args.skip_sha256)
    t0 = time.time()
    with zipfile.ZipFile(args.zip) as zf:
        names, flags = list_cases(zf)
        if args.rows:
            step = len(names) / args.rows
            names = [names[int(i * step)] for i in range(args.rows)]
        partial = os.path.join(out, PARTIAL)
        done = read_rows(partial)
        scratch = tempfile.mkdtemp(dir=args.scratch)
        try:
            with open(partial, 'a') as fh:
                for i, name in enumerate(names):
                    if name in done:
                        continue
                    if args.max_seconds and time.time() - t0 > args.max_seconds:
                        print(f'stopped after {time.time() - t0:.0f} s; {len(done)} of {len(names)} cases done; '
                              'rerun the same command to continue')
                        return 3
                    row = case_inputs(name) + coefficients(zf, name, scratch) + flags[name]
                    fh.write(','.join([name] + row) + '\n')
                    fh.flush()
                    done[name] = row
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    with open(os.path.join(out, TABLE), 'w') as fh:
        fh.write(f'# {DESCRIPTION}\n# ' + ','.join(COLUMNS) + '\n')
        for name in names:
            fh.write(','.join(done[name]) + '\n')
    with open(os.path.join(out, NAMES), 'w') as fh:
        fh.write('\n'.join(names) + '\n')
    print(f'wrote {TABLE} and {NAMES} in {out}: {len(names)} cases ({time.time() - t0:.0f} s this call)')
    if args.check:
        return check(out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
