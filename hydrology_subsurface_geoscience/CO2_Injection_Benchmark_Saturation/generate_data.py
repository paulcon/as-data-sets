"""Write the data files of CO2_Injection_Benchmark_Saturation from the benchmark's own executable.

The saturations are outputs of the authors' finite-volume code, the Linux executable `prog` of the
benchmark's Zenodo archive (doi:10.5281/zenodo.933827, uqBenchmarkData.zip; no source), run on the
archive's 10,000 input samples. They are stored data: this script records how they were made and can
remake them on Linux x86-64; the shipped files stay the record.

    # 1. run prog on the 10,000 samples in parallel chunks (Linux x86-64; about 8 h on 16 cores)
    python generate_data.py run --archive DIR --work WORK [--jobs 16]
    # 2. assemble the chunks into the shipped files (writes here unless --out is given)
    python generate_data.py assemble --work WORK [--out DIR]
    # compare a finished WORK with the shipped files (writes nothing)
    python generate_data.py assemble --work WORK --check
    # rewrite the two stored CSVs from the unzipped archive (lossless; header and values checked)
    python generate_data.py archive --archive DIR [--out DIR]

DIR is the unzipped archive folder `uqBenchmarkData` (it holds `prog`, `InputParameters.txt`,
`InputParameters.csv` and `MCrefs_MeanStd.csv`). WORK is a scratch folder outside this directory.
`run` splits `InputParameters.txt` into chunks `WORK/chunks/in_00`, `in_01`, ... and runs
`prog -pfile WORK/chunks/in_kk -path WORK/out_kk/ -allS -MC<rows>` for each, with the program's
defaults otherwise (250 cells on [0, 250] m, t = 100 days, test case 5023: uncertain injection rate,
relative permeability degree and porosity). `assemble` reads the per-sample saturation files
(`*__S_0000.dat`, one line of 250 values per sample) in chunk order.

Files written by `assemble`:
  saturation_field.npz  S (10000 x 250, float64, as printed by prog with 17 significant digits)
                        and r (250 cell positions in m, from prog's mean-value file)
  outputs.csv           10000 x 5: S at the cells r = 50.2, 75.3, 100.4, 125.5, 150.6 m (cells 50,
                        75, 100, 125, 150), copied from the field

Only numpy and the standard library are imported; `run` needs the executable.
"""
import argparse
import glob
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = (50, 75, 100, 125, 150)
PROG_SHA256 = 'ca3773a458c753e408cb765beaebd369546e1551182cbad9c13eb564334c51b9'
INPUT_SHA256 = 'c5b9e358ff7f54eca20b183110183ef534aac37135f2a4c1fed1a061fbf0707e'
INSTALL = ("The benchmark executable is not available here. `prog` is a statically linked Linux x86-64 "
           "binary from the Zenodo archive doi:10.5281/zenodo.933827 (uqBenchmarkData.zip); unzip it and "
           "rerun this script on a Linux x86-64 machine with --archive pointing to the uqBenchmarkData folder.")


def read_csv(path):
    """Read a CSV of this directory: '#' lines, then comma-separated numbers."""
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def _save(path, A, header):
    np.savetxt(path, A, fmt='%.17g', delimiter=',', header=header)
    if not np.array_equal(read_csv(path), A):
        raise RuntimeError(f'{path} did not round-trip exactly')


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def archive(src, out):
    """Lossless conversion of the two stored files of the Zenodo archive."""
    C = np.loadtxt(os.path.join(src, 'InputParameters.csv'), delimiter=',', skiprows=1)
    _save(os.path.join(out, 'benchmark_input_samples.csv'), C,
          'Koeppel et al. (2019) benchmark: 10,000 coded input samples, InputParameters.csv of '
          'Zenodo 10.5281/zenodo.933827 (BSD 3-clause), values unchanged\n'
          'ir (injection rate, coded),kd (relative permeability degree, coded),po (porosity, coded)')
    with open(os.path.join(src, 'MCrefs_MeanStd.csv')) as fh:
        cols = fh.readline().strip()
    D = np.loadtxt(os.path.join(src, 'MCrefs_MeanStd.csv'), delimiter=',', skiprows=1)
    _save(os.path.join(out, 'mc_reference_mean_std.csv'), D,
          'Koeppel et al. (2019) Monte Carlo reference (10^4 runs of the authors\' finite-volume '
          'code), CO2 saturation mean (e..) and std (s..) at t = 100 d, MCrefs_MeanStd.csv of '
          'Zenodo 10.5281/zenodo.933827 (BSD 3-clause), values unchanged; x in m; al = joint, '
          'ir/kd/po = only that input uncertain\n' + cols)
    print(f'wrote benchmark_input_samples.csv ({C.shape[0]} rows) and mc_reference_mean_std.csv to {out}')


def run(src, work, jobs):
    prog = os.path.join(src, 'prog')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64') or not os.path.exists(prog):
        print(INSTALL, file=sys.stderr)
        sys.exit(3)
    for path, sha in ((prog, PROG_SHA256), (os.path.join(src, 'InputParameters.txt'), INPUT_SHA256)):
        if _sha256(path) != sha:
            sys.exit(f'{path} does not match the SHA-256 recorded in DATA.md; refusing to run')
    os.chmod(prog, 0o755)
    lines = open(os.path.join(src, 'InputParameters.txt')).read().splitlines()
    size = math.ceil(len(lines) / jobs)
    os.makedirs(os.path.join(work, 'chunks'), exist_ok=True)
    procs = []
    t0 = time.perf_counter()
    for k in range(math.ceil(len(lines) / size)):
        part = lines[k * size:(k + 1) * size]
        fin = os.path.join(work, 'chunks', f'in_{k:02d}')
        with open(fin, 'w') as fh:
            fh.write('\n'.join(part) + '\n')
        dout = os.path.join(work, f'out_{k:02d}')
        os.makedirs(dout, exist_ok=True)
        log = open(os.path.join(work, f'prog_{k:02d}.log'), 'w')
        procs.append(subprocess.Popen([prog, '-pfile', fin, '-path', dout + '/', '-allS', f'-MC{len(part)}'],
                                      stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print(f'{len(procs)} chunks, exit codes {sorted(set(codes))}, {time.perf_counter() - t0:.0f} s')


def assemble_arrays(work):
    """S (10000 x 250) and r (250) from the chunk folders, in chunk order."""
    outs = sorted(glob.glob(os.path.join(work, 'out_[0-9][0-9]')))
    if not outs:
        sys.exit(f'no out_kk folders in {work}')
    S, r = [], None
    for d in outs:
        fs = glob.glob(os.path.join(d, '*__S_0000.dat'))
        fm = [f for f in glob.glob(os.path.join(d, '*_0000.dat')) if '__S_' not in f and '__v_' not in f]
        if len(fs) != 1 or len(fm) != 1:
            sys.exit(f'{d}: expected one per-sample file and one mean file')
        S.append(np.loadtxt(fs[0], ndmin=2))
        rk = np.loadtxt(fm[0])[:, 0]
        if r is None:
            r = rk
        elif not np.array_equal(r, rk):
            sys.exit(f'{d}: grid differs from the first chunk')
    return np.vstack(S), r


def write(out, S, r):
    np.savez_compressed(os.path.join(out, 'saturation_field.npz'), S=S, r=r)
    cols = ','.join(f'S_r{r[c]:.1f} (1)' for c in CELLS)
    _save(os.path.join(out, 'outputs.csv'), S[:, list(CELLS)],
          'CO2 saturation at t = 100 d at cells ' + ', '.join(str(c) for c in CELLS) + ' (r in m in the column '
          'names), copied from saturation_field.npz; outputs of the benchmark executable prog '
          '(Zenodo 10.5281/zenodo.933827) on the 10,000 rows of benchmark_input_samples.csv\n' + cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'assemble', 'archive'))
    ap.add_argument('--archive', help='the unzipped uqBenchmarkData folder')
    ap.add_argument('--work', help='scratch folder of the prog runs (outside this directory)')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--check', action='store_true', help='with assemble: compare with the shipped files')
    a = ap.parse_args()
    if a.work and os.path.abspath(a.work).startswith(HERE):
        ap.error('--work must be outside this directory')
    if a.mode == 'archive':
        if not a.archive:
            ap.error('archive needs --archive DIR')
        archive(a.archive, a.out)
    elif a.mode == 'run':
        if not (a.archive and a.work):
            ap.error('run needs --archive DIR and --work WORK')
        run(a.archive, a.work, a.jobs)
    else:
        if not a.work:
            ap.error('assemble needs --work WORK')
        S, r = assemble_arrays(a.work)
        if S.shape != (10000, 250):
            sys.exit(f'assembled {S.shape}, expected (10000, 250)')
        if a.check:
            tmp = tempfile.mkdtemp()
            try:
                write(tmp, S, r)
                old = np.load(os.path.join(HERE, 'saturation_field.npz'))
                print('saturation_field.npz: S identical:', np.array_equal(old['S'], S),
                      '; r identical:', np.array_equal(old['r'], r))
                print('outputs.csv identical:', np.array_equal(read_csv(os.path.join(tmp, 'outputs.csv')),
                                                               read_csv(os.path.join(HERE, 'outputs.csv'))))
            finally:
                shutil.rmtree(tmp)
        else:
            write(a.out, S, r)
            print(f'wrote saturation_field.npz and outputs.csv ({S.shape[0]} samples) to {a.out}')


if __name__ == '__main__':
    main()
