"""Regenerate the files of Wind_Turbine_Inflow_Loads: the design, the per-set values and, with TurbSim and OpenFAST, the runs.

    python generate_data.py                  # writes design.csv, seeds.json, params.json here
    python generate_data.py --out DIR        # the same and sets.csv, recomputed from runs.csv, in DIR
    python generate_data.py --check          # regenerates into a temporary directory and compares (subset of the runs)
    python generate_data.py --check --full   # the same with all 9000 runs (about 400 CPU-hours)
    python generate_data.py --runs A B --out DIR   # runs sets [A, B) x 30 seeds into DIR/runs.csv

What can be regenerated without the outside codes (numpy only, a few seconds):
  design.csv, seeds.json and params.json, from the seed in openfast_driver.SEED, and
  sets.csv, from the shipped runs.csv (openfast_driver.aggregate; the seeds are summed in another order than
  on the machines that made the runs, so the values agree to round-off).
The runs need TurbSim and OpenFAST v3.5.5, built from git tag v3.5.5 with patch_turbsim.py applied to
TurbSim, and the r-test inputs of that tag. Give their locations in the environment:
  TURBSIM   the patched turbsim executable
  OPENFAST  the openfast executable
  OPENFAST_RTEST  reg_tests/r-test/glue-codes/openfast of the v3.5.5 checkout
  DISCON    the DISCON controller library built from the same tag (target regression_test_controllers)
When any is missing, the script does what it can, prints what to install, and exits with code 3.

--check reruns RUNS_CHECK (set 0, seeds 0 and 1; about 3 minutes on two cores) and compares the
per-run statistics with the rows of runs.csv. It prints the largest difference per file and one line
`largest difference from the shipped files: ...`; it fails (exit code 1) only on a structural mismatch:
a missing file, a different shape, status or NaN pattern, or regenerated inputs that differ from the
shipped ones by more than 1e-8 of each input's largest magnitude.

Runs are parallel over processes, each with one BLAS thread (--nproc, default the number of CPUs).
"""
import argparse
import atexit
import csv
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import openfast_driver as drv  # noqa: E402

INPUT_RTOL = 1e-8
RUNS_CHECK = [(0, 0), (0, 1)]
ENV = ('TURBSIM', 'OPENFAST', 'OPENFAST_RTEST', 'DISCON')
INSTALL = ('The OpenFAST runs need TurbSim and OpenFAST v3.5.5 (https://github.com/OpenFAST/openfast, git tag '
           'v3.5.5), with patch_turbsim.py applied to modules/turbsim/src/TSsubs.f90 before building, and the '
           'r-test inputs of that tag (git submodule reg_tests/r-test). Set TURBSIM and OPENFAST to the '
           'executables, OPENFAST_RTEST to reg_tests/r-test/glue-codes/openfast and DISCON to the controller '
           'library built with the target regression_test_controllers.')


def outside_codes():
    """The four locations from the environment, or None if any is missing."""
    loc = {k: os.environ.get(k) for k in ENV}
    if all(v and os.path.exists(v) for v in loc.values()):
        return loc
    return None


def run(tasks, out, loc, nproc):
    """Run (set, seed) tasks; write out/runs.csv with RUN_COLUMNS; return the rows."""
    from concurrent.futures import ProcessPoolExecutor
    X = drv.read_design(out)[:, 1:]
    seeds = json.load(open(os.path.join(out, 'seeds.json')))['turbsim_RandSeed1']
    work = tempfile.mkdtemp(prefix='openfast_')
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    drv.make_template(loc['OPENFAST_RTEST'], work, loc['DISCON'])
    args = [(s, k, X[s].tolist(), seeds[k], work, loc['TURBSIM'], loc['OPENFAST']) for s, k in tasks]
    t0 = time.time()
    with ProcessPoolExecutor(nproc) as ex:
        rows = list(ex.map(drv.run_task, args))
    print(f'{len(rows)} runs in {time.time() - t0:.0f} s; status '
          f'{ {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})} }')
    with open(os.path.join(out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, drv.RUN_COLUMNS, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def compare_design(out):
    """Largest difference of the regenerated design from design.csv, as a fraction of each input's largest magnitude."""
    a = drv.read_design(out)
    b = drv.read_design(HERE)
    if a.shape != b.shape:
        print(f'FAIL: design.csv shape {a.shape} differs from the shipped {b.shape}')
        return None
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b).max(axis=0), 1e-300)))
    same = [f for f in ('seeds.json', 'params.json')
            if json.load(open(os.path.join(out, f))) == json.load(open(os.path.join(HERE, f)))]
    print(f'design.csv: largest difference {rel:.1e} (of each input\'s largest magnitude); '
          f'{", ".join(same)} identical' + ('' if len(same) == 2 else '; FAIL: seeds.json or params.json differs'))
    return rel if len(same) == 2 else None


def compare_runs(rows, shipped):
    """Largest relative difference of rerun statistics from the shipped rows; None on a structural mismatch."""
    idx = {(int(r['set']), int(r['seed'])): r for r in shipped}
    worst = 0.0
    for r in rows:
        s = idx[(int(r['set']), int(r['seed']))]
        if r['status'] != s['status']:
            print(f'FAIL: run ({r["set"]}, {r["seed"]}) status {r["status"]}, shipped {s["status"]}')
            return None
        if int(r['nugget_n']) != int(s['nugget_n']):
            print(f'FAIL: run ({r["set"]}, {r["seed"]}) nugget_n {r["nugget_n"]}, shipped {s["nugget_n"]}')
            return None
        for c in drv.RUN_COLUMNS[7:]:
            if s[c] == '' or c not in r:
                continue
            a, b = float(r[c]), float(s[c])
            worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--out', default=None, help='output directory (default: here; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare with the shipped files')
    ap.add_argument('--full', action='store_true', help='with --check: rerun all 9000 runs')
    ap.add_argument('--runs', nargs=2, type=int, metavar=('A', 'B'), help='run sets [A, B) x 30 seeds')
    ap.add_argument('--nproc', type=int, default=os.cpu_count(), help='worker processes for the runs')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    out_is_here = os.path.abspath(args.out) == HERE
    if args.check and out_is_here:
        ap.error('--check needs --out DIR outside this directory')
    if args.runs and out_is_here:
        ap.error('--runs needs --out DIR outside this directory, so that the shipped runs.csv is not overwritten')
    if args.full and not args.check:
        ap.error('--full goes with --check')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    drv.write_design(args.out)
    shipped = list(csv.DictReader(open(os.path.join(HERE, 'runs.csv'), newline='')))
    agg = drv.aggregate(shipped, drv.read_design(args.out))
    if out_is_here:
        # sets.csv here is the record written on the machines that made the runs; a rerun of the aggregation
        # sums the seeds in another order, so it is compared (--check) rather than written over
        print(f'wrote design.csv, seeds.json and params.json in {args.out} in {time.perf_counter() - t0:.1f} s; '
              'sets.csv (the record of the runs) is left as it is: use --out DIR to write the recomputed one')
    else:
        drv.write_sets(os.path.join(args.out, 'sets.csv'), agg)
        print(f'wrote design.csv, seeds.json, params.json and sets.csv ({len(agg)} sets from the {len(shipped)} rows '
              f'of runs.csv) in {args.out} in {time.perf_counter() - t0:.1f} s')

    if not args.check and not args.runs:
        return
    results, bad = [], False
    if args.check:
        rel = compare_design(args.out)
        if rel is None or rel > INPUT_RTOL:
            bad = True
        else:
            results.append((rel, 'design.csv (inputs)'))
        srel = drv.compare_sets(agg, os.path.join(HERE, 'sets.csv'))
        print(f'sets.csv: largest relative difference {srel:.1e} (recomputed from runs.csv)')
        if not np.isfinite(srel):
            print('FAIL: sets.csv has other columns or rows')
            bad = True
        else:
            results.append((srel, 'sets.csv'))

    loc = outside_codes()
    if loc is None:
        missing = [k for k in ENV if not (os.environ.get(k) and os.path.exists(os.environ[k]))]
        if results:
            print(f'largest difference from the shipped files: {max(results)[0]:.1e} (relative, {max(results)[1]}); '
                  'runs.csv not checked')
        if bad:
            sys.exit(1)
        print(f'runs not rerun: {", ".join(missing)} not set or not found. {INSTALL}', file=sys.stderr)
        sys.exit(3)
    if args.runs:
        A, B = args.runs
        run([(s, k) for s in range(A, B) for k in range(drv.NSEEDS)], args.out, loc, args.nproc)
        return
    tasks = ([(s, k) for s in range(drv.NSETS) for k in range(drv.NSEEDS)] if args.full else RUNS_CHECK)
    print(f'rerunning {len(tasks)} runs: ' + ('all' if args.full else ', '.join(f'set {s} seed {k}' for s, k in tasks)))
    rows = run(tasks, args.out, loc, args.nproc)
    rrel = compare_runs(rows, shipped)
    if rrel is None:
        bad = True
    else:
        print(f'runs.csv: largest relative difference {rrel:.1e} over {len(rows)} rerun rows (statistics columns)')
        results.append((rrel, 'runs.csv'))
    if results:
        print(f'largest difference from the shipped files: {max(results)[0]:.1e} (relative, {max(results)[1]})')
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
