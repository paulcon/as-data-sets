"""Write runs.csv, the per-run file shipped here, as a column subset of the original per-run file of the OpenFAST runs.

    python convert_data.py ORIGINAL_RUNS_CSV      # writes runs.csv here and checks it
    python convert_data.py ORIGINAL_RUNS_CSV --check   # checks the shipped runs.csv without writing
    python convert_data.py ORIGINAL_RUNS_CSV --fail-logs DIR [--check]   # also failures.csv from the failure logs

The original per-run file (9000 rows, 86 columns, 11.4 MB, SHA-256
a7a8cbc97976ccc7480f2fb9e977903c29c798b2c05d9e8ac37bed55f1a661a2) is larger than the repository's 10 MB
limit for one file. runs.csv keeps the columns the notebook and the scripts use (RUN_COLUMNS) and drops the
per-run mean and standard deviation of the load channels and the maxima of the channels that are not loads.
Every kept field is copied as the same text, so no value changes; the script checks this field by field, and
checks that sets.csv is reproduced from runs.csv. Without the original file it exits with code 3.

With --fail-logs DIR (the directory of the OpenFAST messages of the failed runs, r_<set>_<seed>.log, from the
archive of the runs), it also writes failures.csv: one row per failed run with the simulation time at which
OpenFAST stopped and the cause, read from the fatal message ('tower strike', 'Mach number > 1', else 'other'),
and whether ElastoDyn had warned of a large blade deflection (small-angle assumption violated). It checks that
the logs are exactly the runs whose status in runs.csv is not ok.

Imports the standard library and numpy (through openfast_driver).
"""
import argparse
import csv
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import openfast_driver as drv  # noqa: E402

ORIGINAL_SHA256 = 'a7a8cbc97976ccc7480f2fb9e977903c29c798b2c05d9e8ac37bed55f1a661a2'
FAIL_COLUMNS = ['set', 'seed', 'stop_time_s', 'cause', 'large_deflection_warning']


def failure_rows(logdir):
    """One row per failure log r_<set>_<seed>.log in logdir, sorted by set and seed."""
    rows = []
    for f in os.listdir(logdir):
        m = re.fullmatch(r'r_(\d+)_(\d+)\.log', f)
        if not m:
            continue
        t = open(os.path.join(logdir, f), errors='replace').read()
        tm = re.search(r'encountered an error at simulation time ([\d.]+)', t)
        cause = ('tower strike' if 'Tower strike' in t else
                 'Mach number > 1' if 'Mach number exceeds 1.0' in t else 'other')
        rows.append([int(m.group(1)), int(m.group(2)), tm.group(1) if tm else '', cause,
                     int(re.search(r'Small\s+angle\s+assumption\s+violated', t) is not None)])
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('original', nargs='?', default=None, help='path to the original per-run CSV (86 columns)')
    ap.add_argument('--check', action='store_true', help='compare the shipped runs.csv with the original; write nothing')
    ap.add_argument('--fail-logs', default=None, help='directory of the failure logs r_<set>_<seed>.log')
    args = ap.parse_args()
    if args.original is None or not os.path.exists(args.original):
        print('The original per-run file of the OpenFAST runs is needed (86 columns, SHA-256 '
              f'{ORIGINAL_SHA256}); it is not in the repository. See DATA.md, "Files".', file=sys.stderr)
        sys.exit(3)
    digest = hashlib.sha256(open(args.original, 'rb').read()).hexdigest()
    print(f'original {args.original}: SHA-256 {digest} ({"matches" if digest == ORIGINAL_SHA256 else "DIFFERS FROM"} '
          'the recorded digest)')
    with open(args.original, newline='') as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in drv.RUN_COLUMNS if c not in rows[0]]
    if missing:
        print(f'FAIL: columns missing from the original: {missing}')
        sys.exit(1)
    out = os.path.join(HERE, 'runs.csv')
    if not args.check:
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh, lineterminator='\n')
            w.writerow(drv.RUN_COLUMNS)
            for r in rows:
                w.writerow([r[c] for c in drv.RUN_COLUMNS])
        print(f'wrote {out}: {len(rows)} rows, {len(drv.RUN_COLUMNS)} of {len(rows[0])} columns')
    with open(out, newline='') as fh:
        shipped = list(csv.DictReader(fh))
    bad = len(shipped) != len(rows) or list(shipped[0].keys()) != drv.RUN_COLUMNS
    n_diff = 0 if bad else sum(a[c] != b[c] for a, b in zip(rows, shipped) for c in drv.RUN_COLUMNS)
    print(f'runs.csv against the original: {len(shipped)} rows, {len(drv.RUN_COLUMNS)} columns, '
          f'{n_diff} fields differ (text comparison)')
    sets_rel = drv.compare_sets(drv.aggregate(shipped, drv.read_design(HERE)), os.path.join(HERE, 'sets.csv'))
    print(f'sets.csv recomputed from runs.csv: largest relative difference {sets_rel:.1e}')
    if bad or n_diff:
        print('FAIL: runs.csv is not a column subset of the original')
        sys.exit(1)
    if args.fail_logs is None:
        return
    if not os.path.isdir(args.fail_logs):
        print(f'failure-log directory {args.fail_logs} not found; it is in the archive of the runs (DATA.md)',
              file=sys.stderr)
        sys.exit(3)
    frows = failure_rows(args.fail_logs)
    failed = sorted((int(r['set']), int(r['seed'])) for r in shipped if r['status'] != 'ok')
    if [(r[0], r[1]) for r in frows] != failed:
        print(f'FAIL: the {len(frows)} failure logs are not the {len(failed)} failed runs of runs.csv')
        sys.exit(1)
    fout = os.path.join(HERE, 'failures.csv')
    if not args.check:
        with open(fout, 'w', newline='') as fh:
            w = csv.writer(fh, lineterminator='\n')
            w.writerow(FAIL_COLUMNS)
            w.writerows(frows)
        print(f'wrote {fout}: {len(frows)} rows')
    with open(fout, newline='') as fh:
        same = list(csv.reader(fh)) == [FAIL_COLUMNS] + [[str(v) for v in r] for r in frows]
    causes = {c: sum(r[3] == c for r in frows) for c in sorted({r[3] for r in frows})}
    print(f'failures.csv against the {len(frows)} failure logs: {"identical" if same else "DIFFERENT"}; causes {causes}')
    if not same:
        sys.exit(1)


if __name__ == '__main__':
    main()
