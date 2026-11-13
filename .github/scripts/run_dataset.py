"""Run a data set's notebook and scripts the way a reviewer would.

    python .github/scripts/run_dataset.py <category>/<Name>
    python .github/scripts/run_dataset.py <category>/<Name> --skip-scripts

From inside the data set's directory, this:

1. executes the notebook into a temporary directory (never in place), with a 30-minute limit;
2. runs `python generate_data.py --check` when there is a generate_data.py, and reports its
   summary line ("largest difference from the shipped files: ...") as a notice;
3. runs `python check_accuracy.py` when there is one.

A script that exits with code 3 needs an outside code, a download or a package that is not
installed; it is reported as skipped, with its message, and does not fail the run. Any other
non-zero exit (1 is a structural mismatch or an error), or a run over the limit, fails. Data sets and scripts that cannot run in CI are
in SKIP below, with the reason. Standard library only.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 1800        # seconds, for the notebook and for each script
EXIT_MISSING = 3      # a script's exit code for "an outside code or download is missing"
SUMMARY = 'largest difference from the shipped files:'   # generate_data.py --check's last line

# Data sets, or one of their parts, that cannot run in CI: {directory: {part: reason}}, where a
# part is 'notebook', 'generate_data.py' or 'check_accuracy.py'.
SKIP = {
}

GITHUB = os.environ.get('GITHUB_ACTIONS') == 'true'


def notice(kind, msg):
    print(f'::{kind}::{msg}' if GITHUB else f'{kind}: {msg}', flush=True)


def run(cmd, cwd, env, label):
    """Run cmd; return 'ok', 'skipped' or 'failed'."""
    print(f'--- {label}: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, timeout=TIMEOUT, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        notice('error', f'{label} did not finish within {TIMEOUT // 60} minutes')
        return 'failed'
    dt = time.time() - t0
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-6000:], flush=True)
    if r.returncode == 0:
        print(f'--- {label}: ok in {dt:.0f} s', flush=True)
        if label == 'generate_data.py':
            lines = [x.strip() for x in out.splitlines() if SUMMARY in x]
            if lines:
                notice('notice', f'generate_data.py --check: {lines[-1]}')
        return 'ok'
    if r.returncode == EXIT_MISSING and label != 'notebook':
        last = out.splitlines()[-1] if out else ''
        notice('notice', f'{label} skipped: it needs something that is not installed. {last}')
        return 'skipped'
    notice('error', f'{label} failed with exit code {r.returncode} after {dt:.0f} s')
    return 'failed'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('directory', help='the data set, e.g. uq_benchmarks/Golinski_Speed_Reducer')
    ap.add_argument('--skip-scripts', action='store_true',
                    help='run only the notebook')
    args = ap.parse_args()

    d = args.directory.strip('/')
    cwd = ROOT / d
    if not cwd.is_dir():
        sys.exit(f'{d} is not a directory')
    skip = SKIP.get(d, {})
    env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0')
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        nb = f'{cwd.name}.ipynb'
        parts = [('notebook', ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', nb,
                               '--output-dir', tmp, '--output', 'out.ipynb',
                               f'--ExecutePreprocessor.timeout={TIMEOUT}'])]
        if not args.skip_scripts:
            if (cwd / 'generate_data.py').is_file():
                parts.append(('generate_data.py', [sys.executable, 'generate_data.py', '--check']))
            if (cwd / 'check_accuracy.py').is_file():
                parts.append(('check_accuracy.py', [sys.executable, 'check_accuracy.py']))
        for label, cmd in parts:
            if label in skip:
                notice('notice', f'{d}: {label} is not run in CI: {skip[label]}')
                results[label] = 'skipped'
                continue
            results[label] = run(cmd, cwd, env, label)

    print('=== ' + d + ': ' + ', '.join(f'{k} {v}' for k, v in results.items()), flush=True)
    return 1 if 'failed' in results.values() else 0


if __name__ == '__main__':
    sys.exit(main())
