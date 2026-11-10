"""Convert the authors' CovidSim ensemble (Edeling 2023) into this data set's files.

Stored data: the 3000 CovidSim runs were made by W. Edeling on the Altair supercomputer (Poznan
Supercomputing and Networking Center) and published in the MIT-licensed repository
https://github.com/wedeling/deep_active_subspace_data (folder COVID19/). This script only changes
the file layout. It reads

* ``covidsim_samples_only_cont.hdf5``: datasets ``params`` (3000 x 51 inputs) and ``samples``
  (3000 x 801 cumulative death counts, one column per daily output), and
* ``vary.py``: the EasyVVUQ/chaospy input distributions, as text. The file is never imported or
  executed: the ``cp.Uniform(lower, upper)`` entries are parsed with a regular expression and their
  arithmetic (e.g. ``1.5*0.8``) is evaluated by a small AST evaluator that accepts only numbers and
  + - * /.

It checks the SHA-256 of both files and writes

* ``covidsim_inputs.csv``: the 51 inputs of every run, each value written with Python's shortest
  round-trip repr (lossless);
* ``covidsim_input_ranges.csv``: two rows, the lower and upper bounds of the 51 uniform inputs;
* ``covidsim_deaths_day801.csv``: the last column of ``samples`` (the quantity of interest);
* ``covidsim_cumulative_deaths_daily.npz``: the whole ``samples`` array (float64, unchanged) as
  ``deaths``, plus ``day`` = 1..801 (the index of each daily output column).

Usage (``--out`` is required and is never the data set's own directory unless given explicitly):

    python convert_data.py <folder with the two files> --out DIR           # write the four files
    python convert_data.py <folder with the two files> --check --out DIR   # write to DIR, compare

``--check`` writes into DIR (which must be outside this directory) and compares every value with
the shipped files, with tolerance: maximum relative difference <= 1e-15 (the conversion is exact,
so the printed differences are 0).

Needs numpy and h5py (tested with h5py 3.16.0).
"""

import argparse
import ast
import hashlib
import operator
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
SHA256 = {
    'covidsim_samples_only_cont.hdf5': '09182744645145a4c1769b2ae92c4d12a474f12d8a89052d7c8210d69226935f',
    'vary.py': '5b3611b7ef7f6f2cefeff0af071b786d84abb7409d54a70d17e854db8c9f785d',
}
TOL = 1e-15

# Units: CovidSim parameter-file conventions (days for periods and delays, metres for kernel scales);
# "1" = dimensionless factor, rate ratio, proportion or exponent. The article does not list units.
UNITS = {'Latent_period': 'day', 'Delay_from_end_of_latent_period_to_start_of_symptoms': 'day'}
for _k in ['Kernel_scale'] + [f'Kernel_scale_params_for_place_types{_i}' for _i in range(4)]:
    UNITS[_k] = 'm'

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _arith(text):
    """Evaluate a numeric expression with + - * / only (no names, no calls)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f'unsupported expression in vary.py: {text!r}')
    return ev(ast.parse(text.strip(), mode='eval'))


def parse_vary(path):
    """(names, lower, upper) of the active (uncommented) cp.Uniform entries, in file order."""
    names, lo, hi = [], [], []
    pat = re.compile(r'^\s*"(\w+)"\s*:\s*cp\.Uniform\(([^,]+),\s*([^)]+)\)\s*,?\s*$')
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith('#'):
                continue
            mt = pat.match(line)
            if mt:
                names.append(mt.group(1))
                lo.append(_arith(mt.group(2)))
                hi.append(_arith(mt.group(3)))
            elif 'cp.' in line:
                raise ValueError(f'unparsed distribution line in vary.py: {line.strip()}')
    return names, np.array(lo), np.array(hi)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(src):
    try:
        import h5py
    except ImportError:
        sys.exit('convert_data.py needs h5py (tested with 3.16.0): pip install h5py, then rerun.')
    for name, want in SHA256.items():
        got = sha256(os.path.join(src, name))
        if got != want:
            raise ValueError(f'{name}: SHA-256 {got} differs from the recorded {want}')
    names, lo, hi = parse_vary(os.path.join(src, 'vary.py'))
    with h5py.File(os.path.join(src, 'covidsim_samples_only_cont.hdf5'), 'r') as fh:
        X = fh['params'][()]
        D = fh['samples'][()]
    if X.shape != (3000, 51) or D.shape != (3000, 801) or len(names) != 51:
        raise ValueError(f'unexpected shapes {X.shape} {D.shape} {len(names)}')
    if np.any(X < lo) or np.any(X > hi):
        raise ValueError('a stored input lies outside its vary.py range (column order?)')
    cols = ','.join(f'{n} [{UNITS.get(n, "1")}]' for n in names)
    rows = '\n'.join(','.join(repr(float(v)) for v in r) for r in X)
    inputs = ('# Edeling 2023 CovidSim ensemble: 51 uniform inputs of 3000 Monte Carlo runs, physical units, '
              'column order of vary.py (github.com/wedeling/deep_active_subspace_data COVID19/'
              'covidsim_samples_only_cont.hdf5 params; values unchanged)\n# ' + cols + '\n' + rows + '\n')
    ranges = ('# Lower (row 1) and upper (row 2) bounds of the uniform input distributions, from the '
              'cp.Uniform entries of COVID19/vary.py\n# ' + cols + '\n'
              + ','.join(repr(float(v)) for v in lo) + '\n' + ','.join(repr(float(v)) for v in hi) + '\n')
    qoi = ('# Edeling 2023 CovidSim: cumulative deaths after 801 simulated days (last of 801 daily '
           'outputs, HDF5 samples[:, -1]); row i matches row i of covidsim_inputs.csv\n'
           '# deaths_day801 [persons]\n' + '\n'.join(repr(float(v)) for v in D[:, -1]) + '\n')
    return {'covidsim_inputs.csv': inputs, 'covidsim_input_ranges.csv': ranges,
            'covidsim_deaths_day801.csv': qoi}, D


def write(out, texts, D):
    os.makedirs(out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(out, name), 'w') as fh:
            fh.write(text)
    np.savez_compressed(os.path.join(out, 'covidsim_cumulative_deaths_daily.npz'),
                        deaths=D, day=np.arange(1, D.shape[1] + 1))


def _maxrel(a, b):
    if a.shape != b.shape:
        return np.inf
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))


def check(out):
    bad = 0
    for name in ['covidsim_inputs.csv', 'covidsim_input_ranges.csv', 'covidsim_deaths_day801.csv']:
        new = np.loadtxt(os.path.join(out, name), delimiter=',')
        old = np.loadtxt(os.path.join(HERE, name), delimiter=',')
        d = _maxrel(new, old)
        print(f'{name}: max relative difference {d:.1e} (tolerance {TOL:.0e})')
        bad += not d <= TOL
    with np.load(os.path.join(out, 'covidsim_cumulative_deaths_daily.npz')) as a, \
            np.load(os.path.join(HERE, 'covidsim_cumulative_deaths_daily.npz')) as b:
        for k in ['deaths', 'day']:
            d = _maxrel(a[k].astype(float), b[k].astype(float))
            print(f'covidsim_cumulative_deaths_daily.npz[{k}]: max relative difference {d:.1e} '
                  f'(tolerance {TOL:.0e})')
            bad += not d <= TOL
    return int(bad > 0)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src', help='folder with covidsim_samples_only_cont.hdf5 and vary.py')
    ap.add_argument('--out', help='output folder (required)')
    ap.add_argument('--check', action='store_true', help='write to --out and compare with the shipped files')
    a = ap.parse_args(argv)
    if not a.out:
        ap.error('--out DIR is required')
    out = os.path.realpath(a.out)
    if a.check and os.path.commonpath([out, HERE]) == HERE:
        ap.error('--check needs --out outside the data set directory')
    texts, D = build(a.src)
    write(out, texts, D)
    print(f'wrote 4 files to {"the data set directory" if out == HERE else "--out"}: {D.shape[0]} runs')
    return check(out) if a.check else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
