"""Convert the stored PyPSA-Eur runs of Neumann & Brown (2023) into this data set's files.

Stored data: the values were computed by the authors (PyPSA-Eur capacity-expansion LPs solved with
Gurobi on a cluster) and committed to their repository https://github.com/fneum/broad-ranges
(results/ are CC BY 4.0 by the repository's .reuse/dep5). This script only changes the layout. It reads
four files of a clone of that repository at commit b95bbc98 (the files are unchanged since tag v0.1.0,
2022-06-14), checks their SHA-256, and writes

  least_cost_low.csv   the 500 least-cost runs of the low-fidelity model (37 nodes, 4-hourly)
  least_cost_high.csv  the 15 least-cost runs of the high-fidelity model (128 nodes, 2-hourly)
  all_runs.npz         all 50,500 low- and 1,515 high-fidelity runs (least-cost and near-optimal)
  pce_least_cost.csv   the authors' fitted least-cost polynomial chaos surrogates (low-fidelity and
                       multi-fidelity), monomial exponents and coefficients

The CSV files copy every number of the source character for character after two '#' header lines, so
np.loadtxt(path, delimiter=',') reads them. all_runs.npz stores each numeric field as float64 (every
source string is the shortest repr of its float64, which the script verifies, so the text is
recoverable) and each text field (sense, objective, fixed, position) as integer codes with their labels.
The source column 'carrier' is empty in every run and is not stored.

    python convert_data.py <clone of fneum/broad-ranges> --out DIR            # write the four files to DIR
    python convert_data.py <clone of fneum/broad-ranges> --out DIR --check    # ... and compare with the
                                                                               # shipped files
--out is required and must be a folder outside this directory, so the shipped files are never
overwritten. The comparison tolerance is a relative difference of 1e-15 per value (the
conversion is exact; the tolerance only guards the float comparison). numpy and the standard library only.
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-15

SHA256 = {
    'results/dataset_low.csv': 'a1087ace71e403bb7e67b8c3c45217def3ad00585f695b98ac07a013f2640f30',
    'results/dataset_high.csv': 'd1147a893048f4ae6877b99aebcc7d8dc99431b3727c73c881851a93b7ad7ca5',
    'results/pce/polynomial-low-min-cost-0.0.txt':
        '0cb4c12ae5f1edd5a34ec30219f72cc6aebc3237d7ca35ab460d0395c84ff7fe',
    'results/pce/polynomial-high-min-cost-0.0.txt':
        '5d6a6f7082c80c60521116743916c5b81f16e6495153aa380e2fd139b9f7066d',
}

# Inputs: capital-cost multipliers relative to the pessimistic DEA 2050 annuity (config.yaml
# 'uncertainties'); source field -> column name.
INPUTS = [('onwind-cost', 'f_onwind'), ('offwind-cost', 'f_offwind'), ('solar-cost', 'f_solar'),
          ('H2-cost', 'f_H2'), ('battery-cost', 'f_battery')]
# Outputs (units from scripts/collect.py of the source): capacities in GW, transmission volume in TWkm
# (line length km x capacity MW / 1e6), total system cost in bn EUR/a, a Gini coefficient (1).
OUTPUTS = [('onwind', 'onwind', 'GW'), ('offwind', 'offwind', 'GW'), ('wind', 'wind', 'GW'),
           ('solar', 'solar', 'GW'), ('H2', 'H2', 'GW'), ('battery', 'battery', 'GW'),
           ('transmission', 'transmission', 'TWkm'), ('tsc', 'tsc', 'bn EUR/a'), ('gini', 'gini', '1')]
TEXT = ['sense', 'objective', 'epsilon', 'fixed', 'position']
CATEGORIES = {
    'sense': ['min', 'max'],
    'objective': ['cost', 'onwind', 'offwind', 'wind', 'solar', 'H2', 'battery', 'transmission'],
    'fixed': ['none', 'onwind', 'solar', 'battery'],
}
FIDELITY = {'low': '37 nodes, 4-hourly, transport model', 'high': '128 nodes, 2-hourly, linearised load flow'}
# numpoly variable order q0..q4 of the polynomial files = order of 'uncertainties' in config.yaml
PCE_VARS = ['onwind', 'offwind', 'solar', 'H2', 'battery']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_transposed(path):
    """The source CSVs hold one row per field: field name, then one value per run."""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    data = {r[0]: r[1:] for r in rows}
    n = {len(v) for v in data.values()}
    if len(n) != 1:
        raise ValueError(f'{path}: fields of unequal length {sorted(n)}')
    expected = [a for a, _ in INPUTS] + TEXT + ['carrier'] + [a for a, _, _ in OUTPUTS]
    if sorted(data) != sorted(expected):
        raise ValueError(f'{path}: fields {sorted(data)} differ from the expected {sorted(expected)}')
    if any(v != '' for v in data['carrier']):
        raise ValueError(f'{path}: the field carrier is not empty')
    return data, n.pop()


def floats(strings, what):
    vals = np.array([float(s) for s in strings])
    bad = sum(repr(float(s)) != s for s in strings)
    if bad:
        raise ValueError(f'{what}: {bad} strings are not the shortest repr of their float64')
    return vals


def least_cost_csv(data, n, fid):
    rows = [i for i in range(n) if data['objective'][i] == 'cost']
    for i in rows:
        if (data['sense'][i], data['epsilon'][i], data['fixed'][i], data['position'][i]) != \
                ('min', '0.0', 'none', 'none'):
            raise ValueError(f'least-cost run {i} has unexpected settings')
    cols = [f'{b} [1]' for _, b in INPUTS] + [f'{b} [{u}]' for _, b, u in OUTPUTS]
    desc = (f'Neumann & Brown 2023 PyPSA-Eur least-cost runs, {fid}-fidelity model ({FIDELITY[fid]}): '
            f'{len(rows)} Halton samples of 5 capital-cost multipliers and 9 outputs '
            f'(github.com/fneum/broad-ranges results/dataset_{fid}.csv, runs with objective cost; values unchanged)')
    body = [','.join([data[a][i] for a, _ in INPUTS] + [data[a][i] for a, _, _ in OUTPUTS]) for i in rows]
    return f'# {desc}\n# ' + ','.join(cols) + '\n' + '\n'.join(body) + '\n', len(rows)


def npz_arrays(data, n, fid):
    out = {}
    out[f'{fid}_inputs'] = np.column_stack([floats(data[a], f'{fid} {a}') for a, _ in INPUTS])
    out[f'{fid}_outputs'] = np.column_stack([floats(data[a], f'{fid} {a}') for a, _, _ in OUTPUTS])
    out[f'{fid}_epsilon'] = floats(data['epsilon'], f'{fid} epsilon')
    pos = data['position']
    out[f'{fid}_position'] = np.array([np.nan if p == 'none' else float(p) for p in pos])
    if any(p != 'none' and repr(float(p)) != p for p in pos):
        raise ValueError(f'{fid} position: a value is not the repr of its float')
    for k, labels in CATEGORIES.items():
        unknown = set(data[k]) - set(labels)
        if unknown:
            raise ValueError(f'{fid} {k}: unknown labels {unknown}')
        out[f'{fid}_{k}'] = np.array([labels.index(v) for v in data[k]], dtype=np.int8)
    return out


def pce_csv(src):
    """Exponents and coefficients of the authors' least-cost surrogates, text copied unchanged."""
    blocks = {}
    for fid in ['low', 'high']:
        with open(os.path.join(src, f'results/pce/polynomial-{fid}-min-cost-0.0.txt')) as fh:
            lines = fh.read().splitlines()
        head = lines[0]
        if not head.startswith('# numpoly:') or 'names:q0,q1,q2,q3,q4' not in head:
            raise ValueError(f'{fid} polynomial: unexpected header {head[:60]}')
        keys = head.split('keys:')[1].split(' ')[0].split(',')
        exps = [[ord(c) - ord(';') for c in k] for k in keys]
        if any(len(e) != 5 or min(e) < 0 for e in exps):
            raise ValueError(f'{fid} polynomial: bad key')
        names = lines[1][2:].split(' ')
        coef = [ln.split(' ') for ln in lines[2:] if ln.strip()]
        if names != ['onwind', 'solar', 'H2', 'battery', 'transmission', 'offwind', 'wind', 'tsc', 'gini']:
            raise ValueError(f'{fid} polynomial: outputs {names}')
        if len(coef) != len(names) or any(len(c) != len(keys) for c in coef):
            raise ValueError(f'{fid} polynomial: shape')
        [float(v) for c in coef for v in c]
        blocks[fid] = (exps, names, coef)
    if blocks['low'][0] != blocks['high'][0]:
        raise ValueError('low and high polynomials have different monomials')
    exps, names, _ = blocks['low']
    order = [names.index(a) for a, _, _ in OUTPUTS if a != 'gini']
    cols = [f'e_{v} [1]' for v in PCE_VARS]
    for fid in ['low', 'high']:
        cols += [f'{fid}_{names[j]} [{dict((a, u) for a, _, u in OUTPUTS)[names[j]]}]' for j in order]
    rows = []
    for k, e in enumerate(exps):
        vals = [str(x) for x in e]
        for fid in ['low', 'high']:
            vals += [blocks[fid][2][j][k] for j in order]
        rows.append(','.join(vals))
    desc = ('Neumann & Brown 2023 least-cost sparse PCE surrogates (results/pce/polynomial-{low,high}-min-cost-0.0.txt, '
            'coefficients unchanged): row = monomial prod_i f_i^e_i in the raw multipliers f; low = low-fidelity order-3 '
            'Lasso fit, high = low + linear high-fidelity correction (gini not copied)')
    return f'# {desc}\n# ' + ','.join(cols) + '\n' + '\n'.join(rows) + '\n'


def build(src):
    for rel, want in SHA256.items():
        got = sha256(os.path.join(src, rel))
        if got != want:
            raise ValueError(f'{rel}: SHA-256 {got} differs from the recorded {want}')
    texts, arrays = {}, {}
    for fid in ['low', 'high']:
        data, n = read_transposed(os.path.join(src, f'results/dataset_{fid}.csv'))
        texts[f'least_cost_{fid}.csv'], nlc = least_cost_csv(data, n, fid)
        arrays.update(npz_arrays(data, n, fid))
        print(f'{fid}: {n} runs, {nlc} least-cost')
    arrays['input_names'] = np.array([b for _, b in INPUTS])
    arrays['output_names'] = np.array([b for _, b, _ in OUTPUTS])
    arrays['output_units'] = np.array([u for _, _, u in OUTPUTS])
    for k, labels in CATEGORIES.items():
        arrays[f'{k}_labels'] = np.array(labels)
    texts['pce_least_cost.csv'] = pce_csv(src)
    return texts, arrays


def compare(texts, arrays):
    bad = 0
    for name, text in texts.items():
        new = np.loadtxt(text.splitlines(), delimiter=',')
        with open(os.path.join(HERE, name)) as fh:
            old_text = fh.read()
        old = np.loadtxt(os.path.join(HERE, name), delimiter=',')
        ok = new.shape == old.shape and np.all(np.abs(new - old) <= TOL * np.maximum(1.0, np.abs(old)))
        print(f'{name}: {"same within tolerance" if ok else "DIFFERENT"}; text identical: {text == old_text}')
        bad += not ok
    with np.load(os.path.join(HERE, 'all_runs.npz')) as z:
        for k, v in arrays.items():
            w = z[k]
            if v.dtype.kind in 'fi':
                ok = v.shape == w.shape and np.all((np.abs(v - w) <= TOL * np.maximum(1.0, np.abs(w)))
                                                   | (np.isnan(v) & np.isnan(w)))
            else:
                ok = v.shape == w.shape and np.all(v == w)
            bad += not ok
            if not ok:
                print(f'all_runs.npz[{k}]: DIFFERENT')
        extra = set(z.files) ^ set(arrays)
    print(f'all_runs.npz: {len(arrays)} arrays, {"same within tolerance" if not bad and not extra else "DIFFERENT"}')
    return int(bad > 0 or bool(extra))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('src', help='clone of github.com/fneum/broad-ranges (commit b95bbc98 or tag v0.1.0)')
    p.add_argument('--out', required=True, help='folder to write the converted files to')
    p.add_argument('--check', action='store_true', help='compare the converted files with the shipped ones')
    a = p.parse_args()
    out = os.path.realpath(a.out)
    if out == os.path.realpath(HERE) or out.startswith(os.path.realpath(HERE) + os.sep):
        sys.exit('--out must be a folder outside this data set directory, so that the shipped files are never '
                 'overwritten; to replace them, copy the files from DIR after a passing --check')
    texts, arrays = build(a.src)
    os.makedirs(a.out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(a.out, name), 'w') as fh:
            fh.write(text)
    np.savez_compressed(os.path.join(a.out, 'all_runs.npz'), **arrays)
    print(f'wrote {", ".join(texts)}, all_runs.npz to {a.out}')
    if a.check:
        return compare(texts, arrays)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        sys.exit(f'file not found, or not of the expected kind: {err.filename}\n'
                 'The source is a clone of github.com/fneum/broad-ranges (commit b95bbc98); see DATA.md, "Regenerating".')
