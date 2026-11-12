"""Convert the epoxy-resin MD data of Edeling et al. (2024) into this data set's CSV files.

Stored data: the values were computed by the authors (LAMMPS, ELASTIC procedure, 20 replicas per input
point, on SuperMUC-NG) and published in their repository wedeling/MD-active-subspace, tag 1.1 (MIT
licence; archived at Zenodo, doi:10.5281/zenodo.10817481). This script only changes the file layout:
it checks the SHA-256 of the source files, then writes each data row's text unchanged after two ``#``
header lines (a description; column names with units), so ``np.loadtxt(path, delimiter=',')`` reads
the files. Every number is copied character for character. The default values of the 103 sampled
force-field parameters are taken from ``epoxy.params.json`` (``repr`` of the JSON floats, which
round-trips exactly).

    python convert_data.py <epoxy folder of the source> --out DIR          # write the eight CSVs into DIR
    python convert_data.py <epoxy folder of the source> --out DIR --check  # also compare DIR with the shipped files

``<epoxy folder of the source>`` is the ``epoxy/`` folder of a clone of
https://github.com/wedeling/MD-active-subspace at tag 1.1 (or of the extracted Zenodo archive). ``DIR``
must be outside this data set's directory, so the shipped files are never overwritten. ``--check``
compares every value with the shipped file (largest relative difference, tolerance 1e-15) and reports
whether the texts are identical.

Uses numpy (for the check) and the standard library.
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TOL = 1e-15

SHA256 = {
    'data_epoxy/epoxy_inputs.csv': 'ab1989deb15ef572eb03bc2d38a338daa881e2e12743540d79ab912eb908fab4',
    'data_epoxy/epoxy_E11_output.csv': 'c142ba216c4a42eeafca8f16bbf998fcbc3b9081797bbbec890ce7f1e57ea6e3',
    'data_epoxy/epoxy_E22_output.csv': 'fd943c60baab60a3f29d6da9d74e7925838781efebbabc034feaba7334cc9033',
    'data_epoxy/epoxy_E33_output.csv': 'da6c15697ba228e046ccebb5db517f5a6560bd5bc954f5bda6fbbdb81ca4d522',
    'data_epoxy/epoxy_poisson_output.csv': '9495fdd7eb052a6fe0e4330ac54aac26d49249eb8e1bfc45ecee6d45ece650b2',
    'data_epoxy/epoxy_bulk_output.csv': '090c559dcc6b50c73555dbfb248fece9ac466831c5757dbd3fc91715c76c1195',
    'data_epoxy/epoxy_shear2_output.csv': '90598ade21e4c9be92d229a0302fd4a271981024632f7676e37b5461ba66f44a',
    'epoxy.params.json': '68feaa1a31389fe75bcfa5a51d8773c1d99d098f426914a73f72bc64e607fc2a',
}

# Units of the force-field coefficients (LAMMPS "units real"; pair lj/cut/coul/long, bond and angle
# harmonic, dihedral opls; molecular_models/epoxy_elasticity/simulation.in of the source). The role of
# each name (second digit) is inferred from the defaults, which equal the coefficient sets of epoxy.data.
UNITS = {
    ('p', '1'): 'kcal/mol', ('p', '2'): 'Angstrom',
    ('b', '1'): 'kcal/mol/Angstrom^2', ('b', '2'): 'Angstrom',
    ('a', '1'): 'kcal/mol/rad^2', ('a', '2'): 'deg',
    ('d', '1'): 'kcal/mol', ('d', '2'): 'kcal/mol', ('d', '3'): 'kcal/mol', ('d', '4'): 'kcal/mol',
}

OUTPUTS = [
    # (source file, shipped file, unit, description)
    ('epoxy_E11_output.csv', 'epoxy_E11_replicas.csv', 'Pa',
     "Young's modulus E = C11 component used by the authors (E11)"),
    ('epoxy_E22_output.csv', 'epoxy_E22_replicas.csv', 'Pa', 'E22 component'),
    ('epoxy_E33_output.csv', 'epoxy_E33_replicas.csv', 'Pa', 'E33 component'),
    ('epoxy_poisson_output.csv', 'epoxy_poisson_replicas.csv', '1', 'Poisson ratio'),
    ('epoxy_bulk_output.csv', 'epoxy_bulk_replicas.csv', 'Pa', 'bulk modulus'),
    ('epoxy_shear2_output.csv', 'epoxy_shear2_replicas.csv', 'Pa', "shear modulus (authors' label shear2)"),
]
SOURCE = ('Edeling et al. 2024 npj Comput. Mater. 10:87, epoxy resin (LAMMPS); '
          'github.com/wedeling/MD-active-subspace tag 1.1')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unit(name):
    return UNITS[(name[0], name[-1])]


def read_csv(path, ncol):
    """Header labels and data-row strings of a plain comma-separated file."""
    with open(path, encoding='ascii') as fh:
        lines = fh.read().splitlines()
    header = lines[0].split(',')
    rows = [ln for ln in lines[1:] if ln.strip()]
    if len(header) != ncol:
        raise ValueError(f'{path}: {len(header)} columns, expected {ncol}')
    for ln in rows:
        vals = ln.split(',')
        if len(vals) != ncol:
            raise ValueError(f'{path}: a row has {len(vals)} fields')
        [float(v) for v in vals]  # every field is a plain number (raises otherwise)
    return header, rows


def build(src):
    for rel, digest in SHA256.items():
        got = sha256(os.path.join(src, rel))
        if got != digest:
            raise ValueError(f'{rel}: SHA-256 {got} differs from the recorded {digest}')
    texts = {}
    names, rows = read_csv(os.path.join(src, 'data_epoxy', 'epoxy_inputs.csv'), 103)
    if len(rows) != 500:
        raise ValueError('epoxy_inputs.csv: expected 500 rows')
    cols = ','.join(f'{n} [{unit(n)}]' for n in names)
    texts['epoxy_inputs.csv'] = (
        f'# {SOURCE}: 103 force-field parameters of the 500 Monte Carlo input points '
        f'(data_epoxy/epoxy_inputs.csv; physical units; values unchanged)\n# {cols}\n' + '\n'.join(rows) + '\n')
    with open(os.path.join(src, 'epoxy.params.json')) as fh:
        priors = json.load(fh)
    defaults = [priors[n]['default'] for n in names]
    unsampled = [k for k in priors if k not in names]
    if any(priors[k]['default'] != 0.0 for k in unsampled):
        raise ValueError('an unsampled parameter has a nonzero default')
    texts['epoxy_defaults.csv'] = (
        f'# {SOURCE}: default value x_i of each sampled parameter (epoxy.params.json); '
        f'each input is uniform between 0.85 x_i and 1.15 x_i\n# {cols}\n'
        + ','.join(repr(float(v)) for v in defaults) + '\n')
    for srcname, name, u, desc in OUTPUTS:
        header, rows = read_csv(os.path.join(src, 'data_epoxy', srcname), 20)
        if header != [f'replica{k}' for k in range(1, 21)] or len(rows) != 500:
            raise ValueError(f'{srcname}: unexpected header or row count')
        cols = ','.join(f'{h} [{u}]' for h in header)
        texts[name] = (
            f'# {SOURCE}: {desc} of 20 replica runs (random seeds) at each of the 500 input points; '
            f'row i matches row i of epoxy_inputs.csv (data_epoxy/{srcname}; values unchanged)\n'
            f'# {cols}\n' + '\n'.join(rows) + '\n')
    return texts


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if '--out' not in argv or len(args) != 2:
        print(__doc__)
        return 2
    src, out = args[0], os.path.realpath(argv[argv.index('--out') + 1])
    if os.path.commonpath([out, HERE]) == HERE:
        print('--out must be a folder outside this data set directory')
        return 2
    texts = build(src)
    os.makedirs(out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(out, name), 'w') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    if '--check' in argv:
        bad = 0
        for name, text in texts.items():
            a = np.loadtxt(os.path.join(out, name), delimiter=',', ndmin=2)
            b = np.loadtxt(os.path.join(HERE, name), delimiter=',', ndmin=2)
            with open(os.path.join(HERE, name)) as fh:
                same_text = fh.read() == text
            ok = a.shape == b.shape and float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) <= TOL
            rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.shape == b.shape else np.inf
            print(f'{name}: shape {a.shape}, max relative difference {rel:.1e} '
                  f'(tolerance {TOL:.0e}), text identical: {same_text} -> {"OK" if ok else "DIFFERENT"}')
            bad += not ok
        return int(bad > 0)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        print(f'file not found, or not of the expected kind: {err.filename}\n'
              'The source is the folder epoxy/ of github.com/wedeling/MD-active-subspace (tag 1.1); see DATA.md, "Regenerating".', file=sys.stderr)
        sys.exit(3)
