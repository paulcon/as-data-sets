"""Convert the deposited SOCOL-AER selenium files of Feinberg et al. (2020) into this data set's CSV files.

Stored data: the values were computed by the authors with the aerosol-chemistry-climate model SOCOL-AER
(with selenium chemistry) and deposited at the ETH Research Collection, doi:10.3929/ethz-b-000357105
(CC BY 4.0). This script reads the deposited files, checks their SHA-256, and writes:

  se_training_inputs.csv              the 34 inputs of the 400 training runs, physical units: every field of
                                      Parameter_values_training_runs_original_units.csv copied character
                                      for character (lossless)
  se_training_inputs_transformed.csv  the same runs in the authors' transformed units (log10 for the
                                      log-uniform inputs, a conditional CDF value for emiss_aer, a latent
                                      value for lat_aer): Parameter_values_training_runs_transform_units.csv,
                                      copied character for character (lossless)
  se_training_outputs.csv             global totals of the annual-mean fields of sensitivity_runs_data.nc
  se_validation_outputs.csv           the same for validation_runs_data.nc (50 runs; the deposit offered no
                                      input values for them)

The output files are a reduction, not a copy: each annual-mean column burden (kg Se m-2) and each wet or
dry deposition flux (kg Se m-2 s-1) on the T42 Gaussian grid (64 x 128) is integrated over the globe,

    total = sum_{j,i} field[j, i] * A_j,   A_j = a^2 * w_j * (2 pi / 128),   a = 6.371e6 m,

with w_j the Gauss-Legendre weights of the 64 latitudes (numpy.polynomial.legendre.leggauss; the script
checks that their latitudes equal the file's to 1e-10 degrees; sum of A_j over the grid = 4 pi a^2). The
global selenium lifetime is then

    tau = (sum of the 9 burdens) / (sum of the 4 wet + 4 dry deposition rates) / 86400   [days],

the burden divided by total deposition (paper, p. 1373). tau does not depend on a. The float32 fields are
summed in float64; values are written with repr() (shortest round-trip decimal).

    python convert_data.py <folder with the deposited files> --out DIR           # write the four CSVs to DIR
    python convert_data.py <folder with the deposited files> --check --out DIR   # write to DIR (outside this
                                                                                  # directory), compare
--check compares every number with the shipped file: relative difference <= 1e-12 (the input files are
also compared as text). Nothing is written into this directory unless --out names it. Imports numpy and
scipy (scipy.io.netcdf_file reads the NetCDF classic files) and the standard library. About 10 s.
"""

import csv
import hashlib
import os
import sys

import numpy as np
from scipy.io import netcdf_file

HERE = os.path.dirname(os.path.abspath(__file__))
RTOL = 1e-12
EARTH_RADIUS = 6.371e6          # m (ECHAM5's value); cancels in tau
NLAT, NLON = 64, 128            # T42 Gaussian grid

SHA256 = {
    'sensitivity_runs_data.nc': '99dd85465c11bd7afa5687ca37f6166018c94526bbc2cfb0d53a18cc0bbc40a2',
    'validation_runs_data.nc': 'e483d33c28c7d2635710e188ad7274efcfe91e0aa4893b4bd69ab30d8ae3ed0a',
    'Parameter_values_training_runs_original_units.csv':
        'e2b2033bcc2ef90b74a482db17c8f3503afedf114907916dc54edfa9077f94be',
    'Parameter_values_training_runs_transform_units.csv':
        '42cbf7093a50d350e04d85cad901f551723c9bc2cb77079fff45a3ff11940650',
}

# (deposited header with whitespace collapsed, column name, units in the original file, units in the
# transformed file). Table 3 of the paper (p. 1370) gives the distributions.
_K = 'cm^3 molec^-1 s^-1'
INPUTS = ([(f'k{i} ({_K})', f'k{i}', 'cm3 molec-1 s-1', 'log10 of cm3 molec-1 s-1') for i in range(1, 18)]
          + [(f'fk{i}', f'fk{i}', '1', '1') for i in range(18, 22)]
          + [('Emiss_mar (Gg Se yr^-1)', 'emiss_mar', 'Gg Se/yr', 'log10 of Gg Se/yr'),
             ('Emiss_terr (Gg Se yr^-1)', 'emiss_terr', 'Gg Se/yr', 'Gg Se/yr'),
             ('Emiss_ant (Gg Se yr^-1)', 'emiss_ant', 'Gg Se/yr', 'Gg Se/yr'),
             ('Emiss_vol (Gg Se yr^-1)', 'emiss_vol',
              'volcanic map scaling as deposited; 1 unit ~ 1.27 Gg Se/yr emitted (DATA.md)',
              'log10 of the volcanic map scaling'),
             ('%OCSe_ant', 'pOCSe_ant', '%', '%'), ('%CSe2_ant', 'pCSe2_ant', '%', '%'),
             ('%CSSe_ant', 'pCSSe_ant', '%', '%'), ('%H2Se_ant', 'pH2Se_ant', '%', '%'),
             ('%H2Se_vol', 'pH2Se_vol', '%', '%'), ('acc_coeff', 'acc_coeff', '1', '1'),
             ('r_aer (μm)', 'r_aer', 'um', 'log10 of um'),
             ('emiss_aer (kg yr^-1)', 'emiss_aer', 'kg/yr', 'conditional CDF value'),
             ('lat_aer (1 = [90° to 80°S], …, 18 = [80° to 90° N])', 'lat_aer',
              'band 1-18 (1 = 90-80S; 18 = 80-90N)', 'latent value; band = ceil')])
# headers of the transformed file differ only in the units part
TRANSFORMED_UNITS = {'emiss_aer': 'emiss_aer (uniform)',
                     'lat_aer': 'lat_aer (rounded up, 1 = [90° to 80°S], …, 18 = [80° to 90° N])'}

BURDENS = ['bCSSe', 'bCSe2', 'bDMSe', 'bH2Se', 'bOCSe', 'baer_dummy_se', 'baer_se', 'bOX_Se_IN',
           'bOX_Se_OR']
WET = ['waer_dummy_se', 'waer_se', 'wOX_Se_IN', 'wOX_Se_OR']
DRY = ['daer_dummy_se', 'daer_se', 'dOX_Se_IN', 'dOX_Se_OR']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm_header(s):
    return ' '.join(s.split())


def expected_transformed(dep, name):
    if name in TRANSFORMED_UNITS:
        return TRANSFORMED_UNITS[name]
    if 'log10' in dict((n, t) for _, n, _, t in INPUTS)[name]:
        return dep.split(' (')[0] + ' (log10 transform)'
    return dep


def read_inputs(path, transformed):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    header = [norm_header(h) for h in rows[0]]
    want = ['Index', 'Experiment Number'] + [
        expected_transformed(d, n) if transformed else d for d, n, _, _ in INPUTS]
    if header != want:
        bad = [(a, b) for a, b in zip(header, want) if a != b]
        raise ValueError(f'{os.path.basename(path)}: unexpected header fields {bad}')
    data = rows[1:]
    for r in data:
        if len(r) != len(want):
            raise ValueError(f'{os.path.basename(path)}: a row has {len(r)} fields, header {len(want)}')
        for v in r:
            if v != v.strip() or not np.isfinite(float(v)):
                raise ValueError(f'{os.path.basename(path)}: field {v!r} is not a plain finite number')
    idx = [int(r[0]) for r in data]
    if idx != list(range(1, len(data) + 1)):
        raise ValueError(f'{os.path.basename(path)}: Index is not 1..{len(data)}')
    return data


def cell_areas(lat):
    mu, w = np.polynomial.legendre.leggauss(NLAT)
    latg = np.degrees(np.arcsin(mu))[::-1]            # the files run from north to south
    if np.abs(latg - lat).max() > 1e-10:
        raise ValueError('latitudes are not the T42 Gaussian latitudes')
    return EARTH_RADIUS**2 * w[::-1] * (2.0 * np.pi / NLON), np.abs(latg - lat).max()


def integrate(path):
    """Global totals per run of every burden and deposition variable, and tau (days)."""
    nc = netcdf_file(path, 'r', mmap=False)
    V = nc.variables
    runs = np.asarray(V['runs'][:]).astype(int)
    if list(runs) != list(range(1, len(runs) + 1)):
        raise ValueError(f'{os.path.basename(path)}: runs is not 1..{len(runs)}')
    lon = np.asarray(V['lon'][:])
    if len(lon) != NLON or np.abs(np.diff(lon) - 360.0 / NLON).max() > 1e-9:
        raise ValueError('longitudes are not the 128 equally spaced T42 longitudes')
    area, dlat = cell_areas(np.asarray(V['lat'][:]))
    cols = {}
    for k in BURDENS + WET + DRY:
        a = np.asarray(V[k][:], dtype=np.float64)
        if a.shape != (len(runs), NLAT, NLON) or not np.all(np.isfinite(a)) or (a > 1e30).any():
            raise ValueError(f'{k}: unexpected shape, fill values or non-finite values')
        if a.min() < 0:
            raise ValueError(f'{k}: negative values')
        cols[k] = np.einsum('rji,j->r', a, area)
    nc.close()
    B = sum(cols[k] for k in BURDENS)
    W = sum(cols[k] for k in WET)
    D = sum(cols[k] for k in DRY)
    cols['burden_total'], cols['wet_total'], cols['dry_total'] = B, W, D
    cols['tau'] = B / (W + D) / 86400.0
    return runs, cols, dlat


OUT_COLS = ([(k, 'kg Se') for k in BURDENS] + [(k, 'kg Se/s') for k in WET + DRY]
            + [('burden_total', 'kg Se'), ('wet_total', 'kg Se/s'), ('dry_total', 'kg Se/s'),
               ('tau', 'days')])


def build(src):
    for name, h in SHA256.items():
        got = sha256(os.path.join(src, name))
        if got != h:
            raise ValueError(f'{name}: SHA-256 {got} differs from the recorded {h}')
    out = {}
    orig = read_inputs(os.path.join(src, 'Parameter_values_training_runs_original_units.csv'), False)
    trans = read_inputs(os.path.join(src, 'Parameter_values_training_runs_transform_units.csv'), True)
    if [r[:2] for r in orig] != [r[:2] for r in trans]:
        raise ValueError('the two input files list different runs')
    cols = ['run [1]', 'experiment [1]'] + [f'{n} [{u}]' for _, n, u, _ in INPUTS]
    out['se_training_inputs.csv'] = (
        '# Feinberg et al. 2020 SOCOL-AER selenium: 34 inputs of the 400 training runs, physical units '
        '(ETH Research Collection 10.3929/ethz-b-000357105, Parameter_values_training_runs_original_units.csv; '
        'values unchanged)\n# ' + ','.join(cols) + '\n' + '\n'.join(','.join(r) for r in orig) + '\n')
    cols = ['run [1]', 'experiment [1]'] + [f'{n} [{t}]' for _, n, _, t in INPUTS]
    out['se_training_inputs_transformed.csv'] = (
        '# Feinberg et al. 2020 SOCOL-AER selenium: the 400 training runs in the authors\' transformed units '
        '(Parameter_values_training_runs_transform_units.csv of the same record; values unchanged)\n# '
        + ','.join(cols) + '\n' + '\n'.join(','.join(r) for r in trans) + '\n')
    info = {}
    for nc_name, csv_name, what in (('sensitivity_runs_data.nc', 'se_training_outputs.csv', '400 training'),
                                    ('validation_runs_data.nc', 'se_validation_outputs.csv', '50 validation')):
        runs, c, dlat = integrate(os.path.join(src, nc_name))
        info[csv_name] = dlat
        header = ['run [1]'] + [f'{k} [{u}]' for k, u in OUT_COLS]
        lines = [','.join([str(r)] + [repr(float(c[k][i])) for k, _ in OUT_COLS]) for i, r in enumerate(runs)]
        out[csv_name] = (
            f'# Feinberg et al. 2020 SOCOL-AER selenium: global totals of the annual-mean Se burdens and wet/dry '
            f'deposition of the {what} runs (area integrals of {nc_name}, 10.3929/ethz-b-000357105, by '
            f'convert_data.py) and tau = total burden / total deposition\n# ' + ','.join(header) + '\n'
            + '\n'.join(lines) + '\n')
    return out, info


def compare(a_text, b_text):
    """Largest relative difference between the numbers of two CSV texts (inf if their shapes differ)."""
    a = np.loadtxt(a_text.splitlines(), delimiter=',', ndmin=2)
    b = np.loadtxt(b_text.splitlines(), delimiter=',', ndmin=2)
    if a.shape != b.shape:
        return np.inf
    scale = np.maximum(np.abs(a), np.abs(b))
    d = np.abs(a - b)
    return float(np.max(np.where(scale > 0, d / np.where(scale > 0, scale, 1.0), 0.0)))


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if not args or '--out' not in argv or argv.index('--out') + 1 >= len(argv):
        print(__doc__)
        return 2
    out_dir = os.path.abspath(argv[argv.index('--out') + 1])
    src = [a for a in args if os.path.abspath(a) != out_dir][0]
    check = '--check' in argv
    if check and os.path.realpath(out_dir) == os.path.realpath(HERE):
        print('--check needs --out DIR outside the data set directory')
        return 2
    os.makedirs(out_dir, exist_ok=True)
    texts, info = build(src)
    for name, text in texts.items():
        with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    for name, d in info.items():
        print(f'{name}: largest |Gaussian latitude - file latitude| = {d:.1e} degrees')
    if not check:
        return 0
    bad = 0
    for name, text in texts.items():
        with open(os.path.join(HERE, name), encoding='utf-8') as fh:
            shipped = fh.read()
        d = compare(text, shipped)
        ok = d <= RTOL
        extra = ''
        if name.startswith('se_training_inputs'):
            extra = ', text identical' if text == shipped else ', text differs'
        print(f'{name}: largest relative difference {d:.1e} (tolerance {RTOL:.0e}){extra}: '
              f'{"ok" if ok else "DIFFERENT"}')
        bad += not ok
    return int(bad > 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
