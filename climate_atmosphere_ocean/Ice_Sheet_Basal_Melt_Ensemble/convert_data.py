"""Convert the authors' ensemble files of Berdahl et al. (2021) into this data set's two CSV files.

Stored data: the values were computed by the authors with the Community Ice Sheet Model (CISM v2.1)
and published in their repository https://github.com/mberdahl-uw/EmulatorPaper (commit dd8643c, also
archived at Zenodo, doi:10.5281/zenodo.4902149). This script only changes the file layout. It checks
the SHA-256 of the six source files, then

* joins the five per-basin design files ``design_metadata_<basin>.csv`` into one file
  (``ens`` and ``flag`` once, then the six columns of each basin, basins in the authors' order
  amundsen, eais, peninsula, ross, weddell), after checking that ``ens`` and ``flag`` agree in all
  five files;
* copies the 500 data lines of ``SLR.csv`` (ensemble number, then 200 annual values of the ice volume
  above flotation in mm of sea-level equivalent);

and writes each value's text unchanged after two ``#`` header lines (a description; column names
with units), so ``np.loadtxt(path, delimiter=',')`` reads the files. The conversion is lossless:
every number is copied character for character.

    python convert_data.py <EmulatorPaper folder> --out DIR           # write the two CSVs into DIR
    python convert_data.py <EmulatorPaper folder> --out DIR --check   # also compare with the shipped files

DIR must not be this data set's directory (the shipped files are never overwritten by the script);
copy the files from DIR by hand if they are to replace the shipped ones. ``--check`` reads both
versions with numpy and passes when the largest relative difference is at most 1e-12 (it also
reports whether the text is identical).

Standard library plus numpy (numpy only for --check).
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SHA256 = {
    'design_metadata_amundsen.csv': '766f4214327bed8dc8e4ec133590e0975a01d9b27f7752e1270047b18b527742',
    'design_metadata_eais.csv': 'da5fa5c3c4273b59e4f7022d30f95f6812b8913ca7a6fb3c98b06b96b2379fe7',
    'design_metadata_peninsula.csv': '9b4811186229e5a402aff3688d8adbf34ac98354a1841b2bf951c33e6d15b34f',
    'design_metadata_ross.csv': '8ac5d29a27d6af0105747717642717a0cd8684415d1092088be702517b81dcba',
    'design_metadata_weddell.csv': '2f5a6dabbfd05c47e2b6ed9abe90963cf4f34217d36b4f315fed4a9d3a6f9c9a',
    'SLR.csv': 'd774ce52efedc259634bb4cab5c839c9da7a71387294cc979c7f1d1f337d5b46',
}
BASINS = ['amundsen', 'eais', 'peninsula', 'ross', 'weddell']
DESIGN_HEADER = ['ens', 'flag', 'm2200', 'mmax', 't0', 'tau', 'm2200 scaled', 'mmax scaled']
# authors' label -> (column stem, units); the authors' "m2200" is the paper's M200 (melt-rate anomaly at
# simulation year 200), "mmax" its Mmax (asymptote of the sigmoid, Eqs. 1-5); unscaled values are
# fractions of the basin cap, scaled values are in m/yr (cap: Sect. 2.2, p. 2688).
DESIGN_COLS = [('m2200', 'M200frac', '1'), ('mmax', 'Mmaxfrac', '1'), ('t0', 't0', 'yr'),
               ('tau', 'tau', 'yr'), ('m2200 scaled', 'M200', 'm/yr'), ('mmax scaled', 'Mmax', 'm/yr')]
N_RUNS = 500
N_YEARS = 200

DESIGN_OUT = 'basal_melt_design.csv'
SLR_OUT = 'ice_volume_above_flotation.csv'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_lines(path):
    with open(path, encoding='ascii') as fh:
        return [ln.rstrip('\r\n') for ln in fh if ln.strip()]


def build(src):
    for name, want in SHA256.items():
        got = sha256(os.path.join(src, name))
        if got != want:
            raise ValueError(f'{name}: SHA-256 {got} differs from the recorded {want}')

    tokens = {}
    for b in BASINS:
        lines = read_lines(os.path.join(src, f'design_metadata_{b}.csv'))
        if lines[0].split(',') != DESIGN_HEADER:
            raise ValueError(f'design_metadata_{b}.csv: unexpected header {lines[0]!r}')
        rows = [ln.split(',') for ln in lines[1:]]
        if len(rows) != N_RUNS or any(len(r) != len(DESIGN_HEADER) for r in rows):
            raise ValueError(f'design_metadata_{b}.csv: expected {N_RUNS} rows of {len(DESIGN_HEADER)} fields')
        for r in rows:
            [float(v) for v in r]  # every field is a plain number (raises otherwise)
        tokens[b] = rows
    first = tokens[BASINS[0]]
    for b in BASINS[1:]:
        for i, (r0, r) in enumerate(zip(first, tokens[b])):
            if float(r0[0]) != float(r[0]) or float(r0[1]) != float(r[1]):
                raise ValueError(f'row {i + 1}: ens/flag of {b} differ from those of {BASINS[0]}')
    cols = ['ens [1]', 'flag [1]'] + [f'{stem}_{b} [{u}]' for b in BASINS for _, stem, u in DESIGN_COLS]
    out_rows = []
    for i in range(N_RUNS):
        vals = first[i][:2]
        for b in BASINS:
            vals += tokens[b][i][2:]
        out_rows.append(','.join(vals))
    design = ('# Berdahl et al. 2021 (The Cryosphere 15:2683) CISM basal-melt ensemble: sigmoid parameters of '
              'the 500 runs in five LARMIP basins; flag = 1 marks the 9 runs that stopped before year 200 '
              '(github.com/mberdahl-uw/EmulatorPaper design_metadata_<basin>.csv; values unchanged)\n# '
              + ','.join(cols) + '\n' + '\n'.join(out_rows) + '\n')

    lines = read_lines(os.path.join(src, 'SLR.csv'))
    if lines[0] != '# ensemble member':
        raise ValueError(f'SLR.csv: unexpected first line {lines[0]!r}')
    rows = lines[1:]
    if len(rows) != N_RUNS:
        raise ValueError(f'SLR.csv: expected {N_RUNS} data lines, found {len(rows)}')
    for i, ln in enumerate(rows):
        vals = ln.split(',')
        if len(vals) != N_YEARS + 1:
            raise ValueError(f'SLR.csv line {i + 2}: {len(vals)} fields')
        [float(v) for v in vals]
        if float(vals[0]) != float(first[i][0]):
            raise ValueError(f'SLR.csv line {i + 2}: ensemble number differs from the design')
    cols = ['ens [1]'] + [f'V{k} [mm SLE]' for k in range(1, N_YEARS + 1)]
    slr = ('# Berdahl et al. 2021 CISM ensemble: ice volume above flotation of the Antarctic ice sheet, '
           '200 annual values per run in mm sea-level equivalent; row i matches row i of basal_melt_design.csv; '
           'values after a failed run stopped are written as 0 (github.com/mberdahl-uw/EmulatorPaper SLR.csv; '
           'values unchanged)\n# ' + ','.join(cols) + '\n' + '\n'.join(rows) + '\n')
    return {DESIGN_OUT: design, SLR_OUT: slr}


def parse_out(argv):
    if '--out' not in argv:
        return None
    k = argv.index('--out')
    if k + 1 >= len(argv):
        return None
    return os.path.abspath(argv[k + 1])


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    out = parse_out(argv)
    if not args or out is None:
        print(__doc__)
        print('error: give the source folder and --out DIR (outside this data set directory)')
        return 2
    src = args[0] if args[0] != out else (args[1] if len(args) > 1 else None)
    if src is None:
        print('error: give the source folder')
        return 2
    if os.path.commonpath([out, HERE]) == HERE:
        print(f'error: --out must be outside this data set directory ({HERE})')
        return 2
    os.makedirs(out, exist_ok=True)
    texts = build(src)
    for name, text in texts.items():
        with open(os.path.join(out, name), 'w', encoding='ascii') as fh:
            fh.write(text)
        print(f'wrote {os.path.join(out, name)}: {text.count(chr(10)) - 2} rows')
    if '--check' in argv:
        import numpy as np
        bad = 0
        for name, text in texts.items():
            with open(os.path.join(HERE, name), encoding='ascii') as fh:
                same_text = fh.read() == text
            a = np.loadtxt(os.path.join(HERE, name), delimiter=',')
            b = np.loadtxt(os.path.join(out, name), delimiter=',')
            if a.shape != b.shape:
                print(f'{name}: shapes differ {a.shape} vs {b.shape}')
                bad += 1
                continue
            rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
            ok = rel <= 1e-12
            print(f'{name}: shape {a.shape}, max relative difference {rel:.1e} (tolerance 1e-12): '
                  f'{"pass" if ok else "FAIL"}; text identical: {same_text}')
            bad += not ok
        return int(bad > 0)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
