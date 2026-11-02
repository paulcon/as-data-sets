"""Lossless conversion of the authors' Serpent results into microreactor_runs.csv.

Source (primary): ``pmdata/qpower_full.csv`` in the authors' repository
https://github.com/deanrp2/MicroControl (its only licence is MIT, a software licence that does
not mention data; copy in MicroControl_LICENSE.txt; the file is hosted with credit),
commit 7a58e183b51238c9121ba690f2c930fd62308fa1 (2022-01-03), the repository named in the
Data Availability section of D. Price, M. I. Radaideh and B. Kochunas, Nucl. Eng. Des.
393:111776 (2022), doi:10.1016/j.nucengdes.2022.111776. 751 Serpent runs, no symmetry
augmentation. (``pmdata/qpower_211018.csv`` in the same repository is byte-identical.)

Second copy (checked, not used): ``microreactor.csv`` of the pyMAISE benchmark data, Zenodo
record 20140559 (https://doi.org/10.5281/zenodo.20140559, CC BY 4.0). It holds the same 751
runs in the same order, with the flux, fission-rate and time columns rounded half-up to three
significant digits and the angles to nine; ``--check`` verifies this.

What the conversion does, and nothing else:
  * replaces the header line by two '#' header lines (description; column names with units);
  * writes the run label 'sample_00012' as the integer 12 (the label's digits);
  * copies every other field as the same text (no number is parsed and re-printed).
So ``np.loadtxt('microreactor_runs.csv', delimiter=',')`` reads the result.

    python convert_data.py qpower_full.csv                              # write microreactor_runs.csv
    python convert_data.py qpower_full.csv --check [microreactor.csv]   # rebuild in memory, compare;
                                                                        # optionally check the Zenodo copy
"""
import hashlib
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SOURCE_URL = 'https://github.com/deanrp2/MicroControl/blob/7a58e183b51238c9121ba690f2c930fd62308fa1/pmdata/qpower_full.csv'
SOURCE_SHA256 = 'dda6ebc38c31c747de7b063d289a2815d66ec03d5e6eba8d6327d99816c7d003'
ZENODO_URL = 'https://zenodo.org/records/20140559/files/microreactor.csv'
ZENODO_MD5 = 'a548c9ddc384bcb08823a67f33076523'
ZENODO_SHA256 = '032d022709ebfe2b69dd46bc6c5e6dc4f4419bb7c8c21d8e56a6b882e4ef9401'    # = pyMAISE registry
OUT = Path(__file__).with_name('microreactor_runs.csv')

TALLY = ['fluxQ{}', 'fissQ{}', 'fissEQ{}', 'flux_runcertQ{}', 'fiss_runcertQ{}', 'fissE_runcertQ{}']
SOURCE_NAMES = (['', 'k', 'k_uncert', 'cpu_time', 'runtime']
                + [t.format(q) for t in TALLY for q in range(1, 5)]
                + [f'theta{i}' for i in range(1, 9)])
UNITS = (['run number', '-', 'relative sd', 'min?', 'min?']
         + ['n/cm^2/s'] * 4 + ['fission rate ?'] * 4 + ['fission energy ?'] * 4 + ['relative sd'] * 12
         + ['rad; Serpent global orientation'] * 8)
NAMES = ['sample'] + SOURCE_NAMES[1:]
COLUMNS = [f'{n} ({u})' for n, u in zip(NAMES, UNITS)]

DESCRIPTION = ('# HOLOS-Quad microreactor: 751 Serpent runs (8 control-drum angles -> k_eff and quadrant tallies) '
               'of Price, Radaideh and Kochunas, Nucl. Eng. Des. 393:111776 (2022), doi:10.1016/j.nucengdes.2022.111776; '
               'file pmdata/qpower_full.csv of github.com/deanrp2/MicroControl (commit 7a58e18; repository licence MIT, silent on data; hosted with credit), '
               'converted losslessly by convert_data.py (run label sample_NNNNN written as NNNNN); '
               'a rounded copy is in the pyMAISE data, doi:10.5281/zenodo.20140559')


def _read(path, sha=None, md5=None):
    raw = Path(path).read_bytes()
    if sha and hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f'{path}: SHA-256 mismatch')
    if md5 and hashlib.md5(raw).hexdigest() != md5:
        raise ValueError(f'{path}: MD5 mismatch')
    lines = raw.decode('utf-8-sig').splitlines()
    return lines[0].split(','), [ln.split(',') for ln in lines[1:]]


def convert(src):
    header, rows = _read(src, sha=SOURCE_SHA256)
    if header != SOURCE_NAMES:
        raise ValueError('unexpected header line')
    out = [DESCRIPTION, '# ' + ', '.join(COLUMNS)]
    for f in rows:
        if len(f) != 37 or not f[0].startswith('sample_'):
            raise ValueError(f'bad row: {f[:2]}')
        out.append(','.join([str(int(f[0][7:]))] + f[1:]))
    return '\n'.join(out) + '\n'


def _round_half_up(text, digits):
    d = Decimal(text)
    e = d.adjusted()
    return float(d.scaleb(-e).quantize(Decimal(1).scaleb(1 - digits), rounding=ROUND_HALF_UP).scaleb(e))


def check_zenodo(src, zen):
    """The Zenodo copy = the source with some columns rounded; raise if not."""
    hs, rs = _read(src, sha=SOURCE_SHA256)
    hz, rz = _read(zen, sha=ZENODO_SHA256, md5=ZENODO_MD5)
    hz = ['' if h == 'sample number' else h for h in hz]
    if [r[0] for r in rs] != [r[0] for r in rz]:
        raise ValueError('run labels differ')
    report = {}
    for name in SOURCE_NAMES[1:]:
        js, jz = hs.index(name), hz.index(name)
        a, b = [r[js] for r in rs], [float(r[jz]) for r in rz]
        if [float(x) for x in a] == b:
            report[name] = 'equal'
        elif name.startswith('theta'):
            dmax = max(abs(float(x) - y) for x, y in zip(a, b))
            if dmax > 5.0001e-10:
                raise ValueError(f'{name}: differs by {dmax}')
            report[name] = f'equal within {dmax:.1e} rad'
        elif [_round_half_up(x, 3) for x in a] == b:
            report[name] = 'equal after rounding half-up to 3 significant digits'
        else:
            raise ValueError(f'{name}: not a rounding of the source')
    return report


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    new = convert(argv[1])
    if '--check' in argv:
        same = OUT.read_text() == new
        print(f'{OUT.name}: {"identical to" if same else "DIFFERS from"} the conversion of {argv[1]}')
        rest = [a for a in argv[2:] if a != '--check']
        if rest:
            for k, v in check_zenodo(argv[1], rest[0]).items():
                print(f'  Zenodo copy, {k}: {v}')
        sys.exit(0 if same else 1)
    OUT.write_text(new)
    print(f'wrote {OUT.name}: {new.count(chr(10)) - 2} rows')


if __name__ == '__main__':
    main(sys.argv)
