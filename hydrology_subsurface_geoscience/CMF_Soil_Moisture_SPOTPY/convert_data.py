"""Convert the PLOS ONE supporting file S4 of Houska et al. (2015) into this data set's CSV files.

Stored data: the values were computed by the authors with CMF driven by SPOTPY and deposited as
supporting information S4 of doi:10.1371/journal.pone.0145180 (file doi:10.1371/journal.pone.0145180.s004,
CC BY 4.0). S4 is a zip archive holding three zip archives, one per algorithm (MC, LHS, MLE); this
script reads the Monte Carlo and Latin hypercube members straight from the archive (nothing is
extracted or executed), checks the archive's SHA-256, and writes, for each run, the objective value
and the four parameters with their text unchanged, after two ``#`` header lines (a description;
column names with units), so ``np.loadtxt(path, delimiter=',')`` reads the files. The 239 simulated
soil-moisture values per run are not copied (``check_accuracy.py`` reads them from the archive).

Rows whose objective and simulations are all ``#NAME?`` in the deposit (runs SPOTPY scored as minus
infinity because a parameter left the setup's plausibility bounds, see DATA.md) are kept, with their
parameters, ``nan`` as objective and 0 in the ``valid`` column; the other rows get 1.

    python convert_data.py <path of pone.0145180.s004.zip> --out DIR           # write the two CSV files
    python convert_data.py <path of pone.0145180.s004.zip> --out DIR --check   # also compare with the shipped files

``--out`` must name a directory outside this data set's directory. ``--check`` compares the written
values with the shipped files, numerically with tolerance 1e-12 (and the ``nan`` pattern exactly).
Standard library only (csv, hashlib, io, zipfile).
"""

import argparse
import hashlib
import io
import math
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_SHA256 = '0cd46f8f0ba6ba394220991f523900e7cd9432e7c95529874a98da6a981f6a03'
ARCHIVE_SIZE = 9153529
TOL = 1e-12

# (inner zip, member, output file, description)
MEMBERS = [
    ('S4_File_Simulation_results_of_the_CMF_model_with_MC.zip', 'CMF_MC.csv', 'cmf_mc_samples.csv',
     'Houska et al. 2015 PLOS ONE S4 (doi:10.1371/journal.pone.0145180.s004) CMF_MC.csv: SPOTPY Monte Carlo '
     'runs of the CMF soil column; like = -RMSE of topsoil soil moisture 1999 (vol. %); values unchanged; '
     'valid = 0 where the deposit has #NAME?'),
    ('S5_File_Simulation_results_of_the_CMF_model_with_LHS.zip', 'CMF_LHS.csv', 'cmf_lhs_samples.csv',
     'Houska et al. 2015 PLOS ONE S4 (doi:10.1371/journal.pone.0145180.s004) CMF_LHS.csv: SPOTPY Latin '
     'hypercube runs of the CMF soil column; like = -RMSE of topsoil soil moisture 1999 (vol. %); values '
     'unchanged; valid = 0 where the deposit has #NAME?'),
]
SOURCE_COLUMNS = ['like', 'paralpha', 'parn', 'parksat', 'parporosity']
COLUMNS = ('run [-],like [vol. %; = -RMSE],alpha [1/cm (CMF)],n [-],ksat [m/day],porosity [-],'
           'valid [1 = run scored; 0 = objective #NAME? in the deposit]')
BAD = '#NAME?'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_member(archive, inner, member):
    """Return the lines of one CSV member of an inner zip of the S4 archive, read in memory."""
    with zipfile.ZipFile(archive) as outer:
        data = outer.read(inner)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(member).decode('ascii')
    return [ln for ln in text.splitlines() if ln.strip()]


def convert_lines(lines):
    """Return the output data lines (text) for the deposited lines of one algorithm."""
    header = lines[0].split(',')
    if header[:5] != SOURCE_COLUMNS or header[-1] != 'chain':
        raise ValueError(f'unexpected header start {header[:5]} / end {header[-1]}')
    out = []
    for i, ln in enumerate(lines[1:]):
        vals = ln.split(',')
        if len(vals) != len(header):
            raise ValueError(f'row {i}: {len(vals)} fields, header {len(header)}')
        head = vals[:5]
        [float(v) for v in vals[1:5]]  # the parameters are always plain numbers (raises otherwise)
        rest = [vals[0]] + vals[5:-1]  # objective and simulations
        if all(v == BAD for v in rest):
            out.append(','.join([str(i), 'nan'] + head[1:] + ['0']))
            continue
        if BAD in rest:
            raise ValueError(f'row {i}: partly #NAME?')
        [float(v) for v in vals]  # every field is a plain number (raises otherwise)
        out.append(','.join([str(i)] + head + ['1']))
    return out


def parse(lines):
    return [[float(v) for v in ln.split(',')] for ln in lines if not ln.startswith('#')]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('archive', help='path of pone.0145180.s004.zip')
    ap.add_argument('--out', required=True, help='output directory, outside this data set directory')
    ap.add_argument('--check', action='store_true', help='compare the written files with the shipped ones')
    a = ap.parse_args(argv)
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([out, here]) == here:
        sys.exit('--out must be outside the data set directory, subdirectories included '
                 '(the shipped files are the record)')
    if os.path.getsize(a.archive) != ARCHIVE_SIZE or sha256(a.archive) != ARCHIVE_SHA256:
        sys.exit('the archive is not the deposited S4 file (size or SHA-256 differ); see DATA.md')
    os.makedirs(out, exist_ok=True)
    ok = True
    for inner, member, fname, desc in MEMBERS:
        rows = convert_lines(read_member(a.archive, inner, member))
        text = '# ' + desc + '\n# ' + COLUMNS + '\n' + '\n'.join(rows) + '\n'
        with open(os.path.join(out, fname), 'w', newline='\n') as fh:
            fh.write(text)
        nbad = sum(r.endswith(',0') for r in rows)
        print(f'{fname}: {len(rows)} rows ({nbad} with valid = 0)')
        if a.check:
            with open(os.path.join(HERE, fname)) as fh:
                shipped = parse(fh.read().splitlines())
            new = parse(rows)
            same = len(shipped) == len(new) and all(
                len(p) == len(q) and all((math.isnan(x) and math.isnan(y)) or abs(x - y) <= TOL
                                         for x, y in zip(p, q)) for p, q in zip(shipped, new))
            print(f'  check against shipped {fname}: {"identical within 1e-12" if same else "DIFFERENT"}')
            ok = ok and same
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        sys.exit(f'file not found, or not of the expected kind: {err.filename}\n'
                 'The source is the archive pone.0145180.s004.zip; see DATA.md, "Regenerating".')
