"""Convert the training and validation data of Guo and Kramer (2025) to this repository's CSV format.

Y. Guo and B. Kramer, "Risk-based design optimization for powder bed fusion metal additive
manufacturing", Struct. Multidisc. Optim. 68(9):179, 2025, doi:10.1007/s00158-025-04120-w, ran
sequential thermal and mechanical Abaqus models of one electron-beam scan over a Ti-6Al-4V
powder layer. The authors publish their runs as one zip file, Supplementary_material_data.zip, in
the GitHub repository yulin-g/BPOF-optimization-AM. Four of its members are used here:

    Supplementary_material/Training_data/Input_120.csv        120 x 12  inputs of the training runs
    Supplementary_material/Training_data/allStress_120.csv    120 x 448 von Mises residual stress (MPa)
    Supplementary_material/Validation_data/Input_50_50.csv    100 x 12  inputs of the validation runs
    Supplementary_material/Validation_data/allStress_50_50.csv 100 x 448 the same for those runs

The input columns are scanning speed v (mm/s), beam power P (W), preheating temperature T0 (deg C),
yield strength Y (MPa), elastic modulus E (GPa), bulk density rho (kg/m^3), and the six held
coefficients a0, a1, a2 of Cp(T) and b0, b1, b2 of kappa(T) (Sect. 5.1 of the paper). The column
order is not labelled in the source; it follows from the value ranges and from Y (column 4) equal
to the maximum stress of the runs that yield (see DATA.md). The 448 stress columns are the
integration points of the 32 x 14 grid on the x_c-z_c plane (Fig. 2b), in an order the source
does not document.

This script writes four CSV files, each the source member's lines unchanged (every number keeps
its original decimal text) after two '#' lines (a description with the credit; the column names
with units):

    pbf_training_inputs.csv     pbf_training_stress.csv
    pbf_validation_inputs.csv   pbf_validation_stress.csv

It also writes two small files from the cooling-step temperature histories,

    Supplementary_material/Training_data/allTemperature_120/TempCV_<i>_Step_4_all_frames.csv    i = 1..120
    Supplementary_material/Validation_data/allTemperature_50_50/TempCV_<i>_Step_4_all_frames.csv i = 1..100

each holding the temperature (deg C) of five element integration points over the frames of Step-4
(cooling). Only the first frame (t = 0 of the cooling step, i.e. the end of the moving-heat step)
is kept, with its original decimal text:

    pbf_training_cooling_start_T.csv    120 x 5, row i-1 from TempCV_<i>
    pbf_validation_cooling_start_T.csv  100 x 5, row i-1 from TempCV_<i>

File index i is matched to input row i - 1 (the source does not say so; DATA.md gives the checks:
the temperature at x = 0 tracks T0 with correlation 1.000 under this mapping).

The conversion is lossless: the script checks the SHA-256 of the zip and of each member, and that
each written file reads back with np.loadtxt equal, bit for bit, to the source member read the
same way. Row order is kept. Nothing else in the zip (the later frames of the cooling-step temperature
histories and two Abaqus input decks) is shipped; DATA.md lists it.

Usage:
    python convert_data.py --zip /path/to/Supplementary_material_data.zip           # write the CSVs
    python convert_data.py --zip /path/to/Supplementary_material_data.zip --check   # compare only
"""

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_REPO = 'https://github.com/yulin-g/BPOF-optimization-AM'
SOURCE_COMMIT = '29367be258eaf89a56e72c680a56d5aa488ae94c'      # 2025-05-13
SOURCE_ZIP = 'Supplementary_material_data.zip'
SOURCE_SHA256 = '1a1402a602e325bcc401e3131cecb55651359b3148466ae6fe69c5c857357bf6'

# the six varied inputs: names, units, and the bounds the samples were drawn from (DATA.md, "Inputs")
NAMES = ('v', 'P', 'T0', 'Y', 'E', 'rho')
UNITS = ('mm/s', 'W', 'degC', 'MPa', 'GPa', 'kg/m^3')
LOWER = (100.0, 20.0, 585.0, 742.5, 99.0, 3985.2)
UPPER = (1000.0, 200.0, 715.0, 907.5, 121.0, 4870.8)
# Table 1 of the paper prints E in [100, 120] GPa and rho in [3870, 4730] kg/m^3 (see DATA.md)
TABLE1_LOWER = (100.0, 20.0, 585.0, 742.5, 100.0, 3870.0)
TABLE1_UPPER = (1000.0, 200.0, 715.0, 907.5, 120.0, 4730.0)
HELD = ('a0', 'a1', 'a2', 'b0', 'b1', 'b2')
HELD_UNITS = ('J/(kg K)', 'J/(kg K^2)', 'J/(kg K^3)', 'W/(m K)', 'W/(m K^2)', 'W/(m K^3)')
N_STRESS = 448

CREDIT = ('Abaqus runs of Guo and Kramer, Struct. Multidisc. Optim. 68(9):179 2025, '
          'doi:10.1007/s00158-025-04120-w; from github.com/yulin-g/BPOF-optimization-AM '
          f'{SOURCE_ZIP} (sha256 {SOURCE_SHA256[:12]}...), which states no licence; values unchanged')

MEMBERS = {
    'pbf_training_inputs.csv': dict(
        member='Supplementary_material/Training_data/Input_120.csv',
        sha256='01113272075b1253239babda6bc0e1a7d3dd443031306041b9c2ebd588ead100',
        shape=(120, 12), kind='inputs',
        what='120 training runs (Latin hypercube over the 6 inputs); source Training_data/Input_120.csv'),
    'pbf_training_stress.csv': dict(
        member='Supplementary_material/Training_data/allStress_120.csv',
        sha256='30ee6d04c9258f5488c42bdf5627e32fbeab74698541583f09bc36b771464f9e',
        shape=(120, N_STRESS), kind='stress',
        what=('von Mises residual stress at the end of cooling at the 448 integration points of the '
              'x_c-z_c plane (order undocumented), rows as pbf_training_inputs.csv; '
              'source Training_data/allStress_120.csv')),
    'pbf_validation_inputs.csv': dict(
        member='Supplementary_material/Validation_data/Input_50_50.csv',
        sha256='2d4bedf1ef2cd2ede434b4771a57547f7f245c34161ad17606d62a99a1808659',
        shape=(100, 12), kind='inputs',
        what=('100 validation runs, 50 at each optimal design (v, P) = (417.3, 82.97) and (614.5, 115.8) '
              'with random (T0, Y, E, rho); source Validation_data/Input_50_50.csv')),
    'pbf_validation_stress.csv': dict(
        member='Supplementary_material/Validation_data/allStress_50_50.csv',
        sha256='699ee6eeb5f98a840d2634b9bf046a8d6adfaffb3cf73e2850f5ebb4c9446aad',
        shape=(100, N_STRESS), kind='stress',
        what=('von Mises residual stress at the 448 integration points, rows as pbf_validation_inputs.csv; '
              'source Validation_data/allStress_50_50.csv')),
}


# cooling-step temperature files: the five points, as (label, instance, integration point, x, y, z) text
TEMP_POINTS = (('3780', 'BULK', '2', '1.0', '0.75', '0.2992'),
               ('12', 'POWDEROLD-1', '2', '1.9375', '0.75', '0.65'),
               ('1452', 'POWDEROLD-1', '2', '1.0', '0.75', '0.65'),
               ('1524', 'POWDEROLD-1', '2', '1.0', '0.75', '0.6125'),
               ('2988', 'POWDEROLD-1', '2', '0.0', '0.75', '0.65'))
TEMP_WHAT = ('temperature at t = 0 of the cooling step (Step-4; = end of the moving-heat step) at five element '
             'integration points, (x, y, z) in mm: e3780 bulk (1, 0.75, 0.2992); e12 powder (1.9375, 0.75, 0.65), '
             '62.5 um before the end of the scan; e1452 powder (1, 0.75, 0.65), centroid of the top surface; '
             'e1524 powder (1, 0.75, 0.6125); e2988 powder (0, 0.75, 0.65), start of the scan')
TEMP_MEMBERS = {
    'pbf_training_cooling_start_T.csv': dict(
        folder='Supplementary_material/Training_data/allTemperature_120', n=120,
        what='rows as pbf_training_inputs.csv (row i-1 from Training_data/allTemperature_120/TempCV_<i>_Step_4_all_frames.csv, first frame)'),
    'pbf_validation_cooling_start_T.csv': dict(
        folder='Supplementary_material/Validation_data/allTemperature_50_50', n=100,
        what='rows as pbf_validation_inputs.csv (row i-1 from Validation_data/allTemperature_50_50/TempCV_<i>_Step_4_all_frames.csv, first frame)'),
}


def temperature_text(z, spec):
    """First-frame temperatures of the n files of one folder, one row per file, original text."""
    rows = []
    for i in range(1, spec['n'] + 1):
        member = f'{spec["folder"]}/TempCV_{i}_Step_4_all_frames.csv'
        body = z.read(member).decode('ascii')
        if body.count('\r\n') != body.count('\n') or body.count('\r') != body.count('\n'):
            sys.exit(f'{member}: expected CRLF line endings throughout')        # Abaqus export on Windows
        lines = [ln.split(',') for ln in body.strip().split('\r\n')]
        head, recs = lines[0], lines[1:]
        if head[:6] != ['Element Label', 'Instance', 'IntPt', 'x', 'y', 'z'] or float(head[6]) != 0.0:
            sys.exit(f'{member}: unexpected header {head[:7]}')
        if [tuple(r[:6]) for r in recs] != list(TEMP_POINTS) or any(len(r) != len(head) for r in recs):
            sys.exit(f'{member}: unexpected points or ragged rows')
        vals = [r[6] for r in recs]
        if not all(np.isfinite(float(v)) for v in vals):
            sys.exit(f'{member}: non-finite value')
        rows.append(','.join(vals))
    cols = [f'T_e{p[0]} (degC)' for p in TEMP_POINTS]
    return f'# {TEMP_WHAT}; {spec["what"]}; {CREDIT}\n# {",".join(cols)}\n' + '\n'.join(rows) + '\n', rows


def columns(kind):
    if kind == 'inputs':
        return [f'{n} ({u})' for n, u in zip(NAMES, UNITS)] + [f'{n} ({u})' for n, u in zip(HELD, HELD_UNITS)]
    return [f'vm_{i:03d} (MPa)' for i in range(N_STRESS)]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def same_bits(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def build(zip_path):
    """Return {file name: text} built from the zip, after checking its hash and its members."""
    raw = Path(zip_path).read_bytes()
    if sha256(raw) != SOURCE_SHA256:
        sys.exit(f'{zip_path}: sha256 {sha256(raw)} is not the expected {SOURCE_SHA256}')
    texts = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name, spec in MEMBERS.items():
            data = z.read(spec['member'])
            if sha256(data) != spec['sha256']:
                sys.exit(f'{spec["member"]}: unexpected sha256 {sha256(data)}')
            body = data.decode('ascii')
            if '\r' in body or not body.endswith('\n'):
                sys.exit(f'{spec["member"]}: unexpected line endings')
            src = np.loadtxt(io.StringIO(body), delimiter=',', ndmin=2)
            if src.shape != spec['shape'] or not np.all(np.isfinite(src)):
                sys.exit(f'{spec["member"]}: expected a finite {spec["shape"]} array, got {src.shape}')
            text = f'# {spec["what"]}; {CREDIT}\n# {",".join(columns(spec["kind"]))}\n' + body
            back = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
            if not same_bits(src, back):
                sys.exit(f'{name}: the text does not read back bit for bit')
            texts[name] = text
        for name, spec in TEMP_MEMBERS.items():
            text, rows = temperature_text(z, spec)
            back = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
            src = np.array([[float(v) for v in r.split(',')] for r in rows])
            if back.shape != (spec['n'], len(TEMP_POINTS)) or not same_bits(src, back):
                sys.exit(f'{name}: the text does not read back bit for bit')
            texts[name] = text
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--zip', required=True, help=f'path to {SOURCE_ZIP} ({SOURCE_REPO}, commit {SOURCE_COMMIT[:7]})')
    ap.add_argument('--check', action='store_true', help='compare with the shipped CSVs, write nothing')
    args = ap.parse_args()
    texts = build(args.zip)
    for name, text in texts.items():
        path = HERE / name
        if args.check:
            ok = path.exists() and path.read_text(encoding='ascii') == text
            print(f'{name}: {"identical" if ok else "DIFFERS"}')
            if not ok:
                sys.exit(1)
        else:
            path.write_text(text, encoding='ascii')
            print(f'wrote {name}: {len(text.splitlines()) - 2} rows, {path.stat().st_size / 1e6:.3f} MB')
    print('every file reads back bit for bit')


if __name__ == '__main__':
    main()
