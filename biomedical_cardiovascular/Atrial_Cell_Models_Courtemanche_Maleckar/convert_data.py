"""Convert the Figshare design data of Coveney and Clayton (2020) to this repository's CSV format.

S. Coveney and R. H. Clayton, "Sensitivity and Uncertainty Analysis of Two Human Atrial Cardiac
Cell Models Using Gaussian Process Emulators", Front. Physiol. 11:364 (2020),
doi:10.3389/fphys.2020.00364, ran the Courtemanche (1998) and Maleckar (2009) human atrial
myocyte models (MATLAB code generated from CellML, ode15s) on Latin hypercube designs and
published the runs as eight whitespace-separated text files:

    R. Clayton and S. Coveney (2019), "Design data for sensitivity analysis of human atrial
    myocyte models", The University of Sheffield, Dataset, doi:10.15131/shef.data.11309756.v1
    (figshare article 11309756, CC BY 4.0).

    {Courtemanche,Maleckar}Stage{1,2}{Inputs,Outputs}.txt

This script writes one CSV per source file:

    courtemanche_stage1_inputs.csv    295 x 20     courtemanche_stage1_outputs.csv   295 x 11
    courtemanche_stage2_inputs.csv    200 x 10     courtemanche_stage2_outputs.csv   200 x 11
    maleckar_stage1_inputs.csv        286 x 22     maleckar_stage1_outputs.csv       286 x 11
    maleckar_stage2_inputs.csv        195 x 8      maleckar_stage2_outputs.csv       195 x 11

Each CSV has two '#' lines (a description; the column names with units) and then the source rows,
in the source order, with every number copied as the exact decimal text of the source file
(fixed-point, six decimals) and the whitespace replaced by commas. Nothing is rounded or
recomputed, so the conversion is lossless by construction; the script also checks that each CSV
reads back with np.loadtxt equal, bit for bit, to np.loadtxt of the source file.

The module also holds the input names, units, the values printed in Table 1 of the paper, and the
design bounds used by the notebook (LOWER, UPPER), which are inferred from the data where Table 1
is misprinted (see DATA.md, "Inputs").

Usage:
    python convert_data.py --src /path/to/figshare_11309756            # write the CSVs
    python convert_data.py --src /path/to/figshare_11309756 --check    # compare with the shipped CSVs
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_DOI = '10.15131/shef.data.11309756.v1'
SOURCE_URL = ('https://figshare.shef.ac.uk/articles/Design_data_for_sensitivity_analysis_of_human_'
              'atrial_myocyte_models/11309756')
CREDIT = ('Clayton & Coveney (2019), figshare doi:10.15131/shef.data.11309756.v1, CC BY 4.0; '
          'runs for Coveney & Clayton, Front. Physiol. 11:364 (2020)')

# md5 as listed by the figshare API; sha256 computed on the downloaded files (2026-10-31)
SOURCES = {
    'CourtemancheStage1Inputs.txt': ('23fdfad590e19d3d22472901a6959be9',
                                     '5e6cf19e79058dfb39cc85eba29235c8ebff6d5e06cca71ac65a47a76c570817'),
    'CourtemancheStage1Outputs.txt': ('33cd8f4207961b479f3d1e22e4e3f4e1',
                                      '56650670cd14ac37fcd78c3cf1beafddb1af0dbf6bbff62a33b0c715d1fe032e'),
    'CourtemancheStage2Inputs.txt': ('565aafb2516bc1ad45f5b8ae7ced61d7',
                                     '4d5bd8170fb87cbd890b88b2b77fc657c2dcfccdfeb52929633622d4abceef83'),
    'CourtemancheStage2Outputs.txt': ('7117ccbb1cb186f0bb9cbac22571aa76',
                                      '955246cf525ad5bf3b48a3346f4b8de49230b60ae9b03c7290bce1744ddf6dae'),
    'MaleckarStage1Inputs.txt': ('7b2a906afb0315db9c9ad73d3fd58f79',
                                 '2430c8bd0600915cfecb6809be3642b2c7094c33c48d299bacd4e11f7e4a6c16'),
    'MaleckarStage1Outputs.txt': ('10bd9cf818fc40190e6b468518547779',
                                  '55e063b45ed11fb23459f7144f0119f30c4dc3328e294b3b613a9be437252cc1'),
    'MaleckarStage2Inputs.txt': ('3654270661c44a78cd9330864fab72f0',
                                 '2ef7808c9c71cfa565089fe15d0574f997448a884a4b99695f0dbf08f3e64d42'),
    'MaleckarStage2Outputs.txt': ('2fd078545853b05095876de5d7ccdb7f',
                                  '43c61e14217bfe18f3a77c691bf822210ad91e46fe6bb5e30a9cb394df9b9421'),
}

# ---------------------------------------------------------------------------------------------
# Outputs: the 11 biomarkers of Sec. 2.2 (p. 4), in the column order of every *Outputs.txt file.
# Units are not stated in the paper or the data; they are inferred from the values (see DATA.md).
QOI_NAMES = ('dVdt_max', 'Vmax', 'V20', 'V40', 'V60', 'V80', 'APD50', 'APD90', 'RestVm',
             'Ca_min', 'Ca_max')
QOI_UNITS = ('V/s', 'mV', 'mV', 'mV', 'mV', 'mV', 'ms', 'ms', 'mV', 'mM', 'mM')
QOI_LABELS = (r'$dV_m/dt_{max}$', r'$V_{max}$', r'$V_{20}$', r'$V_{40}$', r'$V_{60}$', r'$V_{80}$',
              r'APD$_{50}$', r'APD$_{90}$', r'Rest $V_m$', r'Ca$_{min}$', r'Ca$_{max}$')

# ---------------------------------------------------------------------------------------------
# Inputs. One record per input of Table 1 (p. 3), in the column order of the Stage 1 files:
#   (name, units, Table 1 central value, Table 1 printed range, half-width fraction, centre used,
#    note). The design range is centre * (1 -/+ fraction), except for Maleckar GbCa (see below).
#   'centre used' differs from Table 1 where the data contradict the printed value; it is the
#   centre that puts the Stage 1 points one per stratum of a 300-point Latin hypercube (DATA.md).
_COURTEMANCHE = [
    ('GNa', 'nS/pF', 7.8, '5.85-11.70 (+-50%)', 0.50, 7.8, 'printed lower end is 0.75x centre'),
    ('GK1', 'nS/pF', 0.09, '0.045-0.018 (+-25%)', 0.25, 0.09, 'printed range misprinted'),
    ('Gto', 'nS/pF', 0.165, '0.0826-0.0330 (+-50%)', 0.50, 0.1652, 'printed range misprinted; data reach 0.2476 > 1.5 x 0.165'),
    ('fGKur', '1', 1.0, '0.50-1.50 (+-50%)', 0.50, 1.0, ''),
    ('GKr', 'nS/pF', 0.0294, '0.0147-0.0441 (+-50%)', 0.50, 0.029412, 'centre refined from the LHS strata'),
    ('GKs', 'nS/pF', 0.1294, '0.0647-0.1941 (+-50%)', 0.50, 0.12941, 'centre refined from the LHS strata'),
    ('GCaL', 'nS/pF', 0.1237, '0.0619-0.1856 (+-50%)', 0.50, 0.12375, 'data reach 0.185619 > 1.5 x 0.1237'),
    ('GbNa', 'nS/pF', 0.0006, '0.0003-0.0010 (+-50%)', 0.50, 0.00067444, 'data centre 0.000674, Table 1 rounds to 0.0006'),
    ('GbCa', 'nS/pF', 0.0011, '0.0005-0.0017 (+-50%)', 0.50, 0.001131, 'data reach 0.001695 > 1.5 x 0.0011'),
    ('INaKmax', 'pA/pF', 0.5993, '0.2997-0.8990 (+-50%)', 0.50, 0.59933, 'centre refined from the LHS strata'),
    ('INaCamax', 'pA/pF', 1600.0, '800.00-2400.0 (+-50%)', 0.50, 1600.0, ''),
    ('IpCamax', 'pA/pF', 0.275, '0.1375-0.4125 (+-50%)', 0.50, 0.275, ''),
    ('Krel', '1/ms', 0.30, '0.15-0.45 (+-50%)', 0.50, 30.0, 'data span 15.04-44.91: 100 x the printed values'),
    ('tau_tr', 'ms', 180.0, '90.0-270.0 (+-50%)', 0.50, 180.0, ''),
    ('Iupmax', 'mM/ms', 0.005, '0.0025-0.0075 (+-50%)', 0.50, 0.005, ''),
    ('Kup', 'mM', 0.00092, '0.00046-0.0014 (+-50%)', 0.50, 0.00092, ''),
    ('Cm', 'pF', 100.0, '75.0-125.0 (+-25%)', 0.25, 100.0, ''),
    ('Nao', 'mM', 140.0, '126.0-154.0 (+-10%)', 0.10, 140.0, ''),
    ('Ko', 'mM', 5.4, '4.86-5.94 (+-10%)', 0.10, 5.4, ''),
    ('Cao', 'mM', 1.8, '1.62-1.98 (+-10%)', 0.10, 1.8, ''),
]
_MALECKAR = [
    ('PNa', 'nL/s', 0.0018, '0.0009-0.0027 (+-50%)', 0.50, 0.0018, ''),
    ('GK1', 'nS', 3.1, '2.325-3.875 (+-25%)', 0.25, 3.1, ''),
    ('Gt', 'nS', 8.25, '4.125-12.375 (+-50%)', 0.50, 8.25, ''),
    ('GKur', 'nS', 2.25, '1.125-3.375 (+-50%)', 0.50, 2.25, ''),
    ('GKr', 'nS', 0.5, '0.250-0.750 (+-50%)', 0.50, 0.5, ''),
    ('GKs', 'nS', 1.0, '0.50-1.50 (+-50%)', 0.50, 1.0, ''),
    ('GCaL', 'nS', 6.75, '3.375-10.125 (+-50%)', 0.50, 6.75, ''),
    ('GbNa', 'nS', 0.0605, '0.303-0.909 (+-50%)', 0.50, 0.0606, 'printed range 10x too large; data 0.0304-0.0907'),
    ('GbCa', 'nS', 0.0590, '0.0295-0.0885 (+-50%)', None, None, 'data span 0.0080-0.0862: not +-50% of 0.0590; bounds from the LHS strata'),
    ('INaKmax', 'pA', 68.55, '34.27-102.82 (+-50%)', 0.50, 68.55, ''),
    ('KNaCa', 'pA/mM^4', 0.0750, '0.0375-0.1125 (+-50%)', 0.50, 0.037484, 'data span 0.0188-0.0562: centre is half the printed value'),
    ('IpCamax', 'pA', 4.0, '2.0-6.0 (+-50%)', 0.50, 4.0, ''),
    ('alpha_rel', 'pA/mM', 200000.0, '100,000-300,000 (+-50%)', 0.50, 200000.0, ''),
    ('tau_tr', 's', 0.01, '0.005-0.015 (+-50%)', 0.50, 0.01, ''),
    ('Iupmax', 'pA', 2800.0, '1,400-4,300 (+-50%), central printed 28,00', 0.50, 2800.0, 'printed central 28,00 and upper end 4,300; data 1402-4196'),
    ('Kcyca', 'mM', 0.0003, '0.00015-0.00045 (+-50%)', 0.50, 0.0003, ''),
    ('Ksrca', 'mM', 0.5, '0.25-0.75 (+-50%)', 0.50, 0.5, ''),
    ('Kxcs', '1', 0.4, '0.2-0.6 (+-50%)', 0.50, 0.4, ''),
    ('Cm', 'pF', 50.0, '37.5-62.5 (+-25%)', 0.25, 50.0, ''),
    ('Nao', 'mM', 140.0, '126.0-154.0 (+-10%)', 0.10, 140.0, ''),
    ('Ko', 'mM', 5.4, '4.86-5.94 (+-10%)', 0.10, 5.4, ''),
    ('Cao', 'mM', 1.8, '1.62-1.98 (+-10%)', 0.10, 1.8, ''),
]
MALECKAR_GBCA_BOUNDS = (0.0078656, 0.086548)   # inferred from the Stage 1 Latin hypercube (DATA.md)
DI_RECORD = ('DI', 'ms', 250.0, '50-450 (Sec. 2.3)', 0.80, 250.0, 'diastolic interval of the S2 beat')

# Stage 2 inputs (Sec. 3.2, p. 8; the column order is that of Figs. 7-8 and of the files).
# The text lists 8 Courtemanche inputs; the file and Figs. 7A/8A also have GKr (DATA.md).
STAGE2 = {
    'courtemanche': ('GK1', 'Gto', 'fGKur', 'GKr', 'GCaL', 'GbCa', 'INaKmax', 'INaCamax', 'IpCamax', 'DI'),
    'maleckar': ('GK1', 'Gt', 'GKur', 'GCaL', 'GbCa', 'INaKmax', 'Cm', 'DI'),
}

# Variances the paper assigns to the inputs, in normalized units u in [0, 1] with mean 0.5, for
# its Sobol' indices (Sec. 2.7, p. 6): 0.02 for the +-50% inputs, 0.04 for GK1 and Cm, 0.1 for the
# extracellular concentrations. The variance of DI (Stage 2) is not stated; 0.02 is our choice.
PAPER_VARIANCE = {'GK1': 0.04, 'Cm': 0.04, 'Nao': 0.1, 'Ko': 0.1, 'Cao': 0.1, 'DI': 0.02}
PAPER_VARIANCE_DEFAULT = 0.02


def _records(model):
    return {'courtemanche': _COURTEMANCHE, 'maleckar': _MALECKAR}[model] + [DI_RECORD]


def inputs(model, stage):
    """Input records of a model ('courtemanche' or 'maleckar') and stage (1 or 2), in column order.

    Each record is a dict with keys name, units, table_central, table_range, fraction, centre,
    lower, upper, note, paper_variance.
    """
    recs = {r[0]: r for r in _records(model)}
    names = [r[0] for r in _records(model)[:-1]] if stage == 1 else list(STAGE2[model])
    out = []
    for n in names:
        name, units, tc, tr, frac, centre, note = recs[n]
        if model == 'maleckar' and n == 'GbCa':
            lo, hi = MALECKAR_GBCA_BOUNDS
        else:
            lo, hi = centre * (1 - frac), centre * (1 + frac)
        out.append(dict(name=name, units=units, table_central=tc, table_range=tr, fraction=frac,
                        centre=centre, lower=lo, upper=hi, note=note,
                        paper_variance=PAPER_VARIANCE.get(n, PAPER_VARIANCE_DEFAULT)))
    return out


def bounds(model, stage):
    """(names, lower, upper) of the design of a model and stage, physical units."""
    rec = inputs(model, stage)
    return ([r['name'] for r in rec], np.array([r['lower'] for r in rec]),
            np.array([r['upper'] for r in rec]))


def csv_name(model, stage, kind):
    return f'{model}_stage{stage}_{kind}.csv'


def source_name(model, stage, kind):
    return f'{model.capitalize()}Stage{stage}{kind.capitalize()}.txt'


def _header(model, stage, kind):
    label = {'courtemanche': 'Courtemanche (1998)', 'maleckar': 'Maleckar (2009)'}[model]
    if kind == 'inputs':
        cols = [f"{r['name']} ({r['units']})" for r in inputs(model, stage)]
        what = ('inputs (physical units) of the ' + ('Stage 1 Latin hypercube, 1000 ms pacing' if stage == 1
                else 'Stage 2 Latin hypercube, S1-S2 pacing with diastolic interval DI'))
    else:
        cols = [f'{n} ({u})' for n, u in zip(QOI_NAMES, QOI_UNITS)]
        what = 'biomarkers of the final beat (units inferred) of the ' + (
            'Stage 1' if stage == 1 else 'Stage 2') + ' runs, row-aligned with the inputs file'
    return f'{label} human atrial cell model, {what}; {CREDIT}', cols


def _digest(path, algo):
    h = hashlib.new(algo)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build(src):
    """Return {csv name: text} built from the source directory, after checking hashes and values."""
    src = Path(src)
    texts = {}
    for model in ('courtemanche', 'maleckar'):
        for stage in (1, 2):
            for kind in ('inputs', 'outputs'):
                sname = source_name(model, stage, kind)
                path = src / sname
                md5, sha = SOURCES[sname]
                if _digest(path, 'md5') != md5 or _digest(path, 'sha256') != sha:
                    sys.exit(f'{path}: checksum differs from the figshare file')
                rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
                ncol = {len(r) for r in rows}
                description, cols = _header(model, stage, kind)
                if ncol != {len(cols)}:
                    sys.exit(f'{sname}: expected {len(cols)} columns, found {sorted(ncol)}')
                out = io.StringIO()
                out.write(f'# {description}\n# {",".join(cols)}\n')
                for r in rows:
                    out.write(','.join(r) + '\n')
                text = out.getvalue()
                a = np.loadtxt(path, ndmin=2)
                b = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
                if a.shape != b.shape or not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
                    sys.exit(f'{sname}: the CSV does not read back bit for bit')
                texts[csv_name(model, stage, kind)] = text
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True, help='directory with the eight figshare .txt files')
    ap.add_argument('--check', action='store_true', help='compare with the shipped CSVs, write nothing')
    args = ap.parse_args()
    texts = build(args.src)
    bad = 0
    for name, text in texts.items():
        path = HERE / name
        if args.check:
            same = path.exists() and path.read_text() == text
            bad += not same
            print(f'{name}: {"identical" if same else "DIFFERENT"}')
        else:
            path.write_text(text)
            print(f'wrote {name} ({text.count(chr(10)) - 2} rows)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
