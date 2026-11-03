"""Convert the deposited Zenodo files of Strocchi et al. (2023) into this data set's CSV files.

Stored data: the values were computed by the authors (CARP + CircAdapt four-chamber simulator; ToR-ORd
cell model) and deposited at Zenodo, doi:10.5281/zenodo.7405335 (CC BY 4.0). This script only changes
the file layout: it reads ``parameters.csv`` and ``outputs.csv`` straight from the deposited
``.tar.xz`` archives (nothing is extracted or executed), checks the archives' SHA-256, and writes each
data row's text unchanged after two ``#`` header lines (a description; column names with units), so
``np.loadtxt(path, delimiter=',')`` reads the files. The conversion is lossless: every number is
copied character for character.

    python convert_data.py <folder with the .tar.xz archives>          # write the four CSV files
    python convert_data.py <folder with the .tar.xz archives> --check  # compare with the shipped files

Standard library only (csv, hashlib, tarfile).
"""

import csv
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))

SHA256 = {
    'fourchamber.tar.xz': 'd6c710dbdfdb7ddcd4ea86775fd024bfe3287f0d39262cf12ea77086e553a87d',
    'ventricular_cell_model.tar.xz': '53396893097e6ddc58353b5b960c5df1b293d2864e6f0c1fcb2fda5532140358',
}

# Four-chamber simulator: deposited label -> (column name, units). Units as in the paper (Tables 1-3)
# where it gives them (ms, mm, kPa, mmHg); m/s for CV and kPa/um for kperi are inferred (DATA.md);
# "model" = the units of the cell or tissue model, which the paper does not state; "1" = a scaling factor
# or dimensionless constant.
FC_INPUTS = [
    ('PCa', 'PCa', 'model'), ('[TRPN]', 'TRPN_v', 'model'), ('Gncx', 'GNCX', 'model'),
    ('Tref,v', 'Tref_v', 'model'), ('TRPN50,v', 'TRPN50_v', '1'), ('nTm,v', 'nTm_v', '1'),
    ('nTRPN,v', 'nTRPN_v', '1'), ('rs,v', 'rs_v', '1'), ('rw,v', 'rw_v', '1'), ('Aeff,v', 'Aeff_v', '1'),
    ('ku,v', 'ku_v', 'model'), ('ca50,v', 'ca50_v', 'model'), ('mu,v', 'mu_v', '1'),
    ('Iup(max)', 'Iup_max_a', 'model'), ('[Trpm]max', 'Trpn_max_a', 'model'), ('gCaL', 'gCaL_a', 'model'),
    ('Tref,a', 'Tref_a', 'model'), ('TRPN50,a', 'TRPN50_a', '1'), ('nTm,a', 'nTm_a', '1'),
    ('nTRPN,a', 'nTRPN_a', '1'), ('rs,a', 'rs_a', '1'), ('rw,a', 'rw_a', '1'), ('Aeff,a', 'Aeff_a', '1'),
    ('phi,a', 'phi_a', '1'), ('ca50,a', 'ca50_a', 'model'), ('mu,a', 'mu_a', '1'),
    ('CVf,v', 'CVf_v', 'm/s'), ('kFEC', 'kFEC', '1'), ('CVf,a', 'CVf_a', 'm/s'), ('kBB', 'kBB', '1'),
    ('AVdelay', 'AVdelay', 'ms'), ('Rsys', 'Rsys', '1'), ('Rpulm', 'Rpulm', '1'), ('lAo', 'lAo', 'mm'),
    ('kAo', 'kAo', '1'), ('a,v', 'a_v', 'kPa'), ('bt,v', 'bt_v', '1'), ('a,a', 'a_a', 'kPa'),
    ('bf,a', 'bf_a', '1'), ('bt,a', 'bt_a', '1'), ('kperi', 'kperi', 'kPa/um'),
    ('EDPshift,LV', 'EDPshift_LV', 'mmHg'), ('EDPunload,RV', 'EDPunload_RV', 'mmHg'),
    ('aLvRv', 'aLvRv', '1'), ('TrefLvRv', 'TrefLvRv', '1'),
]
FC_OUTPUTS = [
    ('EDV,lv', 'EDV_LV', 'mL'), ('EDP,lv', 'EDP_LV', 'mmHg'), ('ESV,lv', 'ESV_LV', 'mL'),
    ('pMax,lv', 'pmax_LV', 'mmHg'), ('dp/dtMax,lv', 'dpdt_max_LV', 'mmHg/s'),
    ('dp/dtMin,lv', 'dpdt_min_LV', 'mmHg/s'),
    ('EDV,rv', 'EDV_RV', 'mL'), ('EDP,rv', 'EDP_RV', 'mmHg'), ('ESV,rv', 'ESV_RV', 'mL'),
    ('pMax,rv', 'pmax_RV', 'mmHg'), ('dp/dtMax,rv', 'dpdt_max_RV', 'mmHg/s'),
    ('dp/dtMin,rv', 'dpdt_min_RV', 'mmHg/s'),
    ('EDV,la', 'EDV_LA', 'mL'), ('ESV,la', 'ESV_LA', 'mL'), ('VMax,la', 'EDVvwave_LA', 'mL'),
    ('pMax,La', 'pmax_LA', 'mmHg'),
    ('EDV,ra', 'EDV_RA', 'mL'), ('ESV,ra', 'ESV_RA', 'mL'), ('VMax,ra', 'EDVvwave_RA', 'mL'),
    ('pMax,ra', 'pmax_RA', 'mmHg'),
]
# ToR-ORd ionic model (cell stage, S2 File): units of the ToR-ORd model are not given in the paper.
TO_INPUTS = [
    ('GNa', 'GNa'), ('GNaL', 'GNaL'), ('Gto', 'Gto'), ('PCa', 'PCa'), ('GKr', 'GKr'), ('GKs', 'GKs'),
    ('GK1', 'GK1'), ('Gncx', 'GNCX'), ('GNaK', 'GNaK'), ('Jrel', 'Jrel'), ('Jup', 'Jup'), ('GCa', 'GCa'),
    ('GKb', 'GKb'), ('PNab', 'PNab'), ('PCab', 'PCab'), ('GClCa', 'GClCa'), ('GClb', 'GClb'),
    ('INaCa,SS', 'INaCa_SS'), ('ICaL,SS', 'ICaL_SS'), ('aCaMK', 'aCaMK'), ('bCaMK', 'bCaMK'),
    ('CaMKo', 'CaMKo'), ('[CMDN]', 'CMDN'), ('[TRPN]', 'TRPN'), ('[BSR]', 'BSR'), ('[BSL]', 'BSL'),
    ('[CSQN]', 'CSQN'), ('tau,diffCa', 'tau_diffCa'), ('tau,tr', 'tau_tr'),
]
TO_OUTPUTS = [
    ('Ca,diast,v', 'Ca_diast', 'uM'), ('Ca,ampl,v', 'Ca_ampl', 'uM'), ('TTP,ca,v', 'TTP_Ca', 'ms'),
    ('RT90,v', 'RT90_Ca', 'ms'),
]

FILES = [
    # (archive, member, expected deposited labels, column names with units, output file, description)
    ('fourchamber.tar.xz', 'fourchamber/parameters.csv', [a for a, _, _ in FC_INPUTS],
     [f'{b} [{u}]' for _, b, u in FC_INPUTS], 'fourchamber_inputs.csv',
     'Strocchi et al. 2023 four-chamber electromechanics: 45 inputs of the 407 successful runs '
     '(Zenodo 10.5281/zenodo.7405335 fourchamber/parameters.csv; physical units; values unchanged)'),
    ('fourchamber.tar.xz', 'fourchamber/outputs.csv', [a for a, _, _ in FC_OUTPUTS],
     [f'{b} [{u}]' for _, b, u in FC_OUTPUTS], 'fourchamber_outputs.csv',
     'Strocchi et al. 2023 four-chamber electromechanics: 20 pressure-volume features of the 407 runs, '
     'row i matches row i of fourchamber_inputs.csv (Zenodo fourchamber/outputs.csv; values unchanged)'),
    ('ventricular_cell_model.tar.xz', 'ventricular_cell_model/ionic/parameters.csv',
     [a for a, _ in TO_INPUTS], [f'{b} [model]' for _, b in TO_INPUTS], 'torord_ionic_inputs.csv',
     'Strocchi et al. 2023 ToR-ORd ionic model (S2 File): 29 inputs of 2175 runs '
     '(Zenodo ventricular_cell_model/ionic/parameters.csv; values unchanged)'),
    ('ventricular_cell_model.tar.xz', 'ventricular_cell_model/ionic/outputs.csv',
     [a for a, _, _ in TO_OUTPUTS], [f'{b} [{u}]' for _, b, u in TO_OUTPUTS], 'torord_ionic_outputs.csv',
     'Strocchi et al. 2023 ToR-ORd ionic model: 4 calcium-transient features of the 2175 runs, '
     'row i matches torord_ionic_inputs.csv (Zenodo ventricular_cell_model/ionic/outputs.csv; unchanged)'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_member(archive, member):
    """Return (header labels, list of data-row strings) of one CSV inside a .tar.xz, without extracting."""
    with tarfile.open(archive, 'r:xz') as tf:
        info = tf.getmember(member)
        if not info.isfile():
            raise ValueError(f'{member} in {archive} is not a regular file')
        text = tf.extractfile(info).read().decode('ascii')
    lines = text.splitlines()
    header = next(csv.reader([lines[0]]))
    rows = [ln for ln in lines[1:] if ln.strip()]
    for ln in rows:
        vals = ln.split(',')
        if len(vals) != len(header):
            raise ValueError(f'{member}: a row has {len(vals)} fields, header {len(header)}')
        [float(v) for v in vals]  # every field is a plain number (raises otherwise)
    return header, rows


def build(src):
    out = {}
    for arc in sorted({f[0] for f in FILES}):
        got = sha256(os.path.join(src, arc))
        if got != SHA256[arc]:
            raise ValueError(f'{arc}: SHA-256 {got} differs from the recorded {SHA256[arc]}')
    for arc, member, labels, cols, name, desc in FILES:
        header, rows = read_member(os.path.join(src, arc), member)
        if header != labels:
            raise ValueError(f'{member}: header {header} differs from the expected labels')
        out[name] = f'# {desc}\n# ' + ','.join(cols) + '\n' + '\n'.join(rows) + '\n'
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    texts = build(argv[1])
    if '--check' in argv:
        bad = 0
        for name, text in texts.items():
            with open(os.path.join(HERE, name)) as fh:
                same = fh.read() == text
            print(f'{name}: {"identical" if same else "DIFFERENT"}')
            bad += not same
        return int(bad > 0)
    for name, text in texts.items():
        with open(os.path.join(HERE, name), 'w') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
