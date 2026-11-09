"""Extract this data set's subset of the Urban Building Energy Stock Datasets (Ali et al. 2024).

Stored data: the values were computed by Ali et al. with jEPlus and EnergyPlus (DesignBuilder
construction templates) and deposited at Mendeley Data, doi:10.17632/m6vv9k9gcd.1 (version 1,
CC BY 4.0), as one CSV of 1,030,022 rows and 32 columns inside a zip archive. The full table is
too large to ship, so this script keeps a documented subset:

    rows:    Building_Type == "Semi Detached", Weather_File == "Historical",
             Renewable_Energy_Usage == "No" (49,298 rows), in their original order;
    columns: the 15 numeric inputs (file 1) and the 12 numeric outputs (file 2). The four
             categorical columns are fixed by the row rule, and the two rating labels
             (Simple_Building_Energy_Rating, Building_Energy_Rating) are text classes of the EUI;
             they are not shipped.

The script reads the CSV straight from the deposited zip (nothing is extracted or executed),
checks the archive's SHA-256, and copies the text of every kept field character for character
after two '#' header lines (a description; column names with units), so
np.loadtxt(path, delimiter=',') reads the files. The second file starts with ``source_row``, the
1-based data-row number of the row in the deposited CSV (header excluded), so each shipped row
can be traced. Within the kept rows and columns the conversion is lossless.

    python convert_data.py <zip> --out DIR            # write the two CSV files into DIR
    python convert_data.py <zip> --out DIR --check    # write into DIR and compare with the shipped files

--out is required and must lie outside this directory and its subdirectories, so the script never
writes into the data set (copy new files in by hand after comparing them).
The check compares the files' text and, field by field, their values (relative tolerance 1e-12),
and prints both results. Standard library only (csv, hashlib, io, zipfile).
"""

import argparse
import csv
import hashlib
import io
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

ZIP_NAME = 'Urban Building Energy Stock Datasets.zip'
ZIP_SHA256 = '1f6bb3e1dfcb7ebefbd1a75f93d56dd12db2923aab58717f077ab4a643c42c29'
ZIP_BYTES = 52851275
MEMBER = 'Urban Building Energy Stock Datasets/urban_building_stock_datasets.csv'
N_ROWS_DEPOSIT = 1030022

ROW_RULE = {'Building_Type': 'Semi Detached', 'Weather_File': 'Historical',
            'Renewable_Energy_Usage': 'No'}

# deposited label -> (column name, units). Units as in the paper's Tables 2 and 3 (pp. 4-5);
# HVAC_Efficiency is printed with "%" in Table 2 but holds an efficiency or COP as a fraction
# (0.3-4.5; Table 4 "HVAC Efficiency/COP"), so its unit here is 1.
INPUTS = [
    ('Floor_Insulation_U-Value', 'U_floor', 'W/m2K'),
    ('Door_Insulation_U-Value', 'U_door', 'W/m2K'),
    ('Roof_Insulation_U-Value', 'U_roof', 'W/m2K'),
    ('Window_Insulation_U-Value', 'U_window', 'W/m2K'),
    ('Wall_Insulation_U-Value', 'U_wall', 'W/m2K'),
    ('HVAC_Efficiency', 'HVAC_efficiency', '1'),
    ('Domestic_Hot_Water_Usage', 'DHW_use', 'L/m2/day'),
    ('Building_Orientation', 'orientation', 'deg'),
    ('Lighting_Density', 'lighting_density', 'W/m2'),
    ('Occupancy_Level', 'occupancy', 'persons'),
    ('Equipment_Density', 'equipment_density', 'W/m2'),
    ('Heating_Setpoint_Temperature', 'heating_setpoint', 'degC'),
    ('Heating_Setback_Temperature', 'heating_setback', 'degC'),
    ('Air_Change_Rate', 'air_change_rate', '1/h'),
    ('Window_to_Wall_Ratio', 'window_to_wall_ratio', 'percent'),
]
OUTPUTS = [
    ('Total_Building_Area', 'floor_area', 'm2'),
    ('Heating_Usage', 'heating_usage', 'kWh/yr'),
    ('Water_Systems_Energy', 'water_systems_energy', 'kWh/yr'),
    ('Interior_Lighting_Energy', 'lighting_energy', 'kWh/yr'),
    ('Interior_Equipment_Energy', 'equipment_energy', 'kWh/yr'),
    ('Total_Heating_Energy', 'space_heating_energy', 'kWh/yr'),
    ('Photovoltaic_Power', 'pv_energy', 'kWh/yr'),
    ('Total_Electricity_Energy', 'electricity_energy', 'kWh/yr'),
    ('Heating_Conversion_Factor', 'heating_conversion_factor', '1'),
    ('Electricity_Primary_Conversion_Factor', 'electricity_primary_factor', '1'),
    ('Heating_Primary_Conversion_Factor', 'heating_primary_factor', '1'),
    ('Energy_Use_Intensity', 'EUI', 'kWh/m2/yr'),
]
NOT_SHIPPED = ['Simple_Building_Energy_Rating', 'Building_Energy_Rating'] + list(ROW_RULE)

FILES = {
    'building_inputs.csv':
        'Ali et al. 2024 urban residential building stock (Mendeley Data 10.17632/m6vv9k9gcd.1, '
        'CC BY 4.0): the 15 numeric inputs of the 49298 rows with Building_Type = Semi Detached, '
        'Weather_File = Historical, Renewable_Energy_Usage = No (a subset of 1030022 rows; values '
        'unchanged)',
    'building_outputs.csv':
        'Ali et al. 2024 urban residential building stock: source_row (1-based data row in the '
        'deposited CSV) and the 12 numeric outputs of the same 49298 rows; row i matches row i of '
        'building_inputs.csv (values unchanged)',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(zip_path):
    """Return {file name: text} of the two CSV files built from the deposited archive."""
    got = sha256(zip_path)
    if got != ZIP_SHA256:
        raise ValueError(f'archive SHA-256 {got} differs from the recorded {ZIP_SHA256}')
    with zipfile.ZipFile(zip_path) as zf:
        if zf.namelist() != [MEMBER]:
            raise ValueError(f'unexpected archive members {zf.namelist()}')
        with zf.open(MEMBER) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding='utf-8', newline=''))
            header = next(reader)
            expected = set(a for a, _, _ in INPUTS + OUTPUTS) | set(NOT_SHIPPED)
            if set(header) != expected or len(header) != len(expected):
                raise ValueError(f'deposited header {header} differs from the expected labels')
            col = {name: j for j, name in enumerate(header)}
            rule = [(col[k], v) for k, v in ROW_RULE.items()]
            ji = [col[a] for a, _, _ in INPUTS]
            jo = [col[a] for a, _, _ in OUTPUTS]
            xin, xout, n = [], [], 0
            for n, row in enumerate(reader, start=1):
                if len(row) != len(header):
                    raise ValueError(f'data row {n} has {len(row)} fields, header {len(header)}')
                if all(row[j] == v for j, v in rule):
                    vin = [row[j] for j in ji]
                    vout = [row[j] for j in jo]
                    for v in vin + vout:
                        float(v)  # every kept field is a plain number (raises otherwise)
                    xin.append(','.join(vin))
                    xout.append(str(n) + ',' + ','.join(vout))
    if n != N_ROWS_DEPOSIT:
        raise ValueError(f'deposited CSV has {n} data rows, expected {N_ROWS_DEPOSIT}')
    cols_in = [f'{b} [{u}]' for _, b, u in INPUTS]
    cols_out = ['source_row [1]'] + [f'{b} [{u}]' for _, b, u in OUTPUTS]
    out = {}
    for name, cols, lines in (('building_inputs.csv', cols_in, xin),
                              ('building_outputs.csv', cols_out, xout)):
        out[name] = f'# {FILES[name]}\n# ' + ','.join(cols) + '\n' + '\n'.join(lines) + '\n'
    return out


def values(text):
    return [[float(v) for v in ln.split(',')] for ln in text.splitlines() if ln and ln[0] != '#']


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('zip', help=f'path of the deposited archive "{ZIP_NAME}"')
    p.add_argument('--out', help='directory to write the two CSV files into (required)')
    p.add_argument('--check', action='store_true',
                   help='compare the files written into --out with the shipped files')
    a = p.parse_args(argv)
    if a.out is None:
        sys.exit('give --out DIR: the directory to write the two CSV files into')
    out_real = os.path.realpath(a.out)
    if os.path.commonpath([out_real, os.path.realpath(HERE)]) == os.path.realpath(HERE):
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it), "
                 'so that nothing is written into the data set')
    texts = build(a.zip)
    os.makedirs(a.out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(a.out, name), 'w', newline='\n') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    if not a.check:
        return 0
    bad = 0
    for name, text in texts.items():
        with open(os.path.join(HERE, name)) as fh:
            shipped = fh.read()
        same_text = shipped == text
        new, old = values(text), values(shipped)
        ok = len(new) == len(old) and all(
            len(r) == len(s) and all(abs(x - y) <= 1e-12 * max(abs(x), abs(y)) for x, y in zip(r, s))
            for r, s in zip(new, old))
        print(f'{name}: text {"identical" if same_text else "DIFFERENT"}; values '
              f'{"equal within relative 1e-12" if ok else "DIFFERENT"}')
        bad += not ok
    return int(bad > 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
