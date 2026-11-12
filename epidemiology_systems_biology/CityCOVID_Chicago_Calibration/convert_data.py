"""Convert the CityCOVID files of Robertson et al. (2025) into this data set's CSV files.

Stored data: the values were computed by the authors with the CityCOVID agent-based model (Argonne
National Laboratory; not public) and published in their code repository
https://github.com/sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest
(MIT licence, commit e4c1e0e30e9e99381d3904fc238bcbf9444cce86 of 2025-03-27). This script only changes
the layout. It reads three files of that repository's ``data/`` folder, checks their SHA-256, and writes:

    citycovid_inputs.csv            700 x 9   the nine stored parameter columns, text unchanged
    citycovid_hospitalizations.csv  700 x 95  seed-averaged hospital census, 2020-03-02 .. 2020-06-04
    citycovid_deaths.csv            700 x 95  seed-averaged cumulative deaths, same days
    observed_chicago.csv            70 x 3    day index, observed cumulative deaths and census

Every number is copied character for character, except two stated rules for the observed file: the
date is written as the day index since 2020-03-02 (day 0), and an empty census entry is written as
``nan``. The leading ``instance`` column (1..700) of the parameter and result files is dropped after a
check that it numbers the rows 1..700 in the same order in both files, so row i of every output file
is instance i + 1. Each output has two ``#`` header lines (a description; column names with units), so
``np.loadtxt(path, delimiter=',')`` reads it.

    python convert_data.py <repo checkout> --out DIR            # write the four CSVs into DIR
    python convert_data.py <repo checkout> --out DIR --check    # also compare DIR with the shipped files

``<repo checkout>`` is the root of the authors' repository (the folder holding ``data/``). ``--out`` is
required and must be outside this data set's directory, so the shipped files are never overwritten.
``--check`` compares every value with the shipped file (absolute tolerance 1e-12, NaN equal to NaN)
and prints "same within tolerance" or "DIFFERENT" per file. Standard library only, plus numpy for --check.
"""

import csv
import datetime
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))

SHA256 = {
    'citycovid_parameters.csv': '1d65d3f26674c3a46471e5f8e735d96541ae80a079a5845f76a2a7fa5af7f80d',
    'citycovid_results.csv': '821c3acd0bfa749230933001de410a10f214122c96964abc91e6ec9f12ecf981',
    'observed_chicago.csv': '6b22b399db50262cab71c6cc885404548695e6e2edf4d1b909c5fa42cce0866e',
}

# Stored label -> (column name, units). Units: the paper (p. 5) calls theta1 "the hourly probability
# of getting exposed from an infected individual"; theta3 and theta4 are probabilities (Table 2); the
# paper states no unit for theta2 (DATA.md, "Unknowns"); the five held parameters have no stated units.
INPUTS = [
    ('Number of initial infected', 'n_initial_infected', 'agents'),
    ('Rate of exposure to infected', 'theta1_exposure_rate', 'probability per hour'),
    ('Seasonality multiplier', 'seasonality_multiplier', '1'),
    ('Shielding by susceptible', 'shielding_by_susceptible', '1'),
    ('Proportion of isolated in homes', 'isolation_in_homes', '1'),
    ('Proportion of isolated in nursing homes', 'isolation_in_nursing_homes', '1'),
    ('Time of initial seeding of infection', 'theta2_seeding_time', 'not stated'),
    ('Probability of stay at home', 'theta3_stay_at_home', 'probability'),
    ('Probability of protective behaviors', 'theta4_protective_behaviors', 'probability'),
]
DAY0 = datetime.date(2020, 3, 2)
NDAYS = 95


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def numbers(fields, where):
    for v in fields:
        float(v)  # raises on a non-number
    return fields


def build(src):
    data = os.path.join(src, 'data')
    for name, ref in SHA256.items():
        got = sha256(os.path.join(data, name))
        if got != ref:
            raise ValueError(f'{name}: SHA-256 {got} differs from the recorded {ref}')

    with open(os.path.join(data, 'citycovid_parameters.csv'), newline='') as fh:
        prm = list(csv.reader(fh))
    if prm[0][0] != 'instance' or prm[0][1:] != [a for a, _, _ in INPUTS]:
        raise ValueError(f'unexpected parameter header {prm[0]}')
    prm = prm[1:]
    if [r[0] for r in prm] != [str(i) for i in range(1, 701)]:
        raise ValueError('parameter instances are not 1..700 in order')

    with open(os.path.join(data, 'citycovid_results.csv'), newline='') as fh:
        res = list(csv.reader(fh))
    kinds, dates, idx = res[0], res[1], res[2]
    dates_expected = [(DAY0 + datetime.timedelta(days=k)).isoformat() + ' 00:00:00' for k in range(NDAYS)]
    if (kinds[1:] != ['hospitalizations'] * NDAYS + ['deaths'] * NDAYS
            or dates[1:] != dates_expected * 2 or idx[0] != 'Instance'):
        raise ValueError('unexpected result header')
    res = res[3:]
    if [r[0] for r in res] != [str(i) for i in range(1, 701)]:
        raise ValueError('result instances are not 1..700 in order')

    out = {}
    cols = [f'{b} [{u}]' for _, b, u in INPUTS]
    rows = [','.join(numbers(r[1:], 'parameters')) for r in prm]
    out['citycovid_inputs.csv'] = (
        '# Robertson et al. 2025 CityCOVID surrogate training design: the nine stored parameter columns of '
        '700 parameter sets (row i = instance i+1; authors\' data/citycovid_parameters.csv; values unchanged)\n'
        '# ' + ','.join(cols) + '\n' + '\n'.join(rows) + '\n')
    days = [(DAY0 + datetime.timedelta(days=k)).isoformat() for k in range(NDAYS)]
    for name, lo, label, desc in [
            ('citycovid_hospitalizations.csv', 1, 'H', 'hospital census (occupied beds), mean over 50 seeds'),
            ('citycovid_deaths.csv', 1 + NDAYS, 'D', 'cumulative deaths, mean over 50 seeds')]:
        rows = [','.join(numbers(r[lo:lo + NDAYS], name)) for r in res]
        out[name] = (
            f'# Robertson et al. 2025 CityCOVID runs, Chicago: daily {desc}, one row per parameter set (row i '
            f'matches row i of citycovid_inputs.csv); columns = days 2020-03-02 .. 2020-06-04 '
            f'(authors\' data/citycovid_results.csv; values unchanged)\n'
            '# ' + ','.join(f'{label}_{d} [persons]' for d in days) + '\n' + '\n'.join(rows) + '\n')

    with open(os.path.join(data, 'observed_chicago.csv'), newline='') as fh:
        obs = list(csv.reader(fh))
    if obs[0] != ['date', 'deaths', 'hospitalizations']:
        raise ValueError(f'unexpected observed header {obs[0]}')
    rows = []
    for r in obs[1:]:
        k = (datetime.date.fromisoformat(r[0]) - DAY0).days
        rows.append(','.join([str(k), numbers([r[1]], 'obs')[0], r[2] if r[2] != '' else 'nan']))
        if r[2] != '':
            float(r[2])
    out['observed_chicago.csv'] = (
        '# Observed COVID-19 cumulative deaths and hospital census in Chicago as stored by Robertson et al. 2025 '
        '(authors\' data/observed_chicago.csv; date written as day index since 2020-03-02; empty census = nan)\n'
        '# day [days since 2020-03-02],deaths [persons],hospitalizations [persons]\n' + '\n'.join(rows) + '\n')
    return out


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if '--out' not in argv or len(args) != 2 or argv.index('--out') + 1 >= len(argv):
        print(__doc__)
        return 2
    src = args[0]
    dest = os.path.realpath(argv[argv.index('--out') + 1])
    if os.path.commonpath([dest, HERE]) == HERE:
        print('--out must be outside this data set\'s directory (the shipped files are never overwritten)')
        return 2
    texts = build(src)
    os.makedirs(dest, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(dest, name), 'w') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    if '--check' in argv:
        import numpy as np
        bad = 0
        for name in texts:
            a = np.loadtxt(os.path.join(dest, name), delimiter=',')
            b = np.loadtxt(os.path.join(HERE, name), delimiter=',')
            same = a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True))
            print(f'{name}: {"same within tolerance" if same else "DIFFERENT"} (tolerance 1e-12)')
            bad += not same
        return int(bad > 0)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        print(f'file not found, or not of the expected kind: {err.filename}\n'
              'The source is a clone of github.com/sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest; see DATA.md, "Regenerating".', file=sys.stderr)
        sys.exit(3)
