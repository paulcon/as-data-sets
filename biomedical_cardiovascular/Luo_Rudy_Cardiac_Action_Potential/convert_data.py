"""Convert the PLOS ONE supplements S1 Data and S2 Data to this repository's CSV format.

Chang, Strong and Clayton (2015), doi:10.1371/journal.pone.0130252, supply the
Luo-Rudy 1991 simulator runs as two CSV files with a text header row:

    original/pone.0130252.s002.csv   S1 Data, 200 design runs (Latin hypercube)
    original/pone.0130252.s003.csv   S2 Data, 20 test runs

Each row has the six maximum conductances in normalized units u in [0, 1] and the
eight outputs, all written with 4 decimals. This script writes

    design.csv, test.csv   two '#' header lines, then 14 comma-separated columns:
                           the six conductances in mS/cm^2, then the eight outputs

The conversion is lossless:
  * the outputs are copied character for character;
  * each conductance is G = G0 * (3/4 + u/2), i.e. G0 - G0/4 at u = 0 and G0 + G0/4 at
    u = 1 (paper p. 4), computed in exact decimal arithmetic and written with every
    digit, so u = (G/G0 - 3/4) * 2 recovers the published 4-decimal value exactly.
The script checks both properties and that np.loadtxt reads the files back.

Usage:
    python convert_data.py            # write design.csv and test.csv
    python convert_data.py --check    # rebuild in memory and compare with the shipped files
"""

import argparse
import io
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# LR1991 maximum conductances (mS/cm^2), the centre (u = 0.5) of each range; paper Table 1, p. 5
NAMES = ('GNa', 'Gsi', 'GK', 'GK1', 'GKp', 'Gb')
G0 = tuple(Decimal(s) for s in ('23.0', '0.09', '0.282', '0.6047', '0.0183', '0.03921'))
OUTPUTS = ('dVdt_max', 'Vm_max', 'V_dome', 'APD90', 'V_rest', 'APD50', 'APDr_slope', 'DI_min')
UNITS = ('mV/ms', 'mV', 'mV', 'ms', 'mV', 'ms', '1', 'ms')
# the header of the PLOS files, in the same order
PLOS_HEADER = ('G Na,G si,G K,G K1,G Kp,G b,dVdt max (mV/ms),Vm max (mV),Dome V (mV),'
               'APD90 (ms),Rest v (mV),APD50 (ms),APDr slope,Min DI (ms)')

FILES = {
    'design.csv': ('pone.0130252.s002.csv', 200,
                   'Luo-Rudy 1991 design runs, S1 Data of Chang, Strong & Clayton, PLoS ONE 10(6):e0130252 '
                   '(2015), doi:10.1371/journal.pone.0130252, CC BY 4.0; conductances mapped from [0,1] '
                   'to G0*(3/4 + u/2) by convert_data.py, outputs unchanged'),
    'test.csv': ('pone.0130252.s003.csv', 20,
                 'Luo-Rudy 1991 test runs, S2 Data of Chang, Strong & Clayton, PLoS ONE 10(6):e0130252 '
                 '(2015), doi:10.1371/journal.pone.0130252, CC BY 4.0; conductances mapped from [0,1] '
                 'to G0*(3/4 + u/2) by convert_data.py, outputs unchanged'),
}

LOWER = np.array([float(g * Decimal('0.75')) for g in G0])   # exact range ends, mS/cm^2
UPPER = np.array([float(g * Decimal('1.25')) for g in G0])
NOMINAL = np.array([float(g) for g in G0])


def to_physical(u):
    """Exact decimal G0 * (3/4 + u/2) for a normalized input string u."""
    return [g * (Decimal('0.75') + Decimal(s) / 2) for g, s in zip(G0, u)]


def convert(src, nrows, description):
    """Return the text of the converted file for one PLOS CSV."""
    lines = Path(src).read_text(encoding='ascii').splitlines()
    if lines[0] != PLOS_HEADER:
        raise ValueError(f'{src}: unexpected header {lines[0]!r}')
    rows = [ln.split(',') for ln in lines[1:]]
    if len(rows) != nrows or any(len(r) != 14 for r in rows):
        raise ValueError(f'{src}: expected {nrows} rows of 14 fields')
    out = io.StringIO()
    out.write(f'# {description}\n')
    out.write('# ' + ','.join([f'{n} (mS/cm^2)' for n in NAMES]
                              + [f'{n} ({u})' for n, u in zip(OUTPUTS, UNITS)]) + '\n')
    for r in rows:
        G = to_physical(r[:6])
        back = [(g / g0 - Decimal('0.75')) * 2 for g, g0 in zip(G, G0)]
        if any(b != Decimal(s) for b, s in zip(back, r[:6])):
            raise ArithmeticError(f'{src}: inverse map does not recover {r[:6]}')
        if any(not (Decimal(0) <= Decimal(s) <= Decimal(1)) for s in r[:6]):
            raise ValueError(f'{src}: normalized input outside [0, 1]: {r[:6]}')
        out.write(','.join([format(g.normalize(), 'f') for g in G] + r[6:]) + '\n')
    text = out.getvalue()

    # read-back checks: loadtxt parses it; the inputs map back to the published 4 decimals
    # in floating point too; the outputs are the published values
    A = np.loadtxt(io.StringIO(text), delimiter=',')
    Uf = (A[:, :6] - LOWER) / (UPPER - LOWER)
    U0 = np.array([[float(s) for s in r[:6]] for r in rows])
    Y0 = np.array([[float(s) for s in r[6:]] for r in rows])
    if A.shape != (nrows, 14) or not np.array_equal(A[:, 6:], Y0):
        raise ArithmeticError(f'{src}: outputs changed in conversion')
    if not np.array_equal(np.round(Uf, 4), U0) or np.max(np.abs(Uf - U0)) > 1e-12:
        raise ArithmeticError(f'{src}: inputs do not map back to the published values')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--check', action='store_true',
                   help='compare with the shipped files instead of writing them')
    args = p.parse_args()
    ok = True
    for name, (src, nrows, desc) in FILES.items():
        text = convert(HERE / 'original' / src, nrows, desc)
        if args.check:
            same = (HERE / name).read_text(encoding='ascii') == text
            ok &= same
            print(f'{name}: {"identical to" if same else "DIFFERS from"} the conversion of original/{src}')
        else:
            (HERE / name).write_text(text, encoding='ascii')
            print(f'wrote {name}: {nrows} rows from original/{src}; lossless checks passed')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
