"""Recompute the numbers of DATA.md (Epoxy_MD_Force_Field_Sensitivity) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py                       # checks 1-3, on the shipped CSVs next to this file
    python check_accuracy.py --source REPO_ROOT    # also check 4, on the authors' LAMMPS data file

REPO_ROOT is a clone of https://github.com/wedeling/MD-active-subspace at tag 1.1 (or the extracted
Zenodo archive doi:10.5281/zenodo.10817481); check 4 reads
REPO_ROOT/molecular_models/epoxy_elasticity/epoxy.data, which is not shipped here.

Checks:
 1. Stored precision: the largest number of significant digits written in the input and output files.
 2. Per-run anisotropy: the ratio E11/E22 over all 10,000 runs and over the 500 replica means
    (median, 1st and 99th percentiles).
 3. Spread of the replicas: the median over the 500 points of the replica coefficient of variation of
    E11 and of the replica standard deviation of the Poisson ratio.
 4. (with --source) Each class of sampled defaults (p_k1, p_k2), (b_k1, b_k2), (a_k1, a_k2) and the
    OPLS row (d_k1..d_k4, zeros for the unsampled coefficients) is one coefficient row of the Pair,
    Bond, Angle and Dihedral Coeffs sections of epoxy.data; prints how many classes match and how many
    distinct rows each section has. This is the basis of the inferred roles and units of the inputs.

Deterministic; well under a second. Imports numpy and the standard library.
"""
import hashlib
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(HERE, name)) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
        lines = [ln.strip() for ln in fh if ln.strip()]
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', ndmin=2), cols, lines


def sig_digits(token):
    mant = token.lower().split('e')[0].lstrip('+-').replace('.', '').lstrip('0')
    return len(mant)


def lammps_sections(path):
    sec, out = None, {}
    with open(path) as fh:
        for ln in fh:
            s = ln.split('#')[0].strip()
            head = re.match(r'^(Pair|Bond|Angle|Dihedral|Improper) Coeffs', ln.strip())
            if head:
                sec = head.group(1)
                out[sec] = []
                continue
            if not s:
                continue
            if sec and re.match(r'^[A-Za-z]', s):
                sec = None
                continue
            if sec:
                out[sec].append(tuple(float(v) for v in s.split()[1:]))
    return out


def main(argv):
    X, names, xl = read('epoxy_inputs.csv')
    x0 = read('epoxy_defaults.csv')[0].ravel()
    reps = {q: read(f'epoxy_{q}_replicas.csv') for q in ('E11', 'E22', 'poisson')}

    print('1. stored precision (significant digits, max): inputs',
          max(sig_digits(t) for ln in xl for t in ln.split(',')), '; outputs',
          max(sig_digits(t) for q in reps for ln in reps[q][2] for t in ln.split(',')))

    E11, E22, nu = reps['E11'][0], reps['E22'][0], reps['poisson'][0]
    r_run = (E11 / E22).ravel()
    r_mean = E11.mean(1) / E22.mean(1)
    print('2. E11/E22 per run: median %.3f, 1%%-99%% %.3f to %.3f; per replica mean: median %.3f, 1%%-99%% %.3f to %.3f'
          % (np.median(r_run), *np.percentile(r_run, [1, 99]), np.median(r_mean), *np.percentile(r_mean, [1, 99])))

    cv = E11.std(1, ddof=1) / E11.mean(1)
    print('3. replica spread: median coefficient of variation of E11 %.3f; median sd of the Poisson ratio %.4f'
          % (np.median(cv), np.median(nu.std(1, ddof=1))))

    if '--source' not in argv:
        print('4. skipped: give --source REPO_ROOT (wedeling/MD-active-subspace, tag 1.1) to check the defaults '
              'against epoxy.data')
        return 0
    path = os.path.join(argv[argv.index('--source') + 1], 'molecular_models', 'epoxy_elasticity', 'epoxy.data')
    with open(path, 'rb') as fh:
        print('   epoxy.data SHA-256', hashlib.sha256(fh.read()).hexdigest())
    secs = lammps_sections(path)
    d = dict(zip(names, x0))
    for letter, sec, ncoef in [('p', 'Pair', 2), ('b', 'Bond', 2), ('a', 'Angle', 2), ('d', 'Dihedral', 4)]:
        rows = set(secs[sec])
        classes = sorted({int(n[1:-1]) for n in names if n[0] == letter})
        R = np.array(sorted(rows))
        vec = [np.array([d.get(f'{letter}{k}{j}', 0.0) for j in range(1, ncoef + 1)]) for k in classes]
        hits = sum(bool(np.any(np.all(np.isclose(R, v, rtol=1e-12, atol=0), axis=1))) for v in vec)
        print(f'4. {sec:8s}: {hits} of {len(classes)} sampled classes equal a coefficient row (relative tolerance 1e-12); '
              f'{len(rows)} distinct rows in {len(secs[sec])} types')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
