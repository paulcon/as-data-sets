"""Recompute the numbers of DATA.md (Hall_Thruster_Multi_Component) that neither the notebook nor
`generate_data.py check` prints.

    python check_accuracy.py               all checks; checks 5-8 rerun HallThruster.jl (about 12 runs,
                                           a few minutes after Julia has precompiled the package)
    python check_accuracy.py --skip-julia  only checks 1-4, from the shipped CSV (about a second)

Run it from any directory; it reads hall_thruster_spt100.csv next to this file. Each line prints the
recomputed value; DATA.md quotes these values.

Checks 1-4 (numpy and scipy, from the shipped file):
 1. Physical meaning: the range of V_cc before clipping; whether all averaged currents and thrusts
    are positive and every main-beam divergence angle c2*P_B + c3 is positive; the runs whose
    averaged ion current I_B exceeds the discharge current I_D, with I_B/I_D and their flag.
 2. Distance from the measured thruster (Diamant et al. 2014: I_D = 4.5 A, T = 81.44 mN,
    V_cc = 32.84 V): runs with I_D within 20 % of 4.5 A, below 10 A, the median I_D; runs with T_c
    within 10 % of 81.44 mN; with both; runs with V_cc within 3 V of 32.84 V.
 3. The wall time of the HallThruster.jl runs (column wall_s): minimum, median, maximum, total.
 4. The plume's hemisphere normalisation, 2 pi int_0^{pi/2} exp(-(phi/a)^2) sin(phi) dphi, by the
    400-point Gauss-Legendre rule of spt100_components.py against the closed form with the complex
    error function, at a = 0.2, 0.5, 1, 1.5 and 3 rad.

Checks 5-8 (need Julia, HallThruster.jl v0.18.7 and the B-field file, set up as for the `run` stage
of generate_data.py; JULIA, JULIA_PROJECT and BFIELD as described there). They rerun the design rows
0, 150 and 300 with run_hallthruster.jl in a temporary directory and compare with the shipped file:
 5. the settings of the shipped runs (the simulation is deterministic: relative differences of
    I_D and T);
 6. 2 ms simulated, averaged over the last 1 ms (changes of I_D and T);
 7. ion_wall_losses = false (changes of I_D, T and the exit ion velocity);
 8. 200 cells instead of 100 (changes of I_D and T).
Without Julia the script runs checks 1-4 and then exits with status 1 and says what to install.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_data as gd  # noqa: E402
from spt100_components import TORR_TO_PA, _gauss_norm  # noqa: E402

ROWS = (0, 150, 300)


def load():
    data = np.loadtxt(os.path.join(HERE, gd.CSV), delimiter=',')
    with open(os.path.join(HERE, gd.CSV)) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline().lstrip('# ').strip().split(', ')]
    return data, {c: data[:, i] for i, c in enumerate(cols)}


def physical(col):
    p = {n: col[n] for n in gd.NAMES}
    PB = gd.OPERATING['PB_Torr'] * 1e6                       # uTorr
    raw = p['Vvac'] + p['Tec'] * np.log1p(PB / p['PT']) - p['Tec'] * PB / (p['PT'] + p['Pstar'])
    pos = all(bool(np.all(col[q] > 0)) for q in ('ID', 'IB', 'T', 'Tc'))
    a1 = p['c2'] * gd.OPERATING['PB_Torr'] * TORR_TO_PA + p['c3']
    print(f'1. V_cc before clipping: {raw.min():.2f} to {raw.max():.2f} V (V_d = {gd.OPERATING["Vd"]:g} V); '
          f'all I_D, I_B, T, T_c positive: {pos}; all c2*P_B + c3 > 0: {bool(np.all(a1 > 0))}')
    r = col['IB'] / col['ID']
    for i in np.flatnonzero(r > 1):
        print(f'   row {i}: I_B / I_D = {r[i]:.3f} (I_B {col["IB"][i]:.3f} A, I_D {col["ID"][i]:.3f} A), '
              f'oscillating = {int(col["oscillating"][i])}')
    print(f'   runs with I_B > I_D: {int(np.sum(r > 1))}; largest I_B / I_D of the steady runs: '
          f'{r[col["oscillating"] == 0].max():.3f}')


def near_measurement(col):
    ID, Tc, Vcc = col['ID'], 1e3 * col['Tc'], col['Vcc']
    a = np.abs(ID / 4.5 - 1) <= 0.2
    b = np.abs(Tc / 81.44 - 1) <= 0.1
    print(f'2. I_D within 20 % of 4.5 A: {a.sum()}; I_D below 10 A: {np.sum(ID < 10)}; median I_D {np.median(ID):.1f} A; '
          f'T_c within 10 % of 81.44 mN: {b.sum()}; both: {np.sum(a & b)}; V_cc within 3 V of 32.84 V: '
          f'{np.sum(np.abs(Vcc - 32.84) <= 3)}')


def wall_time(col):
    w = col['wall_s']
    print(f'3. HallThruster.jl wall time per run: {w.min():.2f} to {w.max():.2f} s, median {np.median(w):.2f} s; '
          f'total {w.sum():.0f} s ({w.sum() / 60:.0f} min)')


def plume_quadrature():
    from scipy.special import erf
    errs = []
    for a in (0.2, 0.5, 1.0, 1.5, 3.0):
        # int_0^{pi/2} exp(-(phi/a)^2) sin(phi) dphi = Im int exp(-(phi/a)^2 + i phi) dphi
        c = 0.5 * a * np.sqrt(np.pi) * np.exp(-a * a / 4)
        exact = 2 * np.pi * np.imag(c * (erf(np.pi / (2 * a) - 0.5j * a) - erf(-0.5j * a)))
        errs.append(abs(_gauss_norm(a)[0] / exact - 1))
    print('4. hemisphere normalisation, Gauss-Legendre vs closed form, a = 0.2, 0.5, 1, 1.5, 3 rad: '
          'relative errors ' + ', '.join(f'{e:.1e}' for e in errs))


def julia_rows(settings):
    """Rows of julia_inputs.csv for ROWS, redrawn exactly as generate_data.design writes them, plus the
    optional held settings (ncells, duration, avg_start, ion_wall_losses)."""
    from scipy.stats import qmc
    U = qmc.LatinHypercube(d=len(gd.PARAMS), seed=gd.SEED).random(gd.M)
    p = dict(zip(gd.NAMES, gd.unit_to_physical(U).T))
    Vcc, _ = gd.cathode_coupling_voltage(gd.OPERATING['PB_Torr'], p['Tec'], p['Vvac'], p['PT'], p['Pstar'],
                                         gd.OPERATING['Vd'])
    out = []
    for i in ROWS:
        out.append([i, gd.OPERATING['Vd'], gd.OPERATING['mdot_a'], gd.OPERATING['PB_Torr'], Vcc[i], p['Tec'][i],
                    p['alpha_anom'][i], p['beta_anom'][i], p['z_anom'][i], p['L_anom'][i], p['dz_anom'][i],
                    p['u_n'][i], p['c_w'][i], p['f_n'][i]] + list(settings))
    return np.array(out)


def rerun(julia, settings, tmp, tag):
    fin, fout = os.path.join(tmp, f'in_{tag}.csv'), os.path.join(tmp, f'out_{tag}.csv')
    np.savetxt(fin, julia_rows(settings), delimiter=',', fmt='%.12e')
    subprocess.run([julia, '--startup-file=no', os.path.join(HERE, 'run_hallthruster.jl'), os.environ['BFIELD'],
                    fin, fout], check=True, env={**os.environ, 'JULIA_NUM_THREADS': '1'}, stdout=subprocess.DEVNULL)
    raw = np.loadtxt(fout, delimiter=',', ndmin=2)
    raw = raw[np.argsort(raw[:, 0])]
    return {'status': raw[:, 1], 'ID': raw[:, 2], 'T': raw[:, 4], 'u_exit': raw[:, 5]}


def julia_checks(col):
    julia = gd.require_julia()          # exits with status 1 and the install message when Julia is missing
    idx = np.array(ROWS)
    ship = {q: col[q][idx] for q in ('ID', 'T', 'u_exit')}
    pct = lambda new, old: ', '.join(f'{100 * (n / o - 1):+.1f} %' for n, o in zip(new, old))  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        base = rerun(julia, (100, 1.0e-3, 0.5e-3, 1), tmp, 'base')
        rel = max(np.max(np.abs(base[q] / ship[q] - 1)) for q in ('ID', 'T'))
        print(f'5. rows {ROWS} rerun with the shipped settings: status {base["status"].astype(int).tolist()}; '
              f'largest relative difference of I_D and T from the file {rel:.1e}')
        long = rerun(julia, (100, 2.0e-3, 1.0e-3, 1), tmp, 'long')
        print(f'6. 2 ms, averaged over the last 1 ms: I_D {pct(long["ID"], ship["ID"])}; T {pct(long["T"], ship["T"])} '
              f'(row 0: I_D {ship["ID"][0]:.3f} -> {long["ID"][0]:.3f} A, T {1e3 * ship["T"][0]:.2f} -> '
              f'{1e3 * long["T"][0]:.2f} mN)')
        iwl = rerun(julia, (100, 1.0e-3, 0.5e-3, 0), tmp, 'iwl')
        print(f'7. ion_wall_losses = false: I_D {pct(iwl["ID"], ship["ID"])}; T {pct(iwl["T"], ship["T"])}; '
              f'u_exit {pct(iwl["u_exit"], ship["u_exit"])}')
        fine = rerun(julia, (200, 1.0e-3, 0.5e-3, 1), tmp, 'fine')
        print(f'8. 200 cells: I_D {pct(fine["ID"], ship["ID"])}; T {pct(fine["T"], ship["T"])} '
              f'(row 0: I_D {ship["ID"][0]:.3f} -> {fine["ID"][0]:.3f} A, T {1e3 * ship["T"][0]:.2f} -> '
              f'{1e3 * fine["T"][0]:.2f} mN)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--skip-julia', action='store_true', help='run only checks 1-4 (no Julia needed)')
    args = ap.parse_args()
    _, col = load()
    physical(col)
    near_measurement(col)
    wall_time(col)
    plume_quadrature()
    if not args.skip_julia:
        julia_checks(col)


if __name__ == '__main__':
    main()
