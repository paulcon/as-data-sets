"""Recompute the numbers of the Accuracy section of DATA.md (DTMB_5415_Hull_Resistance).

    python check_accuracy.py

Run it from any directory; it reads the shipped CSVs next to this file and the constants of the
per-run failure rule in l2sea_driver.py. It does not run L2-Sea. Each line prints the recomputed
value; DATA.md quotes these values.

Checks:
 1. The friction line of the failure rule: a straight-line fit of R_f / Rf_ittc (both as the code
    prints them) against Fr on the valid G7 reruns of design_diagnostics.csv (the 41 rows less the
    7 flagged ones); its residuals, and the distance of the flagged rows from the line of
    l2sea_driver.FRICTION_LINE. The same distance for the shape-only study and the valid draft-sweep
    runs.
 2. The wave-cut ratio Rw_wc / R_w of the failure rule: its range on the valid runs of every
    shipped file that records Rw_wc (design_diagnostics.csv, draft_sweep.csv, shape_outputs.csv)
    and on the flagged design rows.
 3. The per-run rule (l2sea_driver.failure_reason) on the draft sweeps: which runs that returned
    numbers fail it with and without the draft-band test.
 4. Composition of fidelity_check.csv (rows rerun on each grid).
 5. The code's own time report against wall time in reference_runs.csv (the code divides a
    millisecond clock count by 10000; DATA.md, Fidelity), and the wall times by fidelity.
 6. The Gauss-Legendre nodes of uq_uniform_quadrature.csv: their drafts against the mapped
    failure bands.

Deterministic. Imports numpy and the standard library.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2sea_driver as drv  # noqa: E402

RHO, NU, G, LPP = 998.5, 1.09e-6, 9.8033, 5.720          # SBDF.nml of the L2-Sea example case
FLAGGED = [3, 63, 189, 207, 305, 323, 343]                # rows left out by the notebook's friction rule
NAMES11 = list(drv.OUTPUT_NAMES[:-1]) + list(drv.DIAG_NAMES)


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def line_resid(fr, ratio):
    return ratio - drv.FRICTION_LINE[0] - drv.FRICTION_LINE[1] * fr


def main():
    X, F = load('inputs.csv'), load('outputs.csv')
    DG, SW, SF = load('design_diagnostics.csv'), load('draft_sweep.csv'), load('shape_outputs.csv')
    FC, REF, QD = load('fidelity_check.csv'), load('reference_runs.csv'), load('uq_uniform_quadrature.csv')

    rows = DG[:, 0].astype(int)
    flag = np.isin(rows, FLAGGED)
    fr = X[rows, 0]
    ratio = F[rows, 6] / DG[:, 2]                         # R_f / Rf_ittc
    wc = DG[:, 1] / F[rows, 5]                            # Rw_wc / R_w
    ok = ~flag
    b, a = np.polyfit(fr[ok], ratio[ok], 1)
    e = ratio[ok] - (a + b * fr[ok])
    print(f'1. friction line refitted on the {ok.sum()} valid G7 reruns of design_diagnostics.csv: '
          f'{a:.5f} + {b:.5f} Fr (l2sea_driver: {drv.FRICTION_LINE[0]} + {drv.FRICTION_LINE[1]} Fr)')
    r0 = line_resid(fr, ratio)
    print(f'   residuals from the l2sea_driver line: valid rows {r0[ok].min():+.4f} .. {r0[ok].max():+.4f}; '
          f'flagged rows |residual| {np.abs(r0[flag]).min():.4f} .. {np.abs(r0[flag]).max():.4f} '
          f'(tolerance {drv.FRICTION_TOL})')
    rs = line_resid(np.full(len(SF), 0.28), SF[:, 6] / SF[:, 10])
    print(f'   shape-only study (256 runs): residuals {rs.min():+.4f} .. {rs.max():+.4f}')

    why_nb = [drv.failure_reason([r[1], r[2]], dict(zip(NAMES11, r[3:])), int(r[0]), band=False)
              if np.isfinite(r[3]) else 'raised' for r in SW]
    why = [drv.failure_reason([r[1], r[2]], dict(zip(NAMES11, r[3:])), int(r[0]))
           if np.isfinite(r[3]) else 'raised' for r in SW]
    num_ok = np.array([w is None for w in why_nb])        # passes the numeric checks (band test off)
    rsw = line_resid(SW[num_ok, 1], SW[num_ok, 9] / SW[num_ok, 13])
    print(f'   draft sweeps, runs that pass the numeric checks ({num_ok.sum()}): residuals '
          f'{rsw.min():+.4f} .. {rsw.max():+.4f}')

    print(f'2. wave-cut ratio Rw_wc / R_w (rule: at most {drv.WAVECUT_MAX_RATIO}); "valid" = passes the numeric checks:')
    print(f'   design_diagnostics.csv, valid rows: {wc[ok].min():.3f} .. {wc[ok].max():.3f}; flagged rows: '
          f'{wc[flag].min():.2f} .. {wc[flag].max():.4g}')
    for f_, fr_ in ((7, 0.28), (7, 0.36), (6, 0.28)):
        m = (SW[:, 0] == f_) & np.isclose(SW[:, 1], fr_) & num_ok
        q = SW[m, 12] / SW[m, 8]
        print(f'   draft sweep G{f_}, Fr {fr_}, valid runs ({m.sum()}): {q.min():.3f} .. {q.max():.3f}')
    q = SF[:, 9] / SF[:, 5]
    print(f'   shape-only study (256 runs): {q.min():.3f} .. {q.max():.3f}')
    allv = np.concatenate([wc[ok], SW[num_ok, 12] / SW[num_ok, 8], q])
    i36 = int(np.flatnonzero((SW[:, 0] == 7) & np.isclose(SW[:, 1], 0.36) & np.isclose(SW[:, 2], -5.61))[0])
    rest = np.concatenate([wc[ok], np.delete(SW[:, 12] / SW[:, 8], i36)[np.delete(num_ok, i36)], q])
    print(f'   all valid runs in these files: {allv.min():.3f} .. {allv.max():.3f}; without the G7, Fr 0.36, '
          f'T = -5.610 m run ({SW[i36, 3]:.2f} N, ratio {SW[i36, 12] / SW[i36, 8]:.3f}; inside the draft band): '
          f'{rest.min():.3f} .. {rest.max():.3f}')

    print('3. per-run rule on draft_sweep.csv:')
    for f_, fr_ in ((7, 0.28), (7, 0.36), (6, 0.28)):
        m = np.flatnonzero((SW[:, 0] == f_) & np.isclose(SW[:, 1], fr_))
        raised = [i for i in m if why[i] == 'raised']
        numbers_bad = [i for i in m if why_nb[i] not in (None, 'raised')]
        Tbad = SW[raised + numbers_bad, 2]
        inside = [i for i in m if why_nb[i] is None and Tbad.min() <= SW[i, 2] <= Tbad.max()]
        print(f'   G{f_}, Fr {fr_}: {len(m)} drafts; raised {len(raised)}; returned numbers that fail the numeric '
              f'checks {len(numbers_bad)} (failed drafts {Tbad.min():.4f} .. {Tbad.max():.4f} m); runs that pass the '
              f'numeric checks between the failed drafts: T = {[round(float(SW[i, 2]), 4) for i in inside]}; '
              f'draft band of the rule {drv.BAD_T[f_]} m contains all failed drafts: '
              f'{bool(np.all((Tbad >= drv.BAD_T[f_][0]) & (Tbad <= drv.BAD_T[f_][1])))}')

    print('4. fidelity_check.csv:', len(FC), 'rows;', end=' ')
    for f_ in sorted(set(FC[:, 1].astype(int))):
        r = FC[FC[:, 1] == f_, 0].astype(int)
        extra = [int(v) for v in sorted(set(r) - set(range(128)))] if f_ == 6 else [int(v) for v in sorted(set(r))]
        print(f'G{f_}: {len(r)} rows' + (f' (rows 0-127 and {extra})' if f_ == 6 else f' {extra}'), end='; ')
    print(f'row 305 on G6 NaN: {bool(np.isnan(FC[(FC[:, 0] == 305) & (FC[:, 1] == 6), 2][0]))}')

    cpu, wall = REF[:, 27], REF[:, 28]
    print(f'5. reference_runs.csv: wall time / code time report = '
          f'{np.array2string(wall / cpu, precision=2, separator=", ")} (hull, fidelity '
          f'{[(int(h), int(f_)) for h, f_ in REF[:, :2]]})')
    for h, f_, w in zip(REF[:, 0].astype(int), REF[:, 1].astype(int), wall):
        print(f'   {"parent" if h == 0 else "optimum"} G{f_}: wall {w:.2f} s')
    p = REF[:, 0] == 0
    w7, w1 = wall[p & (REF[:, 1] == 7)][0], wall[p & (REF[:, 1] == 1)][0]
    print(f'   parent hull, wall G1 / G7 = {w1:.2f} / {w7:.2f} = {w1 / w7:.0f}')

    Tq = np.unique(QD[:, 1])
    print(f'6. quadrature nodes: Fr {np.array2string(np.unique(QD[:, 0]), precision=4)}, T {np.array2string(Tq, precision=4)} m; '
          f'distance of the shallowest node from the G6 band {drv.BAD_T[6]}: '
          f'{drv.BAD_T[6][0] - Tq.max():.3f} m')


if __name__ == '__main__':
    main()
