"""Checks of the stored runs that neither the notebook nor convert_data.py prints (numpy only).

    python check_accuracy.py

1. The two least-cost CSV files equal the least-cost rows of all_runs.npz.
2. Cost constraint of the near-optimal runs: each run with slack epsilon > 0 should satisfy
   tsc <= (1 + epsilon) * tsc of the least-cost run at the same cost sample (Eqs. 15-17 of the paper);
   prints how far the stored costs exceed or fall short of that bound.
3. Ordering: at each sample and slack, the minimized capacity of a technology should not exceed its
   least-cost value and the maximized one should not fall below it; prints the violations.
4. Annuity ratios (optimistic / pessimistic) from the authors' cost files data/costs_optimist.csv and
   data/costs_pessimist.csv (values copied below; discount rate 0.07 from config.pypsaeur.yaml, rooftop PV
   0.04 from the cost files), for comparison with the stored multiplier ranges. The way the technologies are
   combined (6 h of battery energy, 168 h of hydrogen cavern storage, solar half utility and half rooftop)
   follows the paper (pp. 4, 17); this is this data set's reconstruction, not the authors' code.
5. Hydrogen CAPEX sums, electrolysis + fuel cell + 168 h of cavern (EUR/kW), from the same cost files, with
   each file's own cavern cost and with the other file's, for comparison with Table 1 (668-2002 EUR/kW).
6. The range of the stored `gini` output over the least-cost runs of each fidelity.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def annuity(inv, fom, life, rate=0.07):
    """Annual cost per unit of investment inv with fixed O&M fom (fraction/a) and lifetime life (a)."""
    return inv * (rate / (1 - (1 + rate) ** -life) + fom)


# (investment, FOM fraction, lifetime) optimistic | pessimistic, from data/costs_{optimist,pessimist}.csv
COSTS = {
    'onwind': lambda o: annuity(*((800, 0.0118, 30) if o else (1190, 0.0118, 30))),
    'offwind': lambda o: annuity(*((1420, 0.0229, 30) if o else (1950, 0.0229, 30))),
    'solar': lambda o: 0.5 * annuity(*((250, 0.02, 25) if o else (360, 0.02, 25)))
    + 0.5 * annuity(*((590, 0.0155, 25) if o else (880, 0.0155, 25)), rate=0.04),
    'H2': lambda o: annuity(*((200, 0.05, 25) if o else (800, 0.05, 25))) + annuity(*((500, 0.05, 20) if o else (900, 0.05, 20)))
    + 168 * annuity(*((1.8, 0.0, 100) if o else (1.0, 0.0, 100))),
    'battery': lambda o: annuity(*((40, 0.009, 20) if o else (250, 0.009, 20)))
    + 6 * annuity(*((46, 0.0, 20) if o else (176, 0.0, 20))),
}
STORED_LOWER = {'onwind': 0.670, 'offwind': 0.726, 'solar': 0.679, 'H2': 0.429, 'battery': 0.240}


def main():
    print('annuity ratio optimistic / pessimistic from the cost files, against the stored lower bound:')
    for k, fn in COSTS.items():
        print(f'  {k:8s} {fn(True) / fn(False):.3f}  stored {STORED_LOWER[k]:.3f}')
    # electrolysis, fuel cell (EUR/kW) and cavern (EUR/kWh) investment, data/costs_{optimist,pessimist}.csv
    h2 = {'optimist': (200, 500, 1.8), 'pessimist': (800, 900, 1.0)}
    print('hydrogen CAPEX electrolysis + fuel cell + 168 h cavern (EUR/kW), Table 1 gives 668-2002:')
    for f, (el, fc, cav) in h2.items():
        other = h2['pessimist' if f == 'optimist' else 'optimist'][2]
        print(f'  {f:9s} own cavern {cav}: {el + fc + 168 * cav:.1f}; other file cavern {other}: '
              f'{el + fc + 168 * other:.1f}')
    z = np.load(os.path.join(HERE, 'all_runs.npz'))
    outs = list(z['output_names'])
    obj_labels = list(z['objective_labels'])
    sense_labels = list(z['sense_labels'])
    fixed_labels = list(z['fixed_labels'])
    for fid in ['low', 'high']:
        X, Y = z[f'{fid}_inputs'], z[f'{fid}_outputs']
        obj, sense, fixed, eps = z[f'{fid}_objective'], z[f'{fid}_sense'], z[f'{fid}_fixed'], z[f'{fid}_epsilon']
        lc = obj == obj_labels.index('cost')
        csv = np.loadtxt(os.path.join(HERE, f'least_cost_{fid}.csv'), delimiter=',')
        same = np.array_equal(csv, np.column_stack([X[lc], Y[lc]]))
        print(f'[{fid}] least_cost_{fid}.csv equals the least-cost rows of all_runs.npz: {same}')
        gini = Y[lc, outs.index('gini')]
        print(f'[{fid}] gini of the least-cost runs: {gini.min():.2f} to {gini.max():.2f}')
        # map each run to its least-cost run by the cost sample
        keys = {tuple(x): i for i, x in zip(np.flatnonzero(lc), X[lc])}
        ref = np.array([keys[tuple(x)] for x in X])
        tsc = Y[:, outs.index('tsc')]
        no = ~lc
        rel = tsc[no] / ((1 + eps[no]) * tsc[ref[no]]) - 1
        print(f'[{fid}] near-optimal runs: {no.sum()}; tsc / ((1 + eps) tsc_least_cost) - 1: '
              f'min {rel.min():.2e}, median {np.median(rel):.2e}, max {rel.max():.2e}; '
              f'runs above the bound by more than 1e-3: {int(np.sum(rel > 1e-3))}')
        # ordering of single-technology extremes (no fixed second technology)
        free = no & (fixed == fixed_labels.index('none'))
        worst = []
        for k, o in enumerate(obj_labels):
            if o == 'cost':
                continue
            col = outs.index(o)
            for sname, sgn in [('min', 1), ('max', -1)]:
                sel = free & (obj == k) & (sense == sense_labels.index(sname))
                d = sgn * (Y[sel, col] - Y[ref[sel], col])      # > 0 means a violation
                scale = np.maximum(np.abs(Y[ref[sel], col]), 1.0)
                worst.append((np.max(d / scale), o, sname, int(np.sum(d / scale > 1e-3)), int(sel.sum())))
        worst.sort(reverse=True)
        print(f'[{fid}] largest relative ordering violation (extreme vs least cost): '
              f'{worst[0][0]:.2e} ({worst[0][2]} {worst[0][1]}); runs with a violation > 1e-3: '
              f'{sum(w[3] for w in worst)} of {sum(w[4] for w in worst)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
