"""Pilot runs that chose the active-variable proposal of generate_data.py.

For each sampler (ghat cached or recomputed) and each scale s in SCALES, run
one chain of N_PILOT steps with the proposal y' = y + s AV_CHOL e, started at
the posterior mean of y from vanilla chain 1, and print the acceptance rate
and the autocorrelation time of y1 and y2 after BURN steps. generate_data.py
uses the scale with the smallest autocorrelation time for each sampler.

    python tune_proposal.py --workers 6        # about 30 minutes on 6 processes

Seeded (SeedSequence([SEED, 1]), one child per run). The table in DATA.md was
made with the unrounded Cholesky factor of the first file version's vanilla
chain, so a rerun gives statistically equivalent, not identical, numbers.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from as_mcmc import ActiveSubspaceMCMC, chain_diagnostics
from generate_data import AV_CHOL, M, OUT, SEED, _model

SCALES = (0.5, 0.85, 1.2, 1.7)
N_PILOT, BURN = 2500, 300


def _run(args):
    s, recompute, seed = args
    z = np.load(OUT)
    W = z['W']
    model = _model(z['d'], float(z['sig2']))
    y_start = z['vanilla_1_proj'][len(z['vanilla_1_proj']) // 5:, :2].astype(float).mean(axis=0)
    Y, _, acc = ActiveSubspaceMCMC(model.misfit, W[:, :2], W[:, 2:]).sample_active_variables(
        y_start, N_PILOT, M, s * AV_CHOL, np.random.default_rng(seed), recompute=recompute)
    d = chain_diagnostics(Y[BURN:][None])
    return s, recompute, acc.mean(), d['tau']


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    runs = [(s, r) for r in (True, False) for s in SCALES]
    seeds = np.random.SeedSequence([SEED, 1]).spawn(len(runs))
    with ProcessPoolExecutor(args.workers) as ex:
        for s, r, a, tau in ex.map(_run, [(s, r, sd) for (s, r), sd in zip(runs, seeds)]):
            print(f'{"recompute" if r else "cached":>9} s = {s:4.2f}: acceptance {a:.3f}, '
                  f'tau(y1, y2) = {tau[0]:.1f}, {tau[1]:.1f}', flush=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
