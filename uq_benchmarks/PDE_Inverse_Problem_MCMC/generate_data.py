"""Regenerate pde_mcmc_chains.npz, the stored Markov chains of the PDE example.

Run from anywhere with the repository's Python 3 environment:

    python generate_data.py                      # the shipped file, about 45 min
    python generate_data.py --n-van 2000 --n-av 200 --out /tmp/short.npz

Steps, all seeded from SEED (one numpy SeedSequence, spawned into the streams
listed in STREAMS):

1. synthetic data: x_true ~ N(0, I_100), d = m(x_true) + noise with
   sigma^2 = 1e-4 ||m(x_true)||^2;
2. N_GRAD misfit gradients at prior samples, and the active subspace
   (active_subspaces v1, n = 2, first components of w1 and w2 made positive);
3. the vanilla chain: N_VAN steps of random-walk Metropolis-Hastings on all
   100 parameters, proposal standard deviation PROP_STD;
4. the active-variable chain: N_AV steps of Algorithm 1 on y = W1^T x, with
   M misfit evaluations per ghat and proposal standard deviation PROP_STD.

Steps 3 and 4 are independent and run in two processes (--workers 1 runs them
one after the other; the output does not depend on it). Each stream draws
step by step, so a shorter run reproduces the first steps of a longer one.
The script writes the inputs, the subspace and the chains (the vanilla chain
reduced, see DATA.md) to one compressed npz.
"""
import argparse
import hashlib
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy

import active_subspaces as ac
from as_mcmc import ActiveSubspaceMCMC, random_walk_metropolis
from pde_model import PDEMisfit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'pde_mcmc_chains.npz')

SEED = 2779            # the paper's first page, A2779
STREAMS = ('data', 'grad_points', 'bootstrap', 'vanilla', 'active')
N_GRAD = 1000          # gradient samples for the active subspace
NBOOT = 100            # bootstrap replicates for the eigenvalue intervals
N_VAN = 100_000        # vanilla chain length
N_AV = 10_000          # active-variable chain length
M = 10                 # misfit evaluations per ghat
PROP_STD = 0.1         # proposal standard deviation, both chains
NOISE = 1e-4           # sigma^2 = NOISE * ||m(x_true)||^2
N_ACTIVE = 2
BURN_IN = 0.2          # fraction of each chain the notebook discards
THIN = 10              # thinning of the stored full vanilla states
TRACE = (9, 89)        # x_10 and x_90 (0-based), stored at every step
N_PROJ = 4             # stored components of W^T x for the vanilla chain


def streams(seed=SEED):
    """One numpy Generator per entry of STREAMS."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(s) for name, s in zip(STREAMS, children)}


def synthetic_data(model, rng):
    """x_true, d and sigma^2 for the inverse problem (paper section 5.3)."""
    x_true = rng.standard_normal(model.m)
    obs = model.observations(x_true)
    sig2 = NOISE * obs @ obs
    d = obs + np.sqrt(sig2) * rng.standard_normal(obs.size)
    return x_true, d, sig2


def misfit_subspace(G, rng):
    """v1 active subspace of the misfit gradients G, n = 2, 2016 sign rule."""
    sub = ac.active_subspace(space=ac.Gaussian.reference(G.shape[1]), grads=G,
                             nboot=NBOOT, rng=rng, n=N_ACTIVE)
    return ac.orient(sub, by='reference', reference=np.eye(G.shape[1])[:, [0, 0]],
                     columns=N_ACTIVE)


def _progress(label, N, t0):
    def report(k):
        print(f'{label}: {k}/{N} steps, {time.perf_counter() - t0:.0f} s', flush=True)
    return report


def run_vanilla(d, sig2, W, rng, N=N_VAN):
    """The vanilla chain, reduced for storage (see DATA.md)."""
    t0 = time.perf_counter()
    model = PDEMisfit()
    model.set_data(d, sig2)
    x0 = rng.standard_normal(model.m)
    X, F, accept = random_walk_metropolis(model.misfit, x0, N, PROP_STD, rng,
                                          progress=_progress('vanilla', N, t0))
    return {'van_x0': x0,
            'van_trace': X[:, list(TRACE)],
            'van_proj': (X @ W[:, :N_PROJ]).astype(np.float32),
            'van_thin': X[::THIN].astype(np.float32),
            'van_misfit': F,
            'van_accept': accept,
            'van_last': X[-1],
            'wall_van': time.perf_counter() - t0}


def run_active(d, sig2, W, rng, N=N_AV):
    """The active-variable chain (Algorithm 1)."""
    t0 = time.perf_counter()
    model = PDEMisfit()
    model.set_data(d, sig2)
    sampler = ActiveSubspaceMCMC(model.misfit, W[:, :N_ACTIVE], W[:, N_ACTIVE:])
    y0 = rng.standard_normal(N_ACTIVE)
    Y, ghat, accept = sampler.sample_active_variables(y0, N, M, PROP_STD, rng,
                                                      progress=_progress('active', N, t0))
    return {'av_y0': y0, 'av_y': Y, 'av_ghat': ghat, 'av_accept': accept,
            'wall_av': time.perf_counter() - t0}


def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--n-van', type=int, default=N_VAN)
    p.add_argument('--n-av', type=int, default=N_AV)
    p.add_argument('--workers', type=int, default=2, choices=(1, 2))
    p.add_argument('--out', default=OUT)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    rng = streams()
    model = PDEMisfit()
    x_true, d, sig2 = synthetic_data(model, rng['data'])
    model.set_data(d, sig2)
    print(f'data: sigma^2 = {sig2:.4e}, d = {np.round(d, 5)}', flush=True)

    X_grad = rng['grad_points'].standard_normal((N_GRAD, model.m))
    G = np.array([model.misfit_grad(x) for x in X_grad])
    sub = misfit_subspace(G, rng['bootstrap'])
    W = sub.W
    wall_grad = time.perf_counter() - t0
    print(f'{N_GRAD} gradients and subspace: {wall_grad:.0f} s; '
          f'eigenvalues[:4] = {np.round(sub.eigenvalues[:4], 2)}', flush=True)

    jobs = [(run_vanilla, (d, sig2, W, rng['vanilla'], args.n_van)),
            (run_active, (d, sig2, W, rng['active'], args.n_av))]
    if args.workers == 1:
        results = [fn(*a) for fn, a in jobs]
    else:
        with ProcessPoolExecutor(max_workers=2) as ex:
            results = [f.result() for f in [ex.submit(fn, *a) for fn, a in jobs]]

    out = {'seed': np.int64(SEED), 'streams': np.array(STREAMS),
           'x_true': x_true, 'd': d, 'sig2': np.float64(sig2),
           'indz': np.array(model.indz), 'sensor_s1': np.full(len(model.indz), 1 - model.dx),
           'sensor_s2': (np.array(model.indz) + 1) * model.dx,
           'kl_md5': np.str_(_md5(os.path.join(HERE, 'kl-short.npz'))),
           'X_grad': X_grad, 'G': G, 'W': W, 'eigenvalues': sub.eigenvalues,
           'n_active': np.int64(N_ACTIVE), 'prop_std': np.float64(PROP_STD), 'M': np.int64(M),
           'burn_in': np.float64(BURN_IN), 'thin': np.int64(THIN), 'trace_index': np.array(TRACE),
           'wall_grad': np.float64(wall_grad),
           'versions': np.array([f'python {platform.python_version()}', f'numpy {np.__version__}',
                                 f'scipy {scipy.__version__}',
                                 f'active_subspaces {ac.__version__}', platform.platform()])}
    for r in results:
        out.update(r)
    out['wall_total'] = np.float64(time.perf_counter() - t0)
    np.savez_compressed(args.out, **out)
    print(f'vanilla acceptance {out["van_accept"].mean():.4f}, '
          f'active acceptance {out["av_accept"].mean():.4f}', flush=True)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in '
          f'{out["wall_total"]:.0f} s = {out["wall_total"] / 60:.1f} min', flush=True)


if __name__ == '__main__':
    sys.exit(main())
