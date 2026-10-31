"""Regenerate pde_mcmc_chains.npz, the stored Markov chains of the PDE example.

Run from anywhere with the repository's Python 3 environment:

    python generate_data.py --workers 6          # the shipped file (see DATA.md for the time)
    python generate_data.py --n-van 2000 --n-cached 200 --n-recompute 100 --grid-draws 20 \\
        --out /tmp/short.npz                      # a short check
    python generate_data.py --reuse OLD.npz       # copy finished jobs from OLD.npz

Each finished job is saved at once to a checkpoint directory (default: the
output path with '_parts' in place of '.npz'). Rerunning the same command after
an interruption loads those jobs and runs only the rest; a part whose data,
subspace or settings differ from the current run's is run again. The parts are
deleted once the output file is written. wall_total covers the last run only.

Steps, all seeded from SEED: one numpy SeedSequence spawned into the streams
listed in STREAMS, one stream per job.

1. Synthetic data: x_true ~ N(0, I_100), d = m(x_true) + noise with
   sigma^2 = 1e-4 ||m(x_true)||^2.
2. N_GRAD misfit gradients at prior samples, and the active subspace
   (active_subspaces v1, n = 2, first components of w1 and w2 made positive).
3. N_VAN_CHAINS vanilla chains: N_VAN steps each of random-walk Metropolis-Hastings
   on all 100 parameters, proposal standard deviation VAN_PROP_STD, each
   started at a prior draw.
4. N_CHAINS active-variable chains for each of two samplers, each started at
   a prior draw of y, with the proposal y' = y + AV_SCALE[sampler] AV_CHOL e,
   e ~ N(0, I_2) (tuned by tune_proposal.py), and M misfit evaluations per ghat:
   - 'cached': ghat kept with the state, the paper's Algorithm 1 as printed
     (a pseudo-marginal method, exact for pi_M);
   - 'recompute': ghat recomputed at the current state and at the proposal at
     every step, the 2016 code behind the paper's results (a noisy
     Metropolis-Hastings).
5. A grid for the densities that the active-variable chains approximate: the
   misfit at y W1 + W2 z for GRID_N x GRID_N points y and common prior draws z
   (GRID_DRAWS_FINE at every point, GRID_DRAWS at every other point).

Every job runs in one process; --workers sets how many jobs run at once. The
output does not depend on --workers. Each stream draws step by step, so a
shorter run reproduces the first steps of a longer one (the grid draws are one
block, so the first K of a shorter run equal the first K of a longer one).
"""
import argparse
import hashlib
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy

import active_subspaces as ac
from as_mcmc import ActiveSubspaceMCMC, random_walk_metropolis
from pde_model import PDEMisfit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'pde_mcmc_chains.npz')

SEED = 2779            # the paper's first page, A2779
N_CHAINS = 4           # independent chains per active-variable sampler
N_VAN_CHAINS = 20      # independent vanilla chains (4 at first; 16 added 2026-10-31)
# Chains 5-20 have their streams at the end: SeedSequence.spawn gives child i the same seed
# whatever the number of children, so every earlier job keeps its stream.
STREAMS = (('data', 'grad_points', 'bootstrap', 'vanilla_1', 'unused', 'grid')
           + tuple(f'vanilla_{k}' for k in range(2, N_CHAINS + 1))
           + tuple(f'{s}_{k}' for s in ('cached', 'recompute') for k in range(1, N_CHAINS + 1))
           + tuple(f'vanilla_{k}' for k in range(N_CHAINS + 1, N_VAN_CHAINS + 1)))
N_GRAD = 1000          # gradient samples for the active subspace
NBOOT = 100            # bootstrap replicates for the eigenvalue intervals
N_ACTIVE = 2
NOISE = 1e-4           # sigma^2 = NOISE * ||m(x_true)||^2
M = 10                 # misfit evaluations per ghat
BURN_IN = 0.2          # fraction of each chain the notebook discards

N_VAN = 100_000        # steps per vanilla chain
VAN_PROP_STD = 0.1     # vanilla proposal standard deviation (as in the paper)
N_CACHED = 16_000      # steps per cached-ghat chain
N_RECOMPUTE = 5_000    # steps per recompute-ghat chain
# AV proposal y' = y + AV_SCALE[sampler] * AV_CHOL e: AV_CHOL is the Cholesky factor of the
# posterior covariance of (y1, y2) from vanilla chain 1 after burn-in (rounded); the scales
# minimize the autocorrelation time in the pilot runs of tune_proposal.py (DATA.md)
AV_CHOL = np.array([[0.2438, 0.0], [-0.1044, 0.3561]])
AV_SCALE = {'cached': 1.2, 'recompute': 1.7}

THIN_PROJ = 10         # stored W^T x (first N_PROJ components): every 10th vanilla step
N_PROJ = 4
THIN_X = 50            # stored full vanilla states: every 50th step
TRACE = (9, 89)        # x_10 and x_90 (0-based): every step of vanilla chain 1

GRID_CENTER = np.array([0.64, -0.62])   # grid in y: center and half-widths, about
GRID_HALF = np.array([1.0, 1.7])         # 5 standard deviations of the pilot chains
GRID_N = 21            # points per axis (the coarse grid is every other point)
GRID_DRAWS_FINE = 160  # draws of z at every grid point
GRID_DRAWS = 2560      # draws of z at the coarse points


def streams(seed=SEED):
    """One numpy Generator per entry of STREAMS ('unused' fed the first versions' AV chain)."""
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


def grid_points():
    """The grid axes in y1 and y2, GRID_N points each."""
    t = np.linspace(-1, 1, GRID_N)
    return GRID_CENTER[0] + GRID_HALF[0] * t, GRID_CENTER[1] + GRID_HALF[1] * t


def _model(d, sig2):
    model = PDEMisfit()
    model.set_data(d, sig2)
    return model


def run_vanilla(d, sig2, W, rng, N=N_VAN):
    """One vanilla chain from a prior draw, reduced for storage (DATA.md)."""
    model = _model(d, sig2)
    x0 = rng.standard_normal(model.m)
    X, F, accept = random_walk_metropolis(model.misfit, x0, N, VAN_PROP_STD, rng)
    return {'x0': x0, 'trace': X[:, list(TRACE)],
            'proj': (X @ W[:, :N_PROJ]).astype(np.float32)[::THIN_PROJ],
            'thin': X[::THIN_X].astype(np.float32), 'misfit': F[::THIN_PROJ],
            'accept': accept, 'last': X[-1]}


def run_active(d, sig2, W, rng, N, recompute):
    """One active-variable chain from a prior draw of y."""
    model = _model(d, sig2)
    sampler = ActiveSubspaceMCMC(model.misfit, W[:, :N_ACTIVE], W[:, N_ACTIVE:])
    y0 = rng.standard_normal(N_ACTIVE)
    prop = AV_SCALE['recompute' if recompute else 'cached'] * AV_CHOL
    Y, ghat, accept = sampler.sample_active_variables(y0, N, M, prop, rng, recompute=recompute)
    return {'y0': y0, 'y': Y, 'ghat': ghat, 'accept': accept}


_CTX = {}


def _init(d, sig2, W, Z):
    _CTX.update(d=d, sig2=sig2, W=W, Z=Z)


def _job(job):
    """Run one job in a worker; returns (job, result, wall time)."""
    t0 = time.perf_counter()
    kind, k, n = job
    d, sig2, W = _CTX['d'], _CTX['sig2'], _CTX['W']
    if kind == 'vanilla':
        out = run_vanilla(d, sig2, W, streams()[f'vanilla_{k}'], n)
    elif kind in ('cached', 'recompute'):
        out = run_active(d, sig2, W, streams()[f'{kind}_{k}'], n, kind == 'recompute')
    else:                                   # ('grid', list of (i, j), number of draws)
        model = _model(d, sig2)
        g1, g2 = grid_points()
        Z = _CTX['Z'][:n]
        out = np.array([[model.misfit(W[:, :N_ACTIVE] @ np.array([g1[i], g2[j]]) + W[:, N_ACTIVE:] @ z)
                         for z in Z] for i, j in k])
    dt = time.perf_counter() - t0
    print(f'{kind} {k if kind != "grid" else len(k)} done in {dt:.0f} s', flush=True)
    return job, out, dt


def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _fingerprint(d, sig2, W):
    """md5 of everything a job's output depends on besides its name."""
    h = hashlib.md5()
    for a in (d, np.float64(sig2), W, AV_CHOL, GRID_CENTER, GRID_HALF):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((SEED, STREAMS, M, VAN_PROP_STD, sorted(AV_SCALE.items()), THIN_PROJ, N_PROJ,
                   THIN_X, TRACE, GRID_N, _md5(os.path.join(HERE, 'kl-short.npz')))).encode())
    return h.hexdigest()


def _part_name(job):
    """Checkpoint file name of a job: the chain and its length, or the grid batch and its draws."""
    kind, k, n = job
    if kind == 'grid':
        (i, j), size = k[0], len(k)
        return f'grid_{i:02d}_{j:02d}_x{size}_n{n}'
    return f'{kind}_{k}_n{n}'


def _save_part(path, fp, job, out, dt):
    """Write one finished job to path atomically (a partial file never has the final name)."""
    arrays = {'r': out} if job[0] == 'grid' else {f'r_{f}': v for f, v in out.items()}
    tmp = path[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp, fp=np.str_(fp), dt=np.float64(dt), **arrays)
    os.replace(tmp, path)


def _load_part(path, fp, job):
    """(output, wall time) of a checkpointed job, or None if it is missing or from other settings."""
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if str(z['fp']) != fp:
            return None
        out = z['r'] if job[0] == 'grid' else {f[2:]: z[f] for f in z.files if f.startswith('r_')}
        return out, float(z['dt'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--n-van', type=int, default=N_VAN)
    p.add_argument('--n-cached', type=int, default=N_CACHED)
    p.add_argument('--n-recompute', type=int, default=N_RECOMPUTE)
    p.add_argument('--grid-draws', type=int, default=GRID_DRAWS,
                   help=f'draws at the coarse grid points; the fine points get '
                        f'min(this, {GRID_DRAWS_FINE})')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--reuse', metavar='FILE',
                   help='copy chains of the same seed, data and length from FILE instead of running them')
    p.add_argument('--out', default=OUT)
    p.add_argument('--checkpoint', metavar='DIR',
                   help="directory for finished jobs (default: OUT with '_parts' for '.npz')")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    rng = streams()
    model = PDEMisfit()
    x_true, d, sig2 = synthetic_data(model, rng['data'])
    model.set_data(d, sig2)
    X_grad = rng['grad_points'].standard_normal((N_GRAD, model.m))
    G = np.array([model.misfit_grad(x) for x in X_grad])
    sub = misfit_subspace(G, rng['bootstrap'])
    W = sub.W
    print(f'data and subspace: {time.perf_counter() - t0:.0f} s; '
          f'eigenvalues[:4] = {np.round(sub.eigenvalues[:4], 2)}', flush=True)

    n_fine = min(args.grid_draws, GRID_DRAWS_FINE)
    Z = rng['grid'].standard_normal((args.grid_draws, model.m - N_ACTIVE))
    old = np.load(args.reuse) if args.reuse else None
    if old is not None:
        assert int(old['seed']) == SEED and np.array_equal(old['d'], d) and np.array_equal(old['W'], W)

    chains = {'vanilla': args.n_van, 'cached': args.n_cached, 'recompute': args.n_recompute}
    results, jobs = {}, []
    n_chains = {'vanilla': N_VAN_CHAINS, 'cached': N_CHAINS, 'recompute': N_CHAINS}
    for kind, n in chains.items():
        for k in range(1, n_chains[kind] + 1):
            key = f'{kind}_{k}'
            if old is not None and f'{key}_accept' in old.files and len(old[f'{key}_accept']) == n:
                results[(kind, k)] = {f: old[f'{key}_{f}'] for f in _fields(kind, k)}
                print(f'{key}: copied from {args.reuse}', flush=True)
            else:
                jobs.append((kind, k, n))
    coarse = [(i, j) for i in range(0, GRID_N, 2) for j in range(0, GRID_N, 2)]
    fine = [(i, j) for i in range(GRID_N) for j in range(GRID_N) if (i, j) not in set(coarse)]
    if old is not None and 'grid_f_coarse' in old.files and old['grid_f_coarse'].shape[2] == args.grid_draws:
        grid_coarse, grid_fine = old['grid_f_coarse'], old['grid_f_fine']
        print(f'grid: copied from {args.reuse}', flush=True)
    else:
        grid_coarse = grid_fine = None
        jobs += [('grid', [ij], args.grid_draws) for ij in coarse]
        jobs += [('grid', fine[s:s + 8], n_fine) for s in range(0, len(fine), 8)]
    cost = {'vanilla': 1, 'cached': M, 'recompute': 2 * M}
    jobs.sort(key=lambda j: -(j[2] * cost[j[0]] if j[0] != 'grid' else j[2] * len(j[1])))

    ckpt = args.checkpoint or os.path.splitext(args.out)[0] + '_parts'
    os.makedirs(ckpt, exist_ok=True)
    fp = _fingerprint(d, sig2, W)
    walls, grid_out, todo = {}, {}, []

    def collect(job, out, dt):
        if job[0] == 'grid':
            for ij, f in zip(job[1], out):
                grid_out[ij] = f
        else:
            results[job[:2]] = out
            walls[f'{job[0]}_{job[1]}'] = dt

    n_resumed = 0
    for job in jobs:
        part = _load_part(os.path.join(ckpt, _part_name(job) + '.npz'), fp, job)
        if part is None:
            todo.append(job)
        else:
            collect(job, *part)
            n_resumed += 1
            if job[0] != 'grid':
                print(f'{job[0]}_{job[1]}: resumed from {ckpt}', flush=True)
    print(f'{n_resumed} of {len(jobs)} jobs resumed from {ckpt}; running {len(todo)}', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(d, sig2, W, Z)) as ex:
        for fut in as_completed([ex.submit(_job, job) for job in todo]):
            job, out, dt = fut.result()
            _save_part(os.path.join(ckpt, _part_name(job) + '.npz'), fp, job, out, dt)
            collect(job, out, dt)
    if grid_coarse is None:
        grid_coarse = np.array([[grid_out[(i, j)] for j in range(0, GRID_N, 2)]
                                for i in range(0, GRID_N, 2)]).astype(np.float32)
        grid_fine = np.array([[grid_out[(i, j)][:n_fine] for j in range(GRID_N)]
                              for i in range(GRID_N)]).astype(np.float32)

    g1, g2 = grid_points()
    out = {'seed': np.int64(SEED), 'streams': np.array(STREAMS),
           'x_true': x_true, 'd': d, 'sig2': np.float64(sig2),
           'indz': np.array(model.indz), 'sensor_s1': np.full(len(model.indz), 1 - model.dx),
           'sensor_s2': (np.array(model.indz) + 1) * model.dx,
           'kl_md5': np.str_(_md5(os.path.join(HERE, 'kl-short.npz'))),
           'X_grad': X_grad, 'G': G, 'W': W, 'eigenvalues': sub.eigenvalues,
           'n_active': np.int64(N_ACTIVE), 'M': np.int64(M), 'burn_in': np.float64(BURN_IN),
           'van_prop_std': np.float64(VAN_PROP_STD), 'av_chol': AV_CHOL,
           'av_scale_cached': np.float64(AV_SCALE['cached']),
           'av_scale_recompute': np.float64(AV_SCALE['recompute']),
           'thin_proj': np.int64(THIN_PROJ), 'thin_x': np.int64(THIN_X), 'trace_index': np.array(TRACE),
           'grid_y1': g1, 'grid_y2': g2, 'grid_f_coarse': grid_coarse, 'grid_f_fine': grid_fine,
           'wall': np.array([f'{k} {v:.0f} s' for k, v in walls.items()]),
           'versions': np.array([f'python {platform.python_version()}', f'numpy {np.__version__}',
                                 f'scipy {scipy.__version__}',
                                 f'active_subspaces {ac.__version__}', platform.platform()])}
    for (kind, k), r in results.items():
        for f in _fields(kind, k):
            out[f'{kind}_{k}_{f}'] = r[f]
    out['wall_total'] = np.float64(time.perf_counter() - t0)
    np.savez_compressed(args.out, **out)
    for job in jobs:
        path = os.path.join(ckpt, _part_name(job) + '.npz')
        if os.path.exists(path):
            os.remove(path)
    if not os.listdir(ckpt):
        os.rmdir(ckpt)
    for kind in chains:
        acc = np.mean([out[f'{kind}_{k}_accept'].mean() for k in range(1, n_chains[kind] + 1)])
        print(f'{kind}: mean acceptance {acc:.4f}', flush=True)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in '
          f'{out["wall_total"]:.0f} s = {out["wall_total"] / 60:.1f} min', flush=True)


def _fields(kind, k):
    if kind == 'vanilla':
        return ('x0', 'proj', 'thin', 'misfit', 'accept', 'last') + (('trace',) if k == 1 else ())
    return ('y0', 'y', 'ghat', 'accept')


if __name__ == '__main__':
    sys.exit(main())
