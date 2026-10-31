"""Generate the data of Arctic_Snow_Depth_SMRT with SMRT (needs SMRT; the notebook does not).

    python generate_data.py                       # 400 samples, 2 processes, writes here
    python generate_data.py --rows 0:5 --out /tmp/snow --check   # a subset, compared with the files here

Draws N = 400 Monte Carlo samples (numpy default_rng(424), independent uniform on the ranges of
Yan et al. 2026, Table 1, single-layer dry snow) of (depth, density, temperature, radius,
stickiness), runs `snow_smrt_model.SnowOnFYI` (SMRT 1.5.1, IBA with the dense-snow correction +
DORT, 12 AMSR2 channels at 55 deg; Supplement Table S1 of the paper),
and computes the gradient of every TB with respect to the physical inputs by finite differences
with step h = 1e-4 of each input's half-range: central where the stencil fits in the box,
second-order one-sided at the box edges (11 SMRT runs per sample).

TB has jumps (DATA.md, "Accuracy"): DORT's stream count in the snow layer changes with the snow
permittivity (steps of about 0.1 K across isolated surfaces in (density, temperature)), and the
dense-snow correction switches to the inverted medium at density 458.35 kg m^-3 (a step of up to
tens of K). A central stencil that straddles a jump gives a meaningless difference. It is detected by
comparing the two one-sided slopes (g+ and g-, in K per unit of normalized input): if
|g+ - g-| > 0.05 max(|g+|, |g-|) + 0.5 K, the one-sided slope of smaller magnitude is kept and
the entry is flagged in fd_jumps.csv (bit j of column i: input j, channel i).

Files written: inputs.csv, outputs.csv, gradients.csv, fd_jumps.csv. There is no other
randomness, so a run reproduces the files (to the digits written, on the same SMRT version; the
script stops if the installed SMRT is not CONFIG['smrt_version'] unless --any-smrt is given).
Run with one BLAS thread per worker (set below).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snow_smrt_model as sm  # noqa: E402

SEED = 424
N = 400
H = 1e-4                    # finite-difference step, fraction of each input's half-range
JUMP_REL, JUMP_ABS = 0.05, 0.5

_MODEL = None


def draw_inputs(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    return sm.LOWER + (sm.UPPER - sm.LOWER) * rng.random((n, len(sm.NAMES)))


def _init():
    global _MODEL
    import warnings
    warnings.filterwarnings('ignore', message='Using IBA with fraction_volume > 0.5')
    _MODEL = sm.SnowOnFYI()


def sample(x):
    """TB (12,), gradient (12, 5) in physical units, jump bitmask (12,) for one input row."""
    half = (sm.UPPER - sm.LOWER) / 2
    P, kinds = sm.fd_stencil(x, H)
    F = _MODEL(np.vstack([x, P]))
    f0, fp = F[0], F[1:]
    G = sm.fd_gradient(f0, fp, kinds)
    flags = np.zeros(len(f0), dtype=int)
    for j, (k, dlt) in enumerate(kinds):
        if k != 'c':
            continue
        gp = (fp[2 * j] - f0) / dlt * half[j]          # K per unit normalized input
        gm = (f0 - fp[2 * j + 1]) / dlt * half[j]
        bad = np.abs(gp - gm) > JUMP_REL * np.maximum(np.abs(gp), np.abs(gm)) + JUMP_ABS
        if np.any(bad):
            pick = np.where(np.abs(gp) <= np.abs(gm), gp, gm) / half[j]
            G[bad, j] = pick[bad]
            flags[bad] |= 1 << j
    return f0, G, flags


def header(desc, cols):
    return desc + '\n' + ','.join(cols)


def write(out, X, F, G, J):
    m, q = len(sm.NAMES), len(sm.CHANNELS)
    src = (f'SMRT {sm.CONFIG["smrt_version"]} IBA(dense-snow correction)+DORT(32 streams), single-layer dry snow '
           f'(sticky hard spheres) on 2-layer first-year ice, AMSR2 55 deg; Yan et al. 2026 Table 1; '
           f'Monte Carlo uniform, seed {SEED}; written by generate_data.py')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, delimiter=',', fmt='%.17g',
               header=header('inputs, physical units: ' + src,
                             [f'{n} [{u}]' for n, u in zip(sm.NAMES, sm.UNITS)]))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, delimiter=',', fmt='%.12g',
               header=header('brightness temperature TB [K] at the snow-air interface: ' + src,
                             [f'TB{c} [K]' for c in sm.CHANNELS]))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(X), q * m), delimiter=',',
               fmt='%.10g',
               header=header(f'dTB/dx in physical units (K per unit of x_j); column {m}*i + j is '
                             f'channel i, input j; finite differences h = {H:g} of half-range: ' + src,
                             [f'dTB{c}/d{n} [K/{u}]' for c in sm.CHANNELS
                              for n, u in zip(sm.NAMES, sm.UNITS)]))
    np.savetxt(os.path.join(out, 'fd_jumps.csv'), J, delimiter=',', fmt='%d',
               header=header('bitmask per channel: bit j set if a jump (DORT stream count or '
                             'dense-snow switch) was detected in the central stencil of input j and a one-sided slope '
                             'was used (bits: ' + ', '.join(f'{1 << j}={n}' for j, n in
                                                           enumerate(sm.NAMES)) + ')',
                             [f'jumps_TB{c}' for c in sm.CHANNELS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=None, help='a:b subset of the N samples')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--check', action='store_true', help='compare with the files next to this script')
    ap.add_argument('--any-smrt', action='store_true', help='run with another SMRT version')
    a = ap.parse_args()
    from importlib.metadata import version
    have = version('smrt')
    if have != sm.CONFIG['smrt_version'] and not a.any_smrt:
        sys.exit(f"smrt {have} installed; the shipped data use smrt {sm.CONFIG['smrt_version']} "
                 f"(pip install smrt=={sm.CONFIG['smrt_version']}) or pass --any-smrt")
    X = draw_inputs()
    rows = slice(*[int(s) for s in a.rows.split(':')]) if a.rows else slice(None)
    X = X[rows]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        res = pool.map(sample, list(X), chunksize=1)
    F = np.array([r[0] for r in res])
    G = np.array([r[1] for r in res])
    J = np.array([r[2] for r in res])
    write(a.out, X, F, G, J)
    print(f'{len(X)} samples, {len(X) * 11} SMRT runs, {time.time() - t0:.0f} s on {a.workers} '
          f'processes; jump flags in {np.count_nonzero(J)} (sample, channel) entries')
    if a.check:
        ld = lambda p: np.loadtxt(p, delimiter=',', ndmin=2)  # noqa: E731
        for name in ['inputs.csv', 'outputs.csv', 'gradients.csv', 'fd_jumps.csv']:
            new, old = ld(os.path.join(a.out, name)), ld(os.path.join(HERE, name))[rows]
            rel = np.abs(new - old).max() / max(np.abs(old).max(), 1e-300)
            print(f'{name}: max |new - shipped| / max |shipped| = {rel:.1e}')


if __name__ == '__main__':
    main()
