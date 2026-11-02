"""Recompute the accuracy numbers quoted in DATA.md for Arctic_Snow_Depth_SMRT (needs SMRT 1.5.1).

    python check_accuracy.py                  # all checks
    python check_accuracy.py --skip-streams   # without check 6 (64 streams, the slowest part)

Prints, beside the values DATA.md quotes:

1. Step study at 8 shipped samples (340, 71, 10, 255, drawn with numpy default_rng(2026).choice(400, 4);
   339; the first two samples above the dense-snow threshold, 25 and 34; and the smallest depth, 88):
   TB against the stored outputs, and the normalized gradients with steps h = 3e-5 and 3e-4 (fraction
   of each half-range) against the stored h = 1e-4 gradients, as the relative 2-norm difference per
   channel (5-vector).
2. The dense-snow step: TB(458.36) - TB(458.34 kg m^-3) at the midpoint of the other inputs, at the
   corner (0.6 m, 240 K, 0.4 mm, 0.1), and at 40 points of the threshold face (other inputs uniform,
   numpy default_rng(7)); and the notebook's estimate of the 36V jump from the stored data (the step
   term of its degree-5 Legendre surrogate, refitted here) at the same 40 points.
3. Zero depth: TB for d = 0 (bare ice) and d = 1e-7 m at the midpoint of the other inputs.
4. Thin layers: the largest change of TB for d from 1e-7 to 1e-3 m against d = 1e-7 m.
5. The jump detector at a point 0.005 kg m^-3 above the threshold (other inputs of sample 321):
   which channels are flagged, and the kept density slope against sample 321's stored gradient.
6. Stream count: samples 0-39 with 64 DORT streams instead of 32: TB and gradient differences per
   channel, and the TB36V active subspace from those 40 samples with either stream count.

Deterministic (fixed rows and seeds). Needs numpy and SMRT (pip install smrt==1.5.1, the version
that wrote the shipped files); without SMRT the script stops with an error. It reads the files next
to it and writes nothing.
"""
import argparse
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
from numpy.polynomial import legendre as L

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snow_smrt_driver as sm  # noqa: E402
import generate_data as gd  # noqa: E402

HALF = (sm.UPPER - sm.LOWER) / 2
MID = (sm.LOWER + sm.UPPER) / 2
RHO_STAR = sm.DENSE_THRESHOLD
CH = {c: k for k, c in enumerate(sm.CHANNELS)}


def load():
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=',', ndmin=2)  # noqa: E731
    return ld('inputs.csv'), ld('outputs.csv'), ld('gradients.csv').reshape(-1, 12, 5)


def grad(model, x, h):
    P, kinds = sm.fd_stencil(x, h)
    F = model(np.vstack([x, P]))
    return F[0], sm.fd_gradient(F[0], F[1:], kinds)


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


# --- the notebook's surrogate of TB36V (degree-5 Legendre PCE plus a step term), refitted ----------
DEG, DEG_Q = 5, 2
NORM = np.sqrt(2 * np.arange(DEG + 1) + 1)
DP = np.array([np.r_[L.legder(np.eye(DEG + 1)[i]), 0.0] for i in range(DEG + 1)])
Z_STAR = (RHO_STAR - sm.LOWER[1]) / HALF[1] - 1


def multi_indices(deg):
    return np.array([a for a in product(range(deg + 1), repeat=5) if sum(a) <= deg])


ALPHAS, ALPHAS_Q = multi_indices(DEG), multi_indices(DEG_Q)


def basis(Z, alph, grads=True):
    V = np.stack([L.legvander(Z[:, j], DEG) * NORM for j in range(5)], axis=1)
    Phi = np.prod(np.stack([V[:, j, alph[:, j]] for j in range(5)]), axis=0)
    if not grads:
        return Phi
    D = np.stack([np.stack([L.legval(Z[:, j], DP[i]) for i in range(DEG + 1)], 1) * NORM
                  for j in range(5)], axis=1)
    dPhi = np.stack([D[:, j, alph[:, j]] * np.prod(np.stack([V[:, i, alph[:, i]] for i in range(5) if i != j]),
                                                   axis=0) for j in range(5)], axis=1)
    return Phi, dPhi


def surrogate_jump_36v(X, TB, G, Xface):
    """Step term of the notebook's 36V surrogate at the face points (K)."""
    Z = 2 * (X - sm.LOWER) / (sm.UPPER - sm.LOWER) - 1
    H = (Z[:, 1] > Z_STAR).astype(float)
    (P1, dP1), (P2, dP2) = basis(Z, ALPHAS), basis(Z, ALPHAS_Q)
    Phi = np.hstack([P1, H[:, None] * P2])
    dPhi = np.concatenate([dP1, H[:, None, None] * dP2], axis=2)
    k = CH['36V']
    A = np.vstack([Phi, dPhi.reshape(-1, Phi.shape[1])])
    b = np.r_[TB[:, k], (G[:, k] * HALF).reshape(-1)]
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    Zf = 2 * (Xface - sm.LOWER) / (sm.UPPER - sm.LOWER) - 1
    Zf[:, 1] = Z_STAR
    return basis(Zf, ALPHAS_Q, grads=False) @ c[len(ALPHAS):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-streams', action='store_true', help='skip check 6 (64 streams)')
    a = ap.parse_args()
    if not sm.have_smrt():
        sys.exit(f"SMRT is not installed. Install it (pip install smrt=={sm.CONFIG['smrt_version']}, the version "
                 f"that wrote the shipped files) and rerun check_accuracy.py.")
    from importlib.metadata import version
    warnings.filterwarnings('ignore', message='Using IBA with fraction_volume > 0.5')
    warnings.filterwarnings('ignore', category=FutureWarning)
    t0 = time.perf_counter()
    print(f"smrt {version('smrt')} (the shipped data: {sm.CONFIG['smrt_version']}), numpy {np.__version__}")
    model = sm.SnowOnFYI()
    X, TB, G = load()
    Gn = G * HALF

    # 1. step study
    rows = list(np.random.default_rng(2026).choice(len(X), 4)) + [339]
    dense = np.flatnonzero(X[:, 1] > RHO_STAR)[:2]
    rows = [int(i) for i in rows] + [int(i) for i in dense] + [int(X[:, 0].argmin())]
    print(f'\n1. step study at samples {rows} (dense: {dense.tolist()}, smallest depth: {rows[-1]})')
    dTB, E = [], {}
    for h in (3e-5, 3e-4):
        E[h] = np.empty((len(rows), 12))
        for r, i in enumerate(rows):
            f0, g = grad(model, X[i], h)
            E[h][r] = rel(g * HALF, Gn[i])
            if h == 3e-5:
                dTB.append(np.abs(f0 - TB[i]).max())
    print(f'   TB at the 8 samples against the stored outputs: max |difference| {max(dTB):.1e} K (DATA.md: 5e-10 K)')
    for h, quoted in ((3e-5, 'max 1.3e-4 (sample 71, 36H), 8.4e-5 at 36V, 1.4e-8 at 18V, median 8.2e-7; '
                             'smallest depth 6.4e-7'), (3e-4, 'max 1.2e-3 (sample 71), median 1.1e-5')):
        e = E[h]
        r, k = np.unravel_index(e.argmax(), e.shape)
        print(f'   h = {h:.0e}: max {e.max():.1e} (sample {rows[r]}, {sm.CHANNELS[k]}), at 36V {e[:, CH["36V"]].max():.1e}, '
              f'at 18V {e[:, CH["18V"]].max():.1e}, median {np.median(e):.1e}; smallest depth {e[-1].max():.1e} '
              f'(DATA.md: {quoted})')

    # 2. dense-snow step
    def jump(Xo):
        lo, hi = Xo.copy(), Xo.copy()
        lo[:, 1], hi[:, 1] = RHO_STAR - 0.01, RHO_STAR + 0.01
        return model(hi) - model(lo)
    sel = [CH[c] for c in ('18V', '36V', '89V')]
    pts = np.array([MID, [0.6, RHO_STAR, 240.0, 0.4, 0.1]])
    J = jump(pts)[:, sel]
    print('\n2. dense-snow step TB(458.36) - TB(458.34) [K], 18V / 36V / 89V')
    print(f'   midpoint: {J[0, 0]:+.2f} / {J[0, 1]:+.2f} / {J[0, 2]:+.1f} (DATA.md: -0.46 / -6.83 / -41.7)')
    print(f'   corner:   {J[1, 0]:+.2f} / {J[1, 1]:+.2f} / {J[1, 2]:+.2f} (DATA.md: -7.80 / -54.03 / -56.34)')
    face = sm.LOWER + (sm.UPPER - sm.LOWER) * np.random.default_rng(7).random((40, 5))
    face[:, 1] = RHO_STAR
    Jf = jump(face)
    j36 = Jf[:, CH['36V']]
    rms = lambda v: np.sqrt(np.mean(v**2))  # noqa: E731
    print(f'   40 face points: 36V mean {j36.mean():+.2f}, rms {rms(j36):.2f}, range {j36.min():+.2f} to {j36.max():+.2f}; '
          f'18V mean {Jf[:, CH["18V"]].mean():+.2f}, rms {rms(Jf[:, CH["18V"]]):.2f}; 89V mean '
          f'{Jf[:, CH["89V"]].mean():+.2f}, rms {rms(Jf[:, CH["89V"]]):.2f}; largest |jump| below 18 GHz '
          f'{np.abs(Jf[:, :4]).max():.2f} (DATA.md: 36V -8.56, 12.85, -46.23 to -0.17; 18V -0.67, 1.13; 89V -36.94, 40.72)')
    est = surrogate_jump_36v(X, TB, G, face)
    print(f'   notebook surrogate estimate at the same points: correlation with SMRT {np.corrcoef(est, j36)[0, 1]:.3f}, '
          f'rms difference {rms(est - j36):.1f} K (DATA.md: 0.990, 1.7 K)')

    # 3, 4. zero depth and thin layers
    d = np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    P = np.repeat(MID[None], len(d), axis=0)
    P[:, 0] = d
    T = model(P)
    print('\n3. zero depth, midpoint of the other inputs: TB for d = 0 and d = 1e-7 m: ' +
          ', '.join(f'{c} {T[0, CH[c]]:.2f} / {T[1, CH[c]]:.2f}' for c in ('6H', '18H', '36V')) +
          ' (DATA.md: 6H 197.58 / 228.47, 18H 198.34 / 228.95, 36V 262.42 / 260.92)')
    dT = np.abs(T[2:] - T[1]).max(axis=0)
    print(f'4. thin layers, d = 1e-7 to 1e-3 m against 1e-7 m: max |change| below 89 GHz {dT[:10].max():.3f} K, '
          f'89V {dT[CH["89V"]]:.2f} K, 89H {dT[CH["89H"]]:.2f} K (DATA.md: 0.02 K, 89V 0.46 K)')

    # 5. jump detector at a synthetic point just above the threshold
    gd._MODEL = model
    x = X[321].copy()
    x[1] = RHO_STAR + 0.005
    _, g5, flags = gd.sample(x)
    print(f'\n5. jump detector at density {x[1]:.3f} (other inputs of sample 321): density bit set in '
          f'{np.sum(flags & 2 > 0)} of 12 channels; kept density slope against sample 321 stored: '
          f'max relative difference {np.max(np.abs(g5[:, 1] - G[321][:, 1]) / np.abs(G[321][:, 1])):.1e} '
          f'(DATA.md: flagged in all 12 channels, slope matched)')

    # 6. stream count
    if not a.skip_streams:
        m64 = sm.SnowOnFYI(n_max_stream=64)
        G32, G64, T32, T64 = [], [], [], []
        for i in range(40):
            f, g = grad(model, X[i], gd.H)
            f6, g6 = grad(m64, X[i], gd.H)
            T32.append(f); T64.append(f6); G32.append(g * HALF); G64.append(g6 * HALF)
        T32, T64, G32, G64 = map(np.array, (T32, T64, G32, G64))
        dT = np.abs(T64 - T32)
        eg = np.linalg.norm(G64 - G32, axis=2) / np.linalg.norm(G32, axis=2)          # (40, 12)
        lo = [CH[c] for c in sm.CHANNELS if not c.startswith(('36', '89'))]
        hi = [CH[c] for c in sm.CHANNELS if c.startswith(('36', '89'))]
        print('\n6. 64 streams instead of 32, samples 0-39')
        print(f'   recomputed 32-stream TB against the stored outputs: max |difference| {np.abs(T32 - TB[:40]).max():.1e} K')
        print(f'   TB: median |difference| per channel {np.median(dT, 0).min():.2f}-{np.median(dT, 0).max():.2f} K; '
              f'max below 36 GHz {dT[:, lo].max(0).min():.2f}-{dT[:, lo].max(0).max():.2f} K, at 36 and 89 GHz '
              f'{dT[:, hi].max(0).min():.2f}-{dT[:, hi].max(0).max():.2f} K (DATA.md, SMRT 1.7.3: 0.27-0.41, '
              f'0.36-0.61, 3.2-4.6 K)')
        print(f'   gradients: median relative difference per channel {np.median(eg, 0).min():.1e}-{np.median(eg, 0).max():.1e} '
              f'(DATA.md, SMRT 1.7.3: 0.6-6.5 %)')
        k = CH['36V']
        W = {}
        for n, Gs in (('32', G32), ('64', G64)):
            lam, V = np.linalg.eigh(Gs[:, k].T @ Gs[:, k] / 40)
            W[n] = (lam[::-1], V[:, ::-1])
            print(f'   TB36V, {n} streams: lambda1/lambda2 {lam[-1] / lam[-2]:.2f}')
        print(f'   |w1(32) . w1(64)| {abs(W["32"][1][:, 0] @ W["64"][1][:, 0]):.4f} (DATA.md, SMRT 1.7.3: 14.15 with '
              f'either, 1.0000)')
    print(f'\nrun time {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
