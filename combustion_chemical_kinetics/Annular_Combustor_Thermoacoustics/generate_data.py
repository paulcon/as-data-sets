"""Generate the shipped data of Annular_Combustor_Thermoacoustics from annular_combustor_model.py.

    python generate_data.py              # both cases, writes the CSVs next to this script
    python generate_data.py --out /tmp/acx --mc 2000    # smaller Monte Carlo, elsewhere

For each case (weak, strong coupling; Bauerheim et al. 2016, Table 2) this writes

  inputs_<case>.csv      M = 1000 samples of the 38 physical inputs, uniform and independent:
                         n_i on n_bar (1 +- sqrt(3) 0.10), tau_i on tau_bar (1 +- sqrt(3) 0.05)
                         (standard deviations 10 % and 5 % of the mean, JFM p. 13)
  outputs_<case>.csv     per sample: f_Im = max Im(f) of the split pair [1/s], the other
                         component's Im(f) [1/s], and both components' Re(f) [Hz]
  gradients_<case>.csv   exact d f_Im / d x with respect to the physical inputs
  montecarlo_<case>.csv  10 000 further samples (same distribution, own seed): the same
                         four output columns as outputs_<case>.csv (inputs are not stored;
                         `mc_inputs(case)` redraws them from the seed)

and prints the checks quoted in DATA.md: the nominal roots, the Magri et al. Table 2
eigenvalues, the branch check, and the gradients against central differences.
Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import annular_combustor_model as acm  # noqa: E402

M_GRAD = 1000
M_MC = 10000
SEEDS = {'weak': (20161, 20171), 'strong': (20162, 20172)}   # (gradient set, Monte Carlo set)
OUT_COLS = 'f_Im [1/s],f_Im_other [1/s],freq_max [Hz],freq_other [Hz]'


def draw(case, M, seed):
    lo, up = acm.bounds(case)
    return lo + (up - lo) * np.random.default_rng(seed).random((M, 2 * acm.N))


def mc_inputs(case, M=M_MC):
    """The Monte Carlo inputs of montecarlo_<case>.csv, redrawn from the saved seed."""
    return draw(case, M, SEEDS[case][1])


def outputs(roots):
    return np.c_[roots.imag / (2 * np.pi), roots.real / (2 * np.pi)]


def run(X, case, grad, chunk=500):
    F, G, R, C, S = [], [], [], [], []
    for s in range(0, len(X), chunk):
        res = acm.growth_rate(X[s:s + chunk], case, grad=grad, return_info=True)
        info = res[-1]
        F.append(res[0])
        if grad:
            G.append(res[1])
        R.append(info['roots'])
        C.append(info['count'])
        S.append(info['newton_step'])
    roots = np.concatenate(R)
    return (np.concatenate(F), np.concatenate(G) if grad else None, roots,
            np.concatenate(C), np.concatenate(S))


def checks():
    print('--- nominal (symmetric) first azimuthal roots')
    for case, (nb, tb) in acm.CASES.items():
        w = acm.symmetric_root(case)
        print(f'{case}: n = {nb}, tau = {tb*1e3} ms: omega = {w.real:.3f} {w.imag:+.4f}j rad/s, '
              f'f = {w.real/2/np.pi:.3f} Hz, Im f = {w.imag/2/np.pi:+.4f} 1/s')
        X = np.r_[np.full(acm.N, nb), np.full(acm.N, tb)][None]
        _, cnt, _ = acm.split_roots(X[:, :acm.N], X[:, acm.N:], case)
        print(f'   argument-principle count at the nominal point: {cnt[0]:.6f} (double root)')
    print('--- Magri et al. (2016) Table 2, cases A and B (paper: 5.059e3 - 1.392j; 4.272e3 + 3.600e-3j)')
    for c, nt in (('A', (0.5, 6.223e-4)), ('B', (1.75, 7.35e-4))):
        w = acm.symmetric_root(nt)
        print(f'case {c}: n = {nt[0]}, tau = {nt[1]} s: omega = {w.real:.2f} {w.imag:+.4f}j rad/s')


def fd_check(case, X, G, h_frac=1e-5):
    lo, up = acm.bounds(case)
    errs = []
    for x, g in zip(X, G):
        fd = np.empty(2 * acm.N)
        for j in range(2 * acm.N):
            h = h_frac * (up[j] - lo[j])
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            fd[j] = (acm.growth_rate(xp[None], case)[0] - acm.growth_rate(xm[None], case)[0]) / (2 * h)
        errs.append(np.linalg.norm(fd - g) / np.linalg.norm(g))
    return np.array(errs)


def write(path, A, fmt, header):
    np.savetxt(path, A, fmt=fmt, delimiter=',', header=header)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--mc', type=int, default=M_MC, help='Monte Carlo sample size (default 10000)')
    ap.add_argument('--nfd', type=int, default=10, help='samples per case in the finite-difference check')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_all = time.perf_counter()
    checks()
    prov = ('annular_combustor_model.py (ATACAMAC dispersion relation, N = 19; JFM 2016 Table 1 '
            'constants with L_i = 0.0551 m, alpha = 0.862 from Magri et al. 2016 Table 1); '
            'written by generate_data.py')
    for case in acm.CASES:
        nb, tb = acm.CASES[case]
        lo, up = acm.bounds(case)
        dist = (f'{case} coupling, n_bar = {nb}, tau_bar = {tb} s; uniform inputs n_i in [{lo[0]:.6g}, '
                f'{up[0]:.6g}], tau_i in [{lo[-1]:.6g}, {up[-1]:.6g}] s (sd 10 % / 5 % of mean)')
        t0 = time.perf_counter()
        X = draw(case, M_GRAD, SEEDS[case][0])
        f, G, roots, cnt, stp = run(X, case, grad=True)
        t1 = time.perf_counter()
        res = np.abs(acm.dispersion(roots[:, 0], X[:, :acm.N], X[:, acm.N:]))
        print(f'--- {case}: {M_GRAD} gradient samples in {t1 - t0:.1f} s; count in [{cnt.min():.9f}, '
              f'{cnt.max():.9f}], last Newton step <= {stp.max():.2e} rad/s, |D(root)| <= {res.max():.1e}; '
              f'roots within {np.abs(roots - acm.symmetric_root(case)).max():.1f} rad/s of nominal '
              f'(circle radius {acm.RADIUS[case]:.0f}); min split |w1 - w2| = '
              f'{np.abs(roots[:, 0] - roots[:, 1]).min():.2f} rad/s')
        print(f'    fraction f_Im > 0: {100 * (f > 0).mean():.2f} %; mean {f.mean():.3f}, sd {f.std():.3f} 1/s')
        write(os.path.join(args.out, f'inputs_{case}.csv'), X, '%.12e',
              header=f'{M_GRAD} samples of the 38 physical inputs, seed {SEEDS[case][0]}; {dist}\n'
                     + ','.join(n + ('' if n.startswith('n') else ' [s]') for n in acm.NAMES))
        write(os.path.join(args.out, f'outputs_{case}.csv'), outputs(roots), '%.12e',
              header=f'First azimuthal mode, split pair: QoI f_Im = max Im(f) = max Im(omega)/(2 pi). {prov}\n'
                     + OUT_COLS)
        write(os.path.join(args.out, f'gradients_{case}.csv'), G, '%.12e',
              header=f'Exact gradient of f_Im w.r.t. the physical inputs (implicit function theorem). {prov}\n'
                     + ','.join(f'df/d{n} [1/s]' if n.startswith('n') else f'df/d{n} [1/s^2]'
                                for n in acm.NAMES))
        e = fd_check(case, X[:args.nfd], G[:args.nfd])
        print(f'    gradient vs central differences (h = 1e-5 of the range), {args.nfd} samples: '
              f'relative error (2-norm) max {e.max():.2e}, median {np.median(e):.2e}')
        Xmc = draw(case, args.mc, SEEDS[case][1])
        fm, _, rm, cm, sm = run(Xmc, case, grad=False)
        t2 = time.perf_counter()
        print(f'    Monte Carlo {args.mc} samples in {t2 - t1:.1f} s: risk factor P(f_Im > 0) = '
              f'{100 * (fm > 0).mean():.2f} % (binomial s.e. {100*np.sqrt((fm>0).mean()*(1-(fm>0).mean())/args.mc):.2f}); '
              f'mean {fm.mean():.3f}, sd {fm.std():.3f}; count in [{cm.min():.9f}, {cm.max():.9f}], '
              f'Newton step <= {sm.max():.2e}')
        write(os.path.join(args.out, f'montecarlo_{case}.csv'), outputs(rm), '%.10e',
              header=f'{args.mc} Monte Carlo samples, seed {SEEDS[case][1]} (inputs: generate_data.mc_inputs); '
                     f'{dist}. {prov}\n' + OUT_COLS)
    print(f'total {time.perf_counter() - t_all:.0f} s')


if __name__ == '__main__':
    main()
