"""Write the CSVs of the undamped yielding-oscillator limit state.

    python generate_data.py --check                 # writes to a temporary directory, compares
    python generate_data.py --out DIR --check       # the same, keeping the files in DIR (not this directory)
    python generate_data.py --out .                 # rewrites the shipped CSVs (run from this directory)

--out is required without --check, so the script never writes into this directory unless asked to.
`--check` prints the largest differences from the shipped files and fails only on structural
mismatches (shapes, NaN pattern, the yielded flags, inputs).

Draws M = 1000 independent Monte Carlo samples of the six lognormal inputs of
`oscillator_model` (means and standard deviations of Rajashekhar and Ellingwood 1993,
Table 7, p. 218, read as those of x; ln x ~ N(MU, diag(SIGMA^2))) with numpy's default
generator seeded with SEED = 1993, and computes

  * for the yielding oscillator of Fig. 10 (the main model): g = 3 r - z_max, z_max, whether
    spring 2 yields, and the exact gradient of g with respect to the physical inputs
    (forward-mode differentiation of the event-by-event solution);
  * for the closed-form linear benchmark (secondary study): g_lin = 3 r - |z_lin|, z_lin, and its
    exact closed-form gradient.

It checks both gradients against central finite differences in the standardized log inputs
(step 1e-5 in each y_i = (ln x_i - MU_i)/SIGMA_i) at every sample, prints the largest relative
error (row 2-norm), and counts the samples whose finite-difference stencil reaches a switching
event of the yielding model.

Only numpy and the standard library are imported. One process; a few seconds.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oscillator_model as om  # noqa: E402

SEED = 1993
M = 1000
H_Y = 1e-5      # central-difference step in the standardized log inputs, for the check only
# --check: differences relative, elementwise, with |shipped value| floored at FLOOR (absolute near 0)
FLOOR = 0.01
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only

PROVENANCE = ('oscillator_model.py (Rajashekhar and Ellingwood 1993, Example 2: g = 3r - |z_max|, Eq. 12; '
              'z_max of the yielding oscillator of Fig. 10 solved exactly event by event; closed-form '
              'linear z_lin as in UQTestFuns 0.7.0 UndampedOscillator for the secondary study); '
              f'lognormal inputs with the mean and sd of Table 7; numpy default_rng({SEED}); '
              'written by generate_data.py')


def draw(seed=SEED, M=M):
    return om.sample_lognormal(M, np.random.default_rng(seed))


def fd_check(X, Gy, fun, h=H_Y):
    """relative error (row 2-norm) of the gradient in y = (ln x - MU)/SIGMA against central differences."""
    fd = np.empty_like(Gy)
    for j in range(6):
        e = np.zeros(6)
        e[j] = h * om.SIGMA[j]
        fd[:, j] = (fun(X * np.exp(e)) - fun(X * np.exp(-e))) / (2 * h)
    return np.linalg.norm(fd - Gy, axis=1) / np.linalg.norm(Gy, axis=1)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def compute(X):
    res = om.response(X, grad=True)
    gl, Gl = om.limit_state_linear(X, grad=True)
    zl, _ = om.parts_linear(X)
    F = np.column_stack([res['g'], res['z_max'], res['yielded'].astype(float), gl, zl])
    return res, F, res['grad'], Gl


def write(out, X, F, G, Gl):
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the physical inputs, lognormal with mean '
                      f'{om.MEAN.tolist()} and sd {om.SD.tolist()} (Table 7, p. 218). {PROVENANCE}\n'
                      'm (-),c1 (-),c2 (-),r (-),F1 (-),t1 (-)')
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt='%.17g', delimiter=',',
               header='Yielding oscillator (Fig. 10): limit state g = 3r - z_max (failure when g <= 0), peak '
                      'displacement z_max = max |z(t)|, yielded = 1 if spring 2 yields; linear closed form: '
                      f'g_lin = 3r - |z_lin|, z_lin. {PROVENANCE}\n'
                      'g (-),z_max (-),yielded (0/1),g_lin (-),z_lin (-)')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='Exact gradient of g (yielding oscillator) with respect to the PHYSICAL inputs '
                      f'(forward-mode differentiation of the event-by-event solution). {PROVENANCE}\n'
                      + ','.join(f'dg/d{n} (-)' for n in om.NAMES))
    np.savetxt(os.path.join(out, 'gradients_linear.csv'), Gl, fmt='%.17g', delimiter=',',
               header='Exact gradient of g_lin (linear closed form) with respect to the PHYSICAL inputs. '
                      f'{PROVENANCE}\n' + ','.join(f'dg_lin/d{n} (-)' for n in om.NAMES))


FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv', 'gradients_linear.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (required without --check; with --check, a temporary directory by default)')
    ap.add_argument('--check', action='store_true',
                    help='compare with the CSVs in this directory (--out DIR, if given, must be elsewhere)')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            ap.error('--out DIR is required without --check')
    here = os.path.realpath(HERE)
    out_real = os.path.realpath(args.out)
    if args.check and os.path.commonpath([out_real, here]) == here:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    res, F, G, Gl = compute(X)
    write(args.out, X, F, G, Gl)
    g = res['g']
    print(f'wrote {args.out}: M = {len(X)}; yielding oscillator: g in [{g.min():.4f}, {g.max():.4f}], '
          f'{int((g <= 0).sum())} samples with g <= 0, spring 2 yields in {int(res["yielded"].sum())}; '
          f'linear closed form: {int((F[:, 3] <= 0).sum())} samples with g_lin <= 0, w0 t1/2 in '
          f'[{om.parts_linear(X)[1].min():.4f}, {om.parts_linear(X)[1].max():.4f}] (the closed form is the linear peak up to pi/2)')
    sy = om.SIGMA * X
    e = fd_check(X, G * sy, om.limit_state)
    mg = res['margins']
    near = {k: int((v < 2 * H_Y).sum()) for k, v in mg.items()}
    print(f'yielding oscillator, gradient vs central differences (step {H_Y:g} in standardized log inputs), '
          f'all {len(X)} samples: max relative error {e.max():.1e}, median {np.median(e):.1e}; samples whose '
          f'stencil reaches a pulse-end / yield-onset switching set: {near["pulse_end"]} / {near["yield_onset"]}; '
          f'smallest relative gap between the largest and the next peak: {np.min(res["peak_gap"]):.1e}')
    el = fd_check(X, Gl * sy, om.limit_state_linear)
    print(f'linear closed form, gradient vs central differences: max relative error {el.max():.1e}, '
          f'median {np.median(el):.1e}')
    print(f'total {time.perf_counter() - t0:.1f} s')
    if args.check:
        bad, worst = [], (-1.0, '')
        for n in FILES:
            a, b = read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                continue
            if n == 'outputs.csv' and not np.array_equal(a[:, 2], b[:, 2]):
                bad.append(f'FAIL: {n}: the yielded flag differs at {int(np.sum(a[:, 2] != b[:, 2]))} samples')
            rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), FLOOR))
            print(f'{n}: largest difference from the shipped file {rel:.1e} (relative, elementwise, '
                  f'|shipped value| floored at {FLOOR:g})')
            if n == 'inputs.csv' and rel > INPUT_RTOL:
                bad.append(f'FAIL: {n} differs from the shipped inputs by {rel:.1e} > {INPUT_RTOL:g}')
            worst = max(worst, (rel, n))
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)

if __name__ == '__main__':
    main()
