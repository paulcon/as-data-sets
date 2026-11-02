"""Write thermal_compliance.npz: M = 1500 random material distributions, the thermal
energy E and its exact gradient dE/dtheta, from thermal_model.py.

    python generate_data.py                         # all 1500 samples, writes here (about 25 s)
    python generate_data.py --rows 0:50 --check     # recompute 50 rows, compare with the file here
    python generate_data.py --fd-check              # exact gradients vs central differences
    python generate_data.py --casm /path/to/CASM    # reproduce the author's stored files

Design: theta ~ U[0, 1]^512, independent (Monte Carlo), drawn with
numpy.random.default_rng(SEED).random((M, 512)) and rounded to float32. The
model is run at the rounded values (cast back to float64), so the float32
inputs in the file are exactly the inputs of the stored E and gradients.

`--casm DIR` needs a clone of github.com/paluneau/CASM (MIT licence). It runs
the model with the CASM notebook's 124-node boundary list (boundary='casm') at the
author's 512 gradient inputs (np.random.seed(42), as in that notebook) and at
the 50 inputs in Xtest.npy, and compares with dfData.npy (dE/dx, x = 2 theta - 1)
and fData.npy (E - E_max). It prints the comparison and writes no file. To
rebuild the 124-node variant at this data set's inputs (not shipped):
compute(design(), model=tm.ThermalCompliance(boundary='casm')).

Only numpy, scipy and the standard library are imported; one process.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thermal_model as tm  # noqa: E402

SEED = 20250129   #: seed of the input design
M = 1500          #: number of samples
FILE = 'thermal_compliance.npz'


def design(M=M, seed=SEED):
    """The float32 input design, shape (M, 512)."""
    return np.random.default_rng(seed).random((M, tm.M_INPUTS)).astype(np.float32)


def compute(theta32, model=None, verbose=True):
    model = model or tm.ThermalCompliance()
    E = np.empty(theta32.shape[0])
    G = np.empty(theta32.shape, dtype=np.float64)
    t0 = time.time()
    for i, th in enumerate(theta32):
        E[i], G[i] = model.energy_and_grad(th.astype(np.float64))
        if verbose and (i + 1) % 250 == 0:
            print(f'  {i + 1} samples, {time.time() - t0:.1f} s', flush=True)
    return E, G


def write(path, theta32, E, G):
    np.savez_compressed(
        path,
        theta=theta32, E=E, dE_dtheta=G,
        E_max=np.float64(tm.E_MAX),
        description=np.array(
            'Luneau (2025) Sec. 4.3 thermal design model (thermal_model.py, boundary="paper"): '
            f'{len(E)} Monte Carlo samples of theta ~ U[0,1]^512 (seed {SEED}, float32); '
            'E = thermal energy (heat compliance, units of the nondimensional model); '
            'dE_dtheta = exact adjoint gradient w.r.t. the physical theta. Written by generate_data.py.'))


def fd_check(n_samples=3, h=1e-5, seed=7):
    """Relative error of the exact gradient against central differences (step h in theta)."""
    model = tm.ThermalCompliance()
    th_all = design()[:n_samples].astype(np.float64)
    out = []
    for th in th_all:
        _, g = model.energy_and_grad(th)
        fd = np.empty_like(g)
        for e in range(th.size):
            up, dn = th.copy(), th.copy()
            hp = min(h, 1.0 - th[e])
            hm = min(h, th[e])
            up[e] += hp
            dn[e] -= hm
            fd[e] = (model.energy(up) - model.energy(dn))/(hp + hm)
        out.append(np.linalg.norm(g - fd)/np.linalg.norm(g))
        print(f'  ||g - g_fd|| / ||g|| = {out[-1]:.2e} (max entry rel. {np.max(np.abs(g - fd)/np.abs(g)):.2e})')
    return out


def casm_check(casm_dir):
    df = np.load(os.path.join(casm_dir, 'dfData.npy'))
    fdata = np.load(os.path.join(casm_dir, 'fData.npy')).ravel()
    Xt = np.load(os.path.join(casm_dir, 'Xtest.npy'))
    ld = np.linalg.eigvalsh(df.T @ df/len(df))[::-1]
    print(f'dfData.npy itself: rank {np.linalg.matrix_rank(df)}, lambda_1 = {ld[0]:.6e}, '
          f'smallest eigenvalue {ld[-1]:.2e} (eps*lambda_1 = {np.finfo(float).eps*ld[0]:.1e})')
    np.random.seed(42)  # the CASM notebook's legacy global seed and draw order
    X = np.vstack(np.array([np.random.uniform(-1, 1, 512) for _ in range(512)]).T)
    for bnd in ('casm', 'paper'):
        model = tm.ThermalCompliance(boundary=bnd)
        G = np.array([0.5*model.grad((x + 1)/2) for x in X])     # dE/dx = dE/dtheta / 2
        Et = np.array([model.energy((x + 1)/2) for x in Xt]) - tm.E_MAX
        lam = np.linalg.eigvalsh(G.T @ G/len(G))[::-1]
        print(f'boundary={bnd!r}: dfData rel. diff {np.linalg.norm(G - df)/np.linalg.norm(df):.1e}, '
              f'fData max rel. diff {np.max(np.abs(Et - fdata)/np.abs(fdata)):.1e}, '
              f'lambda_1, lambda_2 = {lam[0]:.6e}, {lam[1]:.6e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default=None, help='a:b, a subset of rows')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--check', action='store_true', help='compare with the shipped file')
    ap.add_argument('--fd-check', action='store_true')
    ap.add_argument('--casm', default=None, help='path to a CASM clone')
    a = ap.parse_args()
    if a.fd_check:
        fd_check()
        return
    if a.casm:
        casm_check(a.casm)
        return
    theta = design()
    rows = slice(None) if a.rows is None else slice(*map(int, a.rows.split(':')))
    t0 = time.time()
    E, G = compute(theta[rows])
    print(f'{len(E)} samples in {time.time() - t0:.1f} s')
    if a.check:
        ref = np.load(os.path.join(HERE, FILE))
        for k, v in (('theta', theta[rows]), ('E', E), ('dE_dtheta', G)):
            r = ref[k][rows]
            print(f'  {k}: largest relative difference {np.max(np.abs(v - r))/np.max(np.abs(r)):.1e}')
        return
    if a.rows is not None:
        raise SystemExit('--rows writes nothing without --check')
    os.makedirs(a.out, exist_ok=True)
    write(os.path.join(a.out, FILE), theta, E, G)
    print('wrote', os.path.join(a.out, FILE))


if __name__ == '__main__':
    main()
