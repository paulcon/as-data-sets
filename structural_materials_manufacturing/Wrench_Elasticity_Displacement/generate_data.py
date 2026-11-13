"""Write the mesh, the field basis, the samples and the reference eigenpairs of the wrench model.

    python generate_data.py --out DIR                 # writes every data file to DIR
    python generate_data.py --check                   # writes to a temporary directory and compares with the files here
    python generate_data.py --out DIR --check         # writes to DIR and compares with the files here
    python generate_data.py --n 100 --check           # a quick partial run (first 100 samples, no reference)

DIR must not be a folder inside this directory, and may be this directory itself only without
--check and --n; the shipped files were written with `--out .` from this directory. `--check`
prints the largest differences from the shipped files and fails only on structural mismatches
(shapes, NaN pattern, the set of triangles, inputs). Steps:

1. `wrench_mesh.make_mesh()`: the 2197-triangle mesh (mesh_nodes.csv, mesh_elements.csv,
   mesh_boundary.csv).
2. Sigma_ij = exp(-|c_i - c_j|^2) at the triangle incentres and Phi = U_r D_r^(1/2) for the
   R_KL = 318 largest eigenvalues of Sigma (all eigenvalues >= 10 eps lambda_max on the platform
   where the shipped files were written) -> kl_basis.npz.
3. M = 1000 Monte Carlo draws xi ~ N(0, I_318) (numpy default_rng(SEED)), x = Phi xi, the
   displacement f(x) and the adjoint gradient df/dxi = Phi^T df/dx -> inputs.csv, outputs.csv,
   gradients.csv.
4. M_REF = 10000 further draws (default_rng(SEED_REF)), the matrix
   H = (1/M_REF) sum df/dxi df/dxi^T, its eigenvalues and first five eigenvectors ->
   reference_10000.csv (the analogue of the paper's 10^4-sample reference estimator, p. 14).

Only numpy, scipy and the standard library are imported. One process.
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
import wrench_mesh  # noqa: E402
import wrench_model as wm  # noqa: E402

SEED = 2020
SEED_REF = 2021
M = 1000
M_REF = 10000
R_KL = 318
N_VEC = 5

INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only

FILES = ('mesh_nodes.csv', 'mesh_elements.csv', 'mesh_boundary.csv', wm.KL_FILE,
         'inputs.csv', 'outputs.csv', 'gradients.csv', 'reference_10000.csv')


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write_kl(out, Phi, lam, c):
    np.savez(os.path.join(out, wm.KL_FILE), Phi=Phi, eigenvalues=lam, incentres=c,
             description=np.array(
                 'Phi (2197 x 318): x = Phi xi maps standard normal xi to the log Young modulus per '
                 'triangle; Phi = U_r D_r^(1/2) from the 318 largest eigenpairs of '
                 'Sigma_ij = exp(-|c_i - c_j|^2), c_i = incentres; each column has its '
                 'largest-magnitude entry positive. eigenvalues: D_r. Written by generate_data.py.'))


def write_samples(out, xi, f, G):
    prov = (f'wrench_model.py on the shipped mesh (2197 P1 triangles, nu = 0.3, unit traction); '
            f'xi from numpy default_rng({SEED}).standard_normal(({len(xi)}, {xi.shape[1]})); written by generate_data.py')
    np.savetxt(os.path.join(out, 'inputs.csv'), xi, fmt='%.17g', delimiter=',',
               header=f'{len(xi)} Monte Carlo samples of the standardized field coordinates xi ~ N(0, I) '
                      f'(log E = Phi xi, Phi in kl_basis.npz). {prov}\n'
                      + ','.join(f'xi_{j + 1} (-)' for j in range(xi.shape[1])))
    np.savetxt(os.path.join(out, 'outputs.csv'), f[:, None], fmt='%.17g', delimiter=',',
               header=f'Vertical displacement u_2 at the point of interest (-4.1, 0). {prov}\n'
                      'u2_PoI (nondimensional length)')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header=f'Adjoint gradients d u2_PoI / d xi_j (= Phi^T d u2_PoI / d log E). {prov}\n'
                      + ','.join(f'du2/dxi_{j + 1} (nondimensional length)' for j in range(G.shape[1])))


def reference(model, seed=SEED_REF, n=M_REF, chunk=500):
    """Eigenvalues (descending) and the first N_VEC eigenvectors of (1/n) sum g g^T."""
    rng = np.random.default_rng(seed)
    r = model.Phi.shape[1]
    H = np.zeros((r, r))
    for i in range(0, n, chunk):
        xi = rng.standard_normal((min(chunk, n - i), r))
        _, G, _ = wm.sample(model, xi)
        H += G.T @ G
    H /= n
    lam, V = np.linalg.eigh(H)
    lam, V = lam[::-1], V[:, ::-1]
    V = V[:, :N_VEC]
    # sign: the field Phi v has its largest-magnitude entry positive
    fields = model.Phi @ V
    idx = np.argmax(np.abs(fields), axis=0)
    V = V * np.sign(fields[idx, np.arange(N_VEC)])
    return lam, V


def write_reference(out, lam, V):
    np.savetxt(os.path.join(out, 'reference_10000.csv'), np.c_[lam, V], fmt='%.17g', delimiter=',',
               header=f'Reference active subspace from {M_REF} Monte Carlo gradients (numpy default_rng({SEED_REF}), '
                      f'independent of inputs.csv): row j holds eigenvalue j of (1/{M_REF}) sum df/dxi df/dxi^T '
                      f'and component j of eigenvectors 1-{N_VEC} (each signed so that the field Phi w has its '
                      'largest-magnitude entry positive). Written by generate_data.py.\n'
                      'eigenvalue (nondimensional length^2),' + ','.join(f'w{k + 1} (-)' for k in range(N_VEC)))


def compare(out, n, with_ref):
    """Print the differences from the shipped files; return the structural failures and the largest difference."""
    bad, worst = [], [-1.0, '']

    def rep(name, val, measure, where):
        print(f'  {name}: {val:.1e} ({measure})')
        if val > worst[0]:
            worst[:] = [val, f'{measure}, {where}']

    def load(f, out_rows=None):
        a, b = read_csv(os.path.join(HERE, f))[:n if out_rows else None], read_csv(os.path.join(out, f))
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
            bad.append(f'FAIL: {f}: shape {b.shape} or NaN pattern differs from the shipped {a.shape}')
            return None, None
        return a, b

    p0, t0, s0 = wm.read_mesh(HERE)
    p1, t1, s1 = wm.read_mesh(out)
    # The triangles are compared as a set: the order in which scipy's Delaunay (Qhull) returns them can
    # differ between platforms (on macOS with scipy 1.17.1, 768 of the 2197 keep their index). The rows of
    # Phi follow the element order, so they are compared after mapping the new order onto the shipped one.
    key = lambda a: [tuple(sorted(r)) for r in a.tolist()]
    k0, k1 = key(t0), key(t1)
    same_set = t0.shape == t1.shape and set(k0) == set(k1) and set(key(s0)) == set(key(s1))
    perm = np.array([dict(zip(k1, range(len(k1))))[k] for k in k0]) if same_set else None
    if same_set:
        print(f'  mesh: same {len(t0)} triangles and boundary segments as the shipped mesh '
              f'({int(np.sum(perm == np.arange(len(perm))))} at the same index)')
    else:
        bad.append('FAIL: mesh: the triangles or boundary segments differ from the shipped mesh')
    if p0.shape == p1.shape:
        rep('mesh nodes, max abs difference', np.abs(p0 - p1).max(), 'absolute', 'mesh_nodes.csv')
    else:
        bad.append(f'FAIL: mesh_nodes.csv: shape {p1.shape} differs from the shipped {p0.shape}')
    z0, z1 = np.load(os.path.join(HERE, wm.KL_FILE)), np.load(os.path.join(out, wm.KL_FILE))
    lam0 = z0['eigenvalues']
    if lam0.shape != z1['eigenvalues'].shape or z0['Phi'].shape != z1['Phi'].shape:
        bad.append(f'FAIL: {wm.KL_FILE}: shapes differ from the shipped file')
        return bad, worst
    rep('eigenvalues of Sigma, max abs difference / lambda_1', np.abs(lam0 - z1['eigenvalues']).max() / lam0[0],
        'relative to lambda_1', f'{wm.KL_FILE} eigenvalues')
    if not same_set:
        return bad, worst
    Phi0, Phi1 = z0['Phi'], z1['Phi'][perm]
    lead = lam0 >= 1e-8 * lam0[0]
    dphi = np.linalg.norm(Phi0[:, lead] - Phi1[:, lead], axis=0) / np.linalg.norm(Phi0[:, lead], axis=0)
    rep(f'Phi columns 1-{lead.sum()} (eigenvalue >= 1e-8 lambda_1, rows in the shipped element order), '
        'max relative difference', dphi.max(), 'relative, column 2-norm', f'{wm.KL_FILE} Phi')
    dtrail = np.linalg.norm(Phi0[:, ~lead] - Phi1[:, ~lead]) / np.linalg.norm(Phi0)
    print(f'  Phi columns {lead.sum() + 1}-{R_KL} (eigenvalues below 1e-8 lambda_1): difference / ||Phi||_F = {dtrail:.1e} (not compared)')
    xi0, xi1 = load('inputs.csv', True)
    if xi0 is not None:
        d = np.max(np.abs(xi0 - xi1) / np.abs(xi0))
        rep('inputs, max relative difference', d, 'relative', 'inputs.csv')
        if d > INPUT_RTOL:
            bad.append(f'FAIL: inputs.csv differs from the shipped inputs by {d:.1e} > {INPUT_RTOL:g}')
    f0, f1 = load('outputs.csv', True)
    if f0 is not None:
        rep('outputs, max relative difference', np.max(np.abs(f0 - f1) / np.abs(f0)), 'relative', 'outputs.csv')
    G0, G1 = load('gradients.csv', True)
    if G0 is not None:
        rep('gradients, max relative row difference',
            np.max(np.linalg.norm(G0 - G1, axis=1) / np.linalg.norm(G0, axis=1)), 'relative, row 2-norm', 'gradients.csv')
    if with_ref:
        R0, R1 = load('reference_10000.csv')
        if R0 is not None:
            rep('reference eigenvalues 1-50, max relative difference',
                np.max(np.abs(R0[:50, 0] - R1[:50, 0]) / R0[:50, 0]), 'relative', 'reference_10000.csv eigenvalues')
            rep(f'reference eigenvectors 1-{N_VEC}, max 2-norm difference',
                np.linalg.norm(R0[:, 1:] - R1[:, 1:], axis=0).max(), '2-norm', 'reference_10000.csv eigenvectors')
    return bad, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (required without --check; use . only to rewrite the shipped files; '
                         'with --check, a temporary directory by default)')
    ap.add_argument('--check', action='store_true', help='compare the files written to DIR with the files here')
    ap.add_argument('--n', type=int, default=None, help=f'number of samples (default {M}); skips the reference run')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            ap.error('--out DIR is required without --check')
    out = os.path.abspath(args.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here and \
            (args.check or args.n is not None or os.path.realpath(out) != here):
        ap.error('--out must be outside this data set\'s directory (not a folder in it; the directory itself only '
                 'without --check and --n, to rewrite the shipped files)')
    os.makedirs(out, exist_ok=True)
    n = M if args.n is None else args.n
    t_start = time.perf_counter()

    p, t, segs, poi = wrench_mesh.make_mesh()
    wrench_mesh.write_mesh(out, p, t, segs, poi)
    q = wrench_mesh.quality(p, t)
    t1 = time.perf_counter()
    print(f'mesh: {len(p)} nodes, {len(t)} triangles, angles {q[0]:.1f}-{q[1]:.1f} deg, area {q[4]:.6f} ({t1 - t_start:.0f} s)')

    Sigma = wm.covariance(p, t)
    Phi, lam = wm.kl_basis(Sigma)
    if abs(Phi.shape[1] - R_KL) > 5:
        raise RuntimeError(f'the 10 eps rule keeps {Phi.shape[1]} eigenvalues here, far from R_KL = {R_KL}')
    print(f'Sigma: the 10 eps rule keeps {Phi.shape[1]} eigenvalues here; R_KL = {R_KL} kept')
    lam_all, U = np.linalg.eigh(Sigma)
    order = np.argsort(lam_all)[::-1][:R_KL]
    lam, U = lam_all[order], U[:, order]
    idx = np.argmax(np.abs(U), axis=0)
    U = U * np.sign(U[idx, np.arange(R_KL)])
    Phi = U * np.sqrt(lam)
    write_kl(out, Phi, lam, wm.incentres(p, t))
    # largest dropped eigenvalue: the truncation error ||Sigma - Phi Phi^T||_2 in exact arithmetic; in floating point
    # both are at the rounding level of Sigma
    dropped = np.abs(np.delete(lam_all, order)).max() / lam[0]
    lam_next = np.sort(lam_all)[::-1][R_KL] / lam[0]
    t2 = time.perf_counter()
    print(f'  lambda_1(Sigma) = {lam[0]:.6f}, lambda_{R_KL}/lambda_1 = {lam[-1] / lam[0]:.2e}, '
          f'lambda_{R_KL + 1}/lambda_1 = {lam_next:.2e} (cut-off 10 eps = {wm.TRUNC:.2e}); '
          f'largest dropped eigenvalue / lambda_1 = {dropped:.1e} ({t2 - t1:.0f} s)')

    model = wm.Wrench(out)
    rng = np.random.default_rng(SEED)
    xi = rng.standard_normal((M, R_KL))[:n]
    f, G, _ = wm.sample(model, xi)
    write_samples(out, xi, f, G)
    t3 = time.perf_counter()
    print(f'samples: M = {n}, {t3 - t2:.1f} s ({(t3 - t2) / n * 1e3:.1f} ms per solve + adjoint gradient); '
          f'f in [{f.min():.2f}, {f.max():.2f}]')

    with_ref = args.n is None
    if with_ref:
        lam_ref, V = reference(model)
        write_reference(out, lam_ref, V)
        t4 = time.perf_counter()
        print(f'reference: M_REF = {M_REF}, {t4 - t3:.0f} s; eigenvalues 1-5: '
              + ', '.join(f'{v:.4g}' for v in lam_ref[:5]))
    print(f'total {time.perf_counter() - t_start:.0f} s')
    if args.check:
        print('comparison with the shipped files:')
        bad, worst = compare(out, n, with_ref)
        print(f'largest difference from the shipped files: {worst[0]:.1e} ({worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)


if __name__ == '__main__':
    main()
