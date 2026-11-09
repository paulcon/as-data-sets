"""Write the mesh, the field basis, the samples and the reference eigenpairs of the wrench model.

    python generate_data.py --out DIR                 # writes every data file to DIR
    python generate_data.py --out DIR --check         # writes to DIR and compares with the files here
    python generate_data.py --out DIR --n 100 --check # a quick partial run (first 100 samples, no reference)

DIR must not be a folder inside this directory, and may be this directory itself only without
--check and --n; the shipped files were written with `--out .` from this directory. Steps:

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
import os
import sys
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

# --check tolerances (see DATA.md, "Regenerating")
TOL_NODES = 1e-12      # absolute, node coordinates
TOL_PHI = 1e-6         # relative (column 2-norm), basis columns with eigenvalue >= 1e-8 lambda_1
TOL_EIG_SIGMA = 1e-10  # relative to lambda_1(Sigma), eigenvalues of Sigma
TOL_XI = 1e-15         # absolute, inputs (they come from the generator, not from a computation)
TOL_F = 1e-6           # relative, outputs
TOL_G = 1e-6           # relative (row 2-norm), gradients
TOL_REF_EIG = 1e-6     # relative, the reference eigenvalues 1-50
TOL_REF_VEC = 1e-6     # 2-norm, the reference eigenvectors 1-5 (signs fixed)

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
    """Print the differences from the shipped files and return True if within the tolerances."""
    ok = True

    def rep(name, val, tol, note=''):
        nonlocal ok
        good = val <= tol
        ok &= good
        print(f'  {name}: {val:.1e} (tolerance {tol:g}) {"ok" if good else "EXCEEDED"} {note}')

    p0, t0, s0 = wm.read_mesh(HERE)
    p1, t1, s1 = wm.read_mesh(out)
    same_topology = t0.shape == t1.shape and np.array_equal(t0, t1) and np.array_equal(s0, s1)
    print(f'  mesh connectivity identical: {same_topology}')
    ok &= same_topology
    if p0.shape == p1.shape:
        rep('mesh nodes, max abs difference', np.abs(p0 - p1).max(), TOL_NODES)
    z0, z1 = np.load(os.path.join(HERE, wm.KL_FILE)), np.load(os.path.join(out, wm.KL_FILE))
    lam0 = z0['eigenvalues']
    rep('eigenvalues of Sigma, max abs difference / lambda_1', np.abs(lam0 - z1['eigenvalues']).max() / lam0[0], TOL_EIG_SIGMA)
    lead = lam0 >= 1e-8 * lam0[0]
    dphi = np.linalg.norm(z0['Phi'][:, lead] - z1['Phi'][:, lead], axis=0) / np.linalg.norm(z0['Phi'][:, lead], axis=0)
    rep(f'Phi columns 1-{lead.sum()} (eigenvalue >= 1e-8 lambda_1), max relative difference', dphi.max(), TOL_PHI)
    dtrail = np.linalg.norm(z0['Phi'][:, ~lead] - z1['Phi'][:, ~lead]) / np.linalg.norm(z0['Phi'])
    print(f'  Phi columns {lead.sum() + 1}-{R_KL} (eigenvalues below 1e-8 lambda_1): difference / ||Phi||_F = {dtrail:.1e} (not checked)')
    xi0, xi1 = read_csv(os.path.join(HERE, 'inputs.csv'))[:n], read_csv(os.path.join(out, 'inputs.csv'))
    rep('inputs, max abs difference', np.abs(xi0 - xi1).max(), TOL_XI)
    f0, f1 = read_csv(os.path.join(HERE, 'outputs.csv'))[:n, 0], read_csv(os.path.join(out, 'outputs.csv'))[:, 0]
    rep('outputs, max relative difference', np.max(np.abs(f0 - f1) / np.abs(f0)), TOL_F)
    G0, G1 = read_csv(os.path.join(HERE, 'gradients.csv'))[:n], read_csv(os.path.join(out, 'gradients.csv'))
    rep('gradients, max relative row difference', np.max(np.linalg.norm(G0 - G1, axis=1) / np.linalg.norm(G0, axis=1)), TOL_G)
    if with_ref:
        R0, R1 = read_csv(os.path.join(HERE, 'reference_10000.csv')), read_csv(os.path.join(out, 'reference_10000.csv'))
        rep('reference eigenvalues 1-50, max relative difference', np.max(np.abs(R0[:50, 0] - R1[:50, 0]) / R0[:50, 0]), TOL_REF_EIG)
        rep(f'reference eigenvectors 1-{N_VEC}, max 2-norm difference', np.linalg.norm(R0[:, 1:] - R1[:, 1:], axis=0).max(), TOL_REF_VEC)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory (use . only to rewrite the shipped files)')
    ap.add_argument('--check', action='store_true', help='compare the files written to DIR with the files here')
    ap.add_argument('--n', type=int, default=None, help=f'number of samples (default {M}); skips the reference run')
    args = ap.parse_args()
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
    if args.check:
        print('comparison with the shipped files:')
        ok = compare(out, n, with_ref)
        print('all within tolerance' if ok else 'SOME DIFFERENCES EXCEED THE TOLERANCES')
    print(f'total {time.perf_counter() - t_start:.0f} s')


if __name__ == '__main__':
    main()
