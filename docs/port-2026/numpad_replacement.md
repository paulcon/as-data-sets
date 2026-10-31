# numpad replacement for the elliptic PDE models (Elliptic_PDE, MCMC)

Status: prototype written and validated (2026-10-29). numpy + scipy only. Python 3.11, numpy 2.4.6, scipy 1.17.1, Intel i7-1068NG7 laptop.
Prototype: `prototypes/numpad/pde_model.py` (verbatim copy in Appendix A). Validation scripts sit next to it: `validate.py`, `compare_nb.py`, `perm.py`. Outputs: `validate_out.txt`, `compare_out.txt`.

## 1. Summary

- The legacy model (`Elliptic_PDE/pde_model.py`, `MCMC/pde_model.py`) is linear in u. numpad's `solve` is Newton with `abs_tol = rel_tol = 1e-6` and `max_iter = 10`, starting from u0 = 0. One Newton step does a direct `spsolve` of the exact Jacobian, so the converged numpad u equals the direct sparse solve up to roundoff. `q.diff(x)` uses the adjoint mode because size(x) = 100 > size(q) = 1.
- Replacement: assemble the SPD 5-point matrix A(a) with a fixed CSC pattern, factor it once with `splu`, solve A u = 1, then reuse the factor for one adjoint solve. The gradient is formed from face-wise products, with no loop over the 100 parameters.
- Validation: the legacy residual is reproduced to 3e-12 to 9e-12 absolute (about 1e-15 relative to its terms). The gradient matches central finite differences to 1e-7 or better. The notebook figures are reproduced to plotting resolution (QoI training values, first eigenvectors, Gradient_Samples spectrum).
- Cost: about 16 to 21 ms per `qoi`, `grad_qoi`, `misfit` or `misfit_grad` call. The gradient costs the same as one QoI. The MCMC notebook's numpad timing ("about 10 minutes per 1000 forward model evaluations", cell 18) is about 0.6 s per call, so the replacement is about 30x faster.
- Notebook text vs code: the PDE and BC description is correct in substance. Section 5 lists four discretization-level discrepancies, plus a correlation-length factor of 2 implied by the stored KL data.

## 2. Discretization derived from the legacy `residual`

Grid: `nx = sqrt(B.shape[0]) = 100` cells per side, `dx = 1/nx`. The coefficient `a = exp(B x)` is cell-centred, reshaped C-order to `(nx, nx)`. Axis 0 is the row index i (s2, plotted bottom to top). Axis 1 is the column index j (s1, plotted left to right). This matches MCMC cell 26, which plots `contourf(x, y, a)` with `meshgrid(1..100, 1..100)`.

u lives on the (nx+1) x (nx+1) cell vertices. The unknowns are the interior vertices `U[1:nx, 1:nx]`, a (nx-1) x (nx-1) = 99 x 99 = 9801 array. The padding in `residual` builds `U`:

| padding | vertices | meaning |
|---|---|---|
| `hstack([zeros, u, ...])` | `U[:, 0] = 0` (s1 = 0) | homogeneous Dirichlet, **left** |
| `hstack([..., u, u[:, -1:]])` | `U[:, nx] = U[:, nx-1]` (ghost) | zero flux across the last column of faces: homogeneous Neumann, **right** |
| `vstack([zeros, ..., zeros])` | `U[0, :] = U[nx, :] = 0` | homogeneous Dirichlet, **bottom and top** |

Fluxes: face coefficients are arithmetic means of the two cells that share the face.
- Horizontal flux (axis 1) between `U[i,j]` and `U[i,j+1]`, for i = 1..nx-1 and j = 0..nx-1: `a_hor[i-1,j] = (a[i-1,j] + a[i,j])/2`.
- Vertical flux (axis 0) between `U[i,j]` and `U[i+1,j]`, for i = 0..nx-1 and j = 1..nx-1: `a_ver[i,j-1] = (a[i,j-1] + a[i,j])/2`.
- Residual: `res = div_h(a grad_h u) + f` with `f = 1`. Setting res = 0 is the discrete form of `-div(a grad u) = 1`.

As a matrix, `res = -A u + 1`. A = sum over faces e of `(a_e/dx^2)(e_p - e_q)(e_p - e_q)^T`, with Dirichlet rows and columns dropped and the 99 Neumann faces (j = nx-1) omitted because their flux is identically 0. A is symmetric positive definite with 48,609 nonzeros. It is the standard 5-point vertex-centred finite-volume (equivalently, Q1-lumped) operator.

Where the right boundary sits: the last unknown column is at s1 = 1 - dx = 0.99. The zero-flux face lies between that column and the ghost column at s1 = 1. Its control volume has the full width dx, so the scheme behaves like a cell-centred Neumann face at s1 = 1 - dx/2. This is first-order accurate at the boundary. The last column of cells, `a[:, nx-1]`, enters only through the vertical-face averages `a_ver[:, nx-2]`.

QoIs:
- **Elliptic_PDE**: `q = 0.5*dx*sum(u_bnd[1:] + u_bnd[:-1])` with `u_bnd = u[:, -1]` (s1 = 0.99, s2 = 0.01..0.99). This is a trapezoid rule over s2 in [dx, 1-dx]. Its weights are dx on interior nodes and dx/2 on the two end nodes. It does not divide by |Gamma_2|, but |Gamma_2| = 1. A full trapezoid including the u = 0 corners would weight every node by dx, so the legacy rule underweights the two end nodes by dx/2 each. The effect is tiny, but it is what the code computes, and the replacement keeps it.
- **MCMC**: `obs = u[indz, -1]` with `indz = [20,30,...,80]` (rows of the unknown array). These are vertex rows 21..81, so s2 = 0.21, 0.31, ..., 0.81, all at s1 = 0.99. `misfit = 0.5*||obs - d||^2/sig2`.

## 3. Adjoint gradient

A(a) u = 1, and q = c^T u. The vector c is the trapezoid weights for Elliptic_PDE, and is `(obs - d)/sig2` at the 7 sensor nodes for the misfit. The misfit is quadratic in u, so this c is dq/du.

- Adjoint solve: A^T lam = c. A is symmetric, so this reuses the same `splu` factor.
- dq/da_e = -lam^T (dA/da_e) u = -(u_q - u_p)(lam_q - lam_p)/dx^2 for each face e = (p, q). Dirichlet vertices are taken as 0. Neumann faces contribute 0.
- dq/da (cells): each face derivative is scattered with weight 1/2 to the two cells it averages.
- dq/dx = B^T (a ⊙ dq/da), because da/dx = diag(a) B.

Cost: one LU factorization plus two triangular solves, plus O(N) face work and one 10000x100 mat-vec. Assembly uses a precomputed COO-to-CSC slot map and `np.bincount`, about 1 ms. The `MMD_AT_PLUS_A` column ordering roughly halves fill (3.6e5 vs 6.7e5 nonzeros in L+U) and is about 20% faster than the default COLAMD. Unpreconditioned CG is about 4x slower, and `spsolve` is about 1.5x slower.

## 4. Validation

All runs use `.venv/bin/python` on the real `kl-*.npz` files with seeded `numpy.random.default_rng`.

| check | short (beta = .01) | long (beta = 1) |
|---|---|---|
| (a) max abs legacy-`residual(u, a)` at the new u, 5 random x (terms are O(1e3 to 1e4)) | 3.1e-12 to 4.6e-12 | 2.8e-12 to 9.3e-12 |
| (b) `grad_qoi`, central FD along 5 random unit directions (h = 1e-4), rel. err | 1.3e-10 to 2.3e-6 (median 8e-9) | 4.0e-10 to 4.8e-9 |
| (b) `grad_qoi`, full 100-coordinate central FD (h = 1e-5), ‖g - g_FD‖/‖g‖ | 7.9e-8 | 6.7e-9 |
| (b) `misfit_grad` (synthetic data, sig2 = 1e-4‖obs‖²), 5 random directions (h = 1e-5) | 1.5e-8 to 1.4e-6 | 6.0e-10 to 1.7e-7 |
| q(0) (a ≡ 1, so the same for both cases) | 0.07608 | 0.07608 |

The largest FD errors in the short case are FD truncation and roundoff, because ‖g‖ is about 5e-3 there.

(c) Comparison with stored notebook outputs. The notebooks store no numeric QoI or gradient for known inputs; only figures and `Gradient_Samples.mat`. The checks below were read off the extracted PNGs (`scratchpad/numpad/figs/`):

| stored output | legacy (read from figure) | replacement |
|---|---|---|
| PDE cell 11, "SENS beta = 1" training values q(t e_0), t = -3, -1.5, 0, 1.5, 3. Index 0 is argmax\|w1\| in both runs. | ≈ 0.005, 0.02, 0.075, 0.26, 0.895 | 0.0064, 0.0221, 0.0761, 0.2614, 0.8971 |
| PDE cell 11, "SENS beta = .01" q(t e_0) | ≈ 0.0812, 0.0785, 0.076, 0.0735, 0.0712 | 0.0812, 0.0786, 0.0761, 0.0736, 0.0712 |
| PDE cell 11, "ASM beta = .01" q(t w1) (w1 from 300 samples) | ≈ 0.0916, 0.0835, 0.076, 0.0692, 0.063 | 0.0916, 0.0835, 0.0761, 0.0692, 0.0630 |
| PDE cell 11, QoI range of 300 random test points (different random draws, so this is a sanity check only) | beta = 1: about 0.005..1.07; beta = .01: about 0.063..0.090 | 0.0071..0.7296; 0.0646..0.0905 |
| PDE cell 7, first eigenvector, beta = 1, components 0..4 | ≈ 0.97, 0.18, -0.14, 0.01, -0.11 | 0.966, 0.180, -0.137, 0.011, -0.116 |
| PDE cell 7, first eigenvector, beta = .01, components 0, 4, 5, 6 | ≈ 0.35, -0.25, 0.32, -0.32 | 0.356, -0.254, 0.320, -0.318 |
| MCMC `Gradient_Samples.mat` (1000x100, unknown x_true and data). C-hat eigenvalues 1..4 | 1093, 222, 24.6, 7.8; mean‖g‖ = 32.2 | two synthetic data sets: 1261, 129, 26, 9.9 (mean‖g‖ 31.9) and 2633, 75, 46, 19 (48.5). \|cos(w1, w1_stored)\| = 0.997 and 0.981 |

The `Gradient_Samples.mat` row cannot be an exact match, because the data vector d used to generate it was not saved. Its agreement is at the level of variation between data sets.

Not run: a direct numpad-vs-replacement comparison. numpad was cloned to scratch (commit 81cf268, 2015-08-13). It needs `np.set_numeric_ops`, which numpy 2 removed, so a separate scratch venv with numpy 1.24 / scipy 1.11 was created. Executing the downloaded numpad code was then denied by the session's permission policy. If the author allows it, run `scratchpad/numpad/npvenv/bin/python` with `ext/numpad` on `sys.path` and compare `qoi` and `grad_qoi` at a few seeded x. The expected agreement is about 1e-12 relative.

(d) Timing, mean of 50 calls, single thread:

| call | short | long |
|---|---|---|
| `qoi` | 20.8 ms | 16.4 ms |
| `grad_qoi` | 19.5 ms | 20.2 ms |
| `misfit` | 18.8 ms | 18.4 ms |
| `misfit_grad` | 19.3 ms | 19.4 ms |
| assembly only (exp(Bx) + A) | 1.0 ms | 1.2 ms |

Estimated notebook cost:
- **Elliptic_PDE**: about 4,300 solves, about 1.5 min.
- **MCMC live cells 20, 23, 28**: about 5,000 solves, about 1.6 min.
- **Regenerating the stored 1e5-step vanilla chain** (`generate_data.py`): about 2e5 solves, about 1 h. It calls `misfit(y0)` again every step; caching the current misfit halves that.

## 5. Discrepancies with the notebooks' description of the PDE

| where | notebook says | code / data actually does | confidence |
|---|---|---|---|
| PDE cell 0 | "Homogeneous Dirichlet ... left, top, and bottom ... homogeneous Neumann ... right" | Same. Neumann is imposed by a ghost copy (first order); effective zero-flux face at s1 = 1 - dx/2. | confirmed |
| PDE cell 0 | "f(x) = c^T M u ... M is the symmetric mass matrix ... K u = f" (finite-element language) | 5-point vertex-centred finite volume. QoI is a trapezoid rule on the vertex column at s1 = 0.99 (one dx inside the boundary), over s2 in [0.01, 0.99]. The two end nodes get dx/2 weight, so it is not the full ∫ over [0, 1]. No mass matrix. | confirmed |
| PDE cell 0 | Adjoint formula df/dx_i = -y^T (dK/dx_i) u | Legacy code used numpad AD. The replacement implements exactly the stated adjoint, so the text becomes accurate. | confirmed |
| PDE cell 0 | C(s,t) = exp(-β^{-1}‖s-t‖_1), β = 1 and .01 | The stored `lam` equal the top 100 eigenvalues of the Kronecker product of 1-D matrices exp(-\|s-t\|/ℓ) on a grid with spacing 0.01, with ℓ = **0.02** (max rel. diff 1.3e-6) and ℓ = **2.0** (1.6e-8). That is exp(-‖s-t‖_1/(2β)) in the PDE's [0,1]^2 coordinates. Constantine, Dow & Wang 2014 likely write 2β; check the paper. | data: confirmed; paper: unverified |
| PDE cell 0 | x_i ~ N(0,1), "truncated KL" | K has orthonormal columns; `lam` are discrete covariance eigenvalues (not scaled by cell area). The 100 retained modes carry 13.3% (short) and 99.2% (long) of the field variance. Pointwise std of log a is 0.10..0.42 (short) vs about 1.0 (long). Worth one sentence in the notebook, because it explains why the short case is nearly linear. | confirmed |
| MCMC cell 18 | "observations of u on a few points of the nontrivial boundary" | 7 points at s2 = 0.21, 0.31, ..., 0.81 (`indz` indexes unknown rows, so they are offset by one dx), at s1 = 0.99. | confirmed |
| MCMC cell 18 | "about 10 minutes ... per 1000 forward model evaluations" | about 20 s per 1000 with the replacement. Update the sentence. | confirmed |

## 6. Interface

The prototype keeps the legacy names and signatures:
- **`PDESolver(corrcase='short', datadir=None)`**: attributes `.m`, `.nx`, `.dx`, `.B`, `.f`, and methods `qoi(x)` and `grad_qoi(x)` (Elliptic_PDE).
- **`PDEMisfit(corrcase='short', datadir=None)`**: attribute `.indz`, and methods `observations(x)`, `set_data(d, sig2)`, `misfit(x)` and `misfit_grad(x)` (MCMC).
- **Shared base class**: also exposes `coeff(x)`, `stiffness(a)`, `solve(x)` and the numpy `residual(u, a)` used as a test oracle.

Return types changed:
- **Gradients**: legacy returned `np.matrix` of shape (1, m) from `.todense()`. The prototype returns a 1-D `ndarray` of shape (m,).
- **QoI and misfit**: legacy returned a numpad value; the prototype returns a Python float.

x may be (m,) or (m, 1). `set_data` reshapes d to (7,). Legacy broadcast silently if d was (7, 1).

`datadir` defaults to the module's own directory, so `np.load` no longer depends on the notebook's working directory.

## 7. Recommendation: one module for both case studies

1. **Single implementation, two QoI classes** (as in the prototype): `_EllipticPDE` holds the discretization, solve and adjoint; `PDESolver` holds the boundary average; `PDEMisfit` holds observations and misfit. The MCMC notebook, `pde_mcmc_study.py` and `generate_data.py` change `pdem.PDESolver()` to `pdem.PDEMisfit()`. Alternatively, keep `PDESolver = PDEMisfit` as an alias in MCMC to avoid touching the call sites; the rename is preferred for clarity.
2. **Where the file lives.** The repo convention is self-contained case-study directories that run from their own directory. Choose one of:
   - (a) **Recommended for this repo:** a byte-identical `pde_model.py` in both `Elliptic_PDE/` and `MCMC/`, plus a CI or acceptance check that the two copies are identical (`cmp`). The file is about 170 lines; duplication is cheap and each notebook stays standalone.
   - (b) One copy in `Elliptic_PDE/`, with the MCMC notebook doing `sys.path.insert(0, '../Elliptic_PDE')`. This saves the duplication but couples the directories.
3. **kl-short.npz duplicate.** `MCMC/kl-short.npz` is byte-identical to `Elliptic_PDE/kl-short.npz` (md5 079370cf..., 8.0 MB each). Options:
   - **Keep both** (16 MB total, far below GitHub limits). This is consistent with 2(a). Recommended unless the author prefers 2(b).
   - **Load from `../Elliptic_PDE`** via `datadir=`. Saves 8 MB.
   - **Do NOT regenerate the basis on the fly.** The separable structure would allow it in milliseconds: K is spanned exactly by Kronecker products of 1-D eigenvectors, and B Bᵀ is reproduced to 1e-13, so the random field's law is identical. But 46 (short) and 48 (long) of the 100 eigenvalues come in degenerate pairs. A regenerated basis differs from the stored one by 2x2 rotations within those pairs, which changes the meaning of each x_i. It would also make the stored `Vanilla_Chain.mat`, `AV_Chain.mat` and `Gradient_Samples.mat`, and the eigenvector-vs-index figures, inconsistent with the model. Keep the stored K as the definition of the coordinates.
   - Storing K as float32 would halve the size but perturb results at about 1e-7. Not worth it.
4. **Optional, small:** add `qoi_and_grad(x)`, one factorization for both. It would halve Elliptic_PDE cell 3, which computes values and gradients at the same 300 points separately. v1's gradient estimators accept separate value and gradient arrays either way.

Alternatives rejected:
- **JAX / autograd / PyTorch**: each would give AD "for free", but adds a heavy dependency, violating the numpy/scipy/matplotlib/active_subspaces-only target. Sparse direct solves inside them also need custom VJP rules anyway.
- **Keep numpad (vendored)**: abandoned since 2015, needs `np.set_numeric_ops`, which numpy 2 removed. Rejected.

## 8. Acceptance criteria for the port

- `max|residual(u, a)| < 1e-10` for 5 seeded x in both corrcases.
- Full-coordinate central-FD check of `grad_qoi` and `misfit_grad`: relative error < 1e-6 at a seeded x.
- `PDESolver(c).qoi(np.zeros(100))` = 0.07608 (5 significant digits) for both cases.
- Long case, `q(t e_0)` for t = (-3, -1.5, 0, 1.5, 3) = (0.0064, 0.0221, 0.0761, 0.2614, 0.8971) ±1e-4.
- At most 30 ms per call on the reference laptop.
- No `numpad` import anywhere (`grep -r numpad --include=*.py --include=*.ipynb` only hits prose, if any).

## Appendix A: prototype `pde_model.py` (verbatim)

```python
"""Elliptic PDE model -div(a grad u) = 1 on [0,1]^2, numpy/scipy only.

Drop-in replacement for the numpad-based pde_model.py files in
Elliptic_PDE/ and MCMC/. Reproduces the legacy finite-volume residual
exactly (same grid, same face averaging, same boundary padding) and
computes gradients with a hand-coded discrete adjoint.

Grid: nx x nx cells (nx = 100), coefficient a is cell-centred, u lives on
the (nx+1) x (nx+1) vertices; unknowns are the (nx-1) x (nx-1) interior
vertices. Array axis 0 = "row" i (s2, bottom..top), axis 1 = column j
(s1, left..right).
  u[0, :] = u[nx, :] = 0   (Dirichlet, bottom and top)
  u[:, 0] = 0              (Dirichlet, left)
  u[:, nx] = u[:, nx-1]    (ghost copy -> zero flux on the right faces)
"""
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _EllipticPDE:
    """Shared discretization, solve, and adjoint machinery."""

    def __init__(self, corrcase='short', datadir=None):
        if datadir is None:
            datadir = os.path.dirname(os.path.abspath(__file__))
        kl = np.load(os.path.join(datadir, 'kl-{}.npz'.format(corrcase)))
        K, lam = kl['K'], kl['lam']
        self.B = K * np.sqrt(lam.reshape((1, K.shape[1])))
        self.f = 1.0
        self.nx = int(np.sqrt(self.B.shape[0]))
        self.dx = 1.0 / self.nx
        self.m = K.shape[1]
        self._build_index()

    # ------------------------------------------------------------------
    # static index structure of the stiffness matrix
    # ------------------------------------------------------------------
    def _build_index(self):
        nx, n = self.nx, self.nx - 1
        # node id on the padded (nx+1)x(nx+1) vertex grid; -1 = not an unknown
        nid = -np.ones((nx + 1, nx + 1), dtype=np.int64)
        nid[1:nx, 1:nx] = np.arange(n * n).reshape(n, n)
        # horizontal faces (flux along axis 1) between vertices (i,j),(i,j+1),
        # i = 1..nx-1, j = 0..nx-2 (the j = nx-1 face is the zero-flux one)
        hp, hq = nid[1:nx, 0:nx - 1], nid[1:nx, 1:nx]
        # vertical faces (flux along axis 0) between vertices (i,j),(i+1,j),
        # i = 0..nx-1, j = 1..nx-1
        vp, vq = nid[0:nx, 1:nx], nid[1:nx + 1, 1:nx]
        p = np.concatenate([hp.ravel(), vp.ravel()])
        q = np.concatenate([hq.ravel(), vq.ravel()])
        self._nh = hp.size
        self._p, self._q = p, q
        # COO pattern: +k at (p,p),(q,q); -k at (p,q),(q,p), Dirichlet nodes dropped
        rows, cols, sgn, face = [], [], [], []
        fidx = np.arange(p.size)
        for r, c, s in ((p, p, 1.0), (q, q, 1.0), (p, q, -1.0), (q, p, -1.0)):
            keep = (r >= 0) & (c >= 0)
            rows.append(r[keep]); cols.append(c[keep])
            sgn.append(np.full(keep.sum(), s)); face.append(fidx[keep])
        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._sgn = np.concatenate(sgn)
        self._face = np.concatenate(face)
        self.N = n * n
        # fixed CSC pattern: map every COO entry to its slot in the CSC data array
        key = self._cols * self.N + self._rows
        ukey, self._slot = np.unique(key, return_inverse=True)
        self._csc_indices = (ukey % self.N).astype(np.int32)
        self._csc_indptr = np.searchsorted(ukey // self.N, np.arange(self.N + 1)).astype(np.int32)

    # ------------------------------------------------------------------
    def coeff(self, x):
        """Cell coefficient a = exp(B x), shape (nx, nx)."""
        x = np.asarray(x, dtype=float).reshape(self.m)
        return np.exp(self.B @ x).reshape(self.nx, self.nx)

    def _face_coeff(self, a):
        nx = self.nx
        a_hor = 0.5 * (a[1:, :] + a[:-1, :])        # (nx-1, nx)
        a_ver = 0.5 * (a[:, 1:] + a[:, :-1])        # (nx, nx-1)
        return np.concatenate([a_hor[:, :nx - 1].ravel(), a_ver.ravel()])

    def stiffness(self, a):
        """SPD matrix A with A u = 1 equivalent to legacy residual(u) = 0."""
        k = self._face_coeff(a) / self.dx**2
        data = np.bincount(self._slot, weights=self._sgn * k[self._face],
                           minlength=self._csc_indices.size)
        return sp.csc_matrix((data, self._csc_indices, self._csc_indptr),
                             shape=(self.N, self.N))

    def solve(self, x):
        """Return (u as (nx-1,nx-1) array, a, splu factorization)."""
        a = self.coeff(x)
        lu = spla.splu(self.stiffness(a), permc_spec='MMD_AT_PLUS_A')
        u = lu.solve(np.full(self.N, self.f))
        return u.reshape(self.nx - 1, self.nx - 1), a, lu

    def _adjoint_grad(self, u, a, lu, c):
        """d(c^T u)/dx for c given on the unknowns, via one adjoint solve.

        A(a) u = f, q = c^T u  =>  dq/da_face = -lam^T (dA/da_face) u
                                             = -(du_face)(dlam_face)/dx^2,
        then chain rule through face averaging, a = exp(Bx): dq/dx = B^T (a * dq/da).
        """
        lam = lu.solve(c.ravel())       # A is symmetric: A^T = A
        uu = np.concatenate([u.ravel(), [0.0]])   # index -1 -> Dirichlet zero
        ll = np.concatenate([lam, [0.0]])
        p, q = self._p, self._q
        g_face = -(uu[q] - uu[p]) * (ll[q] - ll[p]) / self.dx**2
        nx = self.nx
        g_hor = np.zeros((nx - 1, nx))
        g_hor[:, :nx - 1] = g_face[:self._nh].reshape(nx - 1, nx - 1)
        g_ver = g_face[self._nh:].reshape(nx, nx - 1)
        g_a = np.zeros((nx, nx))
        g_a[1:, :] += 0.5 * g_hor
        g_a[:-1, :] += 0.5 * g_hor
        g_a[:, 1:] += 0.5 * g_ver
        g_a[:, :-1] += 0.5 * g_ver
        return self.B.T @ (a * g_a).ravel()

    def residual(self, u, a):
        """Legacy numpad residual, plain numpy (for testing)."""
        nx, dx = self.nx, self.dx
        u = np.hstack([np.zeros([nx - 1, 1]), u, u[:, -1:]])
        u = np.vstack([np.zeros([1, nx + 1]), u, np.zeros([1, nx + 1])])
        a_hor = 0.5 * (a[1:, :] + a[:-1, :])
        a_ver = 0.5 * (a[:, 1:] + a[:, :-1])
        a_dudy = a_hor * (u[1:-1, 1:] - u[1:-1, :-1]) / dx
        a_dudx = a_ver * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
        return (a_dudy[:, 1:] - a_dudy[:, :-1]) / dx \
            + (a_dudx[1:, :] - a_dudx[:-1, :]) / dx + self.f


class PDESolver(_EllipticPDE):
    """Elliptic_PDE QoI: trapezoid average of u on the last unknown column."""

    def _qoi_weights(self):
        n = self.nx - 1
        w = np.zeros((n, n))
        w[:, -1] = self.dx
        w[0, -1] = w[-1, -1] = 0.5 * self.dx
        return w

    def qoi(self, x):
        u, _, _ = self.solve(x)
        u_bnd = u[:, -1]
        return 0.5 * self.dx * np.sum(u_bnd[1:] + u_bnd[:-1])

    def grad_qoi(self, x):
        u, a, lu = self.solve(x)
        return self._adjoint_grad(u, a, lu, self._qoi_weights())


class PDEMisfit(_EllipticPDE):
    """MCMC model: sensor observations u[indz, -1] and Gaussian misfit."""

    def __init__(self, corrcase='short', datadir=None):
        super().__init__(corrcase, datadir)
        self.indz = [20, 30, 40, 50, 60, 70, 80]

    def observations(self, x):
        u, _, _ = self.solve(x)
        return u[self.indz, -1]

    def set_data(self, d, sig2):
        self.d = np.asarray(d, dtype=float).reshape(len(self.indz))
        self.sig2 = sig2

    def misfit(self, x):
        obs = self.observations(x)
        r = obs - self.d
        return 0.5 * np.dot(r, r) / self.sig2

    def misfit_grad(self, x):
        u, a, lu = self.solve(x)
        c = np.zeros_like(u)
        c[self.indz, -1] = (u[self.indz, -1] - self.d) / self.sig2
        return self._adjoint_grad(u, a, lu, c)
```

## Appendix B: validation script `validate.py` (verbatim)

```python
import sys, time
import numpy as np
sys.path.insert(0, '.')
from pde_model import PDESolver, PDEMisfit
REPO = '/Users/paulconstantine/PycharmProjects/as-data-sets'
rng = np.random.default_rng(20261029)

for case in ['short', 'long']:
    S = PDESolver(case, datadir=REPO + '/Elliptic_PDE')
    print(f'==== corrcase={case}  nx={S.nx} N={S.N} m={S.m}  nnz(A)={S.stiffness(S.coeff(np.zeros(S.m))).nnz}')
    loga = S.B @ rng.standard_normal((S.m, 2000))
    print(f'  pointwise std of log a over prior: min {loga.std(1).min():.3f} max {loga.std(1).max():.3f}')
    # (a) residual
    res = []
    for k in range(5):
        x = rng.standard_normal(S.m)
        u, a, _ = S.solve(x)
        r = S.residual(u, a)
        res.append((np.abs(r).max(), np.abs(u).max(), a.min(), a.max()))
    print('  (a) max|residual| per sample:', ' '.join(f'{r[0]:.1e}' for r in res),
          ' | max u', f'{max(r[1] for r in res):.3e}', ' a range', f'{min(r[2] for r in res):.2e}..{max(r[3] for r in res):.2e}')
    # (b) FD in random directions, qoi
    errs = []
    for k in range(5):
        x = rng.standard_normal(S.m); v = rng.standard_normal(S.m); v /= np.linalg.norm(v)
        g = S.grad_qoi(x); h = 1e-4
        fd = (S.qoi(x + h*v) - S.qoi(x - h*v)) / (2*h)
        errs.append(abs(fd - g @ v) / abs(g @ v))
    print('  (b) qoi grad vs central FD (h=1e-4), rel err:', ' '.join(f'{e:.1e}' for e in errs))
    # coordinate check, full gradient at one point
    x = rng.standard_normal(S.m); g = S.grad_qoi(x); h = 1e-5
    fdg = np.array([(S.qoi(x + h*e) - S.qoi(x - h*e)) / (2*h) for e in np.eye(S.m)])
    print(f'  (b) full FD gradient (100 coords, h=1e-5): ||g-fd||/||g|| = {np.linalg.norm(g-fdg)/np.linalg.norm(g):.1e}; q={S.qoi(x):.5e} ||g||={np.linalg.norm(g):.3e}')
    # misfit
    M = PDEMisfit(case, datadir=REPO + '/Elliptic_PDE')
    xt = rng.standard_normal(M.m); obs = M.observations(xt)
    sig2 = 1e-4*np.linalg.norm(obs)**2
    M.set_data(obs + np.sqrt(sig2)*rng.standard_normal(obs.shape), sig2)
    errs = []
    for k in range(5):
        x = rng.standard_normal(M.m); v = rng.standard_normal(M.m); v /= np.linalg.norm(v)
        g = M.misfit_grad(x); h = 1e-5
        fd = (M.misfit(x + h*v) - M.misfit(x - h*v)) / (2*h)
        errs.append(abs(fd - g @ v) / abs(g @ v))
    print('  (b) misfit grad vs central FD (h=1e-5), rel err:', ' '.join(f'{e:.1e}' for e in errs))
    # (d) timings
    X = rng.standard_normal((50, S.m))
    for name, fn in [('qoi', S.qoi), ('grad_qoi', S.grad_qoi), ('misfit', M.misfit), ('misfit_grad', M.misfit_grad)]:
        fn(X[0]); t = time.perf_counter()
        for xx in X: fn(xx)
        print(f'  (d) {name:12s} {1e3*(time.perf_counter()-t)/len(X):6.1f} ms/call')
    t = time.perf_counter(); 
    for xx in X: S.stiffness(S.coeff(xx))
    print(f'  (d) assembly only  {1e3*(time.perf_counter()-t)/len(X):6.1f} ms/call')
```
