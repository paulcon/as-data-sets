# Elliptic_PDE: update spec

Status line: `Elliptic_PDE/PDE_notebook.ipynb` (22 cells, kernel `python2`); paper: Constantine, Dow & Wang, "Active subspace methods in theory and practice: applications to kriging surfaces", SIAM J. Sci. Comput. 36(4):A1500–A1524, 2014, doi:10.1137/130916138 (arXiv:1304.2070); m = 100 (KL coefficients, x ~ N(0, I)); QoI = average of u over the Neumann (right) boundary; two fields, "β = 1" (long) and "β = 0.01" (short); legacy API: `Subspaces.compute/partition`, `utils.simrunners.SimulationRunner/SimulationGradientRunner`, `utils.response_surfaces.RadialBasisApproximation`; py2 issues: yes (kernelspec, `pde_study.py` print statements; the notebook cells themselves are py3-clean); hard dependency on numpad (unavailable).

## 1. Summary

The notebook reproduces CDW 2014 §4: the active subspace of a 100-parameter elliptic PDE QoI for two correlation lengths, followed by a kriging (RBF) surface on 1-D and 2-D active subspaces, compared with a coordinate ("SENS") reduction and with a full 100-D surrogate.

What I checked:
- **Solver.** A numpad-free solver whose residual is identical to the legacy one (scratch prototype, §10) reproduces the legacy eigenvector figure: β = 1 gives w1[0] ≈ 0.97, w1[1] ≈ 0.18, w1[2] ≈ −0.14; β = .01 gives w1[0] ≈ 0.35.
- **Paper's Table 1 (CDW p. A1517).** Reproduced: eigenvalue ratio λ2/λ1 = 1.1e-3 (paper 1.0e-3) for β = 1 and 6.2e-3 (paper 5.5e-3) for β = .01.
- **Paper's local-sensitivity coordinates.** Reproduced: x1, x6 for β = .01; x1 plus one member of the degenerate pair x2/x3 for β = 1.
- **Paper's Table 2 errors.** Reproduced within about 25%.

Main problems:
1. **Two code bugs that change the published comparison.**
   - The 2-D SENS second-coordinate index is off by one. For β = 1 it picks x1 twice.
   - The β = 1 length-scale override (`.ell = 1`) is applied after training, so the RBF weights are stale.
2. **The text's correlation function does not match the shipped KL data by a factor of 2.** The data are exp(−‖s−t‖₁/(2β)). The notebook's formula is the book's eq. (3.80). Published CDW (5.3) is the same formula but missing its minus sign.
3. **The model description is the paper's MATLAB finite-element model,** not the code's 99×99 finite-volume grid.
4. **"SENS" is chosen differently from the paper.** The code uses argmax|w1|; the paper uses the gradient at the origin.
5. **The v1 RBF default likelihood degenerates for the β = 1, five-point design** (mean relative error 1.0 instead of 0.13). Fixing `log10_eta` so that ℓ1 = 1 reproduces the paper.

Solver: see `specs/numpad_replacement.md`. This spec covers the notebook-level plan only.

## 2. Inventory

| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| `PDE_notebook.ipynb` | 329,038 B | nbformat 4, kernel python2 | case study | — | Outputs are 4 PNG figures, with no numeric outputs. Execution counts 1–11 are in order. |
| `pde_model.py` | 2,576 B | py2/3 source | `PDESolver(corrcase)` with `qoi`, `grad_qoi` | `from pde_model import PDESolver` | Needs numpad. Loads `kl-{corrcase}.npz` relative to the CWD. |
| `pde_study.py` | 419 B | py2 script | smoke test (one random x) | not used by the notebook | py2 `print q`. |
| `kl-short.npz` | 8,001,158 B | npz (uncompressed), `K` (10000,100) f8, `lam` (100,) f8 | KL basis for "β = .01" | `np.load` | Byte-identical to `MCMC/kl-short.npz` (md5 079370cf…). Zip timestamp 2015-07-25. |
| `kl-long.npz` | 8,001,158 B | same | KL basis for "β = 1" | `np.load` | md5 a8279052…. |

## 3. Text and math review (vs. literature)

Sources (re-checked 2026-10-29 against the published versions):
- **CDW 2014**, SISC 36(4):A1500–A1524 (published PDF). §2–5 read. Page, equation and table numbers below refer to it. It agrees with the arXiv source in every item checked.
- **Constantine 2015 book**, *Active Subspaces*, SIAM. Two parts cover this example:
  - §3.4.2 "Revisiting the parameterized PDE" (pp. 41–44), eqs. (3.78)–(3.81) and Figs. 3.5–3.6;
  - §4.2.3 "Example: The parameterized PDE" (pp. 56–58), Figs. 4.2–4.3.
- **Which source matches the notebook better.** Neither matches throughout.
  - The PDE description in cell 0 follows the **book** almost verbatim. It has eq. (3.80) with the minus sign, which CDW (5.3) lacks.
  - Everything else follows **CDW**: the G/Ĝ/G̃ approximation theory (§3), the kriging-hyperparameter heuristic (§4.2), the SENS comparison, the n = 2 surfaces and the "kriging" framing.
  - The book's §4.2.3 has only n = 1, AS vs. Full. It fits both surfaces' hyperparameters by maximum likelihood (Rasmussen & Williams), not by the eigenvalue heuristic. It uses N = 1 sample per design point, 3M+P = 905 full-space points and 500 test points. It has no SENS comparison.
- The general exposition matches CDW §2–4 except for the rows below.

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "_Active subspaces in theory and practice: applications to kriging surfaces_. SIAM J. Sci. Comput., 36(4), A1500–A1524" | The published title (p. A1500) is "Active subspace **methods** in theory and practice: applications to kriging surfaces". The year (2014) is missing. DOI 10.1137/130916138, as printed on p. A1500. | Correct the title; add "2014" and the DOI. | confirmed |
| 0 | "[R2]: http://bookstore.siam.org/sl02/" | The store link is probably dead (not checked online). The book's ISBN 978-1-611973-85-3 matches the notebook. | Use doi:10.1137/1.9781611973860. | likely (link); confirmed (ISBN) |
| 0 | "$\mathcal C(\mathbf s, \mathbf t) = \exp\left(-\beta^{-1}\|\mathbf s - \mathbf t\|_1\right)$ … $\beta=1$ and $\beta=.01$" | All three publications use a 1/β scale, but the shipped data use 1/(2β). <br>• Book eq. (3.80), p. 42: exp(−β⁻¹‖s−t‖₁), β = 1 and 0.01. The notebook copies this. <br>• Published CDW eq. (5.3), p. A1516: exp(β⁻¹‖s−t‖₁), with no minus sign. That is a typo, because the kernel would grow with distance. <br>• MCMC paper eq. (5.5), p. A2792: exp(−‖·‖₁/β) with β = 0.02. That is the same field as `kl-short`. <br>• Shipped `lam`: top-100 eigenvalues of exp(−‖s−t‖₁/(2β)) on the 100×100 cell centres, with β = 1 and 0.01 (max rel. diff 3e-14). With β⁻¹ they are off by up to 150%. <br>So the CDW/book "β = 0.01" field is not the field the code samples, unless the MATLAB code used 2β internally (unknowable). | Write exp(−‖s−t‖₁/(2β)) with β = 1, .01, or exp(−‖s−t‖₁/ℓ) with ℓ = 2, 0.02 (matches the MCMC paper). Footnote that CDW (5.3) lacks the minus sign. | confirmed (data and all three publications) |
| 0 | "$\log(a(\mathbf s, \mathbf x)) = \sum_{i=1}^mx_i\gamma_i\phi_i(\mathbf s)$ … the $\{\phi_i(\mathbf s),\gamma_i\}$ are eigenpairs" | If γ_i is an eigenvalue, the coefficient must be √γ_i; the code uses `K*sqrt(lam)`. CDW (5.2)–(5.3) (p. A1516) and book (3.79)–(3.80) (p. 42) have the same inconsistency. The MCMC paper (5.4) (p. A2792) writes √σ_i. | Write `x_i √γ_i φ_i`, as in MCMC paper eq. (5.4). | confirmed |
| 0 | "We choose $m=100$…" (no statement about variance captured) | The 100 modes carry 99.2% (long) but only 13.3% (short) of the field variance. This explains why the short case is nearly linear. | Add one sentence. | confirmed |
| 0 | "The matrix equation … $\mathbf K\mathbf u = \mathbf f$ … $f(\mathbf x) = \mathbf c^T\mathbf M\mathbf u$ … $\mathbf M$ is the symmetric mass matrix" | This is the paper's MATLAB P1-FEM model: CDW §5.1 eqs. (5.5)–(5.9), pp. A1516–A1517; book p. 43; 34,320 triangles and 17,361 nodes. The code is a 5-point vertex-centred FV scheme with 99×99 unknowns and no mass matrix. The QoI is a trapezoid rule on the vertex column s1 = 0.99, over s2 ∈ [0.01, 0.99]. | Describe the actual discretization and the adjoint the new solver uses (A u = 1, Aᵀλ = c, ∂f/∂x_i = −λᵀ(∂A/∂x_i)u). Say the paper used FEM. Details: numpad_replacement.md §2, §5. | confirmed |
| 0 | "Homogeneous Dirichlet boundary conditions are used on the left, top, and bottom … homogeneous Neumann … right" | Matches CDW §5.1 and book p. 41, and is correct for the code. `residual` zero-pads column 0 and rows 0 and nx, and ghost-copies the last column. Axis 1 is s1. The Neumann condition is first-order (effective face at s1 = 1 − dx/2). | Keep; optionally add a footnote. | confirmed |
| 0 | "it relies on Qiqi Wang's [Numpad]…; you will need this installed" | numpad is unavailable or obsolete. | Replace with a description of the scipy.sparse solver and its adjoint. | confirmed |
| 0 | "$\approx \frac{1}{\delta^2}E\left[(f(\mathbf x + \delta\mathbf w_1) - f(\mathbf x))^2\right]$" and the next two lines | `w_1` should be `w_i` (three times). The third line drops `E[·]`: "$\frac{1}{\delta^2}\left[(f'(\mathbf x + \delta\mathbf w_1) - f'(\mathbf x))^2\right]$". Published CDW (4.4)–(4.5), p. A1513, has `w_i` and `E`. | Fix both. | confirmed |
| 0 | "if $\chi = [-1,1]^m$ and $\rho$ is uniform, $C_1\leq\frac{2\sqrt m}{\pi}$" | Matches published CDW p. A1514 (after (4.10), citing Bebendorf), where C1 is the **MSE** constant of (4.9): σ² ≤ C1 Σλ. The book states the same bound for the **RMS** form: Lemma 4.1 and Theorem 4.3 (4.19), pp. 50–51, with C1² in the proof (4.23). Bebendorf's d/π (d = 2√m) is an RMS-form constant. So in CDW's MSE form it should be (2√m/π)² = 4m/π². | Use the book's RMS form, or square the constant. Confirm with the author (§11). | likely |
| 0 | "$\int \left(\hat G(\mathbf y) - \tilde G(\mathbf y)\right)^2\rho(\mathbf y\|\mathbf z)\ d\mathbf y \leq C_2\delta\ \forall \mathbf z$" | Matches CDW Assumption 1 (§3.3). | — | confirmed OK |
| 0 | MSE bounds with $(1+\frac1N)$ | Match published CDW Theorem 3.2, eq. (3.12) (p. A1506), and Theorems 3.3–3.6 (MSE form). The book states them in RMS form with (1+N^{-1/2}). | — | confirmed OK |
| 0 | authors "Paul Constantine … <pconstan@mines.edu>", "Eric Dow, MIT, <ericdow@mit.edu>", "Qiqi Wang, MIT, <qiqi@mit.edu>", "Ryan Howard, Colorado School of Mines, <ryhoward@mines.edu>" | Published CDW p. A1500 footnotes: Constantine at Mines, **paul.constantine@mines.edu**; Dow and Wang at MIT (ericdow@mit.edu, qiqi@mit.edu). The notebook's "pconstan@mines.edu" differs from the paper. Ryan Howard is not a paper author. Current affiliations are not checked. | Use the paper's author block, with "notebook by Ryan Howard". Ask about current addresses (§11). | confirmed (vs paper); unverified (current) |
| 2 | "First we draw 300 samples" | Matches CDW §5.2 (M = 300) and book p. 57. | — | confirmed OK |
| 8 | "a subspace consisting of the variable whose weight in the first eignevector is largest (the most locally impotant variable at the origin in the parameter space)" | The code picks argmax\|w1\|, which is a global quantity. Published CDW pp. A1518–A1519: "the largest-in-magnitude components of the gradient ∇x f evaluated at the origin". It reports x1, x3 (β = 1) and x6, x1 (β = 0.01). The prototype's ∇f(0) gives x1, x2 and x1, x6. x2/x3 are a degenerate KL pair (λ2 = λ3), so the choice between them is arbitrary. Typos "eignevector", "impotant". The book has no SENS comparison. | Implement the paper's rule (one adjoint gradient at x = 0) and fix the text. | confirmed |
| 9 | "#Alter the correlation length, l, to agree with the paper" + `RS_long_AS.ell = np.array([[1.0]])` | See §4 (stale weights). CDW p. A1518 says only that the §4.2 heuristic was used (eqs. (4.8)–(4.14), pp. A1513–A1514); no ℓ = 1 is stated. The book (p. 57) fits ℓ by maximum likelihood instead. | Use `log10_eta` in v1 (§6) and say why. | confirmed (code) |
| 12 | "Now we repeat … on … the two most locally important variables, measured by first-eigenvector weight" | The code's second index is wrong (§4). | Fix the code; the text then follows the paper rule. | confirmed |
| 16 | "kriging surfaces with a linear mean term and 1000 training points" | CDW §5.3 (p. A1522) uses 3M+P = 905 (n = 1) and 925 (n = 2) points to match the cost of the AS surrogate (one gradient ≈ 2 solves). Book p. 57 uses 905. Both use 500 independent test points. | Use 3M+P and state the equal-cost rationale. | confirmed |
| 21 | "in the paper, the kriging code used was unable to recognize the linear nature of the quantity of interest" | Neither publication supports this. CDW §5.3 (p. A1522, Fig. 6) says that "the active subspace method produces a more accurate approximation" for the same cost. The book p. 58 (Fig. 4.3(c),(d)) says that "the masses of the error histograms for the active variable construction are to the left … for both correlation lengths". The claim attributes a defect to the paper's code without evidence. Prototype: the full 100-D RBF has mean rel. error 0.97 (905 pts) and 0.71 (925 pts) for β = 1. | Delete the speculation and describe the regenerated results. | confirmed (unsupported by either publication) |
| (none) | CDW Table 1 (p. A1517), normalized "singular values", is not reproduced | The published numbers are eigenvalue ratios, not singular-value ratios. KL β = 1: 0.0978, 0.0975, 0.0282, 0.0280 vs `lam` ratios 0.0975, 0.0975, 0.0281, 0.0281 (√ would be 0.31, 0.17). ASM β = 1: 1.0e-3 vs prototype λ2/λ1 = 1.1e-3 (√ would be 0.033). The book has no such table. Its Fig. 3.5 (p. 42) shows λ1 ≈ 3e-7 and λ2 ≈ 3e-10 for β = 1: the same ratio, but a different absolute scale (FEM QoI), so do not compare absolute eigenvalues with the book. | Optionally add a small table of KL λ_i/λ_1 and AS λ_i/λ_1 for β = 1 and .01, labelled "eigenvalue ratios", with a footnote. | confirmed |
| (none) | CDW Table 2 (p. A1518): mean relative error for n = 1…5 | CDW evaluates this error at the **300 gradient-sample points** (the "300 testing sites"), not at an independent set. The independent 500-point set is used only for the AS vs. Full comparison (§5.3). | Compute the Table-2-style error on the 300 points; use the 500-point set for the Full comparison. | confirmed |
## 4. Code review

Bugs:
1. **cell 13, SENS second index.**
   ```python
   SENS_short_v2 = np.argmax(np.abs([a for i,a in enumerate(list(ss_short.W1)) if i!=SENS_short_v1]))
   ```
   This returns an index into the list *with v1 removed*. It is off by one whenever the true index exceeds v1. Both cases have v1 = 0, so:
   - **β = 1:** v2 = 0 = v1. `x_train_long_SENS[:,0]` is assigned twice, and the "2-D" SENS surrogate is a function of x1 only.
   - **β = .01:** the true second index is 5 (x6); the code uses index 4, where |w1[4]| ≈ 0.001.

   Fix: `np.argsort(-np.abs(w))[:2]`. Better, use the paper's ∇f(0).
2. **cells 9 and 13, β = 1 length scale.** `RS_long_AS.ell = [[1.0]]` and `RS_long_AS2.ell[0,0] = 1` run after `.train`. Legacy `predict` uses `self.ell` with `radial_weights` solved under the old ell. The surface therefore no longer solves its own interpolation system. The plotted "GP mean" in the cell 11 figure is not the ℓ = 1 kriging mean.
3. **cell 3.** Gradients and values come from separate runners, so 1,200 PDE solves are done where 600 suffice. Use one solve per point (`qoi_and_grad`, or v1 `evaluate_with_grads` with `value_and_grad`).

Randomness and reproducibility:
- **No seeds anywhere** (cells 3, 17, 19).
- **Sample generation.** `np.random.multivariate_normal(np.zeros(100), np.eye(100), …)` should be `rng.standard_normal((N, 100))`.

Fragility and readability:
- **Mutable subspace objects.** `ss_*` is re-partitioned back and forth (cells 9, 13, 15, 20); results depend on cell order. v1's `with_dimension` returns new objects.
- **cell 13.** `np.mgrid[-3:3:5j,-3:3:5j]` is rebuilt 50 times inside a loop. Use `np.meshgrid(y1, y1, indexing='ij')`.
- **Naming.** Comments say "Gaussian Process"; the class is an RBF/kriging surrogate. Keep "kriging" in the prose, as the paper does, and note that it is the v1 RBF surrogate.

`pde_model.py`:
- **Duplicate solves.** `qoi` and `grad_qoi` each solve the PDE.
- **CWD-relative data path.** `np.load('kl-…')` depends on the working directory.
- **Return type.** The gradient is returned as `np.matrix` (1, m).
- **Precision workaround.** `_e = 2.718…; _e**a` is a numpad workaround.

All four are fixed by the replacement module (numpad_replacement.md §6).

Cost (new solver, about 30 ms per solve):
- 600 value+gradient evaluations: about 20 s.
- Design points (5 + 25 per case): about 2 s.
- Full-space training (2 × 905–925) and testing (2 × 500): about 85 s.
- Full-space RBF fit: 3.6 s per fit (1000 × 100).
- Total: about 2–3 min.

## 5. Data management review

- **Format.** `.npz` of `K` (orthonormal discrete covariance eigenvectors, ‖K_j‖₂ = 1, KᵀK = I to 9e-15) and `lam` (matrix eigenvalues). The trace of the full matrix is 10,000 (unit pointwise variance), so `B = K√lam` gives the correctly scaled field. There are no headers, units or index columns. The normalization (N(0, I)) is implicit.
- **Provenance.** Covariance exp(−‖s−t‖₁/(2β)) on the 100×100 cell-centre grid ((i+½)/100), β ∈ {1, 0.01}. The span of K equals the span of the top-100 Kronecker products of 1-D eigenvectors (distance 2e-12). Only 4 (long) and 8 (short) columns are themselves rank-1 Kronecker products; the rest are arbitrary rotations within degenerate pairs (λ_i = λ_j). The generating script is not in the repo. Document the construction in the notebook and a README line.
- **Do not regenerate K on the fly.** It would change the coordinates x within degenerate pairs and break comparability with the legacy figures (agrees with numpad_replacement.md §7.3). The stored K defines the coordinates.
- **Redundancy.** `kl-short.npz` is duplicated byte-for-byte in `MCMC/`. 8 MB is well under the 50 MB warning. Deleting one copy does not shrink clone size, because it stays in history. Recommendation: keep both copies (self-contained directories), or load from `../Elliptic_PDE` (author's choice, numpad_replacement.md §7.3).
- **Size.** 16 MB total; no GitHub concern. `np.savez_compressed` gains essentially nothing on float64 eigenvectors.
- **License.** The repo has no LICENSE file; the data and code license is unstated (§11).
- **Pandas-free loading (tested in .venv):**
  ```python
  import numpy as np
  kl = np.load('kl-long.npz')            # keys: 'K' (10000, 100), 'lam' (100,)
  B = kl['K'] * np.sqrt(kl['lam'])       # log a = (B @ x).reshape(100, 100); axis 1 = s1
  ```

## 6. Migration to active_subspaces v1

Space: `space = ac.Gaussian.reference(100)`. x is already N(0, I), so the physical coordinates equal the normalized ones and no gradient scaling is applied.

| legacy call | v1 replacement | notes |
|---|---|---|
| `ac.utils.simrunners.SimulationGradientRunner(solver.grad_qoi).run(x)` + `SimulationRunner(solver.qoi).run(x)` | `model = ac.Model(space=space, value_and_grad=batch_fn)`; `f, G = ac.evaluate_with_grads(model, X=x)` (or a list comprehension over `solver.qoi_and_grad`) | One solve per point. `f` is (M,), not (M,1). |
| `ss = ac.subspaces.Subspaces(); ss.compute(df=grad)` | `sub = ac.active_subspace(space=space, grads=G)` | Eigenpairs equal to rounding. With these gradients EVG picks n = 1 for both cases; the notebook fixes n anyway. |
| `ss.partition(1)` / `ss.partition(2)` | `sub1 = ac.with_dimension(sub, 1)`, `sub2 = ac.with_dimension(sub, 2)` | Immutable, so the cell-order dependence goes away. |
| `ss.W1[:,0]`, `ss.W1[:,1]` | `sub.W[:, 0]`, `sub.W[:, 1]` | **Sign rule changed.** v1 makes the largest-\|·\| component positive; 2016 made the first component positive. w1 is unchanged for both cases (largest component is index 0). The β = 1 second eigenvector is mirrored relative to the legacy figure (first comp. +0.18, largest −0.73). Plots with SENS/AS projections must use the same W. |
| `ss.eigenvals` | `sub.eigenvalues` (m,) | |
| `RadialBasisApproximation(2).train(y_train, f, e=ss.eigenvals)` | `rs = ac.fit_surrogate(X=y_train @ sub_n.W1.T, f=f_train, subspace=sub_n, kind="rbf", degree=2)` | `X` is physical, and here equals `y W1ᵀ`. Eigenvalue-informed ℓ_j = η Σ\|λ\|/\|λ_j\|, nugget relative to the amplitude (D-100), new likelihood (D-101). |
| `RS_long_AS.ell = [[1.0]]` (post-hoc) | `log10_eta=-np.log10(np.abs(lam).sum()/lam[0])` in `fit_surrogate` (gives ℓ1 = 1 and refits properly) | Prototype, β = 1, n = 1: default η gives ℓ1 = 0.014 and mean rel. error 1.01; ℓ1 = 1 gives 0.133 (paper 0.178). For n = 2: 0.435 vs 0.154 (paper 0.149). Same kernel parametrization as 2016 (`exp(-Δ²/ℓ)`). |
| `RadialBasisApproximation(2).train(y_train, f_SENS)` (no `e`) | `ac.fit_surrogate(X=X_sens, f=…, subspace=ac.subspace_from_basis(np.eye(100)[:, idx], space=space), kind="rbf", degree=2)` | Isotropic ℓ = η, nugget floor 1e-6, as 2016. The paper used GPML MLE with an isotropic SE kernel and quadratic basis. |
| `RS.predict(x.dot(ss.W1))[0]` | `ac.predict(rs, x)` (physical) or `rs.evaluate(Y)` | Returns (M,). |
| `RadialBasisApproximation(1).train(x_full, y_full)` | `ac.fit_surrogate(X=x_full, f=y_full, space=space, subspace=None, kind="rbf", degree=1)` | 3.6 s at 1000 × 100. |
| hand-made eigenvector plot (cell 7) | keep the hand plot (two fields per axis), or `ac.plotting.plot_eigenvectors(sub, indices=[0, 1])` per field | Hand plot preferred, because it overlays β cases. |
| `.Rsqr` (not used) | `rs.loo_r_squared` | Optional diagnostic. |

## 7. Associated .py files

- **`pde_model.py`.** Replace with the numpad-free module from numpad_replacement.md (Appendix A), with these changes:
  - Rename the class split: `PDESolver` (QoI) / `PDEMisfit` (MCMC).
  - Add `qoi_and_grad(x)` (one LU, one adjoint).
  - Keep the legacy QoI weights (the trapezoid with half-weighted end nodes) so that numbers are comparable.
  - Byte-identical copy in `MCMC/` (numpad_replacement.md §7.2(a)).
- **`pde_study.py`.** Delete. The notebook's solver-check cell (FD vs. adjoint gradient at a seeded x) replaces it. If kept, port it to py3 as a 10-line smoke test.

## 8. Update plan

1. **Solver.** Add the new `pde_model.py` (from numpad_replacement.md) with `qoi_and_grad`. Delete `pde_study.py`.
2. **Kernel and imports.** Switch the kernelspec to python3; replace imports; add `rng = np.random.default_rng(2014)` and `space = ac.Gaussian.reference(100)`; add a solver-check cell (residual, FD gradient).
3. **Cell 3.** One solve per sample via `qoi_and_grad`. M = 300. Gradients and values for both fields.
4. **Cells 5 and 7.** `ac.active_subspace`, `with_dimension`, eigenvector plot. Optionally add a table of λ_i/λ_1 (KL and AS) vs. paper Table 1.
5. **SENS.** Compute ∇f(0) (one adjoint) and take the top-2 \|component\| indices. Remove the argmax\|w1\| logic and the off-by-one bug.
6. **Cells 9 and 13.** v1 `fit_surrogate`. β = 1 uses `log10_eta` for ℓ1 = 1; β = .01 uses the default. Remove the `.ell` overrides. The 2-D design uses `meshgrid`.
7. **Cells 11, 15 and 20.** Plots via `ac.predict`. Full-space surrogate with 3M+P training points (905 and 925). Independent 500-point test set.
8. **Text fixes (§3).**
   - Correlation function and β convention.
   - √γ_i.
   - Variance captured.
   - FV discretization and adjoint.
   - Remove numpad.
   - w_i / E[·] typos.
   - SENS definition.
   - Equal-cost rationale.
   - Rewrite cell 21 from the actual results.
   - Fix the references.
9. **Execute and record.** Run `jupyter nbconvert --execute` from `Elliptic_PDE/`, commit outputs, and record the runtime in a final cell.
10. **Docs.** README bullet for Elliptic_PDE (currently missing from the README's data-set list); CLAUDE.md note on the shared solver file.

## 9. Acceptance criteria

- **Headless run.** `cd Elliptic_PDE && jupyter nbconvert --to notebook --execute PDE_notebook.ipynb` succeeds in .venv in < 5 min. Imports only numpy, scipy, matplotlib, active_subspaces, and the local `pde_model`. No `numpad`, `pandas` or `print` statements.
- **Seeding.** All randomness comes from one seeded `default_rng`. Two runs give identical outputs.
- **Solver check cell.**
  - max\|residual\| < 1e-10.
  - Central-FD vs adjoint gradient relative error < 1e-6 on 4 coordinates.
  - `qoi(0)` = 0.07608.
- **Eigenvalues.**
  - λ2/λ1 ∈ [5e-4, 2.5e-3] for β = 1 (paper 1.0e-3; prototype 1.1e-3).
  - λ2/λ1 ∈ [3e-3, 1.2e-2] for β = .01 (paper 5.5e-3; prototype 6.2e-3).
- **First eigenvector.**
  - \|w1[0]\| = 0.96 ± 0.02 for β = 1, with \|w1[1]\|, \|w1[2]\| ≈ 0.18, 0.14.
  - \|w1[0]\| = 0.35 ± 0.05 for β = .01.
  - Matches the legacy figure.
- **SENS coordinates (∇f(0)).**
  - β = .01 → {x1, x6} (paper).
  - β = 1 → x1 plus one of {x2, x3}.
- **Mean relative error, AS surrogate.** The paper's numbers (CDW Table 2, p. A1518) are evaluated on the 300 gradient-sample points. The prototype numbers below used an independent 500-point set. Either set is acceptable; record which one is used.
  - β = .01: n = 1, 2 ∈ [5e-3, 1.2e-2] (paper 7.88e-3 / 7.82e-3; prototype 8.1e-3 / 8.3e-3).
  - β = 1: n = 1, 2 ∈ [0.08, 0.27] (paper 0.178 / 0.149; prototype 0.133 / 0.154).
- **Error histograms.** The AS histogram lies left of SENS in all four panels, and left of Full for β = 1.
- **Figures.**
  1. First and second eigenvectors (2 panels).
  2. 1-D sufficient summary with surrogate: ASM/SENS × β (2×2).
  3. log10 relative error histograms ASM vs SENS (2×2).
  4. ASM vs Full (2×2).
- **Text.** Every confirmed row in §3 is resolved.

## 10. Prototype results

Script: `prototypes/g2/fvsolver.py`. It is an independent scipy.sparse vertex-centred FV assembly plus a hand-coded adjoint. Checks and runs, all on .venv with v1:

- **Solver checks** (`check_solver.py`):
  - matrix vs legacy `residual`: 6e-11 absolute;
  - residual at the solution: 2e-12;
  - adjoint vs central FD: relative error 5e-11 to 2e-5 (smallest components);
  - about 30 ms per value+gradient.
- **Correlation convention.** `lam` = top-100 of the Kronecker eigenvalues of exp(−\|Δ\|/(2β)) on cell centres: max rel. diff 3.4e-14 (long) and 8.7e-15 (short). With exp(−\|Δ\|/β): 1.48 and 0.72.
- **`proto_elliptic.py`** (seed 2014, M = 300, `ac.active_subspace(space=Gaussian.reference(100), grads=G)`):
  - β = 1: λ[:5] = 1.18e-2, 1.29e-5, 6.86e-6, 5.81e-6, 3.55e-6; w1[:3] = 0.965, 0.184, −0.138; ∇f(0) top-2 = x1, x2.
  - β = .01: λ[:5] = 2.27e-5, 1.41e-7, 1.23e-7, 1.08e-7, 1.01e-7; w1[0] = 0.355; ∇f(0) top-2 = x1, x6.
  - v1 default RBF mean rel. error: β = .01 n = 1, 2: 8.1e-3, 8.3e-3; β = 1 n = 1, 2: 1.01, 0.43.
- **`proto_long2.py`** (β = 1, ℓ1 fixed to 1 via `log10_eta`):
  - n = 1: 0.133; n = 2: 0.154.
  - Full 100-D RBF, linear tail: 905 pts → 0.97; 925 pts → 0.71.
  - Training values at y = (−3, −1.5, 0, 1.5, 3)·w1: 0.0059, 0.0212, 0.0761, 0.273, 0.983 (legacy figure: about 0, 0.02, 0.08, 0.27, ≈ 1).
- **Legacy figures.** Extracted to `…/scratchpad/g2/figs/pde_c*.png` for visual comparison.

## 11. Open questions for the author

1. **β convention.** All three publications use a 1/β scale: CDW (5.3), p. A1516, has no minus sign; book (3.80); MCMC paper (5.5), β = 0.02. The shipped KL data use 1/(2β). Present the field as exp(−‖s−t‖₁/(2β)) with β = 1 and .01 (the CDW/book labels), or as ℓ = 2 and 0.02 (consistent with the MCMC paper)? Did the MATLAB FEM study in CDW and the book use the 1/β kernel, so that its "β = 0.01" field is half as correlated as `kl-short`? Should CDW (5.3)'s missing minus sign go on an errata list?
2. **SENS rule.** Switch to the paper's ∇f(0) rule (recommended), or keep argmax\|w1\| with the bug fixed?
3. **β = 1 surrogate.** Keep the "ℓ1 = 1 to agree with the paper" choice (via `log10_eta`), or show v1's default likelihood and explain its failure on a 5-point design? Alternatively, model log f.
4. **Uniform Poincaré constant.** CDW p. A1514 uses C1 ≤ 2√m/π in the MSE inequality (4.9). The book (Lemma 4.1, Theorem 4.3, pp. 50–51) uses it as the RMS-form constant. Should the notebook use the book's RMS form, or (2√m/π)² for the MSE form?
5. **v1 documentation.** The `fit_surrogate` docstring calls the eigenvalue length-scale heuristic "a heuristic of the 2016 package with no published reference". It is published CDW 2014 §4.2, eqs. (4.8)–(4.14), pp. A1513–A1514 (ℓ_i² = α Σλ/λ_i, η² = α Σ_{q>n} λ_q). Should the v1 docs cite it?
6. **Authors and data.** Author list and affiliations or emails for the header (Ryan Howard as notebook author?). License for code and data (the repo has none). Can the KL generation script be found (bitbucket `active-subspace-methods-in-theory-and-practice` is MATLAB/FEM, so not the source of these files)?
7. **QoI weights.** Keep the legacy QoI weights (end nodes half-weighted, column s1 = 0.99) for comparability, or switch to the full boundary integral?
