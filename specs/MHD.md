# MHD: update spec

Kind: **both**: the Hartmann problem is a Python model (closed-form `gradients.py`); the 3-D generator runs are stored data from Drekar (Sandia) (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `MHD/MHD.ipynb` | paper: Glaws, Constantine, Shadid, Wildey, *Dimension reduction in magnetohydrodynamics power generation models: Dimensional analysis and active subspaces*, Stat. Anal. Data Min. 10(5) (2017) 312–325, doi:10.1002/sam.11355 (checked against the published PDF; page/equation refs below are to the published version, whose equation numbers differ from arXiv:1609.01255 v1, titled "…MHD power generation models…") | m = 5 log-inputs (log μ, log ρ, log ∂p0/∂x, log η, log B0), uniform in the logs | QoIs: u_avg and B_ind, for (a) the analytic Hartmann problem and (b) the 3-D MHD generator (Drekar, adjoint gradients, 483 runs) | legacy API: `ac.utils.quadrature.gauss_legendre`, `ac.subspaces.sorted_eigh`, `ac.subspaces.Subspaces().compute(df=, nboot=500)/.partition(2)/.e_br`, `ac.utils.plotters.*` | py2 issues: no print statements; pandas `DataFrame.from_csv(...).as_matrix()` (removed in pandas ≥ 1.0); kernelspec python2.

## 1. Summary

The ranges (Tables 2–3, pp.319, 322), the Hartmann formulas (Eqs. 36–37, p.318), the u_avg results (Fig. 2, p.320), the generator setup and the generator conclusions (pp.321–323) match the published paper. The generator numbers (5th eigenvalue 0.00024% / 0.000001% of the sum) reproduce exactly from the shipped data with v1. Main problems:

1. **Hartmann eigenvalues are 32 times too small (confirmed).** The code applies `*2**-5` to weights that already sum to 1. This is the same bug as in Ebola.
2. **Wrong math in the exposition (confirmed).** The cell 0 eigenvalue-sum identity writes an outer product where it needs an inner product, and labels the inactive gradient `∇_x` instead of `∇_z`. The MHD momentum equation shown is not the paper's governing system (published Eq. 21, p.316).
3. **The paper's Hartmann-gap sentence is wrong (refuted by its own Fig. 2A).** The published text (p.319) says a "three-order-of-magnitude gap between the first and second eigenvalues" for Hartmann u_avg. The published Fig. 2A (p.320) shows λ1 ≈ 1.4 and λ2 ≈ 0.05, a ratio of about 28. ℓ = 1 m is now confirmed (p.318), and the code gives 29.8. The notebook text ("at least an order of magnitude") is correct. Fig. 2A's absolute scale (λ1 ≈ 1 rather than 36.6) shows the published figure carries the same `2**-5` scaling bug as the notebook.
4. **Data hygiene.** The two generator files are pandas CSVs with an index column whose labels run 0..499 with 17 gaps. `…_Uavg.txt` contains one exact duplicate record (rows 47 and 48, labels 49 and 50). The two files do not share run labels: rows labelled 50–98 are offset by one between the files. Each file is internally consistent. The effect on results is negligible, but the index column is not a run ID. Units, column meaning and provenance are undocumented in the repo.
5. **Latent bug and fragile import.** `gradients.py` uses `0.0*np.empty(...)` for the ρ-derivative, which can give NaN. `from gradients import *` injects the Hartmann `lb`, `ub` into the notebook, and cell 12 later overwrites them.
6. **Dimensional analysis follows arXiv, not the published version (new).** Published §3.1 (p.317, Eqs. 22–24) uses 8 quantities, with the pressure **gradient** ∇p0 [M L⁻² T⁻²] and μ0 [M L T⁻² C⁻²] added, and gets **four** Π groups: Re = ℓvρ/μ, Ha = ℓB0/(μη)^{1/2}, Rm = ℓvμ0/η, and ℓ∇p0/(v²ρ). The notebook's cell 0 table (pressure p, no μ0) and "3 unitless parameters" (Π3 = p/(ρv²)), and cell 11's "should depend on 3 unitless parameters" and B*_ind with μ0*, follow arXiv. Published Eq. (40) (p.319) has Rm in place of μ0*.
7. **The notebook's Hartmann B_ind does not reproduce the published Fig. 3 (new, unresolved).** Fig. 3B (p.321) w1 ≈ [.38, 0, −.56, .49, −.56]; the notebook/v1 w1 = ±[.381, 0, −.663, .493, −.414]. Components 3 and 5 differ well beyond reading error. Fig. 3A λ1 ≈ 3 and λ1/λ2 ≈ 45, against 0.086 (0.0027 with the bug) and 13.6 here. A smaller ℓ (≈ 0.4–0.5 in the Ha argument) reproduces the published w1, but not the eigenvalue scale or ratio. u_avg (Fig. 2B) matches exactly, so only the B_ind setup differs.

## 2. Inventory

| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| MHD.ipynb | 1.0 MB | nbformat 4, python2 | case study (Hartmann + generator) | — | 16 PNG outputs, no printed numbers |
| gradients.py | 2.9 KB | py (py3-clean) | Hartmann u_avg, B_ind and gradients w.r.t. **normalized** log-inputs; module globals `l=1`, `mu0=1`, `lb`, `ub` | `from gradients import *` | gradient scaling baked in; star import |
| MHD_Generator_Data_Uavg.txt | 67 KB | CSV, header `,0,…,10`, index column | 483 generator runs: cols 1–5 physical inputs (μ, ρ, ∂p0/∂x, η, B0), col 6 u_avg, cols 7–11 ∂u_avg/∂(physical input) | `pn.DataFrame.from_csv(...).as_matrix()` | rows 47 and 48 identical; index labels 0..499 with 17 missing (5, 40, 72, 107, 115, 153, 253, 258, 295, 305, 309, 347, 398, 411, 453, 467, 470) |
| MHD_Generator_Data_Bind.txt | 62 KB | same layout, B_ind | 483 runs | same | X rows match Uavg for 481 runs; labels 49–98 offset by one vs Uavg; 2 runs not in Uavg, 1 Uavg run not here |

## 3. Text and math review (vs. literature)

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "$\lambda_1 + \cdots + \lambda_n = \int\nabla_{\mathbf y} f(\mathbf x)\nabla_{\mathbf y} f(\mathbf x)^T\gamma(\mathbf x)\ d\mathbf x$" (and the same for z) | the left side is a scalar but the right side is an n×n outer product. The identity is $\int \nabla_{\mathbf y} f^T\nabla_{\mathbf y} f\,\gamma\,d\mathbf x$ (CDW 2014 Lemma 2.2; Constantine 2015 §3.2; published Eq. (11), p.315, writes it correctly) | swap the transpose: `\nabla_{\mathbf y} f(\mathbf x)^T\nabla_{\mathbf y} f(\mathbf x)` in both lines | confirmed |
| 0 | "where $\nabla_{\mathbf y}$ and $\nabla_{\mathbf x}$ are gradients with respect to the active and inactive variables" | should be $\nabla_{\mathbf z}$ | fix | confirmed |
| 0 | "let $\mathbf v$ satisfy $\mathbf D\mathbf v = \mathbf u$ and let $\mathbf U$ be a basis … $\Pi_j = \prod_i x_i^{u_{i,j}}$" | **u** is used for both the output-unit vector and the null-space basis entries | write the basis entries as $U_{i,j}$ and its columns as $\mathbf U_{:,j}$, and use $\mathbf U_{:,j}$ in the later derivation | confirmed (notation) |
| 0 | "$\rho\mathbf u\cdot\nabla\mathbf u = -\nabla(p_0+p)+\mu\nabla^2\mathbf u-\frac 1 \eta\mathbf B\times(\mathbf u\times \mathbf B)$" | not in the paper. The published Eq. (21) (p.316; Eq. 20 on arXiv) is the conservative momentum equation with the Maxwell stress (1/μ0)(B⊗B − ½‖B‖²I) plus the induction equation. The notebook form is a low-magnetic-Reynolds (inductionless, E = 0) simplification | replace with published Eq. (21), or label it "inductionless approximation" | confirmed (differs from published) |
| 0 | units table ("Pressure \| $p$ \| $\frac{M}{LT^2}$", no μ0); "The Buckingham Pi Theorem says the system depends on 3 unitless parameters; … $\Pi_1 = \frac{\rho v l}{\mu},\ \Pi_2 = \frac{Bl}{\eta^{1/2}\mu^{1/2}},\ \Pi_3 = \frac{p}{\rho v^2}$" | matches arXiv v1 but **not the published version**. Published §3.1 (p.317) lists 8 quantities (ℓ, v, μ, ρ, ∇p0 with [∇p0] = M L⁻²T⁻², η, B0, μ0 with [μ0] = M L T⁻²C⁻²), D in Eq. (22), and **four** Π groups in Eq. (24): Re, Ha, Rm = ℓvμ0/η, and ℓ∇p0/(v²ρ). "At most 4 linear combinations" still holds (Eqs. 32, 35, p.318) | rewrite the table and Π groups from published Eqs. (22)–(24), with 4 groups | confirmed (differs from published) |
| 0 | Hartmann u_avg, B_ind formulas; ranges table | match published Eqs. (36), (37) (p.318) and Table 2 (p.319) | none | confirmed |
| 0 | "This shows that the dimension of the physical law's active subspace (in terms of its log inputs) has dimension of at most $n+1$" | correct. The published text (p.316) says "has dimension at least n + 1", a typo in the paper: Eq. (20) bounds it from above | keep "at most" | confirmed (paper typo) |
| 0 | author block: Glaws, Constantine "Colorado School of Mines"; Shadid, Wildey "Sandia" | the published version (p.312) lists Glaws and Constantine at Computer Science, CU Boulder; Shadid at Sandia and UNM; Wildey at Sandia | keep the 2016 affiliations with a note, or update (ask author) | confirmed (differs from published) |
| 2 | "we set $l$ and $\mu_0$ to each be 1.0" | ℓ = 1 m is confirmed (published p.318: "We define 𝓁 = 1 m"). μ0 for Hartmann is not stated; μ0 = 1 is stated only for the generator (p.320). μ0 only scales B_ind (eigenvalues ∝ μ0², eigenvectors unchanged), so it cannot explain the Fig. 3 mismatch (see the cell 10/11 B_ind row below) | none for ℓ; ask the author about μ0 and the Fig. 3 setup | ℓ confirmed; μ0 unverified (harmless) |
| 0 | "[1] … arXiv:1609.01255, 2016" | published version exists, and its title spells out "magnetohydrodynamics" | cite SAM 10(5):312–325, doi:10.1002/sam.11355 | confirmed |
| 0 | "[[2]] … _Active subspaces in theory and practice: applications to kriging surfaces_" | title is "Active subspace methods in theory and practice…" | fix | confirmed |
| 4 | "Gauss-Legendre rule with 11 points in each of the 5 dimensions (totaling 161051 points)" | matches paper | none | confirmed |
| 7 (figure) | eigenvalue plot, λ1 ≈ 1 | 32× too small (`2**-5`); correct λ1 = 36.57 | remove the factor | confirmed |
| 8 | "The second eigenvalue is at least an order of magnitude smaller than the first" | true (λ1/λ2 = 29.8). The published text (p.319) claims a "three-order-of-magnitude gap", but the published Fig. 2A (p.320) shows λ1 ≈ 1.4 and λ2 ≈ 0.05, a ratio of about 28. The figure's λ1 ≈ 1.4 equals the code's 36.57/32 = 1.14 within reading error, so the published figure carries the same `2**-5` bug | keep the notebook sentence; do not import the paper's "three orders" | confirmed (paper sentence refuted by its own Fig. 2A) |
| 8 | "neither vector places any weight on log(ρ)" | true (exact 0) | none | confirmed |
| 11 | "$B_{ind}^* = \frac{\partial p_0^*}{\partial x^*}\frac R H\mu_0^*(1 - \frac{2}{H}\tanh(\frac H 2))$"; "The analysis above indicated that the quantities of interest should depend on 3 unitless parameters" | u* matches published Eq. (39). Published Eq. (40) (p.319) has the **magnetic Reynolds number Rm** where the notebook has μ0*, and the published text says **four** unitless quantities (Re, Ha, Rm, pressure gradient) reduced to two by products. The notebook follows arXiv | replace μ0* with Rm (define Rm = ℓvμ0/η); "depend on 4 unitless parameters (Re, Ha, Rm, dimensionless pressure gradient)… several appear only as products" | confirmed (differs from published) |
| 10 (figure), 11 | Hartmann B_ind eigen/eigvec plots; "similar eigenvalue decay compared to u_avg" | the notebook (and v1) w1 = ±[.381, 0, −.663, .493, −.414] and λ1/λ2 = 13.6. Published Fig. 3B (p.321) w1 ≈ [.38, 0, −.56, .49, −.56], and Fig. 3A shows λ1 ≈ 3 with λ1/λ2 ≈ 45. Components 3 and 5 differ. Not explained by μ0; partly by a smaller ℓ (≈ 0.4 matches w1 but gives ratio ≈ 170) | ask the author how Fig. 3 was computed; until resolved, state in the notebook that its B_ind figures differ from the paper's | confirmed (mismatch); cause unverified |
| 11 | "Each run of the model takes about 5 CPU-hours … 483 samples" | published p.321: 5.3 CPU-hours (10 min on 32 cores), M = 483, adjoint gradients (Drekar); generator μ0 = 1, x_on = 4, x_off = 6, δ = 0.1 (p.320) | add "adjoint gradients computed with Drekar", and units and provenance of the data files | confirmed |
| 16 | "most significant decay between the fourth and fifth values … full 4-dimensional subspace" | consistent with recomputation: λ5/Σλ = 2.4e-6 (u), 1.0e-8 (B), as in the published text (pp.322–323: 0.00024%, 0.000001%) | none | confirmed |
| 16 | "the most important variables are viscosity and pressure gradient" | w1(u) = [.83, −.02, −.56, .04, −.05] | none | confirmed |
| 18 | "notable deviation from the trend in the 1-dimensional summary plot when the first active variable is negative" | sign-dependent statement. Under the v1 sign rule (largest component positive) w1(B) = [.49, −.12, −.30, .73, −.36], the same sign as legacy. The published text (p.323) says the same (1-D where y1 > 0, degrading as y1 decreases, "unique up to a sign"), and also that w1's η and B0 components have opposite signs (ours: +.73, −.36) | reword to "for y1 < 0 (with w1 oriented so that its log η component is positive)" | confirmed (published p.323) |
| 0 | "Fluid Magnetic Resistivity η" | the paper calls η "magnetic resistivity"; harmless | none | — |

## 4. Code review

- **Bug (confirmed): `subspace_integrand` uses `*2**-5`**, but legacy GL weights sum to 1. The Hartmann C and its eigenvalues are 1/32 of the true values. The generator path is unaffected.
- **Latent bug: `gradients.py` sets `dudrho = 0.0*np.empty(x.shape[0])`** (and `dBdrho` the same way). `np.empty` may hold NaN or Inf bit patterns, and 0*NaN = NaN. Use `np.zeros`.
- **Wasted computation.** Cells 5 and 9 loop in Python over 161,051 points to build 5×5 matrices; use `(G*w[:,None]).T @ G` or v1. v1 takes 0.2 s for u and 0.1 s for B.
- **Hidden coupling.** `from gradients import *` imports `lb`, `ub`, `l`, `mu0`, `coth`, `sech` and `csch`. Cell 5 depends on that `lb` and `ub`, and cell 12 rebinds `lb` and `ub` to the generator ranges. Use explicit imports and distinct names (`lb_hart`, `lb_gen`).
- **Scaling baked into the model.** `uavg_grad` and `Bind_grad` return gradients w.r.t. *normalized* log-inputs, using the module-level Hartmann `lb` and `ub`. For v1, return ∂f/∂(log x) = x·∂f/∂x and let the space scale it.
- Randomness: `np.random.choice(range(xx.shape[0]), 5000, False)` in cells 7 and 10, and `nboot=500` in cell 13, are unseeded. Use `rng = np.random.default_rng(seed)`; v1 `active_subspace(..., nboot=500, rng=rng)`.
- The analytic gradients were checked against central differences: max relative error 5.6e-10 for all inputs of both QoIs.
- py2: none in code beyond pandas. `DataFrame.from_csv` and `.as_matrix()` were removed in pandas 1.0.
- Hard-coded paths: relative file names only (fine when run from `MHD/`).

## 5. Data management review

- **Format.** pandas `to_csv` output with a header row `,0,1,…,10` and an index column. Columns are not named in the file or the notebook; the meaning is inferred from cell 12. Inputs are in physical units: μ, ρ, ∂p0/∂x, η, B0, with μ0 = 1 and the generator's ℓ, δ fixed. The gradient columns are ∂QoI/∂(physical input). Check: the median of (∂p0/∂x)·∂u/∂(∂p0/∂x)/u is 1.0002, as expected for u roughly linear in the pressure gradient. Normalization is not stored.
- **Problems.**
  - (a) Uavg rows 47 and 48 are bit-identical: the same X, f and gradient, labels 49 and 50. Drop one; M becomes 482 for u_avg. Or ask the author for the missing run.
  - (b) Index labels have 17 gaps (presumably failed runs). They do not identify a run consistently across the two files: labels 50–98 are offset by one. Do not join the files on the index.
  - (c) There is no units or provenance note.
- **Size.** 129 KB total, well under the limits. No redundancy beyond the repeated X columns.
- **Provenance and license.** Sandia Drekar runs (paper §4.2). Ask whether the data may be redistributed under the repo license; the repo has no LICENSE file in this directory.
- **Recommendation.** Keep the files as they are, with the CSV layout unchanged, and add a `MHD/README` or a markdown cell giving column names, units, and the duplicate/gap notes. Optional: rewrite them with a `#`-comment header and named columns.
- **Tested pandas-free loading (venv):**

```python
import numpy as np
D = np.loadtxt('MHD_Generator_Data_Uavg.txt', delimiter=',', skiprows=1)[:, 1:]  # drop index col
Xu, u, du = D[:, :5], D[:, 5], D[:, 6:]          # physical inputs, QoI, d QoI / d physical input
Xu, idx = np.unique(Xu, axis=0, return_index=True); u, du = u[idx], du[idx]  # optional: drop the duplicate (482 rows)
```

(Without dedup: shapes (483,5), (483,), (483,5) for both files; all inputs lie inside the stated log-ranges.)

## 6. Migration to active_subspaces v1

| legacy call | v1 replacement | notes |
|---|---|---|
| `ac.utils.quadrature.gauss_legendre([11]*5)` | `scipy.special.roots_legendre(11)`, weights/2, tensor product | weights sum to 1 |
| hand C loop + `ac.subspaces.sorted_eigh` | `space_h = ac.Uniform(lower=np.log(lo_h), upper=np.log(hi_h), names=['log mu','log rho','log dp0/dx','log eta','log B0'])`; `X = space_h.from_normalized(T)`; `sub = ac.active_subspace(space=space_h, grads=G_log, quad_weights=W)` | "physical" coordinates are the **log** inputs. `G_log` = ∂f/∂log x (do not pre-multiply by (ub−lb)/2). Eigenvalues are 32× the legacy figure |
| `ac.subspaces.Subspaces().compute(df=du, nboot=500)` | `ac.active_subspace(space=space_g, X=np.log(Xu), grads=Xu*du, nboot=500, rng=rng)` | drop the hand `.5*(ub-lb)*Xu*du` and `XXu = 2*(...)-1`; pass `X` only for plotting |
| `.partition(2)`, `.W1` | `ac.with_dimension(sub, 2).W1`, or `plot_eigenvectors(sub, indices=[0,1])` | the EVG default picks n = 1 for all four cases |
| `ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br, label)` | `ac.plotting.plot_eigenvalues(sub, interval="minmax")` | `"minmax"` reproduces the 2016 band; the default percentile band is narrower |
| `ac.utils.plotters.eigenvectors(W1, in_labels=)` | `ac.plotting.plot_eigenvectors(sub, indices=[0,1], labels=in_labels)` | sign: v1 flips Hartmann-u w1 relative to legacy (largest is log ∂p0/∂x, +0.80; legacy had log μ first and positive). Generator w1 has the same sign as legacy |
| `ac.utils.plotters.sufficient_summary(xx[mask].dot(W1), u[mask])` | `ac.plotting.plot_sufficient_summary(sub, X=X[mask], f=u[mask], dims=1)` and `dims=2` | physical (log) X; mask from a seeded `rng.choice` |
| `pn.DataFrame.from_csv(...).as_matrix()` | `np.loadtxt(..., delimiter=',', skiprows=1)[:,1:]` | tested |

## 7. Associated .py files

- `gradients.py`: keep and port.
  - Rename it (e.g. `hartmann.py`) so it does not shadow `active_subspaces.gradients` conceptually. This is optional; it does not collide at import.
  - Replace `0.0*np.empty` with `np.zeros`.
  - Remove the module-level `lb`, `ub` and the normalized-gradient scaling. Return ∂f/∂(log x) = `x * df/dx`, with signature `uavg_grad(logx, l=1.0)`.
  - Keep `l` and `mu0` as keyword arguments.
  - No numpad involved.

## 8. Update plan

1. Text fixes: the outer→inner product (as in published Eq. 11, p.315), ∇_z, the u/U notation, the momentum equation (use published Eq. 21 or label it the inductionless form), citations (published SAM version; CDW title), and "adjoint gradients via Drekar".
1b. Bring the dimensional analysis up to the published version: the units table with ∇p0 and μ0 (Eq. 22), four Π groups (Eq. 24), Rm in B*_ind (Eq. 40), and "4 unitless parameters" in cell 11. Settle the affiliations (§11).
2. Port `gradients.py`: `np.zeros`, gradients w.r.t. log inputs, no globals.
3. Convert the notebook to py3. Use explicit imports and named Hartmann/generator ranges, and add a seeded rng.
4. Hartmann: tensor GL rule plus `ac.active_subspace(quad_weights=)` for u and B; v1 plots.
5. Generator: numpy loading, optional dedup, `ac.active_subspace(..., nboot=500, rng=)`, and `plot_eigenvalues(interval="minmax")`.
6. Add a data note (markdown cell or `MHD/README.md`): column names, units, duplicate, index gaps, provenance.
7. Execute headless and commit the outputs.

## 9. Acceptance criteria

- Headless execution in `.venv` takes < 60 s, with imports limited to numpy, scipy, matplotlib and active_subspaces.
- Hartmann u: λ1, λ2 = 36.57 and 1.2275 (±0.1%), λ3–5 < 1e-12·λ1. w1 ≈ ±[−.572, 0, .802, .062, −.160].
- Hartmann B: λ1, λ2 = 0.08626 and 0.006354 (±0.1%). w1 ≈ ±[−.381, 0, .663, −.493, .414].
- Generator u (483 rows): λ = [28.98, 0.928, 0.202, 0.0306, 7.3e-5] (±0.5%); λ5/Σλ = 2.4e-6. w1 ≈ [.83, −.02, −.56, .04, −.05].
- Generator B: λ = [4.156e-3, 4.94e-4, 1.10e-4, 1.58e-5, 4.9e-11]; λ5/Σλ = 1.0e-8. w1 ≈ [.49, −.12, −.30, .73, −.36].
- Dedup changes these by < 1%.
- Figures: per QoI and problem, 1 eigenvalue plot, 1 two-panel eigenvector plot, and 1-D and 2-D summary plots (16 total, as in legacy).
- A markdown note states that the Hartmann eigenvalues are 32× the 2016 figure values (and the published Fig. 2A).
- The cell 0 and cell 11 dimensional analysis lists 4 Π groups (Re, Ha, Rm, pressure gradient), matching published Eqs. (24) and (39)–(40).
- Hartmann B_ind: either reproduce published Fig. 3B (after the author answers §11.1), or carry a note saying it differs.

## 10. Prototype results

Scripts are in `prototypes/g1/` (`mhd_proto.py`, using a copy of `gradients.py`).

- Gradient check against FD: max relative error 5.6e-10.
- Hartmann on the 11^5 rule in v1, using physical log-gradients `G_norm/((ub−lb)/2)`: u λ = [36.57, 1.2275, 4e-29, …]; B λ = [0.08626, 0.006354, 1e-31, …]. Divided by 32 these give 1.14 and 0.0384, which match the legacy figure (λ1 ≈ 1, λ2 ≈ 0.04), confirming the bug. Runtime 0.2 s.
- Generator in v1 with nboot = 500: the eigenvalues in §9 and the λ5 ratios match the paper exactly. The legacy figure λ1 ≈ 29 matches. The minmax bootstrap band for λ1(u) is [21.0, 36.2].
- Published-figure check, by reading PDF pp.320–321 at 200 dpi: Fig. 2A has λ1 ≈ 1.4 and λ2 ≈ 0.05 (ratio ≈ 28; code /32: 1.14, 0.038), and Fig. 2B evec1 ≈ [.57, 0, −.80, −.06, .17] matches ours up to sign. Fig. 3A has λ1 ≈ 3 and λ2 ≈ 0.07; Fig. 3B evec1 ≈ [.38, 0, −.56, .49, −.56] does **not** match ours ([.381, 0, −.663, .493, −.414] after a sign flip). `prototypes/g1/mhd_bvar.py` scans ℓ ∈ [0.25, 2] and B0 powers {0, 1, 2}: ℓ = 0.4 gives w1 [.380, 0, −.548, .492, −.560] but λ1/λ2 = 170. μ0 only rescales. The Hartmann u range is 0.57–19.2 and B 0.004–1.20, consistent with the axes of published Figs. 2C and 3C.
- Data forensics: the duplicate row, the index gaps and the cross-file label offset were found as described in §5. A quadratic-in-log fit gives similar residuals inside and outside the offset block in both files, so each file is internally consistent.

## 11. Open questions for the author

1. Hartmann B_ind: how was published Fig. 3 computed? Its w1 components 3 and 5 and its eigenvalue scale differ from `gradients.py` with ℓ = μ0 = 1. Was a different ℓ (half-width?) or μ0 used, or a different B_ind expression? The "three-order-of-magnitude" sentence on p.319 is contradicted by Fig. 2A (ratio ≈ 28): erratum? Fig. 2A also shows the 1/32 eigenvalue scaling.
2. Uavg file rows 47 and 48 (labels 49 and 50) are identical. Is one run missing or mislabelled? Can the original Drekar output be recovered?
3. May the generator data be redistributed under the repo license? Who should be credited (Sandia, Drekar)?
4. Show the paper's full governing equations (published Eq. 21 with the induction equation), or keep a simplified momentum equation with a label?
5. Affiliations: keep the 2016 Mines affiliations, or switch to the published ones (CU Boulder; Sandia/UNM)?
6. The published p.316 says the active subspace "has dimension at least n + 1". Confirm it should read "at most" (erratum).
