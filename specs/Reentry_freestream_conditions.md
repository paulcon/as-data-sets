# Reentry_freestream_conditions: update spec

Kind: **stored data**: EXPERT reentry simulations (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `Reentry_freestream_conditions/reentry.ipynb` | paper: the notebook cites "A.F. Cortesi, P.G. Constantine, T.E. Magin, P.M. Congedo, *Heat flux based reconstruction of freestream conditions in atmospheric reentry flows*, Submitted, 2016". **No publication with that title was found** (Crossref, HAL listings, web). The README gives no paper. The published JCP 2020 paper by the same authors (PDF checked) studies a **different configuration** (HEG cylinder, 8 inputs); see §3 for the settled citation. The EXPERT model comes from Tryoen et al., AIAA J. 52(10):2190–2197 (2014), doi:10.2514/1.J052831 (no local copy; not checked). | m = 7 (3 uniform freestream/catalysis + 4 Gaussian log reaction rates) | QoI: stagnation-point pressure p_st and heat flux q_st on the EXPERT vehicle | legacy API: `Subspaces.compute(X, f, sstype='QPHD')`, `.partition(2)`, `utils.misc.BoundedNormalizer`, `utils.plotters.eigenvalues/eigenvectors/sufficient_summary` (imported as `asub`) | py2 issues: no `print`; kernelspec and pandas only

## 1. Summary
Value-only AS (global quadratic, "QPHD") on 7460 samples. The port is trivial except for one method issue:
1. **The text and code disagree on the second moment.** The markdown (cell 0, step 4) correctly says C = bbᵀ + H E[xxᵀ] Hᵀ with a diagonal E[xxᵀ]. The legacy `qphd_subspace` sets a scalar γ = 1 for *all* coordinates whenever max(X) > 1. Uniform coordinates should get 1/3. The legacy plots therefore use Σ = I. With the correct Σ = diag(1/3,1/3,1/3,1,1,1,1), λ₂ drops 3× (p_st) and 2.8× (q_st), and w₁ for q_st changes by about 0.04.
2. v1 has only `Uniform` or `Gaussian` spaces, **no mixed product space**. `ac.Gaussian(0, I)` on hand-normalized X reproduces 2016 exactly but declares the wrong density for 3 of the 7 inputs.
3. Cell 7 says the p_st λ₂ is "an order of magnitude smaller" than λ₁. It is 83× (Σ=I) or 244× (correct Σ), i.e. about two orders.
4. The quadratic fit for q_st has R² = 0.79 only. The notebook does not report it, and the QPHD subspace for q_st is accordingly rough.
5. The dev branch `Reentry/` is the **same model and study** but a **different design** (150-point quasi-random, uniform rates). Its notebook mislabels its own input distributions. Details in §5.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| reentry.ipynb | 560 KB | nbformat 4, python2 kernel | analysis | — | 10 cells, 8 PNG outputs |
| SA_lhs_all.txt | 1.05 MB | CSV, header `run, p_{inf}, M_{inf}, gamma, r1, r2, r3, r4, p_{st}, q_{st}` (spaces after commas), 7460×10, `%e` with 7 significant digits | inputs (physical) + 2 outputs | `pn.DataFrame.from_csv` | No NaNs, no duplicates. run = 1..7460. p∞ ∈ [16.30, 24.30], M∞ ∈ [13.70, 17.30], γ ∈ [0.001, 0.002], all uniform. r1..r4 have mean ≈ 0 and std 0.1200, 0.1200, 0.1000, 0.0999, so Gaussian, untruncated (|r| ≤ 3.6σ). p_st 4.0e3–9.7e3, q_st 1.5e5–5.0e5. |
| *(dev only)* Reentry/reentry.txt | 35 KB | CSV, header `Run,p_{inf},M_{inf},Gamma,r1,r2,r3,r4,wall-press,heat-flux`, 150×10 | see §5 | — | not on master |
| *(dev only)* Reentry/active-subspace-reentry.ipynb | 333 KB | python2 | OLS + local-linear AS | — | not on master |

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "[1] A.F. Cortesi, P.G. Constantine, T.E. Magin, and P.M. Congedo. _Heat flux based reconstruction of freestream conditions in atmospheric reentry flows_. Submitted, 2016" | Not found as published. **Checked the published Cortesi, Constantine, Magin, Congedo, J. Comput. Phys. 407 (2020) 109079, doi:10.1016/j.jcp.2019.109079: it is NOT this study.** It covers a hypersonic flow around a **cylinder in the HEG I configuration** (accepted manuscript p. 4; page numbers below are from the manuscript), solved with the COSMIC code and γ = 1 (fully catalytic, p. 12). Its **8 inputs** are ρ∞ ∈ [1.237e-3, 1.856e-3] kg/m³ and u∞ ∈ [4764.8, 7147.2] m/s, both uniform (Table 3, p. 22), plus 6 log₁₀ reaction pre-exponentials A1–A6, normal with σ 0.12, 0.12, 0.10, 0.15, 0.15, 0.12 (Table 2, p. 21). It uses a 576-point LHS (§5, p. 36) and a global linear model (its Alg. 2 = book Alg. 1.3). The notebook's data have p∞, M∞, γ ∈ [0.001, 0.002], 4 rates and 7460 runs. Only the first two rates (NO+O, NO+N, σ 0.12) coincide. The JCP paper itself credits the EXPERT approach to Tryoen et al. [9] (p. 3): **Tryoen, Congedo, Abgrall, Villedieu, Magin, "Bayesian-based method with metamodels for rebuilding freestream conditions in atmospheric entry flows", AIAA J. 52(10):2190–2197 (2014), doi:10.2514/1.J052831**, which is also the dev notebook's citation. Related: Cortesi, Congedo, Magin et al., AIAA 2016-4196, doi:10.2514/6.2016-4196; A.F. Cortesi, PhD thesis, Univ. Bordeaux, 2018 (theses.fr 2018BORD0021). | **Settled recommendation:** [1] Tryoen et al. 2014 as the source of the EXPERT model and the input ranges and distributions. Describe the data as "simulations by P.M. Congedo / A.F. Cortesi (INRIA Bordeaux), unpublished as a stand-alone AS study". Optionally add "see Cortesi et al., JCP 2020, for a related AS-based freestream calibration on the HEG cylinder (different inputs)". Do **not** cite JCP 2020 as the source of this data. Drop "Submitted, 2016". | confirmed (JCP 2020 PDF); Tryoen content unverified (no copy) |
| 0 | "[[2]] … _Active subspaces in theory and practice_" | Title is "Active subspace **methods** in theory and practice: applications to kriging surfaces". | Fix the title. | confirmed |
| 0 | "fits a global quadratic surrogate … 4. … $\mathbf b\mathbf b^T + \mathbf H\left(\int \mathbf x\mathbf x^T\rho\, d\mathbf x\right)\mathbf H^T$ … the integral simplifies to the diagonal matrix whose $i^{th}$ element is the variance" | **The math is right; the code is not.** Legacy `qphd_subspace` uses `gamma*np.dot(A, A.T)` with scalar γ = 1 when max(X) > 1. The uniform coordinates' variance is 1/3, not 1. (v1 QPHD uses the space's Σ, but v1 has no mixed space.) Also note the identity needs E[x] = 0. | Implement C with Σ = diag(1/3,1/3,1/3,1,1,1,1) (§6), or say explicitly that Σ = I is used as an approximation. Add "(assuming E[x] = 0)". | confirmed (legacy source `upstream/active_subspaces/subspaces.py` l. 275–289) |
| 0 | method name | The global-quadratic estimator is **not** an algorithm in Constantine 2015: the book has Alg. 1.1 (gradients), 1.2 (local linear) and 1.3 (global linear) (pp. 6–7). It is the 2016 library's `QPHD` (in v1, `estimator="qphd"`), which follows Li 1992 (pHd) as documented in v1 `subspaces.QPHD`. | Name it, cite Li (1992) or the library, and note that v1 warns it is valid only for Gaussian inputs. Alternatively use book Alg. 1.3 (OLS), as the JCP 2020 paper does. | confirmed (book) |
| 0 | "chemistry on 5 air species (N, O, NO, N$_2$, and O$_2$), and the catalyticity of the vehicle's surface" | Consistent with Tryoen et al. 2014 (air-5, EXPERT); not re-checked (paywalled). | — | unverified |
| 0 | table "Freestream Pressure \| … \| 16.3 \| 24.3", "Freestream Mach Number \| 13.7 \| 17.3", "γ \| .001 \| .002" | Matches the data. **No units** (p∞ in Pa presumably; p_st in Pa, q_st in W/m² presumably). | Add units after the author confirms. | data confirmed; units unverified |
| 0 | "Gas Reaction \| Distribution of $\log_{10}(k_r)$ \| Normal \| σ_r: 0.12, 0.12, 0.10, 0.10" | Matches the data std. The data stores the **zero-mean perturbation** of log₁₀ k_r about the nominal rate, which the table does not say. | "log₁₀(k_r / k_r,nominal) ~ N(0, σ_r²)". | confirmed (data) |
| 2 | `in_labels = [… r'log($r_1$)', …]` | log base and meaning unclear. | `r'$\log_{10}(k_1/k_1^0)$'` etc. | — |
| 7 | "We can see the second eigenvalue is an order of magnitude smaller than the first" | p_st λ₁/λ₂ = 4.14e6/4.97e4 = **83** (legacy Σ=I); 244 with the correct Σ. | "about two orders of magnitude smaller". | confirmed |
| 7 | "Both eigenvectors place all their weight on the freestream conditions; no weight is placed on γ or the reaction rates" | w₁ = (0.643, 0.766, ~0, ~5e-4 …), w₂ = (0.766, −0.643, 0.003, −0.005 …). Correct. | — | confirmed |
| 9 | "The first two eigenvectors again place most of their weight on freestream conditions; … only the second has appreciable weight on γ" | w₁ = (0.26, 0.96, 0.06, 0.02, …); w₂ γ-component 0.41. Correct. | Add: the quadratic fit for q_st has R² = 0.79, so the estimated subspace is itself uncertain. Optionally cross-check with local-linear gradients (as the dev notebook does). | confirmed |
| 9 | inverse-problem discussion | Reasonable. The JCP 2020 paper (cylinder) carries out exactly this kind of calibration from p_st and q_st with 1-D active variables (§5.1, Fig. 8, p. 37). It also finds (p. 38) "the relative insensitivity of pressure and heat flux to the reaction rates", consistent with cells 7 and 9. | Optionally cite JCP 2020 as an example of the follow-up. | confirmed (JCP 2020) |
| 0 | normalization (cell 3) | The same mixed normalization [−1,1]^k × R^6 is used in JCP 2020 (p. 36: "shift and scale the parameter space to [−1, 1]² × R⁶ … uniform density … times a multivariate standard Gaussian"). This supports option B (§6: mixed Σ). | — | confirmed |

## 4. Code review
- `import active_subspaces as asub`: harmonize to `ac`.
- `Xn = data[:,3:7]` is a view, and `Xn[:,:2] /= .12` mutates `data` in place. Harmless here, but use `.copy()` or build normalized arrays explicitly.
- `pst`, `qst` are reshaped to (M,1); v1 takes (M,).
- `ss_p.partition(2)` mutates. v1 uses `n=2`.
- QPHD is deterministic, so no seeds are needed.
- pandas `from_csv`/`as_matrix` were removed in pandas 1.0.
- No wasted computation. QPHD on 7460×36 least squares takes milliseconds.

## 5. Data management review
- **SA_lhs_all.txt:** 1.05 MB, header with LaTeX-ish names and spaces, 7 significant digits. Inputs are physical except r1..r4, which are zero-mean log-perturbations. Normalization is not stored. No units. The name says LHS; 7460 is an odd size (failed runs removed from 7500?). Provenance: P. Congedo / A. Cortesi, INRIA Bordeaux (commit f04a1ff, R. Howard, 2016). No license.
- **dev `Reentry/` vs. master `Reentry_freestream_conditions/`:** **same study, different data set.**
  - Same model and setup: the same 7 inputs, the same uniform bounds for p∞, M∞, γ, and the same two outputs (wall-press = p_st, heat-flux = q_st; output ranges agree: p 4.4e3–9.6e3, q 1.6e5–4.4e5).
  - Different design: dev has 150 points on a dyadic lattice (spacing = range/256 in every column, quasi-random/Sobol-like), with **r1..r4 uniform on ±2σ** (ranges ±0.238, ±0.238, ±0.197, ±0.198; std = range/√12). Master has 7460 LHS points with **Gaussian** r1..r4.
  - No common rows (nearest-neighbour distance ≥ 0.46 in standardized units).
  - The dev notebook contradicts itself and its data. Its table says r ~ Gaussian with σ = 0.12, **0.10**, 0.10, 0.10. Its code uses `UnboundedNormalizer` with σ = 0.12, 0.12, 0.1, 0.1. Its data is uniform with half-widths 2 × (0.12, 0.12, 0.10, 0.10).
  - The dev notebook does OLS + local-linear gradients (`nboot=2000`); master does QPHD.
  - **Recommendation:** keep `Reentry_freestream_conditions` as the single canonical directory. Do not port dev `Reentry/` as a separate notebook. If the 150-point design is wanted (e.g., to show OLS and local-linear AS on a small budget), add `reentry150.csv` to this directory with a note that its rates are uniform on ±2σ, and add a short section. Author decision (§11).
- **Tested pandas-free loader:**
```python
import numpy as np
with open("SA_lhs_all.txt") as fh:
    header = [h.strip() for h in fh.readline().split(",")]
d = np.loadtxt("SA_lhs_all.txt", delimiter=",", skiprows=1)        # (7460, 10)
lb, ub = np.array([16.3, 13.7, 0.001]), np.array([24.3, 17.3, 0.002])
sigma = np.array([0.12, 0.12, 0.10, 0.10])
Xphys = d[:, 1:8]                                                  # p_inf, M_inf, gamma, r1..r4
X = np.hstack([2 * (Xphys[:, :3] - lb) / (ub - lb) - 1, Xphys[:, 3:] / sigma])   # normalized
pst, qst = d[:, 8], d[:, 9]
```

## 6. Migration to active_subspaces v1
v1 has no mixed uniform/Gaussian space (`active_subspace(space=…)` accepts `Uniform` or `Gaussian`; migration.md E4 says "v1 has no input space for that mix"). Two options, both tested:

| legacy call | v1 replacement | notes |
|---|---|---|
| `BoundedNormalizer(lb, ub)` + `/σ` | hand normalization (loader above) | |
| `ss.compute(X=X, f=pst, sstype='QPHD'); ss.partition(2)` | **(A, reproduces 2016)** `ref = ac.Gaussian(np.zeros(7), np.ones(7), names=…)`; `ac.active_subspace(X=X, f=pst, space=ref, estimator="qphd", n=2, qoi="p_st")` | Bitwise-equivalent to legacy (eigenvalues identical, W to 6e-15). Treats the uniform coordinates as N(0,1), as legacy did. Must be labelled an approximation; geometry and inverse maps from this result are wrong for the uniform inputs. `ac.Gaussian.reference(7)` has no `names=` keyword, so use the explicit constructor. |
| same | **(B, correct second moment, recommended)** numpy: fit `[1, x, x², x_j x_k]` by `lstsq`, then `C = g gᵀ + H Σ H` with Σ = diag(1/3,1/3,1/3,1,1,1,1), then `np.linalg.eigh`. Wrap with `ac.subspace_from_basis(W[:, :2] …)` only if v1 plots are wanted (that result has NaN eigenvalues). | About 12 lines (prototype in §10). Matches v1's own QPHD formula `g gᵀ + H Σ H`. Show A and B side by side, or B only. |
| `plotters.eigenvalues(ss.eigenvals)` | `ac.plotting.plot_eigenvalues(sub)` for (A); matplotlib `semilogy` for (B) | |
| `plotters.eigenvectors(ss.W1, in_labels=…)` | `ac.plotting.plot_eigenvectors(sub)` | Sign rule: w₁ largest component M∞ (+), w₂ largest p∞ (+). Same as the 2016 figures. |
| `plotters.sufficient_summary(X.dot(W1), f)` | `ac.plotting.plot_sufficient_summary(sub, X=X, f=pst)` (A: X is already "physical" for the reference space); for (B) matplotlib scatter of `X @ W[:, :2]` | |

Also consider: an author decision to add a product/mixed `InputSpace` to v1. Out of scope here; listed in §11.

## 7. Associated .py files
None.

## 8. Update plan
1. Convert the kernel to python3, add the pandas-free loader, rename `asub` → `ac`.
2. Implement the QPHD subspace. Option B (correct Σ) as the main result; option A (v1 `Gaussian` reference = 2016) optionally, as a one-cell comparison.
3. Plots: eigenvalues, first 2 eigenvectors, 1-D/2-D summary plots for p_st and q_st. Report the quadratic-fit R² for each QoI.
4. Text fixes: the citation block (§3: Tryoen et al. 2014 as [1]; JCP 2020 only as "related, different configuration"), CDW title, "two orders of magnitude", the meaning and units of r_i, units for p∞, p_st, q_st, and the Σ note in step 4. Name the estimator as the library's global-quadratic (QPHD) estimator, not a book algorithm.
5. (Author decision) Fold in the dev 150-point design or drop it; never port dev `Reentry/` as a separate directory.

## 9. Acceptance criteria
- Executes headless from `Reentry_freestream_conditions/` in .venv in < 30 s; numpy/scipy/matplotlib/active_subspaces only.
- Option A eigenvalues (must equal legacy): p_st 4.1422e6, 4.9687e4, 14.224; q_st 9.7785e9, 4.8538e8, 2.6505e6 (±1e-6 rel). Legacy PNG q_st λ₁ ≈ 1e10, λ₂ ≈ 5e8, λ₃ ≈ 2.6e6, which matches.
- Option B eigenvalues: p_st 4.0355e6, 1.6567e4; q_st 9.1248e9, 1.7390e8. q_st w₁ ≈ (0.306, 0.948, 0.083, 0.012, 0.013, −0.005, 0.002).
- Quadratic-fit R²: p_st 0.99968, q_st 0.788.
- Figures: per QoI, 1 eigenvalue plot, 1 eigenvector plot (2 vectors), 2 summary plots (1-D and 2-D).

## 10. Prototype results
Script: `prototypes/g3/reentry.py Reentry_freestream_conditions/SA_lhs_all.txt`
- v1 `active_subspace(estimator="qphd", space=Gaussian(0, I))` equals the numpy Σ=I computation (eigenvalues identical, |W₁:₂| to 6e-15) and the legacy figures (cell 8 PNGs, checked visually: q_st eigenvalues and eigenvectors). No warnings.
- Correct Σ (option B) vs. Σ=I: p_st λ₂ 4.97e4 → 1.66e4 and w₁ unchanged to 1e-3. q_st λ₂ 4.85e8 → 1.74e8 and w₁ (0.264, 0.962, 0.063, …) → (0.306, 0.948, 0.083, …).
- OLS direction for comparison: p_st (0.644, 0.765, …), q_st (0.329, 0.940, 0.094, …), closer to option B.
- Dev data analysis (distribution, lattice spacing, overlap): see §5. Extracted with `git archive origin/dev Reentry` into `.../scratchpad/g3/dev/`.

## 11. Open questions for the author
1. Citation: JCP 2020 is confirmed to be a different (HEG cylinder, 8-input) study. Was the EXPERT AS study ("Heat flux based reconstruction…") ever published, e.g. in Cortesi's 2018 thesis? If not, confirm Tryoen et al. 2014 as [1] (I have no copy, so the input ranges and distributions are not checked against it).
2. Units of p∞, p_st, q_st (Pa, Pa, W/m²?), and confirm that r_i = log₁₀(k_r/k_r⁰).
3. QPHD second moment: use the correct mixed Σ (option B, which changes the 2016 numbers), or keep Σ = I to reproduce the 2016 figures?
4. Should v1 get a mixed/product input space so that cases like this (and the 2016 borehole) are first-class?
5. Dev `Reentry/` (150-point quasi-random design, uniform rates): fold into this directory as a second data set, or drop it?
6. Why 7460 samples (LHS of 7500 minus failures?)
7. README: add a paper link (after Q1) and fix "atmostpheric" → "atmospheric".
