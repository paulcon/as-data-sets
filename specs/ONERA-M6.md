# ONERA-M6: update spec

Kind: **stored data**: SU2 runs with adjoint gradients (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `ONERA-M6/active-subspace-oneraM6.ipynb` | paper: **Lukaczyk, Palacios, Alonso, Constantine** (order per the published title page), "Active subspaces for shape optimization", 10th AIAA MDO Conf., National Harbor MD, 13–17 Jan. 2014, AIAA 2014-1171, doi:10.2514/6.2014-1171 (published PDF checked). The book (Constantine 2015, §5.3.2 pp. 84–86) describes a **different** ONERA-M6 data set (60 parameters, no gradients); see §1 | m = **50** FFD parameters, each in [−0.05, 0.05] | QoI: C_L and C_D (SU2), with adjoint gradients | legacy API: `Subspaces.compute(df=…, nboot=500)`, `.partition(n)`, `.eigenvals/.W1`, `utils.misc.BoundedNormalizer`, `utils.response_surfaces.RadialBasisApproximation(2)` + `.Rsqr`, `domains.BoundedActiveVariableDomain(ss).vertY`, `utils.plotters.sufficient_summary/zonotope_2d_plot` | py2 issues: minor (`j is 0`; kernelspec; pandas)

## 1. Summary
Gradient-based AS on 297 samples, an R²-vs-dimension study with RBF surrogates, sufficient-summary plots, and zonotope plots. It ports cleanly and runs in about 3 s. Findings:
1. **The README says "60-parameter"; the data, the notebook and the linked AIAA paper have 50.** The AIAA paper (§IV, p. 9) says "X = [−0.05, 0.05]^50 … a 300-point Latin hypercube design … lift coefficient, drag coefficient, and their gradients … computed with an adjoint-based strategy". The file has 297 of those 300 runs. The README's "60" comes from **book §5.3.2** (p. 85): "60 free-form deformation parameters … ±0.05 … we do not have access to gradients … 248 runs", analysed with a global linear model (Alg. 1.3). That data set is not in the repo. The README should say 50 and keep the AIAA link.
2. As in NACA0012, **the gradients are physical** (normalized OLS coefficients / mean gradient = 0.051 for lift, 0.047 for drag). v1 with `Uniform(±0.05)` gives eigenvalues 2.5e-3 × legacy; the eigenvectors are identical.
3. The R² study quotes in-sample RBF R² ("drag starts … around .965"). The v1 RBF LOO R² is **0.70** (drag, n=1), 0.88 (n=2), 0.93 (n=3); lift is 0.980 (n=1). The cell-12 text must be rewritten. The qualitative conclusion (lift 1-D, drag ≥ 2-D) survives and is clearer.
4. v1's default EVG picks n = 1 for drag (λ₁−λ₂ = 5.24e-5 vs λ₂−λ₃ = 4.97e-5). `EVG(gap="log")` picks 49. Fix **n = 2 explicitly** for drag.
5. The notebook's cell-0 author order (Lukaczyk, Palacios, Alonso, Constantine) **matches** the published paper. Its reference [1] (Lukaczyk, Constantine, Palacois, Alonso) does not, and it misspells Palacios. `figs/.DS_Store` is tracked junk.
6. The paper's own dimension choices: lift 1-D (GPR training error 3 %); drag **5-D** by training-error decay, with 2-D used for visualization (training error 5.1 %, Fig. 7). The notebook's "drag 2-D" is defensible but should mention the paper's 5-D choice.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| active-subspace-oneraM6.ipynb | 474 KB | nbformat 4, python2 kernel | analysis | — | 17 cells. Cell 15 lists 4 figure objects but only 2 PNGs (2 blank figures from the legacy `zonotope_2d_plot`) |
| ONERA-M6.txt | 1.06 MB | CSV, header, 297×153 | Run, x01–x50 (physical, ±0.05), Lift, Drag, dlift01–50, ddrag01–50 | `pn.DataFrame.from_csv` | No NaNs, no duplicate rows. X std 0.02886 ≈ 0.05/√3, so uniform. |
| baseline_ffdbox_original.png | 59 KB | PNG 1200×1067 | illustration, cell 0 `<img>` | HTML in markdown | keep |
| baseline_ffdbox_deformed.png | 66 KB | PNG 1200×1067 | illustration, cell 0 | HTML | keep |
| baseline_surface_flow.png | 94 KB | PNG 1200×1067 | illustration, cell 0 | HTML | keep |
| figs/.DS_Store | 6 KB | macOS metadata | none | — | **Delete** (`git rm --cached`). `.gitignore` already lists `.DS_Store`. `figs/` is otherwise empty and unreferenced. |

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | authors "Trent Lukaczyk … Francisco Palacios, The Boeing Company … Juan Alonso … Paul Constantine" | **The order matches** the published title page (Lukaczyk, Palacios, Alonso — Stanford; Constantine — Colorado School of Mines) and book ref. [87]. Crossref's metadata order (Constantine second) is wrong. At publication Palacios was "Engineering Research Associate", Stanford, not Boeing. | Keep the order. Use the affiliations at publication (Stanford), or mark Boeing as current. | confirmed (published PDF p. 1) |
| 0 | "[[1]] T.W. Lukaczyk, P. Constantine, F. Palacois, and J.J. Alonso" | Wrong order and "Palacois" → **Palacios**. | "T.W. Lukaczyk, F. Palacios, J.J. Alonso, and P.G. Constantine … National Harbor, MD, AIAA 2014-1171". | confirmed (published PDF) |
| 0 | "[[2]] … _Active subspaces in theory and practice_" | The title is "Active subspace **methods** in theory and practice: applications to kriging surfaces". | Fix the title. | confirmed |
| 0 | "The wing's shape is determined by 50 free-form deformation (FFD) parameters, all in the range $[-.05, .05]$" | Confirmed by paper §IV p. 9 and the data. The paper says the FFD control points of a Bézier box are the design variables (§II, eq. 2), and gives the flow as Euler (SU2), M∞ = 0.8395, α = 3.06° (p. 3). It does not say which control points or directions. The sample is a 300-point LHS (the file has 297). | Add: "Euler flow at M = 0.8395, α = 3.06°; 300-point Latin hypercube (297 in the file) with adjoint gradients". | confirmed (published PDF) |
| 0 | "we approximate gradients using adjoint methods available in SU2" | Plausible (SU2 continuous adjoint). It does not say the gradients are w.r.t. physical FFD variables. | Add "with respect to the physical FFD variables". | confirmed (physical, from data) |
| 0 | "$\mathbf W = [\mathbf U\ \mathbf V]$" | Notation differs from the other notebooks (W₁, W₂). Not wrong. | Optionally harmonize. | — |
| 10 | "a 2-dimensional active subspace looks more likely for drag" | Consistent with the data: λ₁/λ₂ = 1.82, λ₂/λ₃ = 4.56 for drag; λ₁/λ₂ = 115 for lift. The paper (pp. 9–10) picks **5** dimensions for drag from training-error decay and restricts to 2 for visualization. | Add one sentence citing the paper's 5-D choice. | confirmed (published PDF) |
| 10–12 | "R^2 values of quadratic radial basis response surfaces"; "The value for drag starts fairly high, around .965, and increases significantly from 1 dimension to 2" | The legacy `.Rsqr` is **in-sample**. The v1 `loo_r_squared` is leave-one-out: drag 0.703 / 0.884 / 0.932 / 0.940 … 0.968 (n = 1…10); lift 0.980 → 0.987. The paper reports neither: it uses GPR **training RMS error** scaled to the output range (Fig. 5: lift 3 % at n=1; drag 2-D 5.1 %). The R² study is the notebook's own. | Rewrite with the v1 numbers and say they are LOO and not from the paper. "Quadratic radial basis" → "RBF surrogate with a quadratic polynomial mean". | confirmed |
| 12 | "lift start and remain very high, close to 1" | v1 LOO lift = 0.980 at n = 1. That is still high. | Quote the number. | confirmed |
| 14 | "endpoints are $\pm \|\mathbf u\|^T\mathbf 1$" | Correct for [−1,1]^m (Constantine 2015 §4.2). | — | confirmed |
| 16 | "[[2]][R2] and [3] discuss" | [3] has no link: Jupyter resolves reference links per cell and R3 is not defined in cell 16. | Add `[R3]: …` to cell 16. | confirmed |
| 16 | discussion of constrained optimization "[[1]] discusses these kinds of calculations in some detail" | Correct: the paper minimizes C_D subject to C_L ≥ 0.2864 (eq. 1). It uses inverse maps (eqs. 20–23) and drag/lift polytopes in 2-D (Figs. 8–9). | Optionally quote the constraint C_L ≥ 0.2864. | confirmed (published PDF) |
| 0 | references | The book (ref [3]) §5.3.2 has a different ONERA-M6 study (60 parameters, 248 runs, no gradients). A reader following [3] will find numbers that don't match. | One sentence: "the book's §5.3.2 uses a different, gradient-free 60-parameter data set". | confirmed (book p. 85) |

## 4. Code review
- `F = F_lift if j is 0 else F_drag` (cells 11, 13): `is` with an int literal raises SyntaxWarning in py≥3.8. Use `==`, or loop over `zip(subs, (F_lift, F_drag))`.
- `plt.semilogy(range(1, 51), …)`: hard-coded 50; use `m`.
- `RS` is reused and `subspaces[j].partition(i+1)` mutates in the loop. v1 is functional (`with_dimension`).
- Unseeded bootstrap (`nboot=500`). Use `rng=`.
- Cell 15 produces 2 empty figures (legacy plotter). v1 `plot_active_domain` returns `(fig, ax)`.
- pandas `from_csv`/`as_matrix` were removed in pandas 1.0.
- Cost is trivial: 10 RBF fits × 2 QoIs take ≈ 2.5 s in v1.

## 5. Data management review
- 1.06 MB CSV, `%.16e`, header present, no units (coefficients, dimensionless; FFD displacement units unknown).
- Inputs and gradients are both **physical**. The bounds ±0.05 are stored only in the notebook.
- No redundancy and no NaNs. Run = 1..297.
- Provenance/license: none. Added in commit 79789d9 (R. Howard, 2016); presumably from T. Lukaczyk's SU2 runs (unverified). The PNGs are SU2/Tecplot renderings. Fine to keep.
- Delete the tracked `figs/.DS_Store`.
- **Tested pandas-free loader:**
```python
import numpy as np
with open("ONERA-M6.txt") as fh:
    header = fh.readline().strip().split(",")
d = np.loadtxt("ONERA-M6.txt", delimiter=",", skiprows=1)          # (297, 153)
m = 50
X, F_lift, F_drag = d[:, 1:m+1], d[:, m+1], d[:, m+2]
G_lift, G_drag = d[:, m+3:2*m+3], d[:, 2*m+3:]                      # physical gradients
names = header[1:m+1]
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `BoundedNormalizer(-0.05, 0.05)` | `space = ac.Uniform(np.full(50, -0.05), np.full(50, 0.05), names=names)` | physical X |
| `ss.compute(df=G, nboot=500)` | `ac.active_subspace(grads=G, space=space, nboot=500, rng=0, qoi="Lift")`; for drag add `n=2` | Gradients are physical, so **not** `reference(50)`. Eigenvalues = 2.5e-3 × legacy (λ₁ lift 6.362 → 0.015905). Eigenvectors are identical, including sign: all w₁/w₂ first components and largest components agree. |
| `semilogy(ss.eigenvals)` (both QoIs on one axis) | `ac.plotting.plot_eigenvalues(sub_lift, ax=ax)` then `plot_eigenvalues(sub_drag, ax=ax)`, or plain `ax.semilogy(sub.eigenvalues)` | |
| `ss.partition(i+1)` | `ac.with_dimension(sub, n)` | |
| `RadialBasisApproximation(2).train(y, F); RS.Rsqr` | `ac.fit_surrogate(X=X, f=F, subspace=ac.with_dimension(sub, n), kind="rbf", degree=2).loo_r_squared` | LOO, so lower (see §3). New RBF likelihood. |
| `sufficient_summary(y, F)` | `ac.plotting.plot_sufficient_summary(sub2, X=X, f=F)` | physical X |
| `BoundedActiveVariableDomain(ss).vertY` | `ac.active_domain(ac.with_dimension(sub, 2)).vertices` | Exact zonotope vertices (100 for m=50, n=2); 2016 vertices were approximate |
| `zonotope_2d_plot(vertY, y=y, f=F)` | `ac.plotting.plot_active_domain(dom, points=ac.forward_map(sub2, X=X)[0], values=F)` or `plot_sufficient_summary(sub2, X=X, f=F, domain=dom)` | tested |

## 7. Associated .py files
None.

## 8. Update plan
1. Convert the kernel to python3, add the pandas-free loader, and `git rm --cached ONERA-M6/figs/.DS_Store`.
2. Port the subspaces (physical gradients, Uniform ±0.05, `rng`). Fix the drag dimension at 2 and add a markdown line on why (the linear-gap EVG is borderline).
3. Port the eigenvalue plot and the R² study (v1 LOO R², n = 1..10). Rewrite cells 10–12 with the new numbers.
4. Port the summary and zonotope plots (`plot_active_domain`).
5. Text fixes: author order, "Palacios", CDW title, [R3] link, "physical" gradients, sample count M = 297.
6. README: "60-parameter" → "50-parameter" (50 matches the data and the linked AIAA paper; 60 is the book's different, gradient-free study). This is outside this notebook; route it to the README owner.
7. Text: Euler, M = 0.8395, α = 3.06°, 300-point LHS (297 in the file); the paper's drag n = 5 choice; the book's §5.3.2 being a different data set; Palacios's affiliation at publication.

## 9. Acceptance criteria
- Executes headless from `ONERA-M6/` in .venv in < 30 s; numpy/scipy/matplotlib/active_subspaces only. The three PNGs render.
- λ₁, λ₂: lift 1.5905e-2, 1.3808e-4; drag 1.1601e-4, 6.3601e-5 (±1e-6 rel).
- |w₁ᵀw₁(legacy)| > 1 − 1e-12 for both QoIs. Legacy = eig(GᵀG/M), first component positive.
- LOO R² (rbf, degree 2): lift n=1 ≥ 0.97; drag n=1 ≈ 0.70, n=2 ≈ 0.88 (±0.02).
- Figures: 1 eigenvalue plot (2 curves), 1 R²-vs-n plot, 4 sufficient-summary plots, 2 zonotope plots with points, no blank figures.

## 10. Prototype results
Script: `prototypes/g3/airfoils.py ONERA-M6/ONERA-M6.txt 50 0.05` (full run 2.7 s).
- v1 / legacy eigenvalue ratio = 2.5000e-3. The legacy plot (cell 9 PNG) shows λ₁ ≈ 6.4 (lift) and ≈ 0.046 (drag), matching the unscaled-gradient computation.
- w₁, w₂ for lift and drag agree with legacy to < 5e-15, with the same sign.
- Default EVG: n = 1 for both. `EVG(gap="log")` gives n = 49 for drag (tiny tail eigenvalues), so it is unusable here.
- RBF LOO R², n = 1..10: lift 0.980, 0.983, 0.985, 0.986, 0.986, 0.987, 0.987, 0.987, 0.987, 0.987; drag 0.703, 0.884, 0.932, 0.940, 0.945, 0.951, 0.952, 0.955, 0.955, 0.968. Legacy in-sample (cell 11 PNG): lift ≈ 0.9987–0.9995, drag 0.964 → 0.9976.
- `active_domain(n=2)`: zonotope with 100 vertices. `plot_active_domain` and `plot_sufficient_summary` run under Agg.

## 11. Open questions for the author
1. Resolved: 50 parameters (AIAA paper). Is the book's 60-parameter gradient-free ONERA-M6 data (248 runs) available, and should it be added (e.g. as a second, global-linear-model example)?
2. Which 3 of the 300 LHS runs are missing (failed?). Which FFD control points and directions are x01–x50?
3. Resolved: the notebook's author order matches the published paper. Only reference [1] needs fixing.
4. Fixing n = 2 for drag by judgment (as now) vs. showing `ac.dimension_criterion` output: preference?
