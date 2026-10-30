# Subsurface_Permeability: update spec

Kind: **stored data**: ParFlow runs (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `Subsurface_Permeability/Subsurface_permeability.ipynb` | paper: Gilbert, Jefferson, Constantine, Maxwell, "Global spatial sensitivity of runoff to subsurface permeability using the active subspace method", Adv. Water Resour. 92 (2016) 30–42, doi:10.1016/j.advwatres.2016.03.020 | m = 100 KL coefficients (README 100, correct) | QoI: cumulative runoff volume, Dunnian (initially saturated) and Hortonian (initially unsaturated), 10,000 runs each | legacy API: `Subspaces.compute(sstype='OLS', nboot=500)`, `sstype='QPHD'`, `.partition`, `utils.plotters.{sufficient_summary,eigenvalues}`, `utils.response_surfaces.PolynomialApproximation(1|2|3)` | py2 issues: yes (`print` statements in cell 14; `i/3` used as an index).

## 1. Summary
The inputs are i.i.d. standard Gaussian KL coefficients. Statistical checks on the data: per-column mean |μ| ≤ 0.028, SD 0.987–1.016, max off-diagonal |corr| 0.041, pooled KS against N(0,1) p = 0.45. The notebook correctly uses them **unnormalized**, consistent with the paper's stated ρ ("m-variate Gaussian joint density", p. 33; "100 Gaussian random variables", p. 35). In v1 this is `Gaussian.reference(100)`, and OLS and QPHD are within their validity assumptions (Stein's lemma), so no warnings. Every legacy number I could check reproduces exactly: OLS λ₁, QPHD eigenvalues (in the figure), the max weight differences 0.017 / 0.037, and the Dunnian and Hortonian quadratic-subspace R². Main problems:
- **31% of Hortonian runoffs are exactly 0** (3,137 of 10,000). The log10 summary plot in cell 7 silently drops them (−inf), and its y-axis label does not say log10. The paper filters them explicitly (p. 37: "We filtered out the results from realizations yielding no runoff such that Fig. 7 represents only the non-zero runoff cases"). In v1, `plot_sufficient_summary` raises `input.nonfinite` on these values, so this is a hard migration break.
- **The paper's figures appear to use a different normalization from its stated ρ.**
  - Paper Figs. 6b and 7b (p. 38) show active-variable ranges of about [−1, 1] and [−0.85, 0.9]. With x ~ N(0, I) and ‖w‖ = 1, y ~ N(0, 1); the notebook and data give [−4.1, 3.5] and [−3.5, 3.8].
  - Rescaling each column to [−1, 1] by its sample min/max (a BoundedNormalizer on Gaussian data) reproduces the paper:
    - Dunnian y in [−0.99, 0.92];
    - Hortonian non-zero cases y in [−0.80, 0.91];
    - Hortonian w₁ = −0.646 and w₅ = −0.501, against Fig. 7a ≈ −0.65 and −0.50 (Gaussian scaling gives −0.627 / −0.515).
  - cos(Gaussian w, min/max w) = 0.9992, so the conclusions are unchanged.
  - Keep the Gaussian scaling, which is correct for ρ. Add a note that the paper's figures have a ~4.5× smaller active-variable scale and slightly different weights. Confidence: likely (inferred; the paper does not state its scaling).
- **Sign: the paper orients w = â/‖â‖.** Pp. 37–38: "In both runoff cases, the cumulative outflow increases directly with increasing active variable". Fig. 7a has a negative Hortonian w₁. The legacy notebook's Hortonian weights and summary plot (first component > 0, decreasing trend) are mirrored relative to the paper, and so is the v1 default sign. Orient by â.
- **The "linear subspace" rows of the Hortonian R² table are meaningless.** `ss_unsat.partition(3)` on an OLS subspace (rank 1) adds two arbitrary null-space vectors. Recomputing on another LAPACK gives 0.7412/0.7740 instead of the printed 0.7432/0.7783. The text's claim "improving from .74 to .82 … .78 to .89" compares against those arbitrary directions.
- The weight-difference plots (`ss_quad.W1 - ss.W1`) depend on both vectors sharing a sign convention. Under v1 the rule is different, so align signs explicitly.
- `nboot=500` bootstraps are computed and never used.
- The 25.5 MB input file is under the GitHub limits; keep it as text or convert it to `.npy` (8 MB).

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| Subsurface_permeability.ipynb | 1.2 MB | nbformat 4, python2 kernel | notebook | — | 18 PNG + 1 text output |
| inputs-ShortCorr.GW02.txt | 25.5 MB | space-separated, no header, 10000×100, `%.18e` | KL coefficients X ~ N(0, I) | `pn.DataFrame.from_csv(path, None, ' ', None)` | loads in 0.35 s with `np.loadtxt`. "ShortCorr" probably refers to the paper's correlation lengths (λx = λy = 50 m, λz = 5 m, "from the lower end of the range", p. 33, Table 2). gzip 11 MB; `.npy` 8.0 MB |
| RunOff02_Saturated_10k.txt | 240 KB | one value per line, trailing space | Dunnian runoff: cumulative volume over the 48-h simulation, in m³ (paper p. 35, 38) | `from_csv(header=None)` | range 45,219–46,168, mean 45,483 |
| RunOff02_Unsaturated_10k.txt | 240 KB | same; no final newline (`wc -l` = 9,999) | Hortonian runoff | same | range 0–1,345, mean 66.1; **3,137 exact zeros** |

## 3. Text and math review (vs. literature)
Checked against the published full text (AWR 92:30–42). Relevant locations:
- algorithm: §2.1, p. 32, eq. (5);
- KL parameterization: §2.3–2.4, p. 33, eqs. (7)–(13), Table 2;
- scenarios: §2.6, p. 35;
- results: §3, Figs. 6–9, p. 36–39.

The paper uses only the global linear (OLS) model. The QPHD extension and the R² table are the notebook's own additions; the paper reports no R².

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "the X_i are independent standard Gaussian random variables" / "ρ (a standard Gaussian distribution)" | consistent with the data (see §1) | — | confirmed (data) |
| 0 | "This heuristic tends to work well if f is monotonic in each component of x" | the sentence is the paper's own (p. 32, "This heuristic works well if f is a monotonic function"), so it is not an error. Improvement: for standard Gaussian inputs OLS is not a heuristic in the same sense. By Stein's lemma the population OLS slope equals E[∇f], so ŵ estimates the normalized average gradient, which lies in range(C). It is not guaranteed to be the top eigenvector of C | replace with the Stein's-lemma statement plus the caveat (rank 1; E[∇f] can be ≈ 0 for non-monotone f) | confirmed (math; v1 OLS docstring) |
| 0 | "Draw … where N = O(m)" | fine; here N = 100m | — | — |
| 0 | "Karhunen-Loéve" | the correct spelling is Karhunen–Loève (the paper's abstract also uses "Loéve") | fix | confirmed |
| 0 | "in may applications" | typo → "many" | fix | confirmed |
| 0 | "indtroduce" | typo | fix | confirmed |
| 0 | eigenpair equation ∫C(s,t)φᵢ(s)ds = σᵢ²φᵢ(t) and log κ ≈ μ + Σ σᵢφᵢXᵢ; "We choose m to be 100 since we find this value to result in sufficiently accurate KL representations" | Matches paper eqs. (7)–(10). The paper's basis is not an analytic KL, though: it is the first 100 principal components of 2,000 turning-bands realizations of log permeability (p. 33 §2.4), with geometric mean permeability 0.04465 m h⁻¹ and correlation lengths 50/50/5 m (Table 2). The notebook omits these, the 48-h duration and the precipitation event (3 h at 5 cm/h, p. 35) | add one sentence on the empirical (PCA) KL basis and the Table 2 values; add the scenario details | confirmed (paper) |
| 7 | `sufficient_summary(XX.dot(ss_unsat.W1), np.log10(f_unsat), out_labels[1])`, comment "note the log scale used in the unsaturated case" | 3,137 zero runoffs give −inf and vanish from the plot unnoticed; the y label reads "Hortonian (Unsaturated) Case" with no log10 | plot log10(f) for f > 0 only, as the paper does (p. 37, Fig. 7b, log axis 10⁻⁵–10⁴), say how many are zero, and label the axis "cumulative runoff volume [m³], log scale" | confirmed (data; paper p. 37) |
| 7 | comment "summary plots as presented in the paper" | the notebook's x-axis (Gaussian scaling) spans about ±4, but the paper's spans about ±1 (Figs. 6b, 7b), and the Hortonian panel is mirrored (paper slope > 0, legacy slope < 0) | orient by â; say that the paper's figures use a ~4.5× smaller active-variable scale (likely a per-column min/max rescale) | likely (scale), confirmed (sign) |
| 7 | Hortonian OLS fitted on all 10,000 runs, including zeros | Unclear whether the paper fitted on non-zero runs only ("Fig. 7 represents only the non-zero runoff cases" covers panel a, the weights, too). The difference is small: cos(w_all, w_nonzero) = 0.9984; on the non-zero data, w₁ = −0.646 (min/max scaling) vs paper ≈ −0.65 | state which set is used; recommended: fit on all runs, plot non-zero only. Ask the author | unverified (ambiguous text) |
| 8 | "the last 70 or so components are extremely close to 0" (Dunnian) | Σ_{i>30} w_i² = 0.0032 for Dunnian, 0.048 for Hortonian | — | confirmed |
| 8 | "the Dunnian runoffs span about 45000 - 46000 while the Hortonian range is about 0 - 1000" | actual: 45,219–46,168 and 0–1,345 m³; also, 31% of Hortonian runs produce no runoff. The paper says "about 1000 m³" for both (p. 38) | "0–1,350 m³, with 31% exactly zero"; add units | confirmed |
| 8 | "The majority of weight in the Dunnian case is placed on the first component" | matches the paper ("nearly one", p. 36): w₁ = 0.962 | — | confirmed |
| 8 | "(the output values have a range of about 1000)" | 949 and 1,345 | fine, or give the numbers | confirmed |
| 11 | "large gap between the 1st and 2nd eigenvalues in the Dunnian case and between the 1st and 2nd and 3rd and 4th … Hortonian" | QPHD eigenvalues: Dunnian 12,253 / 631 / 424 / 364; Hortonian 12,979 / 1,236 / 1,103 / 371. Matches | — | confirmed |
| 13 | "maximum difference … less than .02 in the Dunnian case and less than .04 in the Hortonian case" | after sign alignment: 0.0173 and 0.0369 | — | confirmed |
| 14/15 | `ss_unsat.partition(3)` then R² "Linear subspace: 0.4965 0.7432 0.7783"; text "improving from .74 to .82 for the quadratic model and .78 to .89 for the cubic model" | the OLS matrix ââᵀ is rank 1, so columns 2–3 of its eigenvectors are an arbitrary basis of â⊥ (v1 docstring: "the other columns are an arbitrary orthonormal complement"). The printed numbers are not reproducible (0.7412 / 0.7740 here) | compare the 1-D OLS subspace (0.4965 / 0.7385 / 0.7693) with the 3-D QPHD subspace (0.4924 / 0.8230 / 0.8910), or with 1-D QPHD. Rewrite the sentence: ".74 → .82 and .77 → .89" | confirmed |
| 14 | R² values | in-sample (legacy `Rsqr`); the cubic in 3-D has 20 terms against 10⁴ points, so overfitting is negligible (LOO is lower by ≤ 0.001) | report LOO from v1 and say so | confirmed |
| 16 | "approximate gradients with finite differences or other methods" | fine | — | — |
| 0 | ref [1] "Adv. Water Res., 92, 30-42, 2016" | correct; standard abbreviation is "Adv. Water Resour." | optional | confirmed |

## 4. Code review
- py2: the `print '...'` statements in cell 14 are a SyntaxError in py3. `[i/3]` indexes are a TypeError in py3; use `i//3`.
- pandas: `DataFrame.from_csv`, `.as_matrix()` (both removed).
- Bug (methodological): see cell 14 `partition(3)` on OLS (§3).
- Fragile: cell 12 `ss_quad_sat.W1 - ss_sat.W1` assumes identical sign conventions; align with `np.sign(wq @ wl)`.
- Fragile: cells 10, 12 and 14 mutate subspaces through `partition()`, so results depend on cell execution order. In v1, use `with_dimension` to make new objects.
- Waste: `nboot=500` on OLS, unused (cell 5). The cost is small, but drop it.
- Cost: v1 QPHD with m = 100 fits 5,151 coefficients from 10,000 points, about 46 s per QoI on this machine, ≈ 95 s total. It is noisy because the oversampling is only ≈ 1.9×; say so. A cache is not needed.
- `f_sat`/`f_unsat` are `(M,1)`; v1 stores `(M,)`.
- No randomness after nboot is removed, so the notebook is deterministic.

## 5. Data management review
- Text with no header, no units and no normalization needed (already N(0,1)). Not stored anywhere: the KL basis (eigenfunctions φᵢ, σᵢ), the mean and variance of log κ, the correlation length and the domain geometry. Without them the weights cannot be mapped to spatial patterns (the paper's main result). Ask the author for the KL modes (e.g. `kl_modes.npz`) to plot w as a permeability field.
- Size: 25.5 MB is under GitHub's 50 MB warning. Options: keep it as text (portable, already committed; no gain from rewriting history), or replace it with `inputs.npy` (8 MB, `np.load`). Converting adds a new blob without removing the old one from history, so **keep the text file** unless the history is rewritten.
- `RunOff02_Unsaturated_10k.txt` lacks a final newline; harmless for `np.loadtxt`.
- No license or provenance (ParFlow version, run date, GW02 meaning).
- pandas-free loading, tested in .venv:
```python
import numpy as np
X      = np.loadtxt('inputs-ShortCorr.GW02.txt')       # (10000, 100), standard Gaussian KL coefficients
f_sat  = np.loadtxt('RunOff02_Saturated_10k.txt')      # (10000,)  Dunnian
f_uns  = np.loadtxt('RunOff02_Unsaturated_10k.txt')    # (10000,)  Hortonian; 3137 exact zeros
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| (no normalization) | `space = ac.Gaussian.reference(100)` | X is already N(0, I); physical = normalized. No ValidityWarning for OLS or QPHD on a Gaussian space |
| `compute(X=XX, f=f, sstype='OLS', nboot=500)` | `ols = ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)` | λ₁ equal to legacy to 1e-12 rel. Drop nboot |
| `ss.W1` (sign: first component > 0) | `w = ols.W1[:,0]; w *= np.sign(w @ a_hat)` | v1 sign (largest component positive) equals legacy here for both cases, because component 1 is the largest. Both are mirrored relative to the paper for Hortonian. Orient by â (the paper's convention, pp. 37–38): Hortonian w₁ becomes −0.627 and the slope is positive, as in Fig. 7 |
| `compute(X=XX, f=f, sstype='QPHD')` | `q = ac.active_subspace(X=X, f=f, space=space, estimator="qphd")` | v1 C = ggᵀ + HΣH with Σ = I is the same as legacy (γ = 1 because max X > 1). Default EVG → n = 1 |
| `partition(2)`, `partition(3)` | `ac.with_dimension(q, 2)`, `ac.with_dimension(q, 3)` | **never** `with_dimension(ols, k>1)`: the extra columns are meaningless |
| `utils.plotters.eigenvalues(ss.eigenvals[:10], out_label=…)` | `ac.plotting.plot_eigenvalues(q, k_max=10)` | no bootstrap band (nboot = 0) |
| `utils.plotters.sufficient_summary(XX.dot(W1), f)` | `ac.plotting.plot_sufficient_summary(sub, X=X, f=f, qoi_label=…)`; 2-D via `with_dimension(q, 2)` | for log10: pass `X[f>0]`, `np.log10(f[f>0])`, otherwise v1 raises `InputError input.nonfinite` (tested) |
| manual weight plots | `ac.plotting.plot_eigenvectors(sub, indices=[0])`, or keep the custom 100-point plot | — |
| `W1q - W1` | `wq * np.sign(wq @ wl) - wl` | explicit sign alignment |
| `PolynomialApproximation(d).train(Y, f)`, `.Rsqr`, `.predict` | `s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=d)`; `s.loo_r_squared`; in-sample from `s.residuals`; `ac.plotting.plot_surrogate_fit(s, X=X, f=f)` | in-sample values equal legacy for QPHD subspaces (tested, 4 dp) |

## 7. Associated .py files
None. The ParFlow runs and the KL generator are not in the repo.

## 8. Update plan
1. Replace pandas loading with `np.loadtxt` (§5).
2. Port to py3 (print(), `//`) and v1 (`Gaussian.reference(100)`, OLS, QPHD, `with_dimension`), dropping nboot.
3. Orient OLS w by â (matches the paper). Fix the log10 summary: mask zeros as the paper does, report the zero count, label the axis. Add a note that the paper's figures use a ~±1 active-variable scale (per-column min/max rescale) where the notebook uses the Gaussian scale.
4. Fix the R² comparison: 1-D OLS against 1-D and 3-D QPHD; report in-sample and LOO; rewrite the cell 15 text.
5. Align signs before the weight-difference plots.
6. Text fixes: Stein's-lemma justification for OLS on Gaussian inputs, the Hortonian range and zero fraction, units (m³), the Karhunen–Loève spelling, typos. Add the KL/PCA basis description and Table 2 values (geometric mean 0.04465 m h⁻¹, λ = 50/50/5 m, 100 PCs from 2,000 turning-bands fields), the 48-h duration and the 3 h × 5 cm/h storm.
7. Optional: if the author supplies the KL modes, plot w₁ as a spatial field (the paper's key figure).
8. Execute headless (≈ 2 min) and commit with outputs.

## 9. Acceptance criteria
- Executes in `.venv` with `jupyter nbconvert --execute` in < 5 min (QPHD ≈ 2×46 s here), no pandas, no warnings.
- OLS λ₁: Dunnian 11372.3058, Hortonian 6568.8455 (rtol 1e-8). |w_v1 · w_legacy| = 1 to 1e-10.
- QPHD eigenvalues[:4]: Dunnian 12252.5 / 630.56 / 424.28 / 363.64 (all 10,000 runs); Hortonian 16294.24 / 2151.55 / 1833.69 / 701.62 on the 6,863 non-zero runs (author, 2026-10-30; [#46](https://github.com/paulcon/as-data-sets/pull/46)) (rtol 1e-6). The all-runs Hortonian values 12978.8 / 1235.71 / 1103.29 / 371.36 reproduce 2016 (§10).
- Max |w_qphd − w_ols| after alignment: 0.0173 (D) and 0.0369 (H).
- R² table (in-sample, degree 1/2/3): D, OLS 1-D 0.8473 / 0.8771 / 0.8776; D, QPHD 1-D 0.8455 / 0.8763 / 0.8769; H (6,863 non-zero runs), OLS 1-D 0.5319 / 0.7233 / 0.7397; H, QPHD 1-D 0.5171 / 0.7278 / 0.7497; H, QPHD 3-D 0.5224 / 0.8280 / 0.8742. (All runs, as in 2016: H, OLS 1-D 0.4965 / 0.7385 / 0.7693; H, QPHD 3-D 0.4924 / 0.8230 / 0.8910.) LOO is shown next to these.
- The log-scale Hortonian summary plot states "3137 of 10000 runs have zero runoff (not shown)".
- Hortonian OLS w₁ < 0 and the summary slope > 0 (paper Fig. 7 orientation). Dunnian w₁ = +0.962.

## 10. Prototype results
Ran `prototypes/g4/proto_sub.py`, `prototypes/g4/load_sub.py` and `prototypes/g4/proto_misc.py` with the .venv python, 1 min 35 s total:
- Data checks as in §1/§2.
- v1 OLS matches legacy OLS exactly (cos 1.0, same sign).
- v1 QPHD (46.6 s and 45.9 s) gives eigenvalues matching the legacy eigenvalue figure (`sub_c10_1.png`: ≈1.3e4, 1.24e3, 1.10e3, 3.7e2). |cos(w_qphd, w_ols)| is 0.99894 (D) and 0.99374 (H).
- In-sample R² for QPHD subspaces reproduces the legacy printed table to 4 dp. LOO is 0.8454 / 0.8763 / 0.8767 (D, 1-D) and 0.4916 / 0.8221 / 0.8901 (H, 3-D).
- The legacy "linear subspace, k=3" Hortonian row could not be reproduced (0.7412 / 0.7740 against printed 0.7432 / 0.7783), which confirms that those columns are arbitrary.
- `plot_sufficient_summary` with log10 of zeros raises `InputError [input.nonfinite]`.
- Paper cross-check (`prototypes/g4/sub2.py`). Columns: w₁ / w₅, then active-variable range.

  | case | scaling | data | w₁ / w₅ | y range |
  |---|---|---|---|---|
  | Hortonian | Gaussian | all runs | −0.627 / −0.515 | [−3.46, 3.75] |
  | Hortonian | min/max | all runs | −0.648 / −0.491 | [−0.94, 0.93] |
  | Hortonian | Gaussian | non-zero | −0.625 / −0.525 | — |
  | Hortonian | min/max | non-zero | −0.646 / −0.501 | [−0.80, 0.91] |
  | Dunnian | min/max | all runs | w₁ = 0.964 | [−0.99, 0.92] |

  Paper Fig. 7a: w₁ ≈ −0.65, w₅ ≈ −0.50, y ≈ [−0.85, 0.9]. Fig. 6b: y ≈ [−1, 1].

## 11. Open questions for the author
1. The paper drops zero-runoff runs from Fig. 7 (p. 37). Was the Hortonian OLS fitted on the non-zero runs only (the text implies panel a too), or on all 10,000?
2. Did the paper's figures rescale the Gaussian KL coefficients to [−1, 1] by sample min/max? Their active-variable range (≈ ±1) and weights (w₁ ≈ −0.65) match that, not N(0, I). The notebook should keep the Gaussian scale. OK?
3. Can the basis B (100 PCs; Table 2 setup) and the domain grid be added, so w can be shown as a spatial field, as in paper Figs. 8–9?
4. Meaning of "ShortCorr" (λ = 50/50/5 m?) and "GW02"; are long-correlation runs available?
5. Keep the 25.5 MB text file as is (recommended) or switch to `.npy`?
6. Should the notebook keep the quadratic (QPHD) extension, which is not in the paper, given its ≈ 90 s cost?
7. Data license and provenance.
