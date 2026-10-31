# SingleDiodePV: update spec

Kind: **stored data**: Campanelli's MATLAB code. The planned `single_diode.py` re-implementation checks the data; the shipped file stays the record (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `SingleDiodePV/active-subspace-pv.ipynb` | paper: P. G. Constantine, B. Zaharatos, M. Campanelli, "Discovering an active subspace in a single-diode solar cell model", *Stat. Anal. Data Min.* 8(5–6):264–273 (2015), doi:10.1002/sam.11281 (arXiv:1406.7607v3 and, on 2026-10-29, the **published** version: §2.1 p. 266, Table 1 p. 268, eqs. (9)–(10) p. 268, §4.1–4.2 p. 269, Table 2 p. 271, §4.5 p. 272; NREL/JA-5J00-65463; page numbers below refer to the published version) | m = 5 (I_SC, I_S [log-uniform], n, R_S, R_P) | QoI: P_max [W] = max IV subject to the single-diode equation | legacy API: `ac.utils.misc.BoundedNormalizer(xl, xu).normalize`, `ac.subspaces.Subspaces().compute(df=G, nboot=500)`, `.eigenvals/.eigenvecs/.e_br/.sub_br/.partition/.W1`, `ac.utils.plotters.eigenvalues/subspace_errors/eigenvectors(W_br=)/sufficient_summary` | py2 issues: yes (`print` in cell 15; pandas `from_csv`/`as_matrix`)

## 1. Summary
This is the cleanest of the three notebooks: gradients come with the data, there is one QoI, and it runs in seconds. v1 reproduces the legacy w₁ to 4 decimals and the paper's Table 2 exactly. Findings:
1. **Typo in the normalization bounds (cell 5).** `xu[0] = 0.23598`, but the table, the paper and the data say I_SC ≤ **0.23958** (data max 0.239570; published Table 1, p. 268: "ISC Short-circuit current 0.05989 0.23958 amps"). The normalized I_SC then runs up to 1.041, which distorts the summary-plot coordinate y = XX·W₁. The eigen-analysis is unaffected because it uses the stored G.
2. **Gradient coordinates (verified).** The stored gradients `dISC…dRP` are derivatives with respect to the **normalized** inputs, with I_S in log space. I re-implemented the model (Lambert-W, V_th = kT/q at 25 °C = 0.025693 V). It reproduces P_max to 1.3e-6 relative error and the 4 other gradient components to ≤ 2.4e-6; dISC carries a constant 4.7e-5 bias. In v1 they must therefore be passed with an identity space, or divided by (xu−xl)/2 before being passed with the physical space. Passing them as physical with `ac.Uniform(xl, xu)` scales them twice and gives a wrong answer: w₁ becomes R_P-dominated ([.01, −.33, .04, .00, .94]).
3. **Math statement reversed (cell 0, copied from the paper §2.1, p. 266; the wording is identical in the published version).** "The distance between W₁ and Ŵ₁ is inversely proportional to λ_{n+1} − λ_n". The gap is λ_n − λ_{n+1}, and the result is an upper bound proportional to 1/(λ_n − λ_{n+1}), not a proportionality.
4. **Bootstrap settings differ from the paper.** The data have M = 10,000 samples; the paper used M = 1000, 1000 bootstrap replicates and 99 % intervals (published §4.2, p. 269). The notebook uses all 10,000 samples, 500 replicates (eigenvalues) and 10⁴ replicates (eigenvectors), with min/max bands. v1's default band is a 95 % percentile interval. The conclusions do not change, but the text should say which settings were used.
5. **v1 has no eigenvector bootstrap band.** v1 dropped W_br, so the eigenvector band must be hand-coded. Align each replicate's sign with the full-sample vector, not with the v1 "largest entry positive" rule: for w₂ the two largest entries are 0.635 and −0.607, so that rule flips sign from replicate to replicate. **Superseded (V1-02, [#41](https://github.com/paulcon/as-data-sets/pull/41)):** `bootstrap_vectors=` returns aligned replicates and `plot_eigenvectors(..., interval=)` draws the band.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| active-subspace-pv.ipynb | 290,663 B | nbformat 4, kernel python2 | analysis | — | 17 cells, 5 PNGs (eigenvalues, subspace errors, eigenvectors, 1-D and 2-D summary), 1 printed table. Master is newer than dev (affiliation and refs only). |
| SingleDiodePV-Pmax.txt | 2,780,045 B | CSV, 1 header row, 10,000 × 12, `%.16e` | samples | `pn.DataFrame.from_csv` (`Run` → index) | cols `Run, ISC, IS, n, RS, RP, Pmax, dISC, dIS, dn, dRS, dRP`. Inputs are **physical**; I_S spans 2.2e-11..2.2e-7. Gradients are w.r.t. **normalized** x (log I_S). P_max 0.0143–0.2032 W. No NaN. |
| *(history)* `SingleDiodePV/figs/*.eps` | ≤ 2.1 MB | EPS | 2014 figures | — | removed long ago; still in history |

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "The distance between $\mathbf W_1$ and $\mathbf{\hat W}_1$ is inversely proportional to $\lambda_{n+1}-\lambda_n$" | The sign is reversed (λ_{n+1} − λ_n ≤ 0), and the claim is too strong. the subspace-error bound in CDW 2014 §3 (also Constantine 2015 §3.3) is dist(W₁, Ŵ₁) ≤ 4λ₁ε/(λ_n − λ_{n+1}) (ε a relative error in Ĉ); verify the exact corollary number before citing it. The published paper (§2.1, p. 266) has the same wording: "the distance between the n-dimensional subspace defined by W 1 and its approximation defined by Ŵ 1 is inversely proportional to λn+1 − λn [9]". So the error is in the paper too; do not cite the paper for this statement. | "The error in the estimated subspace is bounded by a quantity proportional to $1/(\lambda_n-\lambda_{n+1})$, so a large gap gives an accurate subspace." | confirmed (published p. 266) |
| 0 | "this decomposition will be valid so long as $\lambda_n > \lambda_{n+1}$" | OK (the partition is well defined when there is a gap) | keep | confirmed |
| 0 | "the amount $f$ tends to change along the active subspace is $\int(\nabla_yf)^T(\nabla_yf)\rho d\mathbf x = \lambda_1 + \cdots + \lambda_n$" | Matches paper Lemma 2.2 (CDW 2014 Lemma 2.2) | keep | confirmed |
| 0 | "we use first order finite differences" | The published §4.2 (p. 269) says "a first-order finite difference approximation of the gradient … step size of 10−6 in the normalized domain". The data gradients agree with central differences to ≈ 1e-6, which is consistent. | Add "step 10⁻⁶ in normalized coordinates" | confirmed |
| 0 | table "Short-circuit current … 0.05989 \| 0.23958 \| Uniform" | Matches the published Table 1 (p. 268) and the data. The table has no units (paper: amps, amps, unit-less, ohms, ohms). | Add units | confirmed (published Table 1) |
| 0 | "$V_{th}$ is a fixed thermal voltage for temperature $T = 25^o$C" | Matches the paper (published p. 268: "The thermal voltage Vth is a given, fixed constant for fixed temperature T = 25◦ C"). The published version gives no numerical value either. The data are reproduced with V_th = k_B·298.15 K/q = 0.025693 V (V_th = 0.02585 V, i.e. 27 °C, misses by 0.7 %). | Add "V_th = k_BT/q ≈ 0.025693 V" | confirmed (prototype; the value is absent from the published paper) |
| 0 | "$P_{max} = \max\limits_{I,V}\ IV$" | OK | add "[W]" | confirmed |
| 0 | "__Brian Zaharatos__, University of Colorado Boulder" | The published paper (p. 264) lists both Constantine and Zaharatos at the Department of Applied Mathematics and Statistics, Colorado School of Mines, and Campanelli at NREL. The notebook was changed deliberately (commit 04742b5 on master vs dev), presumably to Zaharatos's later affiliation. | Q5 | confirmed (published = Mines); choice open |
| 2 | "contains 10,000 sets of input parameters and corresponding $P_{max}$ values with gradients" | True. The paper used M = 1000, so the bootstrap ranges here are narrower than in the paper's Fig. 2. | Add a sentence saying so | confirmed |
| 4 | "Normalize the inputs to the interval [-1,1]. The second variable is uniform in the log space." | OK, but the code's I_SC upper bound is wrong (cell 5). | Fix the code | confirmed |
| 10 | "(see also Figures 2 and 3 of [1])" | The paper's figures use 99 % bootstrap intervals with M = M′ = 1000 (published §4.2 and Figs. 2–3, pp. 269–270); the notebook uses min/max bands with M = 10⁴ and 500/10⁴ replicates. | State the settings | confirmed |
| 12 | "We can see a large (order of magnitude) difference between the first and second eigenvalues" | λ₁/λ₂ = 19.4 and λ₂/λ₃ = 8.0, consistent with the paper (p. 269: "a gap of nearly an order of magnitude between both the first and second and the second and third") | keep, or quote the ratios | confirmed |
| 15 (output) | "ISC 0.7674, IS −0.4228, n 0.473, RS −0.09064, RP 0.0207" | Matches the published Table 2 (p. 271: ŵ1 = 0.77, −0.42, 0.47, −0.09, 0.02; Sobol' first-order 0.56/0.17/0.21/0.01/0.00, total 0.61/0.19/0.25/0.01/0.00) and v1 | keep. Label I_S as log I_S. | confirmed |
| 16 | "so their product sales like a product of a linear term and a logarithmic term" | typo | "scales" | confirmed |
| 16 | "the ranges for these variables used in the simulation are smaller than what would be observed in practice while the other ranges are representative of reality" | Paraphrases published §4.5 (p. 272, "ranges for n and I_S are at about their maximum extent … ranges for R_S and R_P are somewhat restricted") | keep | confirmed |
| 16 | "(see Table 2 of [1])" Sobol' ranking | matches the paper's Table 2 | keep | confirmed |
| 0 | "[1] … 8(5-6), 264-273, 2015", doi 10.1002/sam.11281 | Correct (published p. 264: "Statistical Analysis and Data Mining: The ASA Data Science Journal 8: 264–273, 2015", DOI:10.1002/sam.11281, published online 24 July 2015) | Add arXiv:1406.7607 as the open copy. Optionally add the Constantine 2015 book reference (not cited, unlike the other notebooks). | confirmed |
| (not in notebook) | published §4.2 (p. 269): the subspace error "is actually the cosine of the principal angle between the subspaces" | Paper error: ‖W₁ᵀŴ₂‖₂ = ‖W₁W₁ᵀ − Ŵ₁Ŵ₁ᵀ‖₂ is the **sine** of the largest principal angle. Not in the notebook today. | If cell 10/12 gains text on the subspace-error plot, say "sine of the largest principal angle" | confirmed |
| 11 (fig) | eigenvector plot has x-ticks 1.0…5.0 and no parameter names | `in_labels` not passed | `labels` from the space names | confirmed |

## 4. Code review
- **Bug (cell 5):** `xu = np.array([0.23598, …])`. It should be 0.23958, which gives normalized I_SC max 1.0408 instead of ≤ 1. It only affects `XX`, which is used solely for the summary plots.
- **Hard-coded log bounds.** `-24.539978662570231`, `-15.3296382905940`. Use `np.log(2.2e-11)`, `np.log(2.2e-7)`.
- **Cell 9 (eigenvector bootstrap):** 10⁴ replicates of a full `Subspaces.compute`, each also doing sorted_eigh and the default partition. `np.apply_over_axes(np.amin, …)` is a roundabout `w1_boot.min(axis=1)`. Unseeded `np.random.randint`. In 2016 the sign was fixed by the first component; w₂'s first component is 0.63, so that was stable. The v1 sign rule would not be (§1 item 5). Replace with an SVD of the resampled G rows and a sign alignment `v *= np.sign(v @ w_full)`. Prototype: 1000 replicates take 0.8 s.
- `ss.compute(df=G, nboot=500)` is unseeded.
- `ss.partition(2)` then `ss.partition(1)` mutate the same object. Use `ac.with_dimension` in v1.
- Cell 15 uses a py2 `print`. Use an f-string, or `ac.plotting.plot_eigenvectors`, which also labels the axes.
- No hard-coded absolute paths.

## 5. Data management review
- **Format.** CSV with a redundant `Run` index. Inputs are physical (good); gradients are in normalized coordinates, which is undocumented and the main trap for v1. P_max units (W) are not stated.
- **Size.** 2.78 MB; npz float64 (X, f, G) 0.82 MB. No size problem. Converting is optional; consistency with the repo-wide decision matters more.
- **Provenance.** Generated with Campanelli's MATLAB code (paper §3.1, "Requests for the MATLAB code may be sent to … Campanelli"). The model takes about 15 lines of Python with scipy (`scipy.special.lambertw` + `minimize_scalar`). The prototype re-implementation reproduces the data (§10), so the notebook could include it to make the case self-contained and to verify the gradients (open question Q2). License: none stated.
- **README.** Document that the gradients are w.r.t. x ∈ [−1,1]⁵ with x₂ = normalized log I_S, using the Table 1 bounds.
- **pandas-free loading (tested):**
```python
import numpy as np
rec = np.genfromtxt("SingleDiodePV-Pmax.txt", delimiter=",", names=True)   # structured array
in_names = ["ISC", "IS", "n", "RS", "RP"]
X = np.column_stack([rec[k] for k in in_names])        # physical inputs (10000, 5)
f = rec["Pmax"]                                        # [W]
G = np.column_stack([rec["d" + k] for k in in_names])  # d f / d x, x normalized, x2 from log(IS)
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `Y[:,1] = np.log(Y[:,1]); BoundedNormalizer(xl, xu).normalize(Y)` | `Xt = X.copy(); Xt[:, 1] = np.log(X[:, 1])`; `space = ac.Uniform(lower=xl, upper=xu, names=["I_SC", "log I_S", "n", "R_S", "R_P"])` with `xl = [0.05989, log 2.2e-11, 1, 0.16625, 93.75]`, `xu = [0.23958, log 2.2e-7, 2, 0.665, 375]` | v1 has no log-uniform space; work in (I_SC, log I_S, …). Fixes the 0.23598 typo. `space.to_normalized(Xt)` spans [−0.9998, 0.9999]. |
| `ss.compute(df=G, nboot=500)` | `sub = ac.active_subspace(grads=G / ((xu - xl) / 2), space=space, nboot=500, rng=2016)` | **G is already normalized.** Either undo the scaling as shown, so v1 re-applies it (result identical to 2016), or use `space=ac.Uniform(lower=-np.ones(5), upper=np.ones(5), names=…)` with `X` normalized by hand. Never pass `G` with the physical space unchanged: that is double scaling, and w₁ becomes [.01, −.33, .04, .00, .94]. |
| `ss.eigenvals`, `ss.e_br` | `sub.eigenvalues`, `ac.bootstrap_interval(abs(sub.bootstrap_eigenvalues), kind="minmax")` | identical eigenvalues |
| `ss.sub_br` | `ac.bootstrap_interval(sub.bootstrap_distances, kind="minmax")`, `ac.mean_distance(sub)` | |
| `plotters.eigenvalues(..., e_br=)`, `plotters.subspace_errors(ss.sub_br)` | `ac.plotting.plot_eigenvalues(sub, interval="minmax")`, `ac.plotting.plot_subspace_errors(sub, interval="minmax")` | or the default 95 % percentile; the paper used 99 % |
| `plotters.eigenvectors(ss.eigenvecs[:,:2], W_br=w_br)` | `ac.plotting.plot_eigenvectors(ac.with_dimension(sub, 2))` + hand-drawn `fill_between` band from aligned bootstrap replicates | W_br was removed in v1. Sign: v1 makes the largest entry positive. w₁: I_SC (0.767) is both the first and the largest entry, so no change. w₂: 0.635 is the largest entry (vs −0.607), so no change from 2016 either, but this is marginal. **Superseded (V1-02, V1-05, [#41](https://github.com/paulcon/as-data-sets/pull/41)):** `plot_eigenvectors(..., interval=)` draws the band; `ac.orient` sets the paper's rule (first component positive), which equals v1's default here. |
| `ss.partition(2); y = XX.dot(ss.W1); plotters.sufficient_summary(y, f)` | `ac.plotting.plot_sufficient_summary(ac.with_dimension(sub, 2), X=Xt, f=f, qoi_label="$P_{max}$ [W]")` and the same with n = 1 | takes physical (log-transformed) Xt |
| `print "The variable {:5} …"` | f-string over `zip(space.names, sub.W1[:, 0])`; optionally also `sub.physical_directions()` | |
| *(new, optional)* | `ac.fit_surrogate(X=Xt, f=f, subspace=ac.with_dimension(sub, 1), kind="poly", degree=3)`; LOO R² = 0.981 (deg 2: 0.980, deg 1: 0.938) | supports "Using Active Subspaces" (response surfaces) with numbers |

## 7. Associated .py files
None today. Optional: add a `single_diode.py` (about 25 lines: `pmax(params, T=25)` via Lambert W, `to_physical(x)`, `to_normalized(p)`) taken from the prototype `prototypes/g5/pv_model.py`. It would let the notebook (a) check a few stored rows against the model and (b) state the model precisely. It adds no dependency beyond scipy.

## 8. Update plan (one commit each)
1. Notebook to Python 3: numpy loader, `rng = np.random.default_rng(2016)`, f-string print, clear outputs.
2. Fix the I_SC bound typo and use `np.log` for the bounds. Introduce the v1 space on (I_SC, log I_S, n, R_S, R_P).
3. v1 subspace with the G rescaling (§6). Add an assert that `sub.eigenvalues` equal `np.linalg.eigvalsh(G.T @ G / M)[::-1]` to 1e-12, and that `abs(sub.W1[:,0])` equals the legacy values to 1e-4.
4. Replace cell 9 with a seeded, sign-aligned eigenvector bootstrap (1000–10⁴ replicates) and plot the bands with v1 `plot_eigenvectors` plus `fill_between`.
5. Text fixes from §3: the gap statement, units, V_th value, sample-size and bootstrap settings, "scales", log I_S labels.
6. Optional: `single_diode.py` with a verification cell, and a 1-D polynomial surrogate cell.
7. Top-level README unchanged (the 5-parameter count and the link are correct). Execute headless and commit.

## 9. Acceptance criteria
- `cd SingleDiodePV && jupyter nbconvert --to notebook --execute active-subspace-pv.ipynb` succeeds in `.venv` in < 1 min (prototype: subspace with nboot=500 in 0.7 s; 1000 eigenvector replicates in 0.8 s).
- No pandas, no py2 syntax, a seeded rng, no warnings.
- Eigenvalues = [3.3350e-3, 1.7227e-4, 2.1552e-5, 1.3414e-6, 3.9520e-7] (rel. tol 1e-10). λ₁/λ₂ = 19.4.
- w₁ = [0.7674, −0.4228, 0.4730, −0.0906, 0.0207] and w₂ = [0.6346, 0.4290, −0.6074, 0.2088, 0.0288] (tol 1e-4, v1 signs).
- Bootstrap on a seeded M = 1000 subsample with 1000 replicates and 99 % percentile intervals, as in the paper (author, 2026-10-30; this replaced a min/max criterion on all 10,000 samples). The widest 99 % interval of a w₁ component is ≤ 0.03 and of a w₂ component ≤ 0.06 (the port prints 0.027 and 0.056). The mean subspace distance for n = 1 is ≈ 7e-3 (the port prints 7.3e-3; paper Fig. 2b ≈ 7e-3).
- Normalized inputs of the summary plots lie in [−1, 1].
- Figures: eigenvalues, subspace errors, eigenvectors (2, labeled, with bands), 1-D and 2-D summary plots (5, as legacy), plus an optional surrogate fit.

## 10. Prototype results
Run in `.venv` (`prototypes/g5/pv_model.py`, `pv_v1.py`):
- Loading: (10000, 12), no NaN. Data I_SC max 0.239570 (so the code bound 0.23598 is a typo); log I_S range [−24.5395, −15.3299].
- Model check (8 rows): P_max relative error −1.0e-6 to −1.3e-6 with V_th = 0.025693 V. Central-FD gradients in normalized coordinates match the stored G to ≤ 2.4e-6 relative error, except dISC, which has a constant +4.7e-5 bias. The bias is consistent with a slightly different I_SC range in the original generator; negligible. With V_th = 0.02585 V the errors are about 0.7 %. This confirms G is in normalized coordinates.
- v1: the eigenvalues in §9 are identical to `eigh(GᵀG/M)`; w₁ matches the legacy cell-15 output and the paper's Table 2. 500 bootstrap replicates (minmax) give λ₁ ∈ [3.28e-3, 3.39e-3]; `mean_distance` = [0.0022, 0.0039, 0.0025, 0.0067].
- The double-scaling misuse (`grads=G`, physical space) gives w₁ = [0.013, −0.330, 0.039, −0.004, 0.943] and λ₁ = 0.064. This is the main migration trap.
- v1 plots `plot_eigenvalues`, `plot_subspace_errors`, `plot_eigenvectors`, `plot_sufficient_summary` (n = 1, 2) and `plot_surrogate_fit` all run headless (Agg) with no warnings.
- The legacy PNGs (eigenvalues ≈ 3.3e-3, 1.7e-4, 2e-5, 1.3e-6, 4e-7; summary plots) agree.

## 11. Open questions for the author
1. Keep all 10,000 samples (the notebook) or subsample M = 1000 to match the paper's figures and bootstrap widths? Which interval: 99 % (paper), min/max (2016 notebook) or 95 % (v1 default)?
2. Include a Python re-implementation of the single-diode model (`single_diode.py`, verified against the data) so the case is self-contained? Campanelli's MATLAB code is not in the repo.
3. Is the constant 4.7e-5 bias in dISC worth explaining? (It suggests the generator used slightly different I_SC bounds than Table 1.)
4. The data format decision (repo-wide): keep the CSV (2.8 MB) or use `.npz` (0.82 MB)?
5. Header affiliations and emails (Zaharatos: Mines in the published paper p. 264, CU Boulder in the notebook; the Mines emails are from 2016).
6. Record the two published-paper errors (gap wording p. 266; "cosine" for sine p. 269) in a notebook footnote?
