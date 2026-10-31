# Atacamac: update spec

Kind: **stored data**: ATACAMAC annular-combustor model runs (Bauerheim et al.) (see [README.md](../README.md#two-kinds-of-data-sets)).

Status line: `Atacamac/active-subspace-atacamac.ipynb` (dev only; identical to master at `9c1b8a7^`, removed from master in 9c1b8a7) | paper: Bauerheim, Ndiaye, Constantine, Moreau, Nicoud, *Symmetry breaking of azimuthal thermoacoustic modes: the UQ perspective*, J. Fluid Mech. 789:534–566 (2016), doi:10.1017/jfm.2015.730 (open copy: hal-01234985v2; the local "published" PDF in `~/Documents/Claude/JobAppDocs/papers/published/` is HAL **v1**, the author manuscript with HAL cover "Journal of Fluid Mechanics, 2015", whose body is identical to v2; page refs below are its printed manuscript pages, not the journal's 534–566); companion paper with the same data setup: Bauerheim, Ndiaye, Constantine, Iaccarino, Moreau, Nicoud, *Uncertainty quantification of thermo-acoustic instabilities in annular combustors*, Proc. CTR Summer Program 2014, Stanford, pp. 209–218 | m = 38 (n_i, tau_i, i = 1..19), two cases (weak / strong coupling) | QoI: growth rate f_Im of the first azimuthal mode (column `maxFreq`) | legacy API: `UnboundedNormalizer`, `Subspaces.compute(df=, nboot=500)`, `.partition`, `plotters.eigenvalues/subspace_errors`, `response_surfaces.PolynomialApproximation` | py2 issues: yes (`print` statements cells 17, 20; `i/2` used as an index in cell 17)

## 1. Summary
Best-documented of the dev-only case studies, with a published paper that has Constantine as co-author. The gradients come with the data, and the notebook covers AS, a physics-guided change of variables, polynomial surrogates and a risk-factor estimate. Recommend **include in main**.

Problems, in order of severity:
1. The chain rule for the change of variables in cell 13 has a sign error. This is confirmed: the legacy R² values are reproduced exactly with the bug and change once it is fixed.
2. The input-distribution table contradicts **both** papers: JFM p. 13 and Table 2 (p. 14), and CTR 2014 p. 213. Both use uniform inputs with *standard deviations* σ_n = 10 % and σ_τ = 5 % of the mean. The notebook says Gaussian, with σ_n = 8.66 % and σ_τ = 4.33 %. The shipped data are Gaussian with the notebook's σ, so the notebook matches the data but not the papers.
   - Re-checked: the factor √3/2 does not reconcile them. A uniform with σ = 8.66 % would be ±15 %, yet 8.4 % of the data's n values lie outside ±15 % and per-column kurtosis is 2.5–3.8 (uniform: 1.8).
3. One sentence states the stability sign backwards.
4. The risk factors (9.4 % / 27.2 %) are far from the papers' Monte Carlo reference values, 14.77 % / 39.55 % (JFM Table 5, p. 25, and App. A, p. 27; CTR Table 2, p. 216), and nothing in the text explains the difference.
5. (New) The notebook cites only the JFM paper. The data most likely come from the 500-sample gradient set that both papers use: CTR figs. 5–6 ("using 500 samples", pp. 214–215) and JFM fig. 10 / Table 4, M = 500 (pp. 17, 22). The CTR proceedings should be cited as well.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| active-subspace-atacamac.ipynb | 981 KB | nbformat 4, kernel python2 | notebook | — | outputs saved (figures + R² table + RF) |
| atacamac-weak.txt | 896 KB | CSV, 1 header row, 492 rows × 78 cols | weak-coupling samples | `pn.DataFrame.from_csv` (index col `Runs`) | cols: Runs, n01..n19, tau01..tau19, maxFreq, dn01..dn19, dtau01..dtau19; no NaN |
| atacamac-strong.txt | 885 KB | same, 488 rows | strong-coupling samples | same | no NaN |

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "__S. Nicoud__, Université de Montpellier" | First name is Franck: the HAL cover reads "Franck Nicoud", and the reference list in the same cell already says "F. Nicoud". The JFM affiliation is "Université Montpellier 2, I3M UMR CNRS 5149" (p. 1); CTR p. 209 gives "University of Montpellier". The other names and affiliations match JFM p. 1: Bauerheim and Ndiaye CERFACS, Constantine Colorado School of Mines, Moreau Sherbrooke. CTR lists Bauerheim at SNECMA and adds G. Iaccarino. | "F. Nicoud" | confirmed |
| 0 | "FTF amplitudes … \| 0.5 \| 0.0433 \| Gaussian" and "time delays \| 0.000635 \| 2.75e-5 \| Gaussian" (also the strong table, 0.1516 / 3.18e-5) | JFM §3.1, p. 13: "Standard deviations are chosen as 5% (for τ) and 10% (for n) of their mean values … uniform PDFs have been used". Table 2, p. 14: σ_n = 0.05 (W) / 0.175 (S), σ_τ = 31.7 µs (W) / 36.7 µs (S). CTR p. 213: "standard deviations σ_n = 10% n̄ and σ_τ = 5% τ̄", and the MC uses "uniform distributions".

The data are Gaussian: pooled kurtosis 2.95 / 3.08, per-column 2.5–3.8, range/σ ≈ 8. Their σ is 0.0433 / 2.73e-5 (W) and 0.153 / 3.20e-5 (S), i.e. 8.66 % / 4.33 % of the mean, matching the notebook.

The ratio is √3/2, but no reading of the papers' percentages as uniform ranges explains the data. The notebook's σ = 0.05·√3·n̄ is the σ of a ±15 % uniform, yet 8.4 % of the n samples lie outside ±15 %. So the shipped data are **not** drawn from the stated distribution. | Keep the table as the data's distribution, add a sentence saying the shipped samples are Gaussian with these σ and differ from the papers' uniform 10 % / 5 % design, and ask the author (Q1) | confirmed (both papers' text and data statistics) | | Keep the table as the data's distribution, add a sentence saying the shipped samples are Gaussian with these σ and differ from the paper's uniform 10 %/5 % design, and ask the author (Q1) | confirmed (paper text and data statistics) |
| 0 | "$f_{Im} > 0$ implies stability of the mode and $f_{Im} < 0$ implies instability" | Backwards. JFM p. 5: "Each mode corresponding to a positive imaginary frequency (positive growth rate, Im(ω) > 0) is linearly unstable" (convention exp(−jωt)). Eq. 3.2, p. 15: RF = 100∫₀^∞ PDF. CTR p. 210 says the same. So f_Im > 0 means unstable. The next paragraph of the notebook and cell 20 (`ftilde > 0`) already use the correct sign. | "f_Im > 0: unstable; f_Im < 0: stable" | confirmed |
| 0 | "If it were perfectly symmetric, the modes would all be stable" | Not what the paper says. Symmetric configurations can be stable or unstable depending on (n, τ) (fig. 6). Tolerances split the mode and change the growth rate. | "Even a symmetric configuration may be unstable; random tolerances break the symmetry and change the growth rate" | likely |
| 0 | "growth rate of the first thermoacoustic mode" | The paper (§4.2, p. 20) uses "the maximum growth rate of the two azimuthal components", which is what the column name `maxFreq` refers to. Units are s⁻¹ (CTR fig. 4 axis "Im(f) (s⁻¹)", p. 213). The data's mean/σ (W −1.47 / 1.14, S −2.05 / 5.34) are on the same scale as the CTR MC values (W −2.36 / 1.68, S −3.53 / 6.49) and smaller, consistent with the smaller input σ. | Say so, relabel the column as `f_Im` [s⁻¹] in the text | likely |
| 0 | "[1] … J. Fluid Mech., 789, 534-566" | Year missing. The HAL v2 cover gives "Journal of Fluid Mechanics, 2016, 789, pp.534-566. ⟨10.1017/jfm.2015.730⟩". The local PDF (HAL v1) cannot confirm the journal pages. | add "(2016)" and the DOI link text | confirmed (HAL v2 cover) |
| 0 | (references) | Only JFM is cited, but the data are most likely the shared 500-sample gradient set (finite differences: JFM p. 16, CTR p. 214) of the CTR 2014 study and of JFM fig. 10 / Table 4. The notebook's surrogate names FL/RL/RQ/RC follow JFM Table 3 (p. 21); CTR uses L38/L3/Q3/AM. The legacy in-sample R² values .809/.789/.964/.964 (weak) are close to JFM Table 4's validation accuracies at M = 500 (FL .79, RL .80, RQ .96, RC .96; p. 22). | Add [2] Bauerheim et al., Proc. CTR Summer Program 2014, pp. 209–218, as the companion study, and say the data are the M ≈ 500 gradient samples; ask the author to confirm (Q1) | likely |
| 0 | "dscribing", "circumfrential", "aparatus", "paramter", "eigendecomosition" | typos | fix | confirmed |
| 10 | "we expect 1, 3, or 5-dimensional subspaces" | Consistent with the data: weak λ/λ₁ = 1, .313, .282, .021, .019, .004. The paper says 5 before the change of variables and 3 after. | keep; quote the paper | confirmed |
| 12 | "$(n_i,\tau_i)\rightarrow (\Re(n_ie^{j\omega^0\tau_i}),\Im(…))$" | The paper (eq. 4.2, p. 19; CTR eq. 3.3, p. 215) maps to (Re F̂, Im F̂), the **spatial DFT** over the 19 burners of F_i = n_i e^{jω⁰τ_i}. The notebook omits the DFT. The DFT is a real-orthogonal map up to a factor √N, so without per-coordinate standardization the eigenvalues are the same. The eigenvectors differ by that rotation, so the cell-14 stem plot cannot be compared with paper fig. 14. | Add the sentence and ω⁰ = πc₀/L_c (JFM p. 19; c₀ = 832 m/s, L_c = 0.54 m from JFM Table 1, p. 11; matches `omega0 = np.pi*832/.54`). Optionally plot W₁ in Fourier coordinates to reproduce fig. 14 | confirmed |
| 15 | "the active subspace (only 3-dimensional)" | ok (paper: 3 after change of variables) | — | confirmed |
| 18 | "R² … increases from .809 … to .964" | The numbers come from the buggy gradients. With the chain rule fixed: weak 0.797 / 0.979 / 0.980, strong 0.545 / 0.572 / 0.610 (full linear unchanged at 0.809 / 0.583). | Regenerate; the text must cite the computed values | confirmed (prototype) |
| 18/20 | output "weak case has risk factor 9.392% … strong case … 27.16%" | Papers' reference (MC with 10 000 ATACAMAC runs, uniform inputs): 14.77 % / 39.55 % (JFM Table 5, p. 25; App. A, p. 27; CTR Table 2, p. 216; confirmed in both). JFM's surrogate estimates: RQ1000 14.94 / 39.03, RL1000 11.94 / 33.17 (Table 5). In the shipped data the fraction f > 0 is 10.37 % / 37.91 %. The strong-case cubic surrogate (R² ≈ 0.6) underestimates RF by about 10 points, which matches the paper's conclusion that polynomials are not suited to Scase. | Report three numbers per case (data fraction, surrogate MC, paper) and say why they differ: different input distribution, and a poor surrogate in Scase | confirmed |

## 4. Code review
- **Bug, cell 13 (confirmed).** `G_W_prime[:,i] = G_W[:,i]/s_n*np.sqrt(1+(tau/n)**2)**-1 - …`. Here `n` = a = Re F and `tau` = b = Im F, so `(1+(b/a)²)^{-1/2}` = |a|/r, but the chain rule needs ∂n/∂a = a/r. Since ω⁰τ ≈ 3.07 rad, a < 0 in 100 % of rows (both cases), so the n-gradient contribution to ∂f/∂a has the wrong sign. The ∂f/∂b line is correct. Same bug in the strong case.
  - Correct: `Ga = fn*a/r - ft*b/(w0*r**2)`, `Gb = fn*b/r + ft*a/(w0*r**2)`, where `fn, ft` are physical gradients and `r = hypot(a, b)`.
  - Effect: strong-case eigenvalues λ/λ₁ change from 1, .457, .407, .230, .213 (legacy) to 1, .318, .287, .107, .097 (fixed). The gap after 3 becomes visible. The weak case is barely affected.
- Naming: cell 13 reuses `n`, `tau` for Re/Im F. Rename them `a`, `b`.
- Gradient coordinates: the stored `dn*`, `dtau*` are derivatives w.r.t. the **standardized** inputs (x−μ)/σ (likely). Evidence: mean(G)/OLS-slope ≈ σ_τ (2.77e-5 vs 2.73e-5, weak), and cell 13 divides by σ. Cell 7 treats them as normalized gradients, which is consistent.
- Cell 16 repeats the standardization from cell 13. Delete the duplicate.
- Cells 5/19 build `xstd` twice, and also `xmean`. Define them once.
- Randomness: `nboot=500` and `np.random.multivariate_normal` (N = 50 000) are unseeded. Use `rng = np.random.default_rng(<seed>)` and pass `rng=` to v1.
- `np.random.multivariate_normal` with a diagonal covariance is wasteful. Use `mu + sd*rng.standard_normal((N, 38))`.
- `cdf = np.vectorize(lambda x: …len(ftilde[ftilde <= x])…)` is O(N·200). Replace with `np.searchsorted(np.sort(ftilde), x, side='right')/N`.
- Cell 17: `RS[i%2][i/2]` → `i//2`; the `print` statements need py3 syntax.
- Cell 14: `plt.stem(x, y, 'r-', 'rs')` works in mpl 3.11; check that the legend and labels still render.
- No hard-coded paths. The data are loaded relative to the notebook's directory.

## 5. Data management review
- Format: CSV with a header and a 1-based `Runs` index column, %.16e. 78 columns. No NaN, no units in the header. Units: n is dimensionless, τ is in seconds, f_Im is in s⁻¹ (paper: Hz; unverified).
- Normalization: the raw physical inputs are stored. The gradients are stored in standardized coordinates (see §4), and the header does not say so. Document this in a markdown cell, or in a README for the directory.
- Size: 0.9 MB each, far below GitHub limits.
- Provenance: ATACAMAC runs by M. Bauerheim (CERFACS/SNECMA), with gradients by finite differences (JFM p. 16; CTR p. 214).
  - The 492/488 rows match the papers' 500-sample gradient sets (CTR figs. 5–6; JFM fig. 10, Table 4 M = 500), presumably after failed runs were dropped. They do not match the 1 000 of JFM fig. 14 or CTR fig. 7.
  - The distribution does not match the stated design: Gaussian instead of uniform, σ smaller by √3/2. Ask the author whether these are the paper's samples (Q1). License: not stated; ask (Q3).
- Redundancy: none.
- Tested pandas-free loader (.venv, numpy 2.4):
```python
import numpy as np
def load(case):                       # case in {"weak", "strong"}
    fn = f"atacamac-{case}.txt"
    names = open(fn).readline().strip().split(",")
    D = np.loadtxt(fn, delimiter=",", skiprows=1)
    X, f, G = D[:, 1:39], D[:, 39], D[:, 40:]
    return X, f, G, names[1:39]
```
Verified: weak (492, 78) and strong (488, 78), no NaN.

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `ac.utils.misc.UnboundedNormalizer(xmean, xstd**2).normalize(X)` | `space = ac.Gaussian(mean=mu, cov=sd**2)` (cov may be the `(m,)` variances) | v1 normalizes internally; pass physical X |
| `ss.compute(df=G, nboot=500)` (G in standardized coords) | `ac.active_subspace(space=space, X=X, grads=G/sd, nboot=500, rng=seed)`, **or** `ac.active_subspace(space=ac.Gaussian.reference(38), X=(X-mu)/sd, grads=G, …)` | Both give identical eigenvalues (verified). Do **not** pass G with the physical space, or it is scaled twice |
| `ss.eigenvals`, `ss.eigenvecs` | `sub.eigenvalues`, `sub.eigenvectors` | v1 sign rule (largest-magnitude component positive) may flip columns |
| `ss.e_br`, `ss.sub_br` | `ac.plotting.plot_eigenvalues(sub, interval="minmax")`, `ac.plotting.plot_subspace_errors(sub)` | default band is percentile |
| `ss.partition(3)`; `ss.W1` | `ac.active_subspace(..., n=3)`; `sub.W1` | auto rule picks n = 1 here; fix n = 3 per paper |
| change-of-variables subspace | `ac.active_subspace(space=ac.Gaussian.reference(38), X=XXp, grads=Gp, n=3, nboot=500, rng=)` with `XXp` empirically standardized and `Gp = [Ga, Gb]*sd_p` | The transformed inputs are not Gaussian. Using the reference space with empirically standardized coordinates reproduces the legacy computation; say so in the text |
| `PolynomialApproximation(k).train(Y, f)`; `.Rsqr`; `.predict(Y)[0]` | `sur = ac.fit_surrogate(X=XXp, f=f, space=ref, subspace=sub, kind="poly", degree=k)`; R² = 1 − Σ`sur.residuals`²/SST; `sur.evaluate(Y)` with **active coordinates** Y = XXp @ W1 | `evaluate` takes Y (M, n), not X. Full-space linear: `subspace=None` |
| `ac.utils.plotters.*` | `ac.plotting.plot_eigenvalues`, `plot_subspace_errors`, `plot_eigenvectors`, `plot_sufficient_summary` | |

## 7. Associated .py files
None.

## 8. Update plan
1. Restore the directory into main as part of the branch merge (BRANCH_MERGE.md §2). No content change in that commit.
2. Convert to py3. Replace pandas with the loader in §5, and define the constants (μ, σ, ω⁰) once in a single cell.
3. Fix the chain rule in cell 13 and rename `n, tau` → `a, b`. Add a markdown cell with the derivative formulas.
4. Port the ac calls to v1 (table §6). Seed everything with one `default_rng`.
5. Fix the text (table §3): stability sign, Nicoud, the distribution note, DFT remark, the paper year, typos. Add the CTR 2014 proceedings as reference [2] and state that the data are the ~500-sample gradient set.
6. Fix the risk-factor section. Use a vectorized CDF and report the data fraction, the surrogate estimate and the paper reference side by side, with the explanation.
7. Add the Acknowledgments cell used by the master notebooks (DOE DE-SC-0011077), and Ryan Howard's affiliation line as on the other notebooks.
8. Execute headless and commit with outputs. Add the README bullet: "**Atacamac:** a 38-parameter thermoacoustic model of an annular combustor (two coupling regimes). [paper](https://doi.org/10.1017/jfm.2015.730)".

## 9. Acceptance criteria
- `jupyter nbconvert --to notebook --execute` from `Atacamac/` in .venv succeeds in < 2 min, with no pandas import.
- Original coordinates, λ/λ₁[:5]: weak 1, .313, .282, .021, .019 and strong 1, .312, .305, .066, .065 (±0.01 for the point estimates; the bootstrap differs).
- Changed coordinates with the fixed gradients, λ/λ₁[:5]: weak 1, .314, .282, .021, .019 and strong 1, .318, .287, .107, .097.
- In-sample R², fixed gradients, n = 3: weak full-linear .809, RL .797, RQ .979, RC .980; strong .583, .545, .572, .610 (±0.005).
- Risk factor printed for each case: data fraction (10.37 %, 37.91 %), surrogate MC (≈ 9.6 %, ≈ 27.5 % with N = 50 000), paper reference (14.77 %, 39.55 %).
- Figures: 2× eigenvalues, 2× subspace errors, 2× eigenvectors (original coordinates), 2× eigenvalues plus a W₁ stem plot (changed coordinates), an 8-panel predicted-vs-true plot, 2× exceedance curves.
- The text says f_Im > 0 is unstable, everywhere.

## 10. Prototype results
Scripts: `prototypes/g6/ata_proto.py`, `ata_v1.py`, `ata_rf.py` (run with .venv python on the dev extract).
- Loader: OK (see §5).
- Data statistics: Gaussian; σ matches the notebook, not the paper (§3).
- Re-checked against the local PDFs (JFM HAL v1, CTR 2014):
  - The distribution contradiction holds; per-column kurtosis is 2.5–3.8, and 8.4 % of n lie outside ±15 %.
  - RF 14.77 / 39.55 and the stability sign are confirmed in both papers.
  - The legacy in-sample R² values are close to JFM Table 4 at M = 500.
- v1 `active_subspace` with `Gaussian(mean, cov)` and G/σ is identical to the reference-space path.
- **Legacy reproduction with the buggy chain rule:** R² weak .809 / .789 / .964 / .964 and strong .583 / .536 / .546 / .589. These are exactly the legacy cell-17 output, which confirms both the v1 surrogate path and the presence of the bug.
- Fixed chain rule: see §9. Surrogate RF, fixed: 9.62 % / 27.54 %. With the buggy chain rule: 9.63 % / 27.26 %. The legacy notebook printed 9.392 % / 27.16 %; the difference is MC noise and the unseeded bootstrap.

## 11. Open questions for the author
1. Are `atacamac-weak/strong.txt` the 500-sample gradient sets of CTR 2014 (figs. 5–6) and JFM (fig. 10, Table 4)? Their distribution is Gaussian with σ = 8.66 %·n̄ and 4.33 %·τ̄. Both papers state uniform with σ = 10 % and 5 %, and no uniform reading fits the data (§3). Was the gradient set drawn differently from the 10 000-run MC reference? Should the notebook describe the data or the papers?
2. Should the notebook reproduce paper fig. 14 (eigenvectors in Fourier coordinates F̂(k))?
3. Data license and permission from Bauerheim / CERFACS to keep redistributing.
4. Is the column `maxFreq` the maximum of the growth rates of the two split azimuthal components (paper §4.2)? Units: s⁻¹ per CTR fig. 4 (likely); confirm.
