# Hydrology: update spec

Kind: **stored data**: ParFlow-CLM runs (see [README.md](../README.md#two-kinds-of-data-sets)).

Status line: `Hydrology/PF-CLM_Active_Subspaces.ipynb` | paper: Jefferson, Gilbert, Constantine, Maxwell, Comput. Geosci. 83 (2015) 127–138, doi:10.1016/j.cageo.2015.07.001 (reprinted in Comput. Geosci. 90 (2016) 78–89, doi:10.1016/j.cageo.2015.11.002) | m = 19 for grass, 3 for bare soil (the README says 20) | QoI: total latent, sensible and ground heat over 144 h, single-column domain, bare soil and grass (6 OLS subspaces) | legacy API: `Subspaces.compute(sstype='OLS', nboot=500)`, `utils.misc.BoundedNormalizer`, `utils.response_surfaces.PolynomialApproximation` | py2 issues: yes (`i/2` used as a list index in cell 11; pandas `.as_matrix()`).

## 1. Summary
The notebook reproduces the single-column part of the paper. It fits one global linear model (OLS) per output for 300 runs and draws the weights, a quadratic fit with confidence and prediction bands, and a "lai ± 1" / inverse-problem illustration. Both the data and the legacy outputs are consistent: from the shipped files I recomputed every legacy number exactly (weights, and R² = 0.95/0.92/0.99/0.82/0.83/0.96). Main problems:
- The README gives the parameter count as 20; it is 19. The markdown table has 20 rows because z0m appears twice, once for bare soil and once for grass.
- The input space is mixed: 18 uniform inputs plus log z0m, which is Gaussian. v1 now has `ac.Product` (V1-06), and the port uses it ([#40](https://github.com/paulcon/as-data-sets/pull/40)). When this spec was written v1 had no product space.
- The confidence and prediction bands use the leverage formula for a straight line on a quadratic fit, with z = 2.33. They are wrong away from the data mean.
- The nominal point in cell 13 is outside the sampled box: rhol_nir = 0.58 > 0.55, and log z0m is at +3.08σ. The "lai ± 1" predictions are therefore extrapolations.
- Signs: the paper orients w = â/‖â‖. Evidence: p. 18 l. 376–378 says the lai weight is positive for latent heat and negative for ground heat, and every C1 in Table 4 is > 0. The legacy figures (first component > 0) are therefore mirrored relative to the paper for bare GH, grass SH and grass GH; the legacy grass-GH lai weight is +0.69. The v1 rule (largest component > 0) disagrees with â in 4 of 6 cases. Use â orientation. The hard-coded text and arrow positions in cell 13 depend on the legacy sign and data.
- The shipped "updated model evaluations" differ materially from the paper:
  - R²: paper Table 4 0.98/0.92/0.99/0.89/0.84/0.95 vs data 0.95/0.92/0.99/0.82/0.83/0.96.
  - Output scale: paper C0 for bare LH is 8.41e3 "W m⁻²" vs data mean 1.44e6, about 171×.
  - Active variable at the defaults: paper 0.28 (p. 19) vs data 0.236.
  - lai ± 1: paper −7.5% / +8.5% vs data −11.0% / +14.1%.
  - Grass-LH ranking: paper l. 319–321 gives lai, dmx, fc, z0m, wt, wp; data gives fc, lai, wt, wp, sai, dmx, with z0m at 0.06.
  The notebook must not quote paper numbers as if they were reproduced.
- Errors and inconsistencies in the paper itself, so the notebook should not copy them:
  - Step 1 on p. 13 (l. 273) says "draw xi uniformly at random from a 19-dimensional hypercube". Eq. (14) and App. A.1 (l. 486–487) say log z0m is normal; the data and notebook follow A.1.
  - In Table A1, the quadratic R² is below the linear R² for bare GH (0.9900 < 0.9907, single column; 0.9900 < 0.9945, tilted-v). This is impossible for nested in-sample least-squares fits.
  - The default rl_n = 0.58 in Table 2 lies outside the paper's own range U(0.35, 0.55).

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| PF-CLM_Active_Subspaces.ipynb | 397 KB | nbformat 4, python2 kernel | notebook | — | 3 PNG outputs |
| SC_ScatterParameters_grass.txt | 26.7 KB | space-separated, 1 quoted header row, 300×19 | grass inputs (physical) | `pn.read_csv(sep=' ')` | column order differs from the paper; the notebook reorders by name. Values are rounded to 1–2 significant figures (8–89 unique values per column). soilcol is an integer 1–8 |
| SC_ScatterParameters_bare.txt | 4.6 KB | same, 300×3 | bare-soil inputs | same | columns soilcol, z0m, watertable; z0m min 2e-6 (log −13.1) |
| ScatterSummary_grass.txt | 25.3 KB | space-separated, no header, 300×5 | outputs | `pn.read_csv(header=None)` | only columns 0–2 (LH, SH, GH) are used. Columns 3 (~7–22) and 4 (~302–308, probably a temperature in K) are undocumented |
| ScatterSummary_bare.txt | 25.3 KB | same, 300×5 | outputs | same | same |

## 3. Text and math review (vs. literature)
Checked against the author manuscript of the reprint (Comput. Geosci. 90:78–89; same content as 83:127–138), manuscript line numbers "l.", pages "p.". Sources used: the parameter table (Table 2, p. 32), the method (§3, p. 12–14, eqs. 9–10), the normalization (eqs. 13–14, p. 17), App. A.1 (p. 23–24), Table 4 (p. 34), Table A1 (p. 35), and the Fig. 5 and Fig. 8 captions (p. 40, 43).

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| README | "a 20-parameter integrated surface/subsurface hydrologic model" | the paper says 19 (abstract l. 35; l. 123; l. 269 "x contains 19 components"). Paper Table 2 itself has 20 rows because z0m appears twice | "19-parameter" | confirmed (paper) |
| 0 | table rows "Aerodynamic roughness length (bare soil) z0m LN(−8.83, 1.47)" and "(grass) z0m LN(−5.11, 0.52)" | one parameter on two rows makes the table read as 20. LN(μ,σ) is not defined: μ and σ are the mean and SD of log z0m, which the data confirm (sample −8.94/1.53 and −5.10/0.50) | merge into one row with two distributions; add "log z0m ~ N(μ, σ²)". The notebook table copies paper Table 2, including the duplicate row. Table 2 footnote a says "LogNormal(mean, standard deviation)", and eq. (14), x = (log x′ − μ)/σ, fixes μ and σ as moments of log z0m | confirmed (paper Table 2, eq. 14; data) |
| 0 | "Each domain was run with uniform cover of bare soil (which takes only the final 3 parameters)" | correct, but "final 3" depends on the table order: z0m, soilcol, watertable | name them | confirmed |
| 0 | "using a finite difference approximation would be computationally infeasible (requiring 19 additional model evaluations per gradient sample)" | fine | — | — |
| 0 | OLS algorithm paragraph (validity) | it does not say that OLS recovers a direction in the span of E[∇f] only under the linearity condition. With uniform inputs this is a heuristic, which v1 now warns about | add one sentence and cite Constantine 2015 §1.4 / Li 1991 | confirmed (v1 docs) |
| 0 | "We'll also be using updated model evaluations compared to [1] … the figures presented will look slightly different than in the paper. The significant conclusions are the same." | The differences are larger than "slightly": R² for grass LH is 0.82 vs 0.89 (Table 4). Output magnitudes are ~171× the paper's C0 (units differ; Table 4 is "W m⁻²", p. 17 l. 356). For grass LH, z0m has weight 0.06 here but is among the top 4 in the paper (l. 319–320), and dmx drops from 2nd to 6th | state the differences. Keep "conclusions the same" only for bare soil (z0m dominant) and ground heat (lai, sai). Provenance of the runs: ask (§11) | confirmed (paper vs data) |
| 0 | OLS algorithm paragraph | paper p. 13 step 1 says x is uniform on [−1,1]^19, which contradicts its own eq. (14) and App. A.1. The notebook correctly uses the product of uniform and normal | do not copy step 1; state ρ as in A.1 | confirmed (paper) |
| 5 | "normalize the normal inputs (log(z0m)) to be (x−μ)/σ" | correct and matches the code | — | confirmed |
| 6 | "the sstype=5 flag specifies to use the algorithm described above" | stale: the code uses `sstype='OLS'` | "estimator='ols'" | confirmed |
| 8 | "since normalized eigenvectors are unique up to a sign, some plots may appear vertically flipped" | the paper fixes the sign: w = â/‖â‖ (eq. 10; p. 18 l. 374–378, where the lai weight is positive for LH and negative for GH; Table 4 C1 > 0). With â orientation, no panel is flipped relative to the paper | orient by â (§6) and drop the "may appear flipped" caveat | confirmed (paper) |
| 10 | weight interpretation ("lai, fieldcap, wiltpt, watertable strongly affect latent heat", …) | matches the recomputed weights (lai .53, fieldcap −.56, wiltpt −.34, watertable .45); sai −.23 is not mentioned | optionally add sai | confirmed |
| 10/11 | "confidence limits (dark blue shading) and prediction limits … using +/- 2.33 std errors" | `conflim` uses the leverage of a simple linear regression, 1/M + (x−ȳ)²/Sxx, for a quadratic fit. The correct leverage is b(x)ᵀ(BᵀB)⁻¹b(x) with b = [1, y, y²]. 2.33 is the normal 0.99 quantile; for a two-sided band it should be the t quantile with M−3 dof. Bands are drawn on [−3, 3], but the data cover about [−1.5, 1.8] (grass). The paper's Fig. 5 caption (p. 40) shows a "95% confidence interval" only, with no prediction band | compute the leverage from the design matrix and use `scipy.stats.t.ppf(0.975, M-3)` to match the paper's 95%; state the level; restrict to the data range or shade the extrapolation | confirmed (math; paper Fig. 5 caption) |
| 11 | R² shown on the summary plots | paper Table A1 (p. 35) is internally inconsistent: quadratic R² < linear R² for bare GH (0.9900 vs 0.9907 single column; 0.9900 vs 0.9945 tilted-v), which is impossible for nested in-sample least squares. On the shipped data, quadratic ≥ linear as it must be | if the notebook adds a linear-vs-quadratic table, compute it; do not copy Table A1 | confirmed (math) |
| 12 | "suppose we fixed parameters at nominal values (see Table 2 of [1])" / cell 13 `input_nominal` | The 18 defaults match paper Table 2 exactly, including grass z0m = 0.03 m. But Table 2 itself lists rl_n default 0.58 outside its range U(0.35, 0.55) (normalized 1.30). log(0.03) is +3.08σ for grass z0m, and xl = −0.3 is on the boundary. Table 2 gives **no** default for wt ("--"); the notebook's `-1` (the upper bound) is unsourced. The paper's Fig. 8 also shows a z0m example that the notebook omits | say that the default point lies outside the sampled box (rl_n, z0m), so lai ± 1 extrapolates; say wt = −1 is the notebook's choice. Paper values to compare: y_nom 0.28, lai −1/+1 → −7.5%/+8.5% (p. 19 l. 385–390); data gives 0.236 and −11.0%/+14.1% | confirmed (paper Table 2; data) |
| 13 | `axes[0].text(.2, .52, 'lai = 2', transform=axes[0].transAxes)`, `arrow(0, .7, .87, 0, …)` | annotation positions are hard-coded for the legacy sign and data and break when w flips | compute positions from y_nom, f_nom, y_laip1, … | confirmed |
| 0 | author block, five @mines.edu addresses | emails are probably stale (Howard, Gilbert, Jefferson) | ask the author whether to keep them | unverified |
| 0 | ref [1] "Comput. Geosci., 83, 127-138, 2015" | correct; the 2016 reprint (Comput. Geosci. 90:78–89) could also be cited | optional | confirmed |
| 0 | ref [2] Constantine, Dow, Wang 2014 SISC 36(4) A1500–A1524 | correct | — | confirmed |

## 4. Code review
- py2: cell 11 indexes lists with `i/2` (`out_labels[i/2]`, `subspaces[i%2][i/2]`), which is a TypeError in py3; use `i//2`. `in_labels_grass = inputs_grass.keys()` is a pandas Index; replace it with a list.
- pandas: `.as_matrix()` (removed in pandas 1.0), `read_csv`. Replace with `np.loadtxt` (§5).
- Wasted work: `nboot=500` on six OLS fits computes bootstraps that are never used. Remove.
- Cell 11 retrains the module-level `RS` six times, and cell 13 retrains it on grass LH. The surrogate state is global and mutable; give each surrogate its own name.
- `ax.hlines(0, 0, 20, …)` and `range(1,20)`, `range(17,20)` are hard-coded for m = 19. Derive them from `len(names)`.
- `vlines(y_nom, .5e6, …)` and `fill_between(…, .5e6, …)` hard-code a y floor; use `ax.get_ylim()[0]`.
- No randomness after the bootstrap is removed, so the notebook is deterministic.
- There are no hard-coded absolute paths; the notebook uses relative file names.

## 5. Data management review
- Format: plain text, space-separated. Input files have a quoted header row; output files have none. There is no index column. Units are not recorded anywhere. The paper says g returns "total latent, sensible or ground heat per unit area (Wm-2)" (p. 17 l. 355–358), with C0 ≈ 8.4e3 for bare LH, but the shipped values are ~1.4e6. Their units and normalization are unknown (author to confirm). Input units are in paper Table 2 (dmx mm, dl m, z0m m, wt m).
- Normalization is not stored; the bounds live in cell 5 and the markdown table. Add a small `param_ranges.txt` (name, dist, a, b) like the one in Stomatal_Resistance so the bounds are data, not code.
- Output columns 3–4 are unexplained. Document them or drop them (ask the author).
- Size is small (≈80 KB total). There is no license or provenance statement; the whole repo has no LICENSE file.
- pandas-free loading, tested in .venv:
```python
import numpy as np
def read_named(fn):
    with open(fn) as fh:
        names = [h.strip('"') for h in fh.readline().split()]
    return names, np.loadtxt(fn, skiprows=1, ndmin=2)
ORDER_G = ['dewmx','lai','lai0','sai','xl','dleaf','rhol_vis','rhol_nir','rhos_vis','rhos_nir',
           'taul_vis','taul_nir','taus_vis','taus_nir','fieldcap','wiltpt','z0m','soilcol','watertable']
ORDER_B = ['z0m','soilcol','watertable']
hg, Xg = read_named('SC_ScatterParameters_grass.txt'); Xg = Xg[:, [hg.index(n) for n in ORDER_G]]
hb, Xb = read_named('SC_ScatterParameters_bare.txt');  Xb = Xb[:, [hb.index(n) for n in ORDER_B]]
Fg = np.loadtxt('ScatterSummary_grass.txt')[:, :3]   # LH, SH, GH
Fb = np.loadtxt('ScatterSummary_bare.txt')[:, :3]
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `BoundedNormalizer(yl,yu).normalize` + hand `(log z0m − μ)/σ` | keep the hand normalization (helper `normalize(X)`); pass `XX` with `ac.Gaussian(mean=np.zeros(m), cov=np.ones(m), names=ORDER)` | **Superseded: the port uses `ac.Product` (V1-06, [#40](https://github.com/paulcon/as-data-sets/pull/40)).** At the time of writing v1 had no mixed Uniform×Gaussian space. `Uniform.reference` rejects the Gaussian column (`input.out_of_space`, tested). For OLS the space is used only to normalize, so a reference Gaussian with pre-normalized data gives identical numbers (tested: λ₁ and w equal to legacy to rounding) and no ValidityWarning. The cost is that the space is mislabelled; do not call `space.sample`, `active_domain` or `qphd` on it. Alternative: ask for a product space in v1 (§11) |
| `ss.compute(X=XX, f=f[:,None], nboot=500, sstype='OLS')` | `ac.active_subspace(X=XX, f=f, space=sp, estimator="ols", n=1)` | drop nboot. λ₁ = ‖â‖² matches legacy exactly |
| `ss.W1` | `sub.W1[:, 0]`, then orient: `w *= np.sign(w @ a_hat)`, where `a_hat` is the OLS slope (`np.linalg.lstsq` on `[1, XX]`) | v1 makes the largest-magnitude component positive [D-81]. Against legacy it flips grass LH only. Against â it disagrees for bare GH and grass LH, SH and GH. â/‖â‖ is the paper's convention (eq. 10; p. 18), so the figures match the paper's orientation. The legacy notebook was mirrored for bare GH, grass SH and grass GH |
| `PolynomialApproximation()` (N=2) `.train(y, f)`, `.predict(x)[0]`, `.Rsqr` | `s = ac.fit_surrogate(X=XX, f=f, subspace=sub, kind="poly", degree=2)`; `s.evaluate(ygrid[:,None])`; in-sample `1 - sum(s.residuals**2)/sum((f-f.mean())**2)`; `s.loo_r_squared` | in-sample R² equals legacy (0.952/0.917/0.992/0.819/0.826/0.964); LOO is 0.001–0.005 lower. Show LOO and say which one is shown |
| `conflim` | hand-written: `B = np.vander(y, 3)`, `h(x) = b(x)ᵀ (BᵀB)⁻¹ b(x)`, `t = scipy.stats.t.ppf(q, M-3)` | v1 has no prediction intervals |
| `XX_nom.dot(ss_grass_LH.W1)` | `normalize(nominal) @ w` | same; recompute `y_laip1`, `y_laim1` |
| matplotlib weight plot | keep the custom two-colour plot (grass and bare on one axis); `ac.plotting.plot_eigenvectors` cannot overlay two subspaces of different m | — |

## 7. Associated .py files
None.

## 8. Update plan
1. Fix README "20-parameter" → "19-parameter".
2. Replace pandas loading with the §5 snippet; add a `normalize()` helper with bounds in one place (optionally `param_ranges.txt`).
3. Port to py3 (`//`, lists, kernelspec python3) and v1 OLS via `ac.active_subspace`, with the sign orientation from â; drop nboot.
4. Replace the response surface with `ac.fit_surrogate(kind="poly", degree=2)`; report LOO R².
5. Rewrite `conflim` with the correct quadratic leverage and a t quantile; state the level; limit the x-range to the data or mark extrapolation.
6. Cell 13: compute annotation positions from data. Add a sentence that the nominal point lies outside the sampled ranges (rhol_nir, z0m).
7. Text fixes: merge the z0m rows, define LN(μ,σ), change "sstype=5" to `estimator="ols"`, add the OLS validity caveat, state the â sign convention (as in the paper), optionally add the reprint reference. Replace "slightly different" with a short paper-vs-data comparison (R², ranking for grass LH, y_nom and lai ± 1 percentages); state the band level (95%, as in paper Fig. 5).
8. Execute headless and commit with outputs.

## 9. Acceptance criteria
- `cd Hydrology && jupyter nbconvert --to notebook --execute PF-CLM_Active_Subspaces.ipynb` succeeds in `.venv` in < 30 s with no pandas import and no uncaught warnings.
- For all 6 cases, λ₁ equals the legacy ‖â‖² to rtol 1e-10: bare LH 1.98448e10, SH 1.05702e10, GH 2.06168e9; grass LH 6.68147e10, SH 7.68786e10, GH 6.77473e9.
- |w_v1 · w_legacy| = 1 to 1e-10.
- Signs follow the paper (â orientation): grass lai weight > 0 for LH and < 0 for GH (paper p. 18). The quadratic C1 > 0 in all 6 fits (paper Table 4).
- In-sample quadratic R² (2 dp) is 0.95/0.92/0.99 (bare LH/SH/GH) and 0.82/0.83/0.96 (grass); LOO is shown as well.
- Figures: (1) weights for 3 QoIs, grass and bare overlaid; (2) 3×2 summary plots with quadratic fit and correct bands; (3) lai ± 1 and inverse-problem panels with annotations placed from the data.

## 10. Prototype results
Ran `prototypes/g4/proto_hyd.py` (and `prototypes/g4/nom.py` for the default-point check) with the .venv python:
- Loading with numpy only works. Grass is 300×19 and bare is 300×3; uniform columns normalize exactly into [−1, 1].
- `ac.active_subspace(estimator="ols", space=Gaussian(0, I))` on hand-normalized data matches the legacy OLS (numpy lstsq) to rounding for all 6 cases: same λ₁, |cos| = 1.000000000000.
- v1 sign relative to legacy: +,+,+ for bare and −,+,+ for grass. Sign of â relative to v1: +,+,− for bare and −,−,− for grass.
- The quadratic in-sample R² reproduces the R² printed in the legacy figure exactly. Legacy figures were inspected: weights match visually, and the R² values match.
- `ac.Uniform` on the 18 uniform columns raises `ValidityWarning` for OLS (by design). `Uniform.reference` with the Gaussian column raises `input.out_of_space`.
- Nominal check: rhol_nir is out of bounds; the log z0m z-score is 3.08. With â orientation and wt = −1: y_nom = 0.236 (paper 0.28); lai = 1 / 3 → −11.0% / +14.1% (paper −7.5% / +8.5%).
- With â orientation, grass weights are LH lai +0.53, GH lai −0.69. Signs agree with the paper text (p. 18).

## 11. Open questions for the author
1. Should v1 get a product (mixed Uniform/Gaussian) input space, or is a pre-normalized reference Gaussian acceptable here? **Answered:** v1 added `ac.Product` (V1-06), and the port uses it.
2. What are output columns 3 and 4 of `ScatterSummary_*.txt`? What are the units and normalization of LH/SH/GH? The values are ~171× the paper's Table 4 C0, which is labelled W m⁻².
3. Paper Table 2 lists rl_n default 0.58 outside its range U(0.35, 0.55): is this a typo in the range or the default? Table 2 gives no wt default: is −1 m right for the cell-13 example?
4. Provenance of the "updated model evaluations" (date, ParFlow-CLM version), and should the tilted-v runs be added?
5. (Resolved by the paper: w = â/‖â‖, eq. 10 and p. 18.) Confirm that the notebook should follow it.
6. Data license for the repo (no LICENSE file).
