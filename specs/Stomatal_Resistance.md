# Stomatal_Resistance: update spec

Status line: `Stomatal_Resistance/Stomatal_Resistance.ipynb` | papers: J. Jefferson, PhD thesis, Colorado School of Mines, 2016, "Exploring sensitivities of latent heat parameterizations using a coupled, integrated hydrologic model", hdl:11124/170080; journal version: J.L. Jefferson, R.M. Maxwell, P.G. Constantine, "Exploring the sensitivity of photosynthesis and stomatal resistance parameters in a land surface model", J. Hydrometeorol. 18(3) (2017) 897–915, doi:10.1175/JHM-D-16-0053.1 (the notebook title is the journal title) | m = 20 (README 20, correct) | QoI: transpiration (tran_veg) and sunlit and shaded stomatal resistance (rssun, rssha) at Tonzi Ranch, as 12 monthly means and 14 JJA hourly means → 78 OLS fits | legacy API: `Subspaces.compute(sstype='OLS')`, `BoundedNormalizer` | py2 issues: minor (no `print`; pandas `DataFrame.from_csv` / `.as_matrix()` removed).

## 1. Summary
The notebook fits a global linear model per QoI and time slice. It plots the weights per slice, the weights against time, and summary plots. The shipped data reproduce the legacy direction exactly. Main problems:
- **301 samples, not 300.** Row 0 is the default-parameter simulation. All 20 values equal the "Default value" column of JHM Table 1 (p. 901). The paper keeps it separate from "the 300 realizations" (p. 909: "the default simulation and the 300 realizations"). Its effect on w is negligible (cos ≥ 0.9997), but the text is wrong, and strictly row 0 should be excluded from the fits to match the paper.
- **wj1 and wj2 are discrete, not U(1,4.5) / U(2,10.5).** JHM Table 1 (p. 901) lists their "Range" as the sets "1, 4, 4.5" and "2, 8, 10.5" (hard-coded constants; Chen et al. 1999; Oleson et al. 2013). The data take exactly those levels, about 100 runs per level, crossed independently in all 9 combinations. The notebook table's U(1, 4.5) and U(2, 10.5) are wrong. The paper's own step 1 (p. 902), "from the uniform distributions specified in Table 1", is also loose on this point.
- **ppcd**: the notebook's U(355, 400) matches JHM Table 1, which gives ppm. The data and param_ranges store mol/mol (3.55e-4–4e-4). Add the unit; normalization is unaffected.
- **Sign handling; the legacy resistance plots are mirrored relative to the paper.** A hard-coded October flip (`if month_labels[i]=='October': ss.W1 *= -1.0`) patches the legacy first-component sign rule. Under the v1 rule (largest component positive), the October flip becomes wrong and hourly sunlit GMT_22 and GMT_23 flip as well. The paper orients w = â/‖â‖ (p. 902: "positively sloped relationship"). In JHM Fig. 9b,c (p. 910), bp and mp weights for rs are negative (shaded bp −0.7 to −1) and rs increases with the active variable. The legacy notebook shows the opposite sign for both resistances (legacy hourly-shaded bp +0.70 to +0.96, figure `sto_c10_0`). Orienting by â removes all special cases and matches the paper.
- **"Fairly tight univariate trends in each summary plot" is overstated.** Linear-fit R² ranges from 0.37 to 0.99; sunlit resistance in July is 0.46 and in the late JJA hours 0.37–0.44.
- **Mismatch between fit and plot.** The hourly resistance summary plots show log(f) against an active variable that was fitted to f, not log f. The paper (Fig. 9e,f) plots rs on a linear axis, so the log is the notebook's own choice.
- **v1 OLS on a Uniform space emits a ValidityWarning** on each of the 78 fits.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| Stomatal_Resistance.ipynb | 2.6 MB | nbformat 4, python2 kernel | notebook | — | 18 PNG outputs |
| stomata_realization_inputs.txt | 32 KB | space-separated, quoted header, 301×20 | inputs (physical) | `pn.DataFrame.from_csv(sep=' ', index_col=None)` | row 0 is the nominal set; wj1 and wj2 have 3 levels; values are rounded (21–236 unique per column) |
| param_ranges.txt | 192 B | header "min" "max", 20×2 | bounds, same row order as the input columns | same | ppcd 3.55e-4–4e-4 |
| Tonzi_Stomata_SC.mo_avg_{tran_veg,rssun,rssha}.txt | 61–76 KB | quoted header (Sept…Aug), 301×12 | monthly means of daytime values (JHM p. 906), Tonzi Sept 2002–Aug 2003, the "most average year" (p. 906, Fig. 7) | same | water-year order. Units are not stated. tran is in mm s⁻¹: ×86400 gives 0.03–10.4 mm d⁻¹, matching JHM Fig. 9d. rs is in s m⁻¹, matching Fig. 9e,f |
| Tonzi_Stomata_SC.JJA_hr_avg_{tran_veg,rssun,rssha}.txt | 71–88 KB | quoted header GMT_0, GMT_1, GMT_12…GMT_23, 301×14 | June–August 2003 daytime hourly means (JHM Fig. 9 caption) | same, columns reordered by name | the file order is GMT_0, GMT_1, GMT_12, …, so reorder by name |

All values are positive and finite.

## 3. Text and math review (vs. literature)
Checked against the JHM 2017 full text (pp. 897–915). The thesis was not read. JHM context:
- 2,100 runs = 300 realizations × (5 forcing sets + 2 extra scenarios) (p. 902).
- The 5 forcing sets are Breckenridge CO, Little Washita OK, and Tonzi Ranch CA in average, wet and dry years.
- The notebook data are the Tonzi 2002–03 (average-year) set (p. 906) plus the default run.
- JHM Table 1 (p. 901) gives names, ranges, defaults and units.
- Monthly results are in Figs. 6–7; hourly results are in Figs. 8–9.

On the data, the recomputed â-oriented weights reproduce JHM Fig. 9:
- transpiration bp ranges from +0.46 to +0.98;
- shaded rs bp ranges from −0.96 to −0.70;
- max |w_ppcd| = 0.142, consistent with the paper's "< 0.15 for all … quantities" (p. 910).

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "presented in [1]" with [1] = thesis | the notebook title is the JHM 2017 paper title; dev-branch drafts cited the JHM paper | cite both: JHM 18(3):897–915, doi:10.1175/JHM-D-16-0053.1 (primary) and the thesis | confirmed (paper header p. 897, DOI p. 898) |
| 0 | "Our dataset contains 300 input/output pairs" | the files have 301 rows; row 0 equals the default parameter values | say "300 random samples plus the default parameter set (row 0)", and drop row 0 from the fits as the paper does (p. 909) | confirmed (data = JHM Table 1 defaults) |
| 0 | table row "ci Multiplier | wj1 | U(1, 4.5)" and "cp Multiplier | wj2 | U(2, 10.5)" | JHM Table 1 gives the discrete sets "1, 4, 4.5" and "2, 8, 10.5" (hard-coded in the CLM subroutine, footnote a), and the data match. Treating them as continuous U(a,b) puts normalized levels at −1, 0.714, 1 | write "{1, 4, 4.5}, each level equally likely" (and likewise for wj2); keep the linear-model weight, but note that it measures the effect of switching formulations | confirmed (JHM Table 1; data) |
| 0 | "CO2 Partial Pressure | ppcd | U(355, 400)" | matches JHM Table 1 ("Partial pressure of CO2 in the atmosphere", 355–400 ppm, default 355). The data store mol/mol (3.55e-4–4e-4) | add the unit: "U(355, 400) ppm (stored as mol mol⁻¹)". Add a units column for all 20 inputs from JHM Table 1 (vcmx25 µmol m⁻² s⁻¹, kc25/ko25 Pa, hv J mol⁻¹, sv J mol⁻¹ K⁻¹, ecp µmol J⁻¹, bp µmol m⁻² s⁻¹, rsmax0 s m⁻¹) | confirmed (JHM Table 1; data) |
| 0 | names "q10 for vcmx25", "q10 for kc25", "q10 for ko25" | JHM Table 1 calls them "Temperature coefficient for …" | use the paper's wording | confirmed |
| 0 | table order (vcmx25, avcmx, hv, …) | differs from the file order (qe25, ko25, kc25, …), which drives every plot axis | reorder the table to the file order, or plot in table order | confirmed |
| 0 | "explicity", "_sigle_", "paramters" | typos | fix | confirmed |
| 0 | "we don't have gradients for the PF-CLM, so we use the following algorithm to identify the single most important direction" | OLS on uniform inputs is a heuristic: â spans a direction of C only under the linearity condition (Li 1991; Constantine 2015 §1.4). v1 warns about this | add one sentence | confirmed |
| 0 | "Draw {x_j} from ρ" | the design is not iid from the stated ρ (discrete wj1/wj2, plus the nominal row) | qualify | confirmed (data) |
| 2/3 | "model output … averaged by month and by hour" | Monthly values are averages of daytime hours over Sept 2002–Aug 2003; hourly values are June–August 2003 daytime hours in UTC (JHM p. 902, 906, Fig. 9). JHM Fig. 8 (p. 909) labels UTC 12 as 6 am … UTC 1 as 7 pm, i.e. UTC−6. California daylight time is UTC−7, so the paper's local-time labels may be off by an hour | say "monthly daytime means, Sept 2002–Aug 2003; JJA 2003 hourly daytime means (UTC)". Use the paper's local-time labels, or ask the author which offset is right | confirmed (paper); offset unverified |
| 9 | `times = ['Morning', 'Midday', 'Evening']` for GMT 12–15 / 16–21 / 22–1 | per JHM Fig. 8 these are 6–9 am, 10 am–3 pm and 4–7 pm local | acceptable; add the local hours to the labels | confirmed (JHM Fig. 8) |
| 8 | "We can see fairly tight univariate trends in each summary plot" | the paper is more cautious (p. 909: hourly "resistance values spread farther from the perceived univariate trend"). JHM reports no R². Linear-fit R² from the data: transpiration 0.89–0.99; shaded rs 0.75–0.82; sunlit rs 0.37–0.90 (July 0.46, Aug 0.59, GMT_20–23 0.37–0.44) | qualify the claim; print an R² table | confirmed |
| 8 | "bp, mp, and wj1 are the most important … (qe25, kc25, vcmx25, ocr) have small contributions" | matches the top-4 weights recomputed per month | — | confirmed |
| 8 | "the summary plots show the log of stomatal resistance vs the active variable" | w is fitted to f, then log f is plotted. The paper plots rs on a linear axis (Fig. 9e,f) | plot rs as in the paper; optionally add a log-axis view (axis scaling only, w unchanged) | confirmed |
| 5–11 | weight plots, all resistance panels | with the legacy sign rule (and the October hack), the resistance weights are the negatives of the paper's (paper Fig. 9b,c: bp, mp < 0; legacy `sto_c10_0`: bp +0.70 to +0.96) | orient w = â/‖â‖ (§6) | confirmed (paper Fig. 9; legacy output) |
| 12 | "expanding implementations of these models to include gradients could allow us to build better one-dimensional subspaces" | fine | — | — |
| 0 | ref [2] | correct | — | confirmed |

## 4. Code review
- pandas: `DataFrame.from_csv` (removed in pandas 1.0) and `.as_matrix()`. Replace with numpy (§5).
- Each of cells 5, 6, 7, 9, 10 and 11 calls `ss.compute` three times per slice: weights plot, weights-vs-time loop and summary loop. That is 234 fits for 78 distinct ones. Compute once into a dict `W[qoi][slice]`.
- Six cells are near-identical copies, about 300 lines. Write one function `plot_case(F, slice_labels, title, log=False)` and call it six times.
- The `if month_labels[i]=='October': ss.W1 *= -1.0` sign hack appears three times in cell 7. Replace it with orientation by â (§6).
- `colors` is redefined in cell 9 (14 colours) and reused by later cells, so the order of cell execution matters. `Colors`/`styles` give 20 line styles; use a colour-map cycle instead.
- `plt.subplots(...)` followed by `plt.subplot(...)` creates axes twice. Use the returned axes.
- No randomness, no hard-coded absolute paths.

## 5. Data management review
- Formats are consistent: quoted header row, space-separated, no index column. Units are absent everywhere; add a README line or a units row.
- Bounds are stored (param_ranges.txt), which is good. The name column is implicit (row order). Add a name column or document the order.
- Row 0 (nominal) is mixed into the samples. Either document it, or move it to `nominal_inputs.txt` and remove row 0 from all 7 data files.
- The design of wj1/wj2 is not documented.
- Size is small (≈470 KB of data). There is no license or provenance statement.
- pandas-free loading, tested in .venv:
```python
import numpy as np
def read_named(fn):
    with open(fn) as fh:
        names = [h.strip('"') for h in fh.readline().split()]
    return names, np.loadtxt(fn, skiprows=1, ndmin=2)
names, X = read_named('stomata_realization_inputs.txt')        # (301, 20)
_, bounds = read_named('param_ranges.txt'); lb, ub = bounds.T   # (20, 2)
MONTHS = ["September","October","November","December","January","February",
          "March","April","May","June","July","August"]
HOURS = [f"GMT_{h}" for h in (12,13,14,15,16,17,18,19,20,21,22,23,0,1)]
def read_qoi(fn, cols):
    h, F = read_named(fn); return F[:, [h.index(c) for c in cols]]
F = {q: read_qoi(f'Tonzi_Stomata_SC.mo_avg_{q}.txt', MONTHS) for q in ('tran_veg','rssha','rssun')}
G = {q: read_qoi(f'Tonzi_Stomata_SC.JJA_hr_avg_{q}.txt', HOURS) for q in ('tran_veg','rssha','rssun')}
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `BoundedNormalizer(lb, ub).normalize(X)` | `space = ac.Uniform(lower=lb, upper=ub, names=names)`; pass physical `X` | identical normalized coordinates (tested) |
| `ss.compute(X=XX, f=f[:,None], sstype='OLS')` | `ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)` | emits `ac.ValidityWarning` (`validity.assumption`: OLS on a uniform space) on every call, 78 in total. Wrap the loop in `warnings.catch_warnings(); warnings.simplefilter("ignore", ac.ValidityWarning)` and explain the caveat once in markdown, or catch the first one and print it |
| `ss.W1` and the October hack | `w = sub.W1[:,0]; w *= np.sign(w @ a_hat)`, where `a_hat` is the OLS slope (or the sign of `corr(X_norm @ w, f)`) | v1 sign rule [D-81]: relative to legacy it flips monthly-sunlit October and hourly-sunlit GMT_22 and GMT_23. Orienting by â is consistent in every slice and is the paper's convention (JHM p. 902, Fig. 9). Transpiration matches legacy. Resistance plots will be mirrored relative to legacy, but legacy was mirrored relative to the paper |
| `XX.dot(ss.W1)` | `space.to_normalized(X) @ w` (or `ac.forward_map(sub, X=X).Y` when not reoriented) | — |
| summary plots | matplotlib by hand (several slices per axis); `ac.plotting.plot_sufficient_summary` draws one subspace per call | — |

## 7. Associated .py files
None.

## 8. Update plan
1. Replace pandas loading with the §5 snippet.
2. Compute all 78 OLS directions once with v1 (warnings handled), oriented by â; store W[qoi] as (n_slices, 20).
3. Factor the six plotting cells into one function; remove the October hack.
4. Drop row 0 from the fits (the paper's 300 realizations) and report its location as the "default" point if wanted. Plot rs on a linear axis as in the paper.
5. Text: fix the 300 → 301 statement, the wj1/wj2 discrete sets (JHM Table 1), units for all inputs (ppcd ppm), the table order and names, the JHM citation, monthly-daytime / JJA-2003 / UTC labels, the OLS validity caveat, typos; qualify "fairly tight" with an R² table.
6. py3 kernelspec; execute headless; commit with outputs.

## 9. Acceptance criteria
- Executes with `jupyter nbconvert --execute` in `.venv` in < 30 s, without pandas, with no warnings shown except one deliberate ValidityWarning explanation.
- With all 301 rows, |w_v1 · w_legacy| = 1 to 1e-10 for all 78 slices. With row 0 dropped (the target), |cos| ≥ 0.999 to legacy.
- Signs match JHM Fig. 9: hourly bp weight > 0 for transpiration, < 0 for both resistances.
- Weight-vs-time curves are continuous in sign (no single-slice flips); October and GMT_22/23 need no special case.
- An R² table (linear fit of f on the active variable) is printed: transpiration ≥ 0.89, shaded 0.75–0.82, sunlit 0.37–0.90, as in §10.
- Figures: per QoI × {monthly, hourly}: weights per slice, weights vs time, summary plots (18 figures, or fewer combined).

## 10. Prototype results
Ran `prototypes/g4/proto_sto.py` and `prototypes/g4/load_sto.py` with the .venv python:
- Loading works with numpy only. X is 301×20 and bounds are 20×2; every input lies within its bounds.
- v1 OLS gives |cos| = 1.0000000000 to the legacy direction in all 78 slices. v1 sign versus legacy is −1 only for monthly-sunlit October and hourly-sunlit GMT_22 and GMT_23. sign(v1·â) is +1 for all transpiration slices and −1 for all resistance slices except hourly-sunlit GMT_22 and GMT_23 (+1). So orienting by â gives a single consistent rule.
- Dropping row 0 changes w by cos ≥ 0.9997.
- Top-4 weights per month agree with the cell 8 text.
- Linear R² per slice is as listed in §3.
- Legacy figure `sto_c11_1` was inspected: legacy curves are continuous because the first-component rule (qe25 > 0) happens to be stable in the hourly data.
- Paper cross-check (`prototypes/g4/sto2.py`), with â orientation over all slices:
  - bp ranges: monthly tran +0.23 to +0.78, monthly rssha −0.87 to −0.71, monthly rssun −0.54 to −0.16, hourly tran +0.46 to +0.98, hourly rssha −0.96 to −0.70, hourly rssun −0.54 to −0.28.
  - max |ppcd| = 0.142.
  - JJA tran ×86400 gives 0.03–10.4 mm d⁻¹.
  - All are consistent with JHM Fig. 9 and p. 910.
- 78 ValidityWarnings were emitted.

## 11. Open questions for the author
1. Primary reference: JHM 2017, the thesis, or both? Should the wet/dry Tonzi years or the other sites be added (the paper has 7 × 300 runs)?
2. wj1/wj2 discrete sets (resolved by JHM Table 1): should the notebook show them as categorical, e.g. boxplots by level, in addition to the linear weight?
3. Row 0: drop it from the fits to match the paper's 300 realizations (recommended)?
4. Local time at Tonzi: is JHM Fig. 8's UTC−6 labelling correct, or should it be UTC−7 (PDT)?
5. (Resolved: the paper uses w = â/‖â‖, Fig. 9.)
6. Units: tran_veg in mm s⁻¹ and rs in s m⁻¹ (inferred from JHM Fig. 9): confirm.
7. Data license and provenance (PF-CLM version).
