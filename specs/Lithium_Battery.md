# Lithium_Battery: update spec

Kind: **stored data**: Newman-model runs by Hadigol et al. (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `Lithium_Battery/Li_Battery.ipynb` (analysis code is in `battery_figs.py`) | paper: P. G. Constantine, A. Doostan, "Time-dependent global sensitivity analysis with active subspaces for a lithium ion battery model", *Stat. Anal. Data Min.* 10(5):243–262 (2017), doi:10.1002/sam.11347 (checked against the arXiv v2 text and, on 2026-10-29, the **published** version: Table 1 p. 250, §3.1–3.2 pp. 250–251, §4 p. 251, Fig. 2 p. 253, Fig. 8 p. 259, Fig. 9 and Acknowledgments p. 260; page numbers below refer to the published version). Data source: Hadigol, Maute, Doostan, *J. Power Sources* 300:507–524 (2015), doi:10.1016/j.jpowsour.2015.09.060 | m = 19 | QoIs: capacity [mAh cm⁻²] on a 50-point voltage grid, and voltage [V] on a 50-point scaled-time grid t*. Each is given at 3 discharge rates (0.25C, 1C, 4C) | legacy API: none. The analysis uses hand-written `np.linalg.lstsq` (paper Algorithm 1) and pandas only | py2 issues: yes (pandas `DataFrame.from_csv`/`.as_matrix` were removed in pandas 1.0, and non-raw `'$\mathbf{w}$'` strings trigger SyntaxWarning in py ≥ 3.12). The notebook itself has no `print` statements.

## 1. Summary
The notebook is a text wrapper around `battery_figs.py`, a 1,167-line copy-paste module that reproduces paper Figs. 2–7 and 9. It does not reproduce Fig. 8 (Algorithm 2 eigenvalues). The analysis is deterministic except for the bootstrap, and it is cheap: the whole thing runs in under 1 minute with numpy. Porting it is mostly a rewrite of `battery_figs.py`.

Most important findings:
1. **Parameter count.** The README says "18-parameter". The notebook, the paper and the data all have **19** parameters (confirmed).
2. **Input distributions.** The notebook table (from paper Table 1) gives r_a ~ N(2, 0.1354) and r_c ~ N(2, 0.3896). The stored normalized columns for r_a and r_c are **uniform on [−1, 1]**, like all the other columns (excess kurtosis −1.20/−1.22, flat histograms). Either the table or the data description is wrong (open question Q1). The published version (Table 1, p. 250; §4, p. 251: "Each run uses a realization of the input parameters drawn independently according [to] the distributions in Table 1 … we shift and scale the inputs to the hypercube [−1, 1]^19") still says Normal. A shift-and-scale of a normal sample cannot give a uniform sample on [−1, 1], so the conflict between paper and data is confirmed; which side is right remains unverified.
3. **v1 sign rule breaks the time curves.** v1 `estimator="ols"` reproduces |w| to 5e-15, but v1 orients each eigenvector so that its largest entry is positive. Compared with the regression sign used by the legacy code, this flips w at 44 of the 50 voltages (0.25C capacity), 12 of 50 (1C) and 0 of 50 (4C). The w-vs-voltage curves would jump and the sign interpretation would be lost. The sign must be re-aligned (§6). **Superseded (V1-04, [#48](https://github.com/paulcon/as-data-sets/pull/48)):** v1 OLS returns the signed u/‖u‖, equal to the 2016 lstsq weights to 2.6e-13, so no re-alignment is needed.
4. **Silent data edit.** Row 1537 (Run 1538) of `4C.txt` is deleted in 6 places, with no comment. Its final voltage is 2.746 V, an overshoot of the 2.8 V cutoff with |z| = 59. Run 2969 also overshoots (2.791 V) and is kept. The published paper does not mention the exclusion either: §4 (p. 251) says only that each run stopped at the 2.8 V cutoff, and §4.1 (p. 252) says "Each application uses all 3600 evaluations … N = 3600", which is not true of the 4C fits in the paper's own script. Dropping Run 1538 changes the 4C voltage weights by up to 0.020, so this matters and must be documented.
5. **Bootstrap figure clipped.** It is hard-coded to `ylim [0, 0.01]`, while the notebook text claims "all are below 0.010". A seeded rerun gives a maximum standard error of **0.0103** (1C capacity, region B). The value depends on the random draw, and the clipping hides it.
6. **Data size.** The data are 67.2 MB of CSV. 55 % of the columns (3 × 50 concentration columns) are unused, and the 19 input columns are byte-identical in all three files. A lossless `.npz` of everything is 18.9 MB; one with only the used columns is 8.6 MB.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| Li_Battery.ipynb | 971,749 B | nbformat 4, kernel python2 | text + 3 code cells calling `battery_figs` | — | 13 PNG outputs; master is newer than dev (refs and acknowledgments only) |
| battery_figs.py | 41,699 B (1,167 lines) | py2/3 module | all analysis and plotting (paper's figure script) | `from battery_figs import *` | §7 |
| 0.25C.txt, 1C.txt, 4C.txt | 22,392,089 B each (67.2 MB total) | CSV, 1 header row, 3600 × 270 (fixed width, `%.16e`) | samples per rate | `pn.DataFrame.from_csv` (col `Run` → index) | cols: `Run`, 19 inputs (normalized to [−1,1], stored with 8 decimals), `Vol01..Vol50`, `Cap01..Cap50`, `ConA01..50`, `ConS01..50`, `ConC01..50`. No NaN. The input block is identical in all 3 files. |
| V_025C.txt, V_1C.txt, V_4C.txt | 1,150 B each | 50 floats, one per line | voltage grid [V] for the capacity QoI (descending, ends at 2.8) | `np.loadtxt` | 0.25C: 4.133→2.8; 1C: 4.026→2.8; 4C: 3.724→2.8 |
| README | 183 B | text | cites the paper | — | no column documentation |

Column semantics (inferred, not documented anywhere):
- `Cap*` raw values are the **elapsed discharge time in seconds** at each grid voltage. The 0.25C maximum is 13,906 s = 3.86 h ≈ 4 h, and the 1C maximum is 3,442 s ≈ 1 h. The code multiplies by `CAPFAC = rate·30 A m⁻²/3600/10`, which gives capacity in mAh cm⁻², the unit on the paper's figure axes and in the published §3.1 (p. 250: capacity is "measured in milliampere-hours per square centimeter (mAh ⋅ cm−2)"). The unit is confirmed. That the raw column is time in seconds is still inferred: the paper does not describe the file layout. Confidence: likely.
- `Vol50` is about 2.8 V for every run (std 9e-5), so the last t* column is the cutoff, i.e. t* ≈ 0. The code labels the grid `tstar = linspace(1, 99, 50)[::-1]`, which ends at 1 (open question Q2).

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| README | "an 18-parameter model of a lithium-ion battery" (top-level README) | The published paper (§3.2 p. 250, "Hadigol et al. [25] modeled 19 of the Newman model's input parameters"; Table 1 p. 250 has 19 rows; §5 p. 260 "The model contained 19 input parameters"), the notebook and the data all have 19. | "19-parameter" | confirmed |
| 0 | "We call $\mathbf W_1^T\mathbf x \equiv \mathbf y$ the active variables and their range of values the active subspace." | Wrong definition. The active subspace is range(W₁), the span of the first n eigenvectors (Constantine 2015 §3.1; CDW 2014 §2). It is not the range of values of y. | "…the active variables; the subspace spanned by the columns of $\mathbf W_1$ is the active subspace." | confirmed |
| 0 | "we could use this new $\mathbf C$'s Schur compliment" | typo | "Schur complement" | confirmed |
| 0 | "chellenging", "charcterize", "contians" | typos | fix | confirmed |
| 0 | "Anode Particle Size \| $r_{s, a}$ \| $N(2, 0.1354$" and "Cathode Particle Size \| … \| $N(2, 0.3896$" | Missing ")". The distribution matches paper Table 1 but **not** the data, where the normalized r columns are uniform (§1 item 2). The table also omits the units and nominal values that paper Table 1 gives. | Close the parentheses. Add units and nominal values from paper Table 1 (p. 250). Add a note on how r was normalized once Q1 is answered. | confirmed (typo; table = published Table 1 p. 250; data uniform); unverified (which is right) |
| 0 | "Anode Solid Diffusion Coefficient \| $D_{s,a}$", "Anode Particle Size \| $r_{s,a}$" | The plot legends (`input_labels()`) use $D_a$, $r_a$; the CSV header uses `$D_a$`, `$r_a$`. Inconsistent notation. | Use one notation everywhere (the legend form is shorter) | confirmed |
| 0 | table order (anode, cathode, separator, misc) | The data column order is ε_a, ε_s, ε_c, brugg_a, brugg_s, brugg_c, t₊⁰, D, D_a, D_c, σ_a, σ_c, k_a, k_c, r_a, r_c, L_a, L_s, L_c. Readers who index `X[:, j]` need this order. | Add a line giving the column order | confirmed |
| 0 | "$0.25C, 1C$, and $4C$, where $C$ is the rate at which a battery discharges from its full capacity" | Imprecise. 1C is the current that discharges the nominal capacity in one hour. | "1C discharges the nominal capacity in 1 h, 0.25C in about 4 h and 4C in about 15 min" | confirmed |
| 0 | "Our dataset has 3600 pairs of inputs/outputs for each discharge rate … totaling 10800 runs." | True of the files and of the published text (§4 p. 251; "N = 3600", p. 252). The 4C analysis silently uses 3599 runs (Run 1538 dropped), and the paper does not mention this. | Add: "one 4C run (Run 1538) overshoots the 2.8 V cutoff and is excluded" | confirmed |
| 0 | "The outputs are 50 voltage/capacity and 50 time/voltage pairs." | The files also hold 150 concentration columns (anode, separator, cathode) that are never described or used. | Document them, or drop them from the shipped data (Q4) | confirmed |
| 0 | "we let $t^* = 100$ at physical time 0 and $t^* = 0$ at the physical time when voltage reaches $2.8V$" | Matches published §4 (p. 251), which gives only the endpoints. The code's grid is `linspace(1, 99, 50)` (Q2). The figures label the summary panels "t* = 95, 23, 5", which match the paper. | Keep; add the grid actually used | likely |
| 0 | "While these are possible, the relative smoothness of physical models … makes it very unlikely that we would observe these cases in practice." | Stronger than the paper (§2.8, p. 249, unchanged in the published version: "In practice, we have never experienced such a false positive. We suspect this is because…"). | Paraphrase the paper's hedged wording | confirmed |
| 0 | "[[1]] … arXiv:1606.08770, 2016" | The paper is published | "*Stat. Anal. Data Min.* 10(5):243–262, 2017, doi:10.1002/sam.11347 (arXiv:1606.08770)" | confirmed |
| 0 | (missing) | The data come from Hadigol et al. (2015) (paper §3.2, §4) | Add the reference: Hadigol, Maute, Doostan, *J. Power Sources* 300:507–524, 2015, doi:10.1016/j.jpowsour.2015.09.060 | confirmed |
| 0 | (missing) | The paper also applies Algorithm 2 (global quadratic model, Fig. 8). It is the evidence for "one-dimensional" at A/B/C. | Optional new cell with v1 `estimator="qphd"`. Prototype: 0.25C capacity at A/B/C gives log₁₀(λ₁/λ₂) = 1.90 / 0.85 / 3.53, matching the paper's "differ by two, one, and nearly four orders of magnitude". | confirmed (prototype) |
| 0 | header "__Paul Constantine__, Colorado School of Mines, <pconstan@mines.edu>", "__Ryan Howard__ … <ryhoward@mines.edu>" | The published paper (p. 243) lists Constantine at the Department of Computer Science, University of Colorado Boulder (paul.constantine@colorado.edu) and Doostan at Smead Aerospace Engineering Sciences, CU Boulder. The notebook's 2016 Mines affiliation predates publication. Whether to use 2016, 2017 or current affiliations is the author's call. | Ask the author (Q7) | confirmed (published differs); choice open |
| 0 | (missing) | Published Acknowledgments (p. 260): "We thank Dr. Mohammad Hadigol for providing us with the Li battery data set". | Credit Hadigol as data provider in the notebook and the README | confirmed |
| 3 | "The tightest trend appears for discharge rate $0.25C$ and voltage $3.1V$ and the loosest for $1C$ and $3.2V$." | Agrees with the paper (Figs. 2g, 3f) and the data: linear-fit R² is 1.000 at 0.25C/3.10 V and 0.840 at 1C/3.18 V | keep | confirmed |
| 3 | axis labels "Capacity, 0.25C" (figures) | No units. The paper's axes use mAh·cm⁻². | Add units to all QoI axes | confirmed |
| 5 | "In all cases (for both capacity and voltage and all discharge rates), porosities, lengths, and bruggeman coefficients remain the most important parameters" | False for 4C voltage. At t* = 5, the largest component is D (−0.59), and r_c is −0.28 to −0.43 across A/B/C. The published paper says "Generally, porosities and Bruggeman coefficients are among the most important parameters" and "The separator's porosity is by far the most important parameter … the salt diffusion coefficient suddenly becomes important" for 4C (§4.1 bullets, pp. 257–258). | "Generally … except at 4C, where ε_s dominates and D and r_c become important late in the discharge" | confirmed (data + published pp. 257–258) |
| 7 | "All bootstrap errors are fairly low (all are below 0.010 and most are below 0.005)" | The plot clips at 0.01. A seeded rerun gives a maximum of 0.0103 (1C capacity, region B). 20.7 % of the 1C capacity SEs exceed 0.005. The published Fig. 9 (p. 260) is clipped at 0.010 too; its 1C capacity curves peak near 0.009. | Remove the clip. Say "about 0.01 or less; at 0.25C and 4C mostly below 0.005" and compute the statement from the run. | confirmed (prototype, published Fig. 9) |
| 7 | "there are 2 notable spikes in standard error: region B for both capacity with $0.25C$ and capacity with $1C$" | Agrees with arXiv v2 ("region B in Figures 9a and 9c"), with the published Fig. 9A/9C (p. 260: the peaks sit in band b) and with the rerun (peaks at indices 17 and 34). **New:** the published text (§4.1, p. 259) says "region c in Figures 9A and 9C". That contradicts its own figure; it is an erratum in the published paper. | keep "B". Do not "correct" it to the published text. | confirmed |
| 6 (fig) | bootstrap x-labels "Capacity, 0.25C" on a voltage axis | The axis is voltage (or t*). The paper's Fig. 9 uses the x-label "Voltage [V]" with the panel title "Capacity, I = 0.25C". The right-hand column has no y-label. | xlabel "Voltage [V]" or "$t^*$"; title = QoI and rate | confirmed |

The general AS exposition (C matrix, λᵢ = E[(∇fᵀwᵢ)²], the linear-model heuristic C ≈ aaᵀ, the time-dependent C(t)) matches paper §2 and Constantine 2015 Ch. 1–3. There are no other math errors.

## 4. Code review (battery_figs.py)
- **py3 / removed APIs.** `pn.DataFrame.from_csv` and `df.as_matrix()` were removed in pandas 1.0. The labels `'$\mathbf{w}$ components…'`, `'$\mathbf{w}^T\mathbf{x}$'` are non-raw strings with `\m`, which is an invalid escape (SyntaxWarning in 3.12+, a future error). `import pdb` is unused. `np.linalg.lstsq(A, f)` has no `rcond`; numpy 2 changed the default, but this has no effect here (full rank, M = 3600 ≫ 20).
- **Hard-coded outlier removal.** `np.delete(…, 1537, 0)` is applied for `cflag==2` in 6 places, without a comment. Move it to one place with a named constant and an explanation.
- **Duplication.** `Cap_025C/1C/4C` and `Vol_025C/1C/4C` are 6 near-identical 120-line functions, and `bootstrap_SE` repeats the block 6 times. They differ only in the file, the rate factor, the shaded indices (cap 0.25C: 2/17/38; cap 1C and 4C: 5/34/48; vol: 2/38/47) and the `fig.text` label positions.
- **Wasted computation.** `bootstrap_SE` computes the point estimates `a` 6 times and never uses them. Each data file is re-parsed 4 times, 12 parses in all. That is cheap with `np.loadtxt` (0.3 s/file), but pandas `from_csv` was slower.
- **Randomness.** `np.random.randint` in `get_coefficients_boot` is unseeded, so the bootstrap figure and the text's "below 0.010" claim are not reproducible. Use `rng = np.random.default_rng(SEED)`.
- **Fragile layout.** The A/B/C labels use `fig.text(0.148, 0.85, 'A')` and similar absolute figure fractions tuned for a 2016 matplotlib. Use `ax.text(x_center_of_band, 0.9, 'A', transform=ax.get_xaxis_transform())`. Legends use `bbox_to_anchor=(1.265, …)` outside the axes; check them in mpl 3.11.
- **Stale comments.** "summary plots 95% / 65% / 23%" do not match the indices used.
- **`get_coefficients_boot`** clamps SE < 1e-15 to 1e-15. This is a leftover from log-scale plotting and has no effect on linear axes. Remove it.
- **Index bookkeeping.** In the Vol figures, `a[:, i]` holds column `i+1` and is plotted against `tstar[1:-1]`. This is consistent, but write it as `cols = range(1, 49)` explicitly. The capacity grid uses all 50 columns; column 0 has nonzero variance (std 9.2 s at 0.25C), so that is fine.
- **Physical meaning not in code.** `CAPFAC*` are unexplained magic numbers. Name them, e.g. `I_1C = 30.0  # A m^-2`, `capacity_mAh_cm2 = rate * I_1C * t_s / 3600 / 10`. Confirm the value with the author (Q3).
- No hard-coded absolute paths. Files are read relative to the notebook's directory.

## 5. Data management review
- **Format.** Comma-separated text with LaTeX column names in the header (`$\epsilon_a$`, `$\text{brugg}_a$`). Column 0 `Run` is 1..3600. The inputs are stored normalized only; physical values and bounds are not stored. They could be rebuilt from paper Table 1 for the 17 uniform inputs, but not unambiguously for r_a and r_c (Q1).
- **Units.** None in the files. Capacity is stored as time in s (inferred). Concentration units are unknown.
- **Redundancy.** The 19 input columns (3600 × 19) are identical in all 3 files. The 150 concentration columns per file are unused.
- **Size.** 22.4 MB per file, under the GitHub 50 MB warning; 67.2 MB total. The git history also holds them once (the blobs never changed). `.git` is 225 MB for the whole repo, including a deleted 64.5 MB `NREL_Wind/windData500k.mat`. Measured alternatives:

| option | size | lossless | note |
|---|---|---|---|
| current CSV ×3 | 67.2 MB | — | |
| `gzip -9` CSV (`*.txt.gz`; `np.loadtxt` reads .gz directly, tested 0.3 s) | 21.2 MB | yes | smallest change; keeps headers and text |
| one `.npz` (`savez_compressed`), all columns, X stored once | 18.9 MB | yes | |
| one `.npz`, X + voltage + capacity only | 8.6 MB | yes for used data | drops concentrations |
| same, float32 | 4.0 MB | no | max \|Δw\| = 4e-8; inputs have only 8 significant digits anyway |

  Recommendation: ship `battery_data.npz` (X once, with `names`, `units`, `vgrid_<rate>`, `tstar`, `voltage_<rate>`, `capacity_<rate>` already converted to mAh cm⁻², and `dropped_runs`) at 8.6 MB. Deposit the original 3 CSVs (with concentrations) on Zenodo with a DOI and Hadigol et al. as data authors. Do **not** make the notebook download at run time, because the acceptance criterion is a headless offline run. Shrinking the clone would also need a history rewrite (`git filter-repo`); that is the author's call (Q5).
- **Provenance and license.** Generated by Hadigol et al. (2015) on CU's JANUS cluster (paper §4: 20 min/run, 3600 CPU-h). No license file in the repo. Ask about redistribution (Q6).
- **pandas-free loading (tested in .venv, numpy 2.4):**
```python
import numpy as np
RATES = {"0.25C": ("0.25C.txt", "V_025C.txt", 0.25),
         "1C":    ("1C.txt",    "V_1C.txt",   1.0),
         "4C":    ("4C.txt",    "V_4C.txt",   4.0)}
DROP = {"4C": [1537]}          # Run 1538: final voltage 2.746 V overshoots the 2.8 V cutoff
I_1C = 30.0                    # A m^-2 (inferred from CAPFAC = rate*30/36000)

def load(rate):
    fname, vname, c = RATES[rate]
    with open(fname) as fh:
        names = fh.readline().strip().split(",")[1:20]
    d = np.loadtxt(fname, delimiter=",", skiprows=1)        # (3600, 270), 0.3 s
    d = np.delete(d, DROP.get(rate, []), axis=0)
    X = d[:, 1:20]                                          # normalized inputs in [-1, 1]
    V = d[:, 20:70]                                         # voltage [V] at t* grid
    Q = c * I_1C * d[:, 70:120] / 3600 / 10                 # capacity [mAh cm^-2] at vgrid
    vgrid = np.loadtxt(vname)
    return X, V, Q, vgrid, names
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `pn.DataFrame.from_csv(f); df.as_matrix()` | `np.loadtxt(f, delimiter=",", skiprows=1)` or `np.load("battery_data.npz")` | §5 |
| `get_coefficients(X, f)` (OLS, `u/‖u‖`) | `sub = ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)`; `w = sub.W1[:, 0]`; **then** `w *= np.sign(np.cov(X @ w, f)[0, 1])` | `space = ac.Uniform(lower=-np.ones(19), upper=np.ones(19), names=labels)` (identity map with names; `Uniform.reference(m)` takes no names). \|w\| matches lstsq to 5e-15. **Sign:** v1 makes the largest entry positive, which flips w at 44/50 (0.25C), 12/50 (1C), 0/50 (4C) capacity voltages compared with the regression sign. Re-align with the sign of cov(Xw, f), which equals the sign of uᵀw and keeps "f increases with wᵀx". Every OLS call on a uniform space raises `ValidityWarning` (`validity.assumption`): 300 calls. Wrap the loop in `warnings.catch_warnings()` with `filterwarnings("ignore", category=ac.ValidityWarning)` and explain why in a markdown cell (paper §2: Algorithm 1 is a heuristic). Alternatively keep the 3-line `np.linalg.lstsq`, which is exactly paper Algorithm 1 and needs no sign fix; then v1 is used only for QPHD and plots. Pick one (Q8). **Superseded (V1-04, V1-12, [#48](https://github.com/paulcon/as-data-sets/pull/48)):** v1 OLS returns the signed direction, so drop the cov(Xw, f) re-alignment. Q8 chose v1 OLS, with `Uniform.reference(19, names=)`. |
| `get_coefficients_boot(X, f, nboot=100)` | hand-coded, `rng = np.random.default_rng(SEED)`; SE of the **sign-aligned** w over 100 resamples | v1 `nboot` returns `bootstrap_eigenvalues/distances`, not eigenvector replicates (W_br was dropped). For a rank-one OLS, `bootstrap_distances[:, 0]` (the sine of the angle between w and w⁽ᵇ⁾) is a usable one-number stability measure per grid point. It could be added next to the per-component SE plot. |
| *(paper Fig. 8, not in notebook)* | `ac.active_subspace(X=X, f=f, space=space, estimator="qphd")`; plot `abs(sub.eigenvalues[:5])` | v1 QPHD uses C = ggᵀ + HΣH with Σ = I/3, which is paper eq. (14) for uniform x. 2016 used γ estimated from the data, so the eigenvalues differ by a constant factor. Prototype gaps match the paper (§3). Also warns `ValidityWarning`. |
| hand-made summary plots `ax.plot(X @ aa, ff, 'ko')` | keep matplotlib (2×3 grids with zoom). `ac.plotting.plot_sufficient_summary(sub, X=X, f=f)` is available for single panels | the v1 plot recomputes Xw with the v1 sign, so the plot can be mirrored relative to the weights figure. Keep the hand plot with the aligned w. **Superseded (V1-04, [#48](https://github.com/paulcon/as-data-sets/pull/48)):** the v1 OLS sign is the weights' sign, so the v1 plot is not mirrored. |
| — | `ac.plotting.plot_eigenvalues(sub_qphd)` for the optional Fig. 8 | |

## 7. Associated .py files
- **battery_figs.py: rewrite** (do not port line by line). Either inline it into the notebook (preferred, since the notebook is now almost empty) or reduce it to a `battery_analysis.py` of about 150 lines with:
  - `load(rate)` (§5)
  - `weights(X, F, cols)` → (len(cols), 19), sign-aligned
  - `bootstrap_se(X, F, cols, rng, nboot=100)`
  - `plot_weights(ax, grid, A, bands, label_style)`, with one `input_labels()` table kept as module constants (labels, linestyles, colors, linewidths, groups)
  - `plot_summaries(X, F, idx, grid, qoi_label)`
  - one `SPEC` dict per (QoI, rate) holding shaded indices and axis labels.

  No pandas, no pdb, raw strings for TeX, and no `plt.rc` side effects at import (set rcParams in the notebook). Keep the paper's line-style legend design (groups Anode/Cathode/Separator/Miscellany).
- README: replace it with a short data dictionary (columns, units, grid, dropped run, provenance, citation).

## 8. Update plan (one commit each)
1. Data (after Q5): write `battery_data.npz` (or `*.txt.gz`) with a conversion script `make_battery_npz.py`. Keep the CSVs until step 6. Update `Lithium_Battery/README` with the data dictionary.
2. Rewrite `battery_figs.py` as `battery_analysis.py` (or inline it), in Python 3 with numpy only. Add an assert that its w equals the old lstsq path to 1e-12.
3. Convert the notebook to Python 3: kernelspec `python3`, clear outputs, replace the `from battery_figs import *` cells with explicit calls, `rng = np.random.default_rng(2016)`.
4. Introduce v1: `ac.active_subspace(estimator="ols")` with sign alignment, or keep lstsq (Q8). Add an optional QPHD eigenvalue cell (paper Fig. 8).
5. Text fixes from §3: the active-subspace definition, the typos, units, column order, the dropped run, the 4C statement, the bootstrap statement, published reference + Hadigol reference, and credit Hadigol as data provider (published Acknowledgments, p. 260). Do not import the three published-paper errata (§10, Q10).
6. Figure fixes: units on axes, no ylim clip on the bootstrap figure, correct bootstrap x-labels, A/B/C labels in axis coordinates.
7. Remove the CSVs if the author approves (Q5), and fix the top-level README ("18" → "19", DOI link).
8. Execute headless (`jupyter nbconvert --to notebook --execute --inplace Li_Battery.ipynb`) and commit with outputs.

## 9. Acceptance criteria
- `cd Lithium_Battery && jupyter nbconvert --to notebook --execute Li_Battery.ipynb` succeeds in `.venv` in < 3 min (prototype: about 11 s per rate for all weights, both bootstraps and the v1 calls; loading takes 0.3 s per file).
- Imports: only numpy, scipy (optional), matplotlib, active_subspaces. No `pandas`, no `pdb`. No warnings in the output other than an explicitly filtered `ValidityWarning`.
- Weights match the 2016 lstsq to ≤ 1e-12 at every grid point, with sign convention cov(Xw, f) > 0. Spot values (rounded to 0.01):
  - 0.25C capacity, idx 38 (3.10 V): ε_a −0.77, L_a +0.64, all others |·| ≤ 0.01
  - 1C capacity, idx 34 (3.18 V): ε_c 0.62, brugg_c −0.60, L_a 0.36
  - 4C voltage, idx 47 (t* = 5): D −0.59, ε_s 0.49
- Linear-fit R² of the summary plots: 0.25C capacity A/B/C = 0.992 / 0.931 / 1.000; 1C capacity idx 34 = 0.840 (the loosest).
- The seeded bootstrap SE has its maximum in region B for 0.25C (idx 17) and 1C (idx 34); overall max ≤ 0.011.
- Optional QPHD: log₁₀(λ₁/λ₂) for 0.25C capacity at A/B/C ≈ 1.9 / 0.85 / 3.5.
- Figures (15; author, 2026-10-30): 6 weight-vs-grid figures, 6 summary grids (2×3), 1 bootstrap grid (3×2), paper Fig. 8 (QPHD eigenvalues, 3×2), and 1 v1 bootstrap-band figure (`plot_eigenvectors(..., interval="percentile")`, 1C capacity at A/B/C). Every QoI axis has units.
- Data in the repo ≤ 20 MB for this directory (if the npz/gz option is accepted).

## 10. Prototype results
Run in `.venv` (scripts in `prototypes/g5/li*.py`):
- Loading: `np.loadtxt` parses each 22 MB file in 0.3 s. Shapes (3600, 270), no NaN, the `Run` column is 1..3600, and the inputs are identical across the 3 files.
- Inputs: all 19 normalized columns are uniform on [−1, 1] (std 0.571–0.583, excess kurtosis −1.18 to −1.23), **including r_a and r_c** (Q1).
- Outliers: the only large |z| is in 4C `Vol50`: row 1537 (z = 59, V = 2.746) and row 2968 (z = 9.5, V = 2.791). 0.25C and 1C have none.
- v1 `estimator="ols"`, `n=1`: max ‖|w_v1| − |w_lstsq|‖∞ = 4.8e-15 / 4.2e-15 / 1.4e-15 (0.25C/1C/4C capacity). Sign disagreements: 44/50, 12/50, 0/50.
- Bootstrap (100 reps, seeded): max SE for capacity 0.0053 / 0.0103 / 0.0044, for voltage 0.0028 / 0.0031 / 0.0052.
- Effect of dropping Run 1538 (4C): capacity weights change by ≤ 3.2e-4, voltage weights (cols 1–48) by ≤ 0.020.
- QPHD (v1) 0.25C capacity, idx 2/17/38: |λ|₁…₅ = [3.9e-4, 4.9e-6, …], [6.2e-3, 8.7e-4, …], [2.4e-2, 7.1e-6, …].
- Paper Figs. 2, 5, 9 (arXiv v2, and the published Figs. 2 and 9, pp. 253 and 260) agree with the legacy notebook figures and with the prototype values.
- The published Fig. 8A (p. 259), read off the log axis, gives a ≈ 4e-4/6e-6, b ≈ 5e-3/8e-4 and c ≈ 2.5e-2/7e-6 (λ₁/λ₂). The v1 QPHD prototype above matches within plot-reading accuracy, so `estimator="qphd"` reproduces paper Algorithm 2 for this case.
- Published-paper issue (not in the notebook): §3.1 (p. 250) says capacity "is inversely proportional to the cell voltage". The data show capacity is a decreasing, strongly nonlinear function of voltage (e.g. 0.25C: 0.04 → 2.9 mAh cm⁻² between 4.13 V and 2.8 V), not ∝ 1/V. Do not copy the sentence into the notebook.
- Paper-internal inconsistency (does not affect the notebook): paper §4.1 says that at 0.25C/3.1 V "all weights are nearly zero except … anode porosity and … anode Bruggeman coefficient", and that ε_a's weight is positive. The paper's own Fig. 2a and the data show ε_a = −0.77 and **L_a** = +0.64 (brugg_a ≈ 0). The physics in paper §3.2.1 ("increasing porosity … lowers capacity") supports the negative sign. This is confirmed against the published version: the text on pp. 253–254 is unchanged, and the published Fig. 2A on p. 253 shows ε_a ≈ −0.77 and L_a ≈ +0.64. If the notebook gains a sentence on this case, use the data, not the paper text.

## 11. Open questions for the author
1. r_a and r_c: Table 1 says N(2, 0.1354) and N(2, 0.3896), but the stored normalized values are uniform on [−1, 1]. Were they sampled uniformly, or are the stored x CDF-transformed? Is 0.1354 a standard deviation or a variance? What text should the notebook use?
2. t* grid: is column k at t* = 99 − 2k (code) or at 100 − 100k/49 (`Vol50` ≡ 2.8 V suggests t* = 0)? The answer changes the axis labels slightly (e.g. B: 23 vs 22.4).
3. Confirm I_1C = 30 A m⁻², and that the `Cap*` columns are elapsed time in seconds.
4. Keep the 150 concentration columns in the shipped data (unused, undocumented), or move them to an archival deposit only?
5. Data format and storage, a repo-wide decision: `.npz` vs `.txt.gz`; delete the CSVs; Zenodo deposit; any history rewrite (`.git` is 225 MB).
6. License and permission to redistribute the Hadigol et al. data. The published Acknowledgments (p. 260) confirm Hadigol provided the data set. Should Hadigol be credited in the notebook header?
10. Should the errata in the published paper be recorded somewhere (e.g. a note in the notebook)? They are: "region c" for "region b" (p. 259); the ε_a sign and brugg_a/L_a mix-up (pp. 253–254); "inversely proportional to the cell voltage" (p. 250).
7. Current affiliations and emails for the header (Mines addresses from 2016).
8. Use v1 `estimator="ols"` (with a sign fix and a filtered ValidityWarning) or keep the explicit lstsq Algorithm 1? Add the paper's Fig. 8 via QPHD?
9. Run 1538 is dropped but Run 2969 (final voltage 2.791 V) is kept. Is that intended?
