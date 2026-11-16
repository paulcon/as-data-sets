# Irrigation_Water_Withdrawal: data

Kind: **Python model**. The model is `iww_model.py`, a closed form: Eq. 16 of the Supplementary Materials of Puy et al. (2022), evaluated as the authors' archived code evaluates it. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun reproduces them exactly (see Regenerating).

**Which form of Eq. 16.** Eq. 16 as printed (Supplementary Materials, p. 18) does not reproduce the published results. At the midpoint of Table S4 it gives a reference evapotranspiration of 55.6757 mm for one January day and a withdrawal of 102852.3 m³, against the paper's 8600 m³. The authors' code in the same Zenodo record (`code_ghm_auditing.R`, function `full_model`, v1.0.4) differs from the printed Eq. 16 in two ways: it divides ET₀ by 31 ("To get a daily value"), and its Penman–Monteith expression divides the numerator by Δ alone and then adds γ(1 + 0.34w), instead of dividing by Δ + γ(1 + 0.34w). With both, the midpoint gives 8530.7 m³. **The data here use the archived code's form**, which is what produced the published numbers. The printed form is kept for comparison (`iww(X, form='printed')`, and `form='printed31'` with the division by 31; 3205.3 m³ at the midpoint). These values are printed by the notebook ("The model").

## Files

| file | contents | role |
|---|---|---|
| `Irrigation_Water_Withdrawal.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2022; outputs committed |
| `iww_model.py` | `iww(X, form)`, `grad_iww(X)`, `et0(X, form)`, `sample(n, rng)`; `NAMES`, `SYMBOLS`, `UNITS`, `DESCRIPTIONS`, `LOWER`, `UPPER`, `NOMINAL`, `FORMS` | the model and its input ranges (numpy) |
| `sobol_tools.py` | `base_matrices`, `evaluate`, `indices`, `bootstrap`, `g_function`, `g_function_indices` | Sobol' indices with the paper's estimators, used in the notebook's comparison (numpy, scipy) |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files | regenerates the data |
| `check_accuracy.py` | checks the model against a second transcription of the archived code's expression, and `sobol_tools.py` against exact indices (see Accuracy) | check |
| `inputs.csv` | 2000 × 12 physical inputs, columns in the order of `NAMES` (units in the header) | model input |
| `outputs.csv` | 2000 × 1: irrigation water withdrawal y in m³ (one day, the grid cell) | model output |
| `gradients.csv` | 2000 × 12: ∂y/∂x_j with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description, then the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly.

## Model

**Source.** A. Puy et al., "The delusive accuracy of global irrigation water withdrawal estimates", *Nature Communications* 13:3183 (2022), doi:10.1038/s41467-022-30731-8 (the Comment; pp. 1–4). The model, the inputs and the results used here are in its Supplementary Materials, its ref. 4: Puy et al., Zenodo, doi:10.5281/zenodo.5528844, version 1.0.4 (the latest version on 2026-11-15). The record holds `supplementary_materials.pdf` (25 pages, cited here by its printed page numbers) and the R code `code_ghm_auditing.R` (also at https://github.com/arnaldpuy/ghm_auditing). Section 2.5 (pp. 17–19) describes the case: wheat in the grid cell at x = −99.7084, y = 29.4583 in Uvalde County, Texas, January 6–7, 2007 (the Comment's Fig. 1c caption, p. 3, and the code give the longitude as −99.7083). The authors' R code (`code_ghm_auditing.R`, function `full_model` and the chunks that build the input bounds) was read, to establish the units and the form the authors evaluated. `iww_model.py` is a Python reimplementation of that model, written from the equations; no code was copied into it. As an independent check, `check_accuracy.py` (check 1) transcribes the code's one-line expression for y into Python. Check 4 rebuilds the code's input bounds from a few archived values typed in as constants.

**Equations.** Supplementary Materials Eq. 1 (p. 3): y = I_a (ET_c − P)/E_p; Eq. 2 (p. 4): ET_c = k_c ET₀; Eq. 4 (p. 5), FAO-56 Penman–Monteith: ET₀ = (0.408 Δ A + γ (900/(T_a + 273)) w v) / (Δ + γ(1 + 0.34 w)); Eq. 5 (p. 8): E_p = E_a E_c M_f. Eq. 16 (p. 18) combines them, printed as y = I_a [k_c ET₀ − P]/(E_a E_c M_f), with ET₀ from Eq. 4. The form used here (archived code) is

  E = (0.408 Δ A + γ (900/(T_a + 273)) w v)/Δ + γ(1 + 0.34 w),  ET₀ = E/31,
  y = 10 I_a (k_c ET₀ − P)/(E_a E_c M_f)  [m³],

with I_a in ha and ET₀ and P in mm (units from the code's comments; 1 mm over 1 ha is 10 m³). P is subtracted inside the I_a factor, as in Eqs. 1 and 16 and in the code. **Units.** Eq. 16 is printed without units. Eq. 1 (p. 3) gives I_a in m², ET_c and P in m and y in m³. That conflicts with the units the supplement uses elsewhere: irrigated areas in ha (p. 17, and Table S4's range for I_a) and ET₀ in mm (p. 4). This data set uses ha and mm, as the code does. The text after Eq. 16 says "we use the Priestley-Taylor formula for ET₀", but Eq. 16 is the Penman–Monteith formula, and the text after Table S4 (p. 18) says so too.

**Inputs** (m = 12, independent uniforms; Table S4, p. 18, in its order):

| name | symbol | units | description | distribution | source |
|---|---|---|---|---|---|
| `Delta` | Δ | kPa °C⁻¹ | slope of the saturation vapour pressure curve (Table S4: "vapour pressure"; Eq. 4 text, p. 5: "gradient of saturated vapour pressure") | U(0.0796, 0.0804) | Table S4 |
| `gamma` | γ | kPa °C⁻¹ | psychrometric constant | U(0.065, 0.066) | Table S4 |
| `A` | A | MJ m⁻² (see Unknowns) | net radiation minus soil heat flux | U(297.550, 402.448) | Table S4 (same range as Table S1, p. 8, for the Bosque del Apache, New Mexico) |
| `T_a` | T_a | °C | mean air temperature | U(9.9, 10.1) | Table S4 |
| `w` | w | m s⁻¹ | wind speed | U(2.67, 2.95) | Table S4 |
| `v` | v | kPa | vapour pressure deficit | U(0.26, 0.29) | Table S4 |
| `k_c` | k_c | – | crop coefficient (wheat) | U(0.45, 1.14) | Table S4; Ko et al. data for January 6–7, 2007 (p. 19, Fig. S11) |
| `I_a` | I_a | ha | irrigated area of the cell | U(42.9, 144.5) | Table S4; GRIPC and FAO-GMIA maps (pp. 17, 19) |
| `E_a` | E_a | – | field application efficiency | U(0.49, 0.88) | Table S4; Bos and Nugteren, sprinkler (p. 19) |
| `E_c` | E_c | – | conveyance efficiency | U(0.64, 0.96) | Table S4; Bos and Nugteren, sprinkler (p. 19) |
| `M_f` | M_f | – | management factor | U(0.50, 0.97) | Table S4; the range reported by Bos and Nugteren, with no significant difference between systems above and below 10,000 ha (p. 19) |
| `P` | P | mm | precipitation | U(0, 0.1) | Table S4; Uvalde, January 6–7, 2007 (p. 19) |

`NOMINAL` in `iww_model.py` is the midpoint of every range, the mean of each input; the paper's point estimate is y at the mean values of the inputs (Fig. S13a caption, p. 21).

**Quantity of interest.** y, the irrigation water withdrawal in m³ for one day in the grid cell. Over the 2000 samples it ranges from 1554 to 30729 m³ (mean 9146, sd 4779 m³; printed by the notebook and by `generate_data.py`), and it is positive everywhere on the box, since k_c ET₀ always exceeds P.

**Samples outside physical meaning.** None: `generate_data.py` counts 0 samples with y ≤ 0 or k_c ET₀ ≤ P. All values lie inside the Table S4 bounds (checked in the notebook).

**Gradients.** Exact, with respect to the physical inputs (`iww_model.grad_iww`, archived form only). With c = 900/(T_a + 273), ∂y/∂E = 10 I_a k_c/(31 E_p); ∂E/∂Δ = −γ c w v/Δ², ∂E/∂γ = c w v/Δ + 1 + 0.34 w, ∂E/∂A = 0.408, ∂E/∂T_a = −γ w v 900/((T_a + 273)² Δ), ∂E/∂w = γ c v/Δ + 0.34 γ, ∂E/∂v = γ c w/Δ; ∂y/∂k_c = 10 I_a ET₀/E_p, ∂y/∂I_a = y/I_a, ∂y/∂E_a = −y/E_a (likewise E_c, M_f), ∂y/∂P = −10 I_a/E_p.

**Sampling design.** Monte Carlo: M = 2000 draws, LOWER + (UPPER − LOWER) · `numpy.random.default_rng(2022).random((2000, 12))` (`iww_model.sample`). The notebook's Sobol' indices use a separate design, not shipped: base matrices A and B of N = 2¹⁵ rows from one scrambled Sobol' sequence in 24 dimensions (`scipy.stats.qmc.Sobol`, seed 2022), 753664 model runs in all, recomputed by the notebook in a few seconds.

**What the notebook finds** (seed 2022): a one-dimensional active subspace, λ₁/λ₂ = 38.8 (λ₁ holds 95.89 % of the trace), w₁ ≈ (I_a +0.602, k_c +0.491, M_f −0.422, E_a −0.361, E_c −0.243, A +0.173, P −0.015; Δ, γ, T_a, w, v below 10⁻³ in magnitude), within |cos| = 0.9976 of the normalized gradient of log y at the midpoint. A cubic in y₁ has leave-one-out R² = 0.9848. The Sobol' indices agree with the read-offs from Fig. S13b–c, and the activity scores rank the six influential inputs in the same order as the total-order indices.

## Regenerating

```bash
python generate_data.py           # writes the three CSVs here
python generate_data.py --check   # reruns into a temporary directory and compares with the shipped files
python check_accuracy.py          # the checks of the Accuracy section
```

`generate_data.py` takes 0.05 s (0.2 s with Python start-up) and also prints the gradient check and the count of samples outside physical meaning. The shipped files were written on 2026-11-15 in the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS 14 on x86_64). There, `--check` reports relative differences of 0.0 for `inputs.csv`, `outputs.csv` and `gradients.csv` (largest difference from the shipped files: 0.0e+00) in 0.2 s. On another platform, differences at rounding level (about 10⁻¹⁶ relative) would be expected. `--check` exits with code 1 only on a structural mismatch: a missing file, a different shape or NaN pattern, or inputs that differ from the shipped ones by more than 10⁻⁸ of each input's largest magnitude. `check_accuracy.py` takes about 1 s. The notebook runs in 13–18 s (17–25 s with `jupyter nbconvert` start-up), and two runs print the same values apart from the run time.

## Accuracy

- **Values.** Closed form in double precision. `iww_model.iww` reproduces `outputs.csv` exactly (largest relative difference 0.0e+00; printed in the notebook).
- **Transcription.** `check_accuracy.py` (check 1) writes the archived code's expression out a second time, in the order the code evaluates it and with no code shared with `iww_model.py`. It agrees with `iww_model.iww` and with `outputs.csv` to a relative 3.8e-16 on the 2000 shipped inputs, and gives 8530.6674 m³ at the midpoint.
- **Gradients.** Exact. Checked by `generate_data.py` and by the notebook at all 2000 samples against central differences with step 10⁻⁶ of each input's range: largest relative error (row 2-norm) 1.6e-07, the size expected from rounding in the differences.
- **Sobol' estimators.** `check_accuracy.py` (check 2) applies `sobol_tools.py` to the Sobol' G function in 12 dimensions with N = 2¹⁵, as in the notebook, against its exact indices: largest absolute errors 1.2e-03 (first order), 4.5e-03 (total order) and 1.4e-03 (second order, seven pairs).
- **Table S4 against Eqs. 17–22.** `check_accuracy.py` (check 3) evaluates the supplement's formulas for Δ, v and γ at Table S4's T_a; see Unknowns.
- **Table S4 against the archived code's bounds.** `check_accuracy.py` (check 4) rebuilds the bounds as the code computes them and evaluates y at their midpoint; see "How these files relate to the paper".
- **Sobol' intervals.** In the notebook, the 95 % bootstrap intervals (±1.96 bootstrap standard deviations, 1000 resamples) of the irrigation model's indices have half-widths of at most 0.0219 for S_i, 0.0086 for T_i and 0.0075 for S_ij (printed in the notebook).

## How these files relate to the paper

- **New runs of the authors' model, not their runs.** The paper's samples were not published. The data here are new samples of the archived code's form of Eq. 16 on the Table S4 ranges.
- **Eq. 16 as printed vs the archived code.** See "Which form of Eq. 16" above: the printed equation gives 102852.3 m³ at the midpoint and the archived form 8530.7 m³, against the paper's 8600 m³ (Comment p. 2; Supplementary Materials p. 19). The divisor 31 and the Penman–Monteith denominator of the archived code are not explained in the paper.
- **Reproduced.** The point estimate (8531 m³ at the Table S4 midpoint, −0.8 %; 8635.9 m³ at the code's bounds, +0.4 %; against 8600 m³); the 2.5–97.5 % range of y (2825–21073 against 3000–21,000 m³, p. 19); the range of y (1234–40952 m³ over 2¹⁵ samples, against 1400–40,000 m³, Comment p. 2); the share of interactions (1 − ΣS_i = 0.0888 against "c. 10%", p. 19); and the first-, total- and second-order Sobol' indices of Fig. S13b–c (p. 21), which the paper shows only as plots and which we compare with approximate read-offs (table in the notebook).
- **The point estimate.** At the Table S4 midpoint, y is 8530.7 m³, 0.8 % below the paper's 8600 m³. The archived code does not use Table S4's printed bounds: it computes them from archived data. `check_accuracy.py` (check 4) rebuilds those bounds from archived values typed in as constants (the climate file's days 6–7, the 8 k_c points with 50 < x < 53, the sprinkler rows of `bos.dt.csv`, and the code's A = 350 ± 15 % and w = 2.81 ± 5 %; I_a from the text, p. 17, since the code's map rasters are not archived). At their midpoint, y = 8635.9 m³, 0.4 % above 8600 m³, which is within the rounding of "8600". The paper also evaluates y at the sample means of its inputs rather than the exact midpoints. Of the 1.2 % between the two midpoints, the k_c bounds alone account for +0.68 % and the weather bounds (Δ, γ, T_a, w, v) alone for +0.55 %. **This data set keeps Table S4 as its input space by choice**, since it is the distribution the paper states.
- **Sampling.** Section 2.2 (p. 14) says Sobol' quasi-random numbers were used. The archived code sets `type <- "R"` and passes it to sensobol's `sobol_matrices`. The sensobol reference manual (CRAN, version 1.2.0, published 2026-07-17, `sobol_matrices`, p. 23) says `type = "R"` "uses random numbers" and the default `type = "QRN"` uses Sobol' quasi-random numbers, so the code disagrees with section 2.2. We read the current version; the authors' 2021 run used an earlier version of sensobol, which we did not read. The notebook uses a scrambled Sobol' sequence.

## Unknowns

- Why the archived code divides ET₀ by 31 and divides the Penman–Monteith numerator by Δ alone. Which form the authors intended for Eq. 16 is a question for the authors.
- The units of A. Eq. 4 (p. 5) gives MJ m⁻² d⁻¹, but the division by 31 suggests the values were treated as monthly totals. The range of A in Table S4 is identical to that of Table S1 (p. 8), which describes a mid-morning June day in the Bosque del Apache, New Mexico, not Uvalde in January. The archived code uses 350 ± 15 % (297.5–402.5), where Table S4 prints 297.550–402.448.
- The origin of the Table S4 ranges for Δ, γ, T_a and v. The archived code computes them from the archived climate file `uvalde_climate_data_january.csv` (mean of January 6 and 7) with Eqs. 17–22 (pp. 18–19). Read as coded, that file gives T_a = 11.0 °C, against Table S4's 10 °C (9.9–10.1), and a relative humidity of 0.6865, against the h_rel = 0.77 stated on p. 19. The code's bounds then differ from Table S4's midpoints by 9.1 % (T_a), 33.2 % (v), 1.6 % (γ) and 8.3 % (Δ), relative to the code's values. The code also gives γ a range of ±0.1 % (0.066519–0.066653), where Table S4 prints 0.065–0.066, ±0.76 % (`check_accuracy.py`, check 4). Even the supplement's own Eqs. 17–22, evaluated at Table S4's T_a = 10 °C with h_rel = 0.77 and z = 9 m, give Δ = 0.0823, v = 0.2824 and γ = 0.0665: 2.9 %, 2.7 % and 1.6 % above the Table S4 midpoints, and for Δ and γ outside the Table S4 ranges (`check_accuracy.py`, check 3). Why Table S4 prints these values is not known. These inputs carry Sobol' indices of about 0 in the paper and here, so the effect on the analysis is small, but their bounds shift the midpoint value of y by +0.55 % (check 4). Table S4's k_c bounds are the range of the archived Ko et al. data (`kc_wheat_new.csv`, the 8 points with 50 < x < 53, as the code selects them), 0.45643–1.14422, truncated to two decimals (0.45–1.14; rounding would give 0.46 for the lower bound). The text gives z = 9 m for the measurement height (p. 19); the code uses 9.1 m.
- The units the authors intended for Eq. 16: Eq. 1 (p. 3) gives m² and m, while the supplement gives irrigated areas in ha (p. 17) and ET₀ in mm (p. 4), and the code uses ha and mm.
- The I_a bounds the code used: it takes them from map rasters that are not in the archive, so check 4 uses the text's 42.9–144.5 ha (p. 17).
- Whether y is for one day or two. The Comment says "a single day in January" (p. 2), its Fig. 1 caption says "January 6–7, 2007" (p. 3), and the code divides by 31 "to get a daily value". We read y as one day, with the weather inputs averaged over January 6–7.
- The exact sample of the paper's histogram and quantiles (the code uses the first N = 2¹⁵ rows of its sample matrix, generated by sensobol), so the paper's extremes (1400 and 40,000 m³) cannot be matched draw for draw.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the equation and the input ranges published in the Supplementary Materials. No file from the authors' archive is redistributed, and none of their code (whose Zenodo record states the license "other-open" and whose GitHub repository has no license file) was copied. Model and input ranges: A. Puy and co-authors. Cite:

- A. Puy, R. Sheikholeslami, H. V. Gupta, J. W. Hall, B. Lankford, S. Lo Piano, J. Meier, F. Pappenberger, A. Porporato, G. Vico, A. Saltelli, "The delusive accuracy of global irrigation water withdrawal estimates", *Nature Communications* 13:3183, 2022, doi:10.1038/s41467-022-30731-8.
- A. Puy et al., "Supplementary Materials for The delusive accuracy of global irrigation water withdrawal estimates", Zenodo, v1.0.4, 2021, doi:10.5281/zenodo.5528844.
