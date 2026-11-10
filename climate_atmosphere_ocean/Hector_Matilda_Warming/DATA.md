# Hector_Matilda_Warming: data

Kind: **stored data (generated here with Hector 3.1.1 and Matilda v1.0.0, outside codes run from R), with a
regeneration script.** Hector is a C++ model with an R interface (GPL-3) and Matilda is an R package (MIT); neither
is a numpy/scipy model, and Hector is not ported. The files were written by `generate_data.py`, which drives R
through `hector_driver.py`; the shipped CSVs are the record. **Regenerating the data or rerunning the accuracy
checks needs R with hector 3.1.1 and matilda v1.0.0** (recipe below); `generate_data.py` and `check_accuracy.py`
stop with an error naming them when they are missing. The notebook needs only numpy and `active_subspaces`: it
reads the shipped files and the prior definitions in `hector_driver.py`.

## Files

| file | contents | role |
|---|---|---|
| `Hector_Matilda_Warming.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2024; outputs committed |
| `hector_driver.py` | `PARAMS` (names, units, prior mean and sd, normal/lognormal), `z_to_x`, `x_to_z`, `run_x`, `run_z`, `fd_stencil`, `fd_gradients` | the Python side of the driver; constants and maps import without R |
| `hector_worker.R` | runs Hector (via Matilda's `set_params`) for a table of parameter sets | the R side of the driver |
| `matilda_score.R` | Matilda's `score_runs(criterion_gmst_obs(), score_bayesian)` on the stored GMST series | weights |
| `generate_data.py` | writes all CSVs (`--out` required, `--rows`, `--check`) | regenerates the data (needs R, hector 3.1.1, matilda v1.0.0) |
| `check_accuracy.py` | prints the numbers of "Accuracy" below (`--paper`: also the paper-ensemble check; `--weights`: only the Fig. 6B emulation, numpy only) | accuracy check (same needs, except `--weights`) |
| `inputs.csv` | 1000 × 12: $z_1..z_6$ (scaled, standard normal), then α, β, ECS (°C), K_eff (cm² s⁻¹), NPP₀ (Pg C yr⁻¹), Q₁₀ | Monte Carlo inputs, seed 20240502 |
| `outputs.csv` | 1000 × 5: warming (°C), CO₂ in 2100 (ppmv), mean GMST 1850–1900 (°C), RMSE (–), weight (–) | model outputs; one failed run is NaN (row 568) |
| `gmst_1950_2023.csv` | 1000 × 74: raw Hector GMST 1950–2023 (°C) | the series Matilda scores |
| `gradients.csv` | 400 × 12: ∂warming/∂z_j (°C), ∂CO₂₂₁₀₀/∂z_j (ppmv), rows 0–399 of `inputs.csv` | central differences, step 0.25 in z; row 336 has NaN in the NPP₀ entries |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are written with 17 significant digits, the digits the model was run with.

## Model

### Citation

J. K. Brown, L. Pressburger, A. Snyder, K. Dorheim, S. J. Smith, C. Tebaldi, B. Bond-Lamberty, "Matilda v1.0: An R
package for probabilistic climate projections using a reduced complexity climate model", *PLOS Climate* 3(5):
e0000295 (2024), [doi:10.1371/journal.pclm.0000295](https://doi.org/10.1371/journal.pclm.0000295) (Crossref-checked). The PDF read is the publisher's version (CC BY).
Hector: K. Dorheim et al., *Geosci. Model Dev.* 17, 4855–4869 (2024), doi:10.5194/gmd-17-4855-2024 (the paper cites
its discussion version, "Hector V3.1.1", doi:10.5194/egusphere-2023-1477, ref. 22); C. A. Hartin et al.,
*Geosci. Model Dev.* 8, 939–955 (2015), doi:10.5194/gmd-8-939-2015.

### Model and scenario

Hector, a global, annual-step carbon-cycle climate model (paper pp. 2–3), run under SSP2-4.5, the scenario of the
paper's worked example (Sec. 2, pp. 3–4) and one of the four in its case study (Sec. 3, pp. 16–18), with Hector's
bundled `hector_ssp245.ini` (emissions-driven CO₂, default solver and spin-up settings). Each run starts from
`reset(core, 0)` (spin-up redone) and runs to 2100. Matilda's `iterate_model` runs to Hector's end year 2300;
the years to 2100 are the same (`check_accuracy.py`, part 1).

### Inputs (paper Table 1, p. 5)

| input | Hector name | units | prior | z ↦ x |
|---|---|---|---|---|
| α, aerosol forcing scaling | `AERO_SCALE` | – | Normal(1.0, 0.23) | 1.0 + 0.23 z |
| β, CO₂ fertilization | `BETA` | – | Normal(0.55, 0.10) | 0.55 + 0.10 z |
| ECS | `ECS` | °C | lognormal, mean 3.0, sd 0.65 | exp(1.0757 + 0.2142 z) |
| K_eff, ocean heat diffusivity | `DIFFUSIVITY` | cm² s⁻¹ | Normal(1.16, 0.118) | 1.16 + 0.118 z |
| NPP₀ | `NPP_FLUX0` | Pg C yr⁻¹ | Normal(56.2, 14.3) | 56.2 + 14.3 z |
| Q₁₀ | `Q10_RH` | – | lognormal, mean 2.2, sd 1.0 | exp(0.6945 + 0.4334 z) |

- The paper gives the priors as "mean ± standard deviation" (Table 1) and says that for lognormal ones "(μ, σ) is
  substituted by log(μ, σ)" (p. 5). Matilda v1.0.0's code (`lognorm()` in `R/iterate_model.R`, used by
  `generate_params()` in `R/generate_param.R`) treats 3.0 ± 0.65 and 2.2 ± 1.0 as the arithmetic mean and sd of
  the parameter; this data set follows the code. The means are Hector 3.1.1's defaults, which equal Table 1
  (Hector 3.2.0 and later changed β, Q₁₀ and K_eff to 0.53, 1.76 and 2.38, so the version matters).
- The priors are independent (p. 5). **Scaled coordinates:** every input is a function of one standard normal
  $z_j$ (the last column above), so each is scaled through its own prior; gradients and the subspace are in z.
- The normal priors are unbounded. Of the 1000 samples, none has α, β, K_eff or NPP₀ ≤ 0; 51 have Q₁₀ < 1
  (respiration falling with warming), which Hector accepts and which are kept. One run fails: row 568, NPP₀ = 2.28
  Pg C yr⁻¹ (z = −3.77); Hector stops and the worker writes NaN. The notebook drops it by the stated rule
  "non-finite output".

### Quantities of interest

- `warming`: mean GMST 2081–2100 minus that run's mean GMST 1850–1900 (°C), the case-study metric "long-term
  (2081–2100) global mean surface temperature change relative to a pre-industrial reference (1850–1900)" (p. 17).
  Subtracting each run's own 1850–1900 mean is this data set's reading; the paper's code differs (see "How these
  files relate to the paper"). `gmst_1850_1900` is stored, so the raw 2081–2100 mean is `warming + gmst_1850_1900`.
- `co2_2100`: atmospheric CO₂ concentration in 2100 (ppmv), a secondary output (Matilda's other scoring variable,
  pp. 9–10, Fig. 4).
- `rmse`, `weight`: Matilda's `score_bayesian` with the defaults the case study keeps (p. 17): RMSE against
  `criterion_gmst_obs()` (1950–2023, HadCRUT5-based anomalies in Matilda v1.0.0) with σ_Y = sd of the observations,
  likelihood exp(−½(RMSE/σ_RMSE)²) with σ_RMSE = sd of the RMSE values, normalised over the 999 finite runs
  (paper Eqs. 4–6, p. 9).

### Gradients

Central differences in z with step h = 0.25 (a quarter of a prior standard deviation) for every input, 12 extra
runs per sample, at rows 0–399 (row 336 has NaN in its NPP₀ components because the stencil point at lower NPP₀
failed; the notebook drops that row). The step is large on purpose. Hector's outputs are not smooth at small scales. Scans of 41 points on ±0.5 in z_j
(spacing 0.025) at rows 376, 249 and 273 (`check_accuracy.py`, part 3) show a sawtooth: a single 0.025 step departs from
a quadratic fit by up to 1.3e-2 °C in warming and 7.4 ppmv in CO₂, typically 5e-3 to 9e-3 °C and 5–6 ppmv at rows 249
and 273 (smaller at row 376: up to 3.4e-3 °C and 2.2 ppmv). The jumps are not specific to NPP₀: they appear along
NPP₀, Q₁₀ and ECS at all three rows, and along every input at rows 249 and 273. Inside one tooth the local slope
differs from the smoothed slope, so the derivative is not defined at small scales and the finite-difference value
depends on the step. At row 249 the NPP₀ component of the CO₂ gradient is −18.9 (shipped h = 0.25), −25.0 (h = 0.5),
−42.0 (h = 0.1) and −25.9 ppmv per unit z for the slope of the quadratic fit on ±0.5 (part 3). Tightening the spin-up
and solver tolerances (`eps_spinup`, `eps_abs`, `eps_rel`) did not remove the roughness in exploratory runs (external
run, not reproducible from the repository). The gradients are therefore smoothed slopes, not exact derivatives: between
steps the warming gradients differ by a few per cent (median 2–3 %, up to 8.5 % per row), and the CO₂ gradients by about
10 % in the median, with some rows up to 46 % and single components 25–50 % (Accuracy).

### Sampling design

Plain Monte Carlo: 1000 independent draws of z ~ N(0, I₆) (numpy `default_rng(20240502)`), as in Matilda's
`generate_params` (independent marginal draws, p. 5), but not the paper's draws (R's `set.seed(245)` in the
paper's scripts). Gradients at the first 400 draws.

### Published results (paper)

No active-subspace or sensitivity results. Probabilistic projections from 1000 weighted members (Sec. 3, pp. 16–18;
Fig. 6): SSP2-4.5 "very likely (92%) to be warmer by 1.7˚C–3.8˚C" (p. 18); SSP1-2.6 94 % for 1.1–2.6 °C (p. 18);
SSP1-1.9 97 % for 1.0–2.0 °C and SSP3-7.0 92 % for 2.4–4.8 °C (p. 17). The 25-member SSP2-4.5 example gives
"~90% probability" for 2.0–4.0 °C (p. 16). Run time: "~100 minutes" for 1000 sets × 4 scenarios, serially (p. 17).

### What the notebook finds (from these files)

- Warming: λ₁/λ₂ = 36.8, λ₁ = 95.8 % of the trace, mean bootstrap subspace distance 0.013 (k = 1) and 0.17 (k = 2);
  the next eigenvalue ratios are 2.88, 1.96, 3.05, 2.30. n = 1 is chosen. The ratio depends on the finite-difference
  step: scaling the step-sensitive NPP₀ gradient column by 0.7 or 1.3 moves it to 47.0 or 28.1 (w₁ moves by
  |w₁ᵀw₁'| = 0.996), so it is ≈ 28–47, "about 30–45", rather than exactly 36.8. w₁ = (α, β, ECS, K_eff, NPP₀, Q₁₀) =
  (−0.171, −0.252, +0.862, −0.064, −0.288, +0.277); ECS carries 71.5 % of E[(∂f/∂z)²]. A quadratic in y₁ has LOO R²
  0.9834 over the 999 runs (0.9911 in two variables). QPHD and OLS from function values alone give the same direction
  (|w₁ᵀw| = 0.9999 and 0.9998).
- CO₂ in 2100: λ₁/λ₂ = 8.57, a carbon-cycle direction (Q₁₀ +0.606, NPP₀ −0.545, β −0.516); LOO R² 0.9317 in one and
  0.9746 in two active variables. QPHD and OLS from the 999 values give the same direction (|w₁ᵀw| = 0.9996 and
  0.9986); scaling the NPP₀ gradient column by 0.7 / 1.3 gives |w₁ᵀw₁'| = 0.984 / 0.987 and λ₁/λ₂ = 12.21 / 7.61.
- Weighted P(1.7 < warming ≤ 3.8 °C) = 96.9 % (bootstrap 95.7–98.0 %) against the paper's 92 % (see below). An
  emulation of the paper's Fig. 6B script with misaligned weights gives 93.2 % and bins that match Fig. 6B; this is
  the likely, unconfirmed cause of the difference ("The 92 %" below).

## Regenerating

External run (the shipped files were written in a Linux container on 2026-11-09/10, R 4.4.3 from conda-forge,
hector 3.1.1, matilda v1.0.0). Recipe:

```bash
# R and a C++ compiler (conda-forge; CRAN binaries work too)
micromamba create -p ./renv -c conda-forge r-base=4.4 r-rcpp r-bh gxx_linux-64 make
export PATH=$PWD/renv/bin:$PATH
git clone https://github.com/JGCRI/hector   && git -C hector checkout v3.1.1    # commit 4337eebc
git clone https://github.com/JGCRI/matilda  && git -C matilda checkout v1.0.0   # commit 495e95c3
# hector's Makevars asks for C++11; recent Boost (r-bh) headers need C++14:
printf 'CXX11STD = -std=gnu++14\n' > Makevars.user && export R_MAKEVARS_USER=$PWD/Makevars.user
mkdir rlib && R CMD INSTALL -l rlib hector && R CMD INSTALL -l rlib matilda
export R_LIBS=$PWD/rlib          # or HECTOR_RSCRIPT=/path/to/Rscript
python generate_data.py --out DIR                       # all files
python generate_data.py --out DIR --rows 0:5 --check    # a subset vs the shipped files
```

Matilda v1.0.0's DESCRIPTION reports version 0.1.0; the git tag is what identifies it. The run cost
5800 Hector runs (1000 base, 4800 for the gradients), about 0.4 s of CPU each plus about 1 s of R start-up per chunk:
the run was interrupted once by a container restart and resumed from its chunk cache; the wall time, with up to six
other jobs sharing two CPUs, was about 2 h (base runs about 350 s per 250 samples, gradients 345–712 s per 40
samples, from the generator's log). On an idle core we estimate about 45 min (5800 × 0.4 s plus start-up). The paper reports ~100 minutes for 4000 runs
to 2300, serially (p. 17). `--check` compares with relative tolerance 1e-8 (same build); `generate_data.py --out DIR --rows 0:3 --check` reproduced all four files at those rows (maximum relative difference printed as 0.00e+00; 41 s). Results on
another platform or compiler may differ at the level of Hector's jumps (up to about 1e-2 °C and several ppmv; Gradients).

## Accuracy

All numbers from `check_accuracy.py` (needs R, hector 3.1.1, matilda v1.0.0; part 5 needs only numpy, `--weights`).
Parts 1–4 were run in the Linux container where the files were written. Part 5 was also rerun in the repository's
`.venv` (`python check_accuracy.py --weights`, under 1 s) and printed the values given in the table:

| check | result | script |
|---|---|---|
| Matilda's own path (`iterate_model` to 2300, then the 1850–1900 baseline) vs `outputs.csv` warming, rows 0–4 | below 1e-15 °C (printed 0.0e+00) | `check_accuracy.py`, part 1 |
| warming gradients, step 0.125 vs 0.25, rows 0–9: relative difference of the gradient vector | median 0.031, max 0.085 | part 2 |
| warming gradients, step 0.5 vs 0.25 | median 0.022, max 0.042 | part 2 |
| warming, largest component difference between steps 0.125 and 0.5, as a fraction of the gradient norm | α 0.044, β 0.043, ECS 0.045, K_eff 0.006, NPP₀ 0.043, Q₁₀ 0.053 | part 2 |
| CO₂ gradients, step 0.125 vs 0.25 / 0.5 vs 0.25 | median 0.094 / 0.084, max 0.463 / 0.192 | part 2 |
| CO₂, largest component difference (0.125 vs 0.5) / gradient norm | α 0.238, β 0.251, ECS 0.246, K_eff 0.036, NPP₀ 0.268, Q₁₀ 0.266 | part 2 |
| warming / CO₂ gradients at rows 376, 249, 273, step 0.1 vs 0.25: relative difference per row | 0.008 / 0.072 / 0.065; CO₂ 0.094 / 0.437 / 0.296 | part 2 |
| same, step 0.5 vs 0.25 | 0.013 / 0.031 / 0.036; CO₂ 0.084 / 0.187 / 0.178 | part 2 |
| row 249, NPP₀ component of the CO₂ gradient: h = 0.25 / 0.5 / 0.1 / slope of a quadratic fit on ±0.5 | −18.9 / −25.0 / −42.0 / −25.9 ppmv per unit z (27 % error at the shipped step) | parts 2–3 |
| roughness, 41 points on ±0.5 (spacing 0.025) along each z_j, rows 376, 249, 273: largest single-step departure from a quadratic fit, warming | 1.3e-02 °C (row 273, NPP₀); 5e-3 to 9e-3 °C along every input at rows 249 and 273; ≤ 3.4e-3 °C at row 376 | part 3 |
| same, CO₂ | 7.4 ppmv (row 273, NPP₀); 5–6 ppmv along most inputs at rows 249 and 273; ≤ 2.2 ppmv at row 376 | part 3 |
| same scans, largest departure of f itself from the quadratic fit | warming 9.2e-03 °C, CO₂ 4.5 ppmv | part 3 |
| slope of the quadratic fit at the sample vs shipped gradient, / gradient norm | warming median 0.007, max 0.016; CO₂ median 0.037, max 0.085 | part 3 |
| paper's draws (seed 245) under SSP2-4.5 (weights from the SSP2-4.5 series alone), weighted P(1.7 < T ≤ 3.8 °C) for raw / ensemble-baseline / own-baseline metric | 95.2 % / 96.2 % / 95.2 % (unweighted 92.2 / 93.1 / 92.6 %); 0 of 1000 runs failed; weights' effective sample size 578 | `check_accuracy.py --paper`, part 4(b) |
| paper's draws under SSP3-7.0: sets that fail | sets 114 and 930 (NPP₀ 14.2 and 13.9 Pg C yr⁻¹; the smallest NPP₀ that runs is 17.6); 0 fail under SSP2-4.5 | part 4(a) |
| paper's draws, SSP2-4.5, raw metric, weights misaligned as in part 5 with the sets of 4(a) | P(1.7 < T ≤ 3.8) = 92.7 % (paper 92 %); bins 0.018, 0.185, 0.389, 0.275, 0.100, 0.029, 0.003 (Fig. 6B ≈ 0.02, 0.185, 0.39, 0.275, 0.095, 0.03, 0.005); 27 min wall for (a)–(c) | part 4(c) |
| Fig. 6B pipeline emulated on the 999 finite runs (numbered 1–999, runs 114 and 930 dropped, `metric_calc` renumbering, merge by run number), raw-GMST metric: P(1.7 < T ≤ 3.8) | correct weights 96.5 %, unweighted 92.6 %, misaligned 93.2 % | part 5 |
| same, 0.5 °C bins (1,1.5] … (4,4.5] | correct 0.007, 0.216, 0.491, 0.243, 0.042, 0.001, 0.000; unweighted 0.022, 0.202, 0.373, 0.259, 0.110, 0.024, 0.005; misaligned 0.028, 0.194, 0.394, 0.261, 0.101, 0.016, 0.003; published Fig. 6B SSP2-4.5 (read from the figure, p. 19, ±0.01) ≈ 0.02, 0.185, 0.39, 0.275, 0.095, 0.03, 0.005 | part 5 |
| same, largest bin difference from Fig. 6B | correct 0.101, unweighted 0.017, misaligned 0.014 | part 5 |
| pre-2024-02-07 `score_bayesian` likelihood exp(−½·rmse²), P(1.7 < T ≤ 3.8) | 93.8 % (per-run baseline), 93.5 % (raw metric); effective sample size 974 (current 608) | part 5 |

The notebook checks the gradient coordinates (least-squares slope on z vs mean gradient: cosine 0.9996 for warming,
0.9990 for CO₂) and that the stored weights follow from the stored RMSE (difference of order 1e-19). The subspace
does not depend on the step error: QPHD and OLS from the 999 function values give |w₁ᵀw| = 0.9999 and 0.9998, and the
two halves of the gradient rows 0.9997 (notebook).

## How these files relate to the paper

- **Not the paper's runs.** The paper's ensemble came from `set.seed(245); generate_params(<SSP3-7.0 core>, 1000)` in R
  (scripts at github.com/JGCRI/Brown-etal_2024-PLOSClimate, commit c9a9b49, MIT; Zenodo doi:10.5281/zenodo.10695259),
  run under four SSPs. These files are a new numpy-seeded Monte Carlo sample of the same priors under SSP2-4.5 only,
  with gradients, which the paper does not have. `check_accuracy.py --paper` reruns the paper's own draws.
- **QoI.** The paper states the metric as 2081–2100 warming relative to 1850–1900 (p. 17). Its script normalises by
  one ensemble-wide 1850–1900 mean but computes the metric with `metric_calc`, which reads the raw `value` column,
  so the published probabilities use raw Hector GMST (zero in 1745). This data set subtracts each run's own 1850–1900
  mean and stores that mean (`gmst_1850_1900`), so all three versions can be formed; the ensemble mean is −0.026 °C
  with sd 0.028 °C across runs (notebook), so they differ little.
- **Weights.** The paper's script scores the 4000 runs of all four scenarios in one call (`score_runs` splits by run
  number only, so each set's four 1950–2023 series are stacked and the observations recycled); here each run is
  scored on its SSP2-4.5 series alone. The scenarios differ only after 2014.
- **The 92 %.** Our weighted probability of 1.7–3.8 °C is 96.9 % (notebook) and 95.2–96.2 % with the paper's own
  draws (`check_accuracy.py --paper`), against the paper's 92 % (p. 18); the unweighted fractions (92.2–93.1 %) are
  close to 92 %. **Likely explanation (from the authors' code; not confirmed by them):** a misalignment of weights and
  runs in the paper's scripts. (1) With the paper's draws, sets 114 and 930 (NPP₀ ≈ 14 Pg C yr⁻¹) fail under SSP3-7.0
  but run under the other scenarios (part 4(a)). (2) `score_runs` on the stacked four-scenario frame
  (`02_running_matilda_case_study.R`) then gives them RMSE and weight NA. (3) `Fig_06B_plot_probability_bar.R` calls
  `na.omit(matilda_weighted)`, which drops both sets in every scenario, and Matilda v1.0.0's `metric_calc` renumbers the
  remaining runs `run_number <- 1:nrow(df)` (`R/metric_calc.R`); the following `merge(df, use_weights, by =
  "run_number")` therefore gives each run from 115 on the weight of a neighbouring, independent run, which scrambles
  the weighting. (4) Emulating this on our 999 runs (raw-GMST metric, the same two positions) lowers P(1.7–3.8) from
  96.5 % to 93.2 %, and the 0.5 °C bins then match the published Fig. 6B for SSP2-4.5 (largest difference 0.014,
  against 0.101 with correct weights; part 5, and the notebook). The unweighted bins match nearly as well (0.017), as
  expected for scrambled weights. With the paper's own draws the same misalignment gives 92.7 % and bins 0.018, 0.185,
  0.389, 0.275, 0.100, 0.029, 0.003, within the reading error of the figure (part 4(c); weights there are scored on the
  SSP2-4.5 series alone, not on the four stacked scenarios). A second candidate, also close to
  unweighted, is the `score_bayesian` of Matilda before 2024-02-07 (likelihood exp(−½·RMSE²) in these units; 93.8 % on
  this data set, effective sample size 974 against 608; part 5); but the authors'
  commit 8a4772b (2024-02-14) regenerated Fig. 6 for the new `score_bayesian`, and the published figure is that version,
  so the misalignment is the better-supported cause. We did not run the authors' unmodified four-scenario scripts, so this remains a reconstruction.
- **Hector version.** The paper cites "Hector V3.1.1" (ref. 22) and Table 1's means are its defaults; the paper does not
  name the version it ran. Matilda's case-study run time (~100 min, p. 17) is the only cost figure.
- The paper text says GMST was extracted for 1960–2100 (p. 17), but Fig. 6A shows 1950–2100 and Matilda v1.0.0's
  `criterion_gmst_obs()` covers 1950–2023; the script saves all years. This data set scores 1950–2023.

## License and credit

Data files: CC BY 4.0 (repository data licence), generated here with Hector (GPL-3, github.com/JGCRI/hector; called,
not copied or ported) and Matilda (MIT, github.com/JGCRI/matilda). The R scripts and Python modules are new code
(MIT). Matilda's observed-temperature series is used inside the package only and is not shipped. Credit: Brown et
al. (2024) for the study design and priors; the Hector team for the model.

## Unknowns and open questions

- **Likely explained, unconfirmed:** the paper's weighted 92 % (1.7–3.8 °C, SSP2-4.5, p. 18), lower than the 95–97 %
  found here, is probably the result of NA weights for sets 114 and 930 (which fail under SSP3-7.0), `na.omit` and the
  `metric_calc` renumbering, which misalign weights and runs ("The 92 %"). Questions for the authors: did sets 114 and
  930 fail under SSP3-7.0 in your run, and so are Fig. 6B and the quoted 92 / 94 / 97 / 92 % effectively unweighted?
  Which `score_bayesian` version (before or after 2024-02-07) produced the text percentages?
- The run-number hazard in Matilda v1.0.0's `metric_calc` (`run_number <- 1:nrow(df)` ignores the input run numbers),
  which misaligns any later merge by run number once runs have been removed.
- Which Hector version the paper ran (3.1.1 is cited and matches Table 1's means; this data set uses it).
- Whether "relative to 1850–1900" was meant per run (as here) or as the ensemble-wide normalisation of the scripts,
  and whether the raw-GMST metric in the scripts was intended.
- The years scored: the text says GMST was extracted for 1960–2100 (p. 17); the package criterion is 1950–2023.
- The source of Hector's small-scale jumps: a sawtooth with jumps of up to about 1e-2 °C and 5–8 ppmv per 0.025 step
  in z, along NPP₀, Q₁₀ and ECS and at some samples along every input (Gradients). The gradients are smoothed slopes
  with a step of 0.25 prior sd, and λ₁/λ₂ of warming depends on the step (≈ 28–47).
- Q₁₀ < 1 in 51 samples (5 %) and Hector's failure at low NPP₀ are properties of the unbounded priors. The failure
  limit depends on the scenario: under SSP2-4.5 runs fail at NPP₀ = 2.28 and near 8 Pg C yr⁻¹ while 11.7 runs, but
  under SSP3-7.0 the paper's sets at NPP₀ ≈ 14 Pg C yr⁻¹ fail (part 4(a));
  the paper does not discuss them.
