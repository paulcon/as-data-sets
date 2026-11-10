# CMF_Soil_Moisture_SPOTPY: data

Kind: **stored data**, converted from outside files. The runs were made by Houska et al. (2015) with the
Catchment Modelling Framework (CMF) driven by their calibration package SPOTPY, and deposited as the PLOS ONE
supporting file S4 ([doi:10.1371/journal.pone.0145180.s004](https://doi.org/10.1371/journal.pone.0145180.s004),
CC BY 4.0). They cannot be regenerated here (the forcing and observation data are not public) and are not
altered: `convert_data.py` copies the objective and the four parameters of every Monte Carlo and Latin
hypercube run character for character.

## Files

| file | contents | role |
|---|---|---|
| `CMF_Soil_Moisture_SPOTPY.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 615; outputs committed |
| `cmf_mc_samples.csv` | 10,001 × 7: run index, `like` (= −RMSE, vol. %), alpha (1/cm), n (–), ksat (m/day), porosity (–), `valid` (1/0) | S4 `CMF_MC.csv` (SPOTPY Monte Carlo) |
| `cmf_lhs_samples.csv` | 10,001 × 7, same columns | S4 `CMF_LHS.csv` (SPOTPY Latin hypercube) |
| `convert_data.py` | reads the two CSV members from the S4 zip (inner zips read in memory, nothing extracted or executed), checks size and SHA-256, writes the two CSVs; `--check` compares | the conversion (standard library only) |
| `check_accuracy.py` | the objective check against the deposited simulations, the failed-run check, stored precision | check |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Deposited column → column here: `like` → `like`,
`paralpha` → `alpha`, `parn` → `n`, `parksat` → `ksat`, `parporosity` → `porosity`. Rows whose objective and
simulations are `#NAME?` in the deposit (29 MC, 23 LHS) keep their parameters, get `nan` as `like` and
`valid` = 0; the notebook drops them (rule: `valid` = 0). Not shipped: the 239 simulated soil-moisture
values per run (read from the deposit by `check_accuracy.py`), the `chain` column (1 in every row), and the
S4 member with the MLE runs (an optimizer trajectory, not a design).

## Model

**Citation.** T. Houska, P. Kraft, A. Chamorro-Chavez, L. Breuer, "SPOTting Model Parameters Using a
Ready-Made Python Package", *PLOS ONE* 10(12):e0145180 (2015), doi:10.1371/journal.pone.0145180 (publisher
PDF read; CC BY 4.0; checked on Crossref). Affiliations: Institute for Landscape Ecology and Resources
Management (IFZ), Justus Liebig University Giessen; Breuer also Centre for International Development and
Environmental Research, JLU Giessen.

**Simulator** (pp. 12–13). CMF (Kraft et al. 2011, doi:10.1016/j.envsoft.2010.12.009; GPL-3.0) as a
one-dimensional soil column, Shuttleworth–Wallace evapotranspiration, Richards-equation soil water flux, van
Genuchten–Mualem retention curve (van Genuchten 1980). Forcing: meteorological and groundwater-table data of
the Linden FACE grassland site A1 (Fluvic Gleysol). Simulation 01/06/1998–01/01/2000, the first seven months
burn-in; evaluation against daily topsoil (0–0.1 m) soil moisture in 1999. The paper prints no model
equations; the retention and conductivity curves of CMF are quoted in the notebook from CMF 2.0.2's
documentation (alpha in 1/cm, ksat in m/day). The CMF version of the runs is not stated. The SPOTPY 1.0.13
example `spot_setup_cmf1d.py` (MIT) sets the model up with 5 layers of 0.01 m; whether the paper's runs used
that setup is not stated.

**Inputs** (m = 4; p. 12, Table 1 p. 7, Fig. 9 p. 14).

| input | units | prior used here (MC sample moments; this data set's choice) | SPOTPY 1.0.13 example prior | plausibility bounds (example) | Fig. 9 axis range, read from the page image |
|---|---|---|---|---|---|
| alpha | 1/cm | N(0.2982, 0.0984²) | N(0.3, 0.1²) | [0.0001, 0.6] | 0–0.7 |
| n | – | N(1.2000, 0.0352²) | N(1.2, 0.035²) | [1.05, 1.4] | 1.05–1.35 |
| ksat | m/day | N(1.0001, 0.2965²) | N(1.0, 0.3²) | [0.01, 3] | 0–2.5 |
| porosity | – | N(0.5503, 0.0397²) | N(0.55, 0.04²) | [0.4, 0.7] | 0.35–0.75 |

The paper gives the priors only as bell curves in Fig. 9 ("based on results from [49]", p. 12) and Table 1
calls MC "Normal random sampling" and LHS "Normal sampling along the HyperCube matrix". The means and sds
used here are those of the 9,972 valid MC runs (printed by the notebook with 95 % intervals); they agree with
the example setup's priors. Alpha's sd interval lies just below 0.1, which the cut at the bounds explains:
N(0.3, 0.1²) cut to [0.0001, 0.6] has sd 0.0987, with 27.0 of 10,000 draws expected outside, against 29 failed MC
rows (notebook). Fig. 9 values are the axis tick ranges read from page 14 rendered at 150 dpi, with the prior curves
centred near 0.3, 1.2, 1.0 and 0.55; they are graphical, not stated bounds.

**Quantity of interest.** RMSE (vol. %) between the 239 simulated and observed topsoil soil-moisture
values of 1999 (`like` = −RMSE in the deposit, since SPOTPY maximizes; the example setup defines
`likelihood = -rmse`). Confirmed on the deposit by `check_accuracy.py` (Accuracy).

**Gradients.** None. CMF computes none, and neighbouring runs cannot give finite differences at the stored
2-decimal precision.

**Sampling design.** MC: independent draws from the normal prior (10,001 rows; the paper says 10,000
iterations, p. 13, and the S4 description says "10,000 model iterations", p. 18; the extra row is not explained).
The deposit was not made by SPOTPY 1.0.13's MC sampler with its CMF example setup unchanged: that sampler
(`algorithms/mc.py`) runs the setup's `optguess` set (0.2, 1.22, 2.0, 0.6) first and then repetitions − 1
random sets, 10,000 in all, and neither deposited file contains the `optguess` row (notebook). The SPOTPY
version and setup of the runs are therefore not known. Runs outside the plausibility bounds were not simulated
and score minus infinity (written `#NAME?` in the deposit, apparently by a spreadsheet program; 29 rows).
LHS: although Table 1 says "Normal", the deposited LHS inputs are uniform (kurtosis 1.80–1.81, notebook) on a
box of about prior mean ± 2.9–3.2 sd, as SPOTPY 1.0.13's `lhs` builds it (uniform Latin hypercube between
the minimum and maximum of 1,001 prior draws; only the uniform marginals can be checked, not the
stratification, at the stored 2-decimal precision); 23 rows outside the bounds (alpha < 0.0001). All values are
stored with 2 decimals, so n and porosity take 29 and 30 distinct values in the MC runs.

**Published results.** No active subspace. Best RMSE 3.2096 (ROPE, SCE-UA, MLE), the other algorithms
3.2098–3.2153 (p. 13); Fig. 8 best runs; Fig. 9 priors, best-10 % posteriors and optimal values; optimal
porosity 0.6–0.63; "high equifinality" (p. 13).

## Regenerating

The runs cannot be regenerated from the repository: CMF is installable (`pip install cmf`), but the FACE
forcing and observation files are not public. The CSVs are regenerated from the deposit:

```
python convert_data.py <path of pone.0145180.s004.zip> --out DIR           # writes the two CSVs into DIR
python convert_data.py <path of pone.0145180.s004.zip> --out DIR --check   # also compares with the shipped files
```

`--out` must be outside this directory (subdirectories and links into it are refused, with or without
`--check`); `--check` compares numerically with tolerance 1e-12 (and the `nan` pattern exactly). Run on
2026-11-09 in the Linux container where the shipped files were written, it printed "identical within 1e-12" for
both files (under 1 s). The archive is not in this repository, so the check was not repeated in `.venv`.

**Source.** PLOS ONE supporting file S4, "Simulation results of the CMF model with MC, LHS and MLE",
doi:10.1371/journal.pone.0145180.s004, file `pone.0145180.s004.zip`, 9,153,529 bytes, SHA-256
`0cd46f8f0ba6ba394220991f523900e7cd9432e7c95529874a98da6a981f6a03` (downloaded 2026-11-09). It holds three zips
(`S4_File_..._MC.zip`, `S5_File_..._LHS.zip`, `S6_File_..._MLE.zip`, named S4–S6 inside although the article
numbers them as parts of S4); the archive is not shipped.

In the repository's `.venv` the notebook runs in about 17 s under nbconvert (9.7 s by its own timer in the
committed run, with other jobs running).

## Accuracy

- Conversion: lossless (`convert_data.py --check`, above).
- Objective (`check_accuracy.py <zip>`, which needs the deposited archive; run on 2026-11-09 in the Linux
  container where the shipped files were written, not repeated in `.venv`): one fixed observation vector, fitted to all 19,950 valid runs,
  reproduces the stored RMSE as the RMSE of each run's 239 stored simulations with rms difference 0.0029 (MC
  and LHS) and maximum 0.0067 (MC) and 0.0085 (LHS); the objective's 2-decimal rounding alone gives rms
  0.0029. So `like` is −RMSE against the same observations in both designs.
- Failed runs (`check_accuracy.py`; this and the next check need only the shipped files and print the same
  numbers in `.venv`, in under 1 s): all 29 MC and 23 LHS rows with `valid` = 0 have a stored parameter
  outside or on a plausibility bound of the example setup; no valid row is outside by more than half a
  rounding step.
- Stored precision (`check_accuracy.py`): 2 decimals for every value; a rounding step is 0.102, 0.284,
  0.034 and 0.252 MC sd for alpha, n, ksat and porosity. Local-linear gradients suffer from it (notebook), so
  the notebook uses QPHD.
- No gradients, so no gradient check.

## How these files relate to the paper

They are the paper's MC and LHS runs of the CMF case study (Figs 8–9), with their objective values. The
paper calibrates; the notebook estimates the active subspace of the RMSE under the prior (MC runs, QPHD):
$\lambda_1/\lambda_2$ = 108.0, a leading direction of mainly n and porosity, which explains the paper's
equifinality and its well-constrained n and porosity against a poorly constrained ksat. The gap belongs to a
quadratic with LOO R² 0.8338, which misses the steep rise of the RMSE as alpha → 0: the 36 MC runs with
alpha < 0.05 1/cm (0.0036 of the runs) hold 0.0635 of the sum of squares, and a second, alpha-only direction
describes them (LOO R² 0.8913 in one active variable, 0.9609 in two). So the RMSE is one-dimensional for the
bulk of the prior and two-dimensional (alpha) if that corner matters; on log RMSE, QPHD gives
$\lambda_1/\lambda_2$ = 167.7 with the same leading direction. The LHS runs reach the corner much more often
(787 runs) and are two-dimensional on all rows (ratios 2.6, 84.1). Leaving out small alpha gives a ratio that
depends on the threshold: 15.5 (alpha ≥ 0.03), 202.2 (≥ 0.05, where the later ratios are 2.7 and 46.7) and 107.7
(≥ 0.10). These LHS ratios fit the quadratic to the uniform box but average $\mathbf{C}$ under the Gaussian
prior; averaging over the LHS points instead gives 3.8, 102.4 (all rows) and 273.2 (alpha ≥ 0.05), with the
same conclusion (notebook). The paper's text (p. 13)
gives optimal ksat "from 0.8 (MLE) to 1.9 (MC)", while Fig. 9 draws MC's optimum near 0.8 and MLE's near
1.95; a stored MC run tied at the best stored RMSE has ksat 0.79 and matches the MC lines of Fig. 9, so the
text appears to swap the two algorithms. The paper's Table 1 calls the LHS sampling normal; the stored LHS
inputs are uniform.

## License and credit

Data: © Houska et al. 2015, PLOS ONE supporting information S4, CC BY 4.0 (the article's licence, Crossref).
The forcing and observations behind the runs are the FACE project data provided to the authors by C. Müller
and L. Grünhage (Acknowledgments, p. 18). Cite the paper when using these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this
repository). No CMF (GPL-3.0) or SPOTPY code is included; the SPOTPY 1.0.13 example setup (MIT) is cited for
its prior and bound values only.

## Unknowns

- The exact priors of the runs: not printed in the paper; the MC sample moments are used, and they agree with
  the SPOTPY 1.0.13 example setup, which may not be the paper's exact setup.
- SPOTPY and CMF versions of the runs; why each file has 10,001 rows rather than 10,000, and why neither file
  holds the `optguess` run that SPOTPY 1.0.13's MC sampler makes first (so the deposit is not the output of
  that sampler with the example setup unchanged).
- The column depth of the model (the example uses 5 × 0.01 m; the observations are 0–0.1 m).
- The observation vector and forcing data (not deposited), so no run can be repeated.
- Stored values are rounded to 2 decimals; whether that rounding happened only on export (the runs used
  unrounded parameters) is not stated, but the objective check above is consistent with it.

## Open questions

- For the authors: the SPOTPY version and setup file of the CMF runs, given 10,001 rows per file and no
  `optguess` run; whether the priors were exactly N(0.3, 0.1²), N(1.2, 0.035²), N(1, 0.3²), N(0.55, 0.04²); why
  the LHS sampling is uniform while Table 1 says "Normal sampling along the HyperCube matrix" (was a normal LHS
  intended?); whether the text's "0.8 (MLE) to 1.9 (MC)" is swapped relative to Fig. 9; the model's layer
  depth (5 × 0.01 m in the SPOTPY example) against the 0–0.1 m observations.
