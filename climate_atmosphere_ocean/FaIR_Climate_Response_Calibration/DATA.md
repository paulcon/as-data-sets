# FaIR_Climate_Response_Calibration: data

Kind: **stored data (generated with FaIR 2.1.3), with a regeneration script.** The model is FaIR, an outside
code (github.com/OMS-NetZero/FAIR, Apache-2.0, `pip install fair==2.1.3`), so this is not a numpy/scipy
"Python model" in the repository's sense, even though `fair_driver.py`, which sets FaIR up, is Python. The
parameter sets are the published fair-calibrate v1.4.1 posterior, converted losslessly by `convert_data.py`.
The model outputs and gradients were computed for this repository with FaIR 2.1.3 in a Linux container
(x86-64, Python 3.11.18) on 2026-11-03 and 2026-11-08, by `generate_data.py`. The shipped CSVs are the record.
**Regenerating the data, or rerunning the accuracy checks, needs FaIR 2.1.3**: `generate_data.py` and
`check_accuracy.py` stop with an error that says what to install when it is missing. The notebook needs only
numpy, matplotlib and active_subspaces: it reads the shipped files and the numpy-only part of `fair_driver.py`.

## Files

| file | contents | role |
|---|---|---|
| `FaIR_Climate_Response_Calibration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2024; outputs committed |
| `fair_driver.py` | numpy-only part: parameter names, units, groups, the coordinates u and v and their gradient maps (`to_analysis`, `to_erfaci_coordinates`, `from_erfaci_coordinates`, `erfaci_unscaled`), file loaders; `FairEnsemble` (runs FaIR 2.1.3) and `fd_gradients` | model set-up (a FaIR driver, not a numpy/scipy model); its constants import without FaIR |
| `LICENSE-fair-calibrate.txt` | fair-calibrate's MIT licence | covers the set-up in `fair_driver.py` and the harmonisation settings in `generate_data.py`, both taken from fair-calibrate v1.4.1 |
| `convert_data.py` | writes the three converted input files from the Zenodo record and fair-calibrate (`--check`) | lossless conversion, numpy only |
| `generate_data.py` | stage `runs`: `qoi.csv`, `qoi_internal_variability.csv`, `gradients_*.csv`; stage `emissions`: `ssp245_emissions_1750-2100.csv` (`--out`, `--rows`, `--check`) | regenerates the data (needs FaIR 2.1.3; the emissions stage also aneris and scmdata) |
| `check_accuracy.py` | the numbers of "Accuracy" below | accuracy check (needs FaIR 2.1.3) |
| `posterior_parameters.csv` | 841 × 47: index in the 1.6-million prior ensemble, the 45 parameters (physical units, FaIR names), seed | converted from `calibrated_constrained_parameters.csv` (Zenodo) |
| `fixed_calibrations.csv` | 1 × 9: methane lifetime and its 6 sensitivities, land-use and LAPSI factors (row `historical_best`) | converted from `CH4_lifetime.csv`, `landuse_scale_factor.csv`, `lapsi_scale_factor.csv` (Zenodo) |
| `solar_volcanic_forcing_1750-2101.csv` | 352 × 3: time bound (yr), solar and volcanic ERF (W m-2) | converted from fair-calibrate v1.4.1 `data/forcing/` |
| `ssp245_emissions_1750-2100.csv` | 351 × 52: mid-year time (yr), emissions of 51 FaIR species (units in the header) | built by `generate_data.py --stage emissions` |
| `qoi.csv` | 841 × 6: prior index, the four QoIs (K), CO2 in 2022 (ppm); internal variability off | FaIR output (deterministic runs; used for the subspace) |
| `qoi_internal_variability.csv` | 841 × 6: as `qoi.csv`, internal variability on with each member's seed | FaIR output (as in the paper; used for the Table 6 check) |
| `gradients_dT_ssp245_2081-2100.csv`, `gradients_ECS.csv`, `gradients_TCR.csv`, `gradients_GMST_2003-2022.csv` | 841 × 45 each: ∂f/∂u_j (K per unit of u_j) | FaIR output (central differences, deterministic runs) |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. The largest file is 800 kB; the data set is 4.9 MB.

## Model

### Citation

C. Smith, D. P. Cummins, H.-B. Fredriksen, Z. Nicholls, M. Meinshausen, M. Allen, S. Jenkins, N. Leach,
C. Mathison, and A.-I. Partanen, "fair-calibrate v1.4.1: calibration, constraining, and validation of the FaIR
simple climate model for reliable future climate projections", *Geoscientific Model Development* 17(23) (2024)
8569–8592, doi:10.5194/gmd-17-8569-2024 (Crossref-checked). The PDF read is the publisher's version. Data:
C. Smith, fair calibration data (v1.4.1), Zenodo, doi:10.5281/zenodo.10566813 (cited in the paper, p. 8588).
Code: github.com/chrisroadmap/fair-calibrate, tag v1.4.1 (commit 5a31144b58d6b9e2b23a845f68f73c7eda98701c).

### Physics

- FaIR is an emission-driven simple climate model: emissions of about 50 species drive gas cycles, gases and
  aerosols give effective radiative forcings (ERF), and the forcing drives a three-layer energy balance model
  (pp. 8572–8573, Eqs. 1–4), with ECS and TCR computed from its impulse-response coefficients (p. 8574).
- Carbon cycle: four-time-constant impulse response with the 100-year integrated airborne fraction
  I100 = r0 + rU ΔC_U + rT ΔT + rA ΔC_A (Eq. 7, p. 8577). Methane lifetime: Eq. (6), p. 8575, with the Table 1
  calibration, p. 8577. Aerosol–cloud ERF: Eq. (8), p. 8578, over SO2, BC and OC. Aerosol–radiation and
  ozone ERF: linear in the precursors (Tables 3, 4). Other forcings carry the scale factors of Table 5.
- **FaIR 2.1.3 and the printed Eq. 4.** In FaIR's code (`energy_balance_model.py`), the first state variable
  relaxes to the forcing at rate γ and drives the surface layer, so γ sets a lag of 1/γ years even with
  internal variability off. The paper's equations, with η = 0, would make γ irrelevant (p. 8573). The shipped
  data are FaIR as it runs: the warming gradient with respect to γ is nonzero in all 841 members (notebook).
- Set-up (`fair_driver.FairEnsemble`): as fair-calibrate v1.4.1 `05_constrained-ssp-projections.py`
  (methane method Thornhill2021; species list without Halon-1202, NOx aviation and contrails; the v1.4.1 1750
  baseline emissions; volcanic efficacy 0.6; solar trend ramp 0 in 1750 to 1 in 2020), restricted to SSP2-4.5
  and the years 1750–2101 (the script runs to 2300).

### Inputs

m = 45 ("In total, 45 parameters are sampled", p. 8580). The paper lists them in its supplement Table S3,
which was not available here; the list is the column list of the posterior file. The posterior is the 841
members selected in the reweighting step (Sect. 3.3.2, p. 8582) from a 1.6-million-member Monte Carlo prior
(p. 8578), of which 224 342 pass the historical-warming filter (p. 8581). The notebook prints each
parameter's posterior 5 %, 50 % and 95 % values.

| # | parameters (FaIR names) | units | prior in the paper | source |
|---|---|---|---|---|
| 0–10 | clim_gamma, clim_c1, clim_c2, clim_c3, clim_kappa1, clim_kappa2, clim_kappa3, clim_epsilon, clim_sigma_eta, clim_sigma_xi, clim_F_4xCO2 | yr-1; W yr m-2 K-1 (×3); W m-2 K-1 (×3); 1; W m-2 yr-1/2 (×2); W m-2 | 11-D Gaussian KDE of the fits to 49 CMIP6 models; negative values and κ1 < 0.3, C1 < 1.8, C3 < C2, C2 < C1, γ < 0.5 redrawn | Sect. 3.2.1, p. 8580 |
| 11–14 | cc_r0, cc_rU, cc_rT, cc_rA | yr; yr (Gt CO2)-1; yr K-1; yr (Gt CO2)-1 | 4-D KDE of the 11 models of Leach et al. (2021) | Sect. 3.2.4, p. 8580 |
| 15–23 | ari_BC, ari_OC, ari_Sulfur, ari_NOx, ari_VOC, ari_NH3, ari_CH4, ari_N2O, ari_EESC | W m-2 per unit emission (Mt yr-1) or concentration (ppb; ppt for EESC) | Gaussian, e.g. BC 0.0279 ± 0.0239, SO2 −0.00308 ± 0.00229 (90 % ranges), scaled to the AR6 −0.3 ± 0.3 W m-2 | Table 3, p. 8581; Sect. 3.2.3, p. 8580 |
| 24–26 | aci_shape_so2, aci_shape_bc, aci_shape_oc | (Mt SO2 yr-1)-1, (Mt BC yr-1)-1, (Mt OC yr-1)-1 | correlated KDE of log s from the 13 model fits of Table 2 | Sect. 3.2.2, p. 8580; Table 2, p. 8579 |
| 27 | aci_beta | W m-2 | set so that ERFaci 2005–2014 is a draw from a trapezoid (−2.2 to +0.2, plateau −1.6 to −0.4 W m-2) | Sect. 3.2.2, p. 8580 |
| 28–33 | o3_CH4, o3_N2O, o3_EESC, o3_CO, o3_VOC, o3_NOx | W m-2 per ppb, ppt (EESC) or Mt yr-1 | Gaussian, e.g. NOx 1.19e-3 ± 1.17e-3 (90 %) | Table 4, p. 8582; Sect. 3.2.5 |
| 34–43 | fscale_CH4, _N2O, _minorGHG, _Stratospheric water vapour, _Land use, _Volcanic, _solar_amplitude, _solar_trend, _Light absorbing particles on snow and ice, _CO2 | 1 (solar trend W m-2) | Gaussian ±0.20, 0.14, 0.19, 1.00, 0.50, 0.25, 0.50 (5–95 %); solar trend +0.01 (−0.06 to +0.08) W m-2; BC on snow skew normal (0.00, 2.25); CO2 ±0.12 by quantile mapping from F4xCO2 | Table 5, p. 8582; Sect. 3.2.6, p. 8581 |
| 44 | cc_co2_concentration_1750 | ppm | Gaussian 278.3 ± 2.9 (5–95 %) | Sect. 3.2.4, p. 8580 |

Fixed (not sampled): methane lifetime 16.788 yr in 1750 and its sensitivities (Table 1, p. 8577, printed to
three figures; the file has the full values), the land-use and LAPSI factors, the 1750 baseline emissions.
Table 3 prints the EESC unit as W m-2 ppb-1; FaIR's EESC is in ppt (Table 4 uses ppt), so the file and the
notebook use W m-2 ppt-1.

### Quantities of interest

The paper's Table 6 definitions (p. 8585); means over time bounds with half weights at the ends, as in
fair-calibrate (`fair_driver.qois_from_temperature`):

1. `dT_ssp245_2081-2100` (K): GMST 2081–2100 relative to 1995–2014 under SSP2-4.5 harmonised to 2022 (p. 8575).
   The main QoI.
2. `ECS` (K) and 3. `TCR` (K) of the energy balance model (FaIR's `ebms.ecs`, `ebms.tcr`).
4. `GMST_2003-2022` (K): GMST 2003–2022 relative to 1850–1900.

Plus `CO2_2022` (ppm), the mean of the time bounds 2022 and 2023, as a check against Table 6.

**Internal variability.** The paper runs with internal variability on and a fixed seed per parameter set
(p. 8580). A QoI that holds a different noise realization in each member is not a function of the 45
parameters, so the gradients and the subspace use deterministic runs (`stochastic_run = False`), in which
σ_η and σ_ξ have no effect. `qoi_internal_variability.csv` repeats the paper's set-up for the Table 6 check.

### Gradients and coordinates

- Stored gradients: central differences of the deterministic QoIs with respect to the coordinates
  **u** = `fair_driver.to_analysis(x)`: ln s for the three aerosol–cloud shapes, asinh(β / 1 W m-2) for the
  scale, the physical value for the other 41. Step h_j = 10^-3 × (sd of u_j over the 841 members), 90 extra
  runs of the whole ensemble. Units: K per unit of u_j.
- Notebook coordinates **v** (this data set's choice, not the paper's): u with asinh β replaced by
  ERFaci_2005–2014 = β D(s), with D(s) the mean over the emission years 2005–2014 of
  log(1 + Σ s_i A_i) − log(1 + Σ s_i A_i,1750) (Eq. 8) from the shipped emission file. This is the quantity
  whose prior sets β (p. 8580); its posterior 5/50/95 % values, −1.6627, −0.9592, −0.3436 W m-2 (notebook),
  agree with the ERFaci row of Table 6 (−1.66, −0.96, −0.35) to within 0.0064 W m-2 (the 95 % value rounds
  to −0.34, not −0.35). Gradients in v come from the stored ones by the
  chain rule (`fair_driver.to_erfaci_coordinates`); D(s) > 0 in every member.
- Normalization: one Gaussian approximation of the posterior of v (sample mean and covariance), z =
  L^-1 (v − mean), Cholesky L, through `ac.Gaussian`; the expectation in C is the plain average over the 841
  members.

### Sampling design

No new design: the inputs are the 841 posterior members of the paper, a subset selected by likelihood
reweighting (Sect. 3.3.2) from a seeded Monte Carlo prior of 1.6 million (p. 8580), used here with equal
weights. No row is duplicated and no value is outside physical meaning; β > 0 in 4 members (notebook).

### Published results (paper)

- Table 6 (p. 8585): reweighted-posterior 5/50/95 % of the constrained metrics and SSP projections, e.g.
  ECS 2.01, 2.96, 4.99 K; TCR 1.31, 1.79, 2.38 K; GMST 2003–2022 0.86, 1.03, 1.13 K; CO2 2022 416.1, 417.0,
  417.8 ppm; ERFaci 2005–2014 −1.66, −0.96, −0.35 W m-2; SSP2-4.5 2081–2100 1.06, 1.71, 2.66 K.
- Fig. 8 (p. 8584): metric distributions by constraint step; Fig. 9 (p. 8585): SSP2-4.5 projections by step;
  Fig. 10 (p. 8586): prior and posterior of the 45 parameters (multimodal aerosol–cloud shapes, p. 8583);
  Fig. 11: SSP projections.
- No active-subspace or sensitivity results.

### What the notebook finds (from these files)

- Table 6 is reproduced by the runs with internal variability to within 0.005 K for the four temperatures
  (SSP2-4.5: 1.064, 1.706, 2.658 K) and 0.04 ppm for CO2.
- SSP2-4.5 warming, coordinates v: λ1/λ2 = 4.86, λ2/λ3 = 1.44, λ3/λ4 = 1.19, λ4/λ5 = 5.28; subspace distances
  0.045 (n = 1) and 0.042 (n = 4), 0.164 and 0.281 for n = 2, 3. LOO R² of a quadratic: 0.6368 (n = 1),
  0.8515 (n = 2), 0.9166 (n = 4). By these gradient-based criteria n = 4. w1 loads on κ1 (−0.69), ln s_OC
  (+0.50), ERFaci (−0.49), ln s_BC (−0.39), κ3 (−0.34), C3 (−0.32). Gradient shares: climate response 0.493,
  aerosol–cloud 0.403.
- **A single direction does as well.** Under 10-fold cross-validation with every direction re-estimated on
  the training folds (seed 2024), a quadratic in the least-squares (OLS) direction of the warming on the
  whitened inputs has R² 0.917, the same as a quadratic in the four-dimensional active subspace (0.917), and
  far above one in w1 (0.635); a linear function of all 45 inputs gives 0.873. The OLS direction loads on κ1
  (−0.87) and ERFaci (−0.69), with no aerosol–cloud shape parameter among its five largest loadings; its
  cosine with w1 is 0.827, and with the span of w1..w4 0.955. So the warming is close to a one-dimensional
  ridge, which the leading gradient eigenvector does not find. The likely cause is the posterior: 0.415 of
  the members have ln s_BC < −20 (no BC aerosol–cloud effect), the members where the BC term switches on
  (−20 < ln s_BC < 0, 0.401 of all) are 0.667 of the 5 % with the largest gradients, and on this multimodal,
  heavy-tailed sample (kurtosis of z up to 26.98) gradient-based eigenvalues need not rank directions by
  how much f varies across the posterior. The least-squares slope and the mean whitened gradient, equal under
  a Gaussian measure, differ (cosine 0.890 for the warming, 0.638 for GMST).
- ECS: λ1/λ2 = 89.70, LOO R² 0.9712 in y1 (10-fold CV: 0.970 in w1, 0.955 in the OLS direction); TCR: 5.09,
  0.8401. GMST 2003–2022: λ1/λ2 = 2.26 and LOO R² −0.0005 in y1, but 10-fold CV R² 0.635 for a quadratic in
  its OLS direction (cosine 0.105 with w1) and 0.770 in the four-dimensional active subspace: it has
  low-dimensional structure that the leading gradient eigenvectors do not rank first.
- In the coordinates u (β as input), the warming's λ1 is 26.2 instead of 0.52 and its w1 is ln s_BC, but a
  quadratic in y1 has LOO R² −0.0037: at fixed β, a change of s_BC changes the present-day forcing that the
  prior and the constraints pin down. The notebook therefore uses v.

## Regenerating

```bash
# run from this directory; the venv and every --out go to a directory outside it
OUT="$HOME/fair_regen"
# FaIR environment (Linux x86-64, Python 3.11 was used)
python3.11 -m venv "$OUT/fairvenv"
# the environment of the shipped files (pip freeze)
$OUT/fairvenv/bin/pip install aneris-iamc==0.3.1 certifi==2026.10.23 cftime==1.6.3 \
    charset-normalizer==3.5.2 contourpy==1.3.3 cycler==0.12.1 et_xmlfile==2.0.0 fair==2.1.3 \
    fonttools==4.68.0 globalwarmingpotentials==0.13.2 idna==3.20 kiwisolver==1.5.1 \
    matplotlib==3.11.2 numpy==1.26.4 openpyxl==3.1.5 openscm-units==0.5.4 packaging==26.3 \
    pandas==2.1.4 pillow==13.0.0 Pint==0.23 Pint-Pandas==0.5 platformdirs==4.14.3 pooch==1.8.0 \
    pyomo==6.10.1 pyparsing==3.3.3 python-dateutil==4.0.0 pytz==2026.6 PyYAML==6.0.3 \
    requests==2.34.2 scipy==1.11.4 scmdata==0.16.1 six==1.17.0 tqdm==4.70.1 \
    typing_extensions==4.16.0 tzdata==2026.6 urllib3==2.8.0 xarray==2023.12.0 xlrd==2.0.2
# the model outputs and gradients (resumable: a cache per parameter in $OUT/runs/_cache_0-841)
$OUT/fairvenv/bin/python generate_data.py --stage runs --out $OUT/runs
$OUT/fairvenv/bin/python generate_data.py --check                    # stage runs, all 841 members vs the shipped files
$OUT/fairvenv/bin/python generate_data.py --rows 0:20 --check        # a subset vs the shipped files (about 30 s)
# the emissions (needs the two files below)
$OUT/fairvenv/bin/python generate_data.py --stage emissions --fair-example extensions_1750-2500.csv \
    --rcmip rcmip-5-1-0-corrected-nox.csv --out $OUT/emissions --check
# the converted inputs (numpy only)
python convert_data.py --zenodo ZENODO_DIR --fair-calibrate FAIR_CALIBRATE_CLONE --out $OUT/converted --check
$OUT/fairvenv/bin/python check_accuracy.py                                                   # about 1 min
```

`--out` may not be inside this directory; it may be this directory itself only with `--overwrite` (which
replaces the shipped files; the log and the gradient cache then go to a temporary folder that is removed at
the end, so nothing else is written here), and `--check` and `--rows` always need an `--out` outside it. With
`--check`, `--stage` defaults to `runs` and `--out` to a temporary directory that is removed at the end.
`generate_data.py --check` prints, for each file, the maximum |new − shipped| relative to each column's
largest |shipped| value (leaving out the config-id or time column, which must match exactly); the largest
values observed, measured relative to the whole file's largest value, are those of the provenance note below (Linux, Python 3.11, fair 2.1.3: up to 1.9e-17 for the
QoI files and 1.9e-9 for the gradients); FaIR is not installed on the macOS machine, so nothing was measured
there. It exits 1 only on a structural mismatch: a different shape or NaN pattern, or different config ids
(or time points, for the emissions). `convert_data.py --check` uses a tolerance of 1e-15 for the converted files.

Outside inputs (not shipped in their original form):

| file | source | sha256 |
|---|---|---|
| `calibrated_constrained_parameters.csv`, `CH4_lifetime.csv`, `landuse_scale_factor.csv`, `lapsi_scale_factor.csv` | Zenodo 10.5281/zenodo.10566813 (v1.4.1, CC BY 4.0); 755 kB, 199 B, 51 B, 41 B | 76e2b915…1848c, 32e51797…4848, b730032c…3c3c, 7dbc6a52…720d (full values in `convert_data.py`) |
| `data/forcing/solar_erf_timebounds.csv`, `data/forcing/volcanic_ERF_1750-2101_timebounds.csv` | fair-calibrate v1.4.1 (MIT) | 129e6d00…b5b4, a051cea4…2f91 (full values in `convert_data.py`) |
| `data/emissions/rcmip-5-1-0-corrected-nox.csv` (344 kB) | fair-calibrate v1.4.1 (MIT; RCMIP v5.1.0 emissions, Nicholls et al. 2020, with the NOx correction of p. 8572) | 2275ef4ff97229537ff0e1bff730df1d8a89b9d778c5a8698191d1b632025b67 |
| `examples/data/calibrated_constrained_ensemble/extensions_1750-2500.csv` (2.0 MB) | FaIR v2.2.0 (commit 453ac814e1d25f5dfdb112969166db9bc36f5589, Apache-2.0); its history rows to 2022 | fcf994bb97bd08aae04bb4373ee7b8639e007f44055af1aa3d458aefc14b61ae |

The paper's own harmonised emission file was not available here (the 3.8 GB `fair_calibrate.zip` of the
Zenodo record, which holds the intermediate data, was not downloaded). The emissions here were rebuilt from the files above with the harmonisation settings of
fair-calibrate's `calibration/12_harmonize-emissions.py` (aneris, 2022 harmonisation year, its overrides);
the Table 6 check in the notebook confirms the reconstruction (SSP2-4.5 warming within 0.005 K).

Provenance and run time of the shipped files: Linux container (x86-64; 2 CPUs shared with other
jobs), Python 3.11.18, fair 2.1.3, numpy 1.26.4, scipy 1.11.4, pandas 2.1.4, xarray 2023.12.0. The
gradient run was resumed once: the gradients of parameters 0–39 were written on 2026-11-03 (35.6 min wall,
under load), those of 40–44 and the two QoI files on 2026-11-08 (4.2 min wall, 80 s of CPU for 12 ensemble
runs of 841 members). That is about 92 ensemble runs and, at the measured 6.7 s of CPU per ensemble run,
about 10 min of CPU in all. On 2026-11-08, `--rows 0:20 --check` recomputed the first 20 members from
scratch (33 s wall) and reproduced all six run files (largest difference printed as
0.0e+00); `--stage emissions --check` reproduced the emission file to within 1e-10 of its largest value (10 s), and
`convert_data.py --check` the three converted files (0.0e+00). A later
`--rows 0:3 --check` run (24 s wall) gave largest differences 1.5e-18 (`qoi.csv`), 1.9e-17
(`qoi_internal_variability.csv`) and 1.9e-9 (gradients). The committed notebook run took 11.4 s
(the repository's `.venv`: Python 3.11.16, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2,
active_subspaces 1.0.0.dev0; 2026-11-09).

## Accuracy

All numbers in this section are printed by `check_accuracy.py` (FaIR 2.1.3, run 2026-11-08, about 1 min), except
where another script is named. It uses the members with the smallest, median and largest deterministic
SSP2-4.5 warming: rows 733 (0.705 K), 541 (1.711 K) and 509 (3.493 K).

- **QoIs.** Rerunning FaIR at the three rows reproduces `qoi.csv` to rounding level (relative differences of
  order 1e-16 to 1e-14; the exact values depend on the platform and the run).
- **Gradients, step study** (relative 2-norm difference of the whitened gradient L^T g_u between successive
  steps δ = 1e-1, …, 1e-5 posterior sd; shipped δ = 1e-3). SSP2-4.5 warming: δ 1e-3 vs 1e-4 differ by 1.2e-4,
  2.9e-5, 2.3e-5 at the three rows (1e-4 vs 1e-5: 1.1e-3, 2.5e-4, 2.0e-4, where rounding grows); GMST
  2003–2022: 6.3e-5, 1.6e-5, 5.3e-6; ECS: 2.1e-8 to 2.0e-7; TCR: 1.1e-7 to 9.0e-7. So the shipped
  gradients are accurate to about 1e-4 (warming, GMST) and 1e-6 or better (ECS, TCR). The largest steps
  differ much more at the lowest-warming row (1.1 and 1.9 between δ = 1e-1 and 1e-2), so the QoIs are not
  smooth on the scale of 0.1 sd there.
- **Gradients in v** (chain rule from the shipped files, as in the notebook, against central differences
  taken directly in v, whitened, step 1e-3 sd of v_j): warming 3.5e-5, 6.4e-6, 1.3e-6; GMST 8.4e-5, 1.4e-4,
  1.2e-6; ECS and TCR at rounding level (order 1e-14 to 1e-12, depending on the run).
- **ECS closed form** (notebook): FaIR's ECS equals F4xCO2/(2 κ1) to rounding (order 1e-15), and the finite-difference ECS
  gradient matches the closed-form gradient with a median relative error of 7.9e-08 (maximum 5.2e-07).
- **Regeneration** (`generate_data.py --check`, above): rerunning the first 20 members reproduces the
  shipped files.

## How these files relate to the paper

- The parameter sets are the paper's posterior, unchanged. The model, its version (FaIR 2.1.3, p. 8572), its
  set-up and the seeds are the paper's, so the runs with internal variability reproduce the paper's Table 6.
- The paper's runs were not published at this level of detail; the QoI files are a regeneration for
  SSP2-4.5 only, and the deterministic runs and the gradients are new (the paper computes neither).
- The SSP2-4.5 emissions were rebuilt rather than taken from the paper's archive (above).
- The coordinate v (ERFaci in place of β) and the whitening are this data set's choices.

## Unknowns

1. Supplement Table S3 (the paper's list of the 45 parameters with their model roles) was not read; the
   list here is the column list of the posterior file, which has 45 parameter columns.
2. The SSP2-4.5 emissions are a reconstruction from fair-calibrate's inputs, not the paper's file; Table 6 is
   matched within 0.005 K.
3. ari_EESC has zero gradient for every QoI. The cause is in FaIR 2.1.3 itself: within each time step,
   `fair.py` computes the aerosol–radiation forcing (its step 6) before the EESC concentration of that step
   (step 8), so the EESC value it reads is still unset (NaN) and `np.nansum` drops the term. The paper's runs
   use FaIR v2.1.3 (p. 8570), so the parameter presumably had no effect there either; whether it was meant
   to act is a question for the authors.
4. Table 3 prints the EESC unit as ppb; ppt is assumed (FaIR's EESC is in ppt).
5. The printed Eq. (4) and FaIR 2.1.3's code differ in the role of γ without noise (above); the data follow
   the code.
6. The 841 members are used with equal weights, as the paper's posterior file gives them; the paper does
   not state whether the selection was with or without replacement (no row is duplicated).

## License and credit

Code (`*.py`, the notebook): MIT. `fair_driver.py` and `generate_data.py` carry set-up values and
harmonisation settings from fair-calibrate v1.4.1 (MIT, © 2021 Chris Smith; `LICENSE-fair-calibrate.txt`).
FaIR (Apache-2.0) is not shipped. Data: the posterior and fixed calibrations come from the Zenodo record
(CC BY 4.0; credit C. Smith); the outputs and gradients are CC BY 4.0. `solar_volcanic_forcing_1750-2101.csv`
is converted from fair-calibrate (MIT), and `ssp245_emissions_1750-2100.csv` combines FaIR's example file
(Apache-2.0) with RCMIP emissions as shipped in fair-calibrate (MIT; RCMIP, Nicholls et al. 2020); both are
listed in the exceptions of `LICENSE-DATA`. Model and study: C. Smith et al. (2024); FaIR: N. J. Leach et
al., *Geosci. Model Dev.* 14 (2021) 3007–3036, doi:10.5194/gmd-14-3007-2021.
