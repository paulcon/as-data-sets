# Raven_Salmon_River_Blended_Model: data

Kind: **stored data**. The outputs come from an outside code, the Raven hydrological modelling framework (C++), version string "3.0" (commit `cb745e1`), built from source here. The files are the only record of the runs; `generate_data.py` and `raven_salmon.py` document how they were made and can rerun them with a Raven executable, the xSSA forcing and streamflow files and the xSSA deck templates, none of which is in this repository (the templates are downloaded at run time from a pinned commit; see "Provenance of the input deck").

## Files

| file | contents | role |
|---|---|---|
| `Raven_Salmon_River_Blended_Model.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2020`; outputs committed |
| `inputs.csv` | 4096 × 43: x01 … x35 (the paper's x1 … x35, Table C2 units; x04 and x11 are log10 of rates in 1/d) and r1 … r8 (dimensionless, [0, 1]) | stored inputs, physical units |
| `weights.csv` | 4096 × 13: the process-option weights computed from r1 … r8 by the paper's Eq. (A1) (infiltration HMETS, VIC_ARNO, HBV; quickflow LINEAR_ANALYTIC, VIC, TOPMODEL; soil evaporation ALL, TOPMODEL; baseflow LINEAR_ANALYTIC, POWER_LAW; snow balance HMETS, SIMPLE_MELT, HBV) | derived from `inputs.csv`, for reference; not independent inputs |
| `outputs.csv` | 4096 × 15: mean_Q (m³/s), NSE (–), KGE (–), Q_Jan … Q_Dec (m³/s) | Raven output, reduced to scalars |
| `fd_gradients_meanQ.csv` | 3 × 44: design row (0, 1, 2), then the finite-difference gradient of mean_Q with respect to the 43 normalized inputs (m³/s per normalized unit; step 1e-3 of each half-range) | check of the QPHD estimate (notebook) |
| `raven_salmon.py` | `fetch_templates(folder)`, `load_templates(folder)`, `placeholder_values(x)`, `write_deck(x, folder, templates)`, `run(x, raven_exe, obs_dir, templates)`, `weights(x)`, `stick_breaking_weights(r)`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `DESCRIPTIONS`, `PROCESS_GROUPS`, `WEIGHT_NAMES`, `BASIN`, `XSSA_COMMIT`, `XSSA_TEMPLATE_URL`, `XSSA_TEMPLATE_SHA256` | downloads and verifies the xSSA deck templates, fills their placeholders, runs Raven and reads its output (numpy and the standard library; contains no template text) |
| `generate_data.py` | draws the design, runs Raven block by block (resumable cache), `--assemble` writes the CSVs, `--check` reruns a subset, `--fd` writes `fd_gradients_meanQ.csv` | provenance of the files |

The CSVs are comma-separated numbers after two `#` lines (a description and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs and weights are written with 17 significant digits (exact doubles); outputs and gradients with 10.

## Model

**Citation.** J. Mai, J. R. Craig, and B. A. Tolson, "Simultaneously determining global sensitivities of model parameters and model structure", *Hydrology and Earth System Sciences* 24(12):5835–5858, 2020, doi:10.5194/hess-24-5835-2020 (CC BY 4.0; publisher PDF read; a corrigendum was not checked). The paper introduces the extended Sobol' sensitivity analysis (xSSA) for groups of parameters, process options and processes, tests it on two analytic benchmarks (not used here), and applies it to a Raven model of the Salmon River.

**The Raven model** (Sect. 2.1.2–2.1.3; Appendix C, Tables C1 and C2, pp. 5854–5855). Raven (Craig et al., Environ. Model. Softw. 129:104728, 2020) in lumped mode (one HRU, 4230 km²) for the Salmon River near Prince George, BC (Water Survey of Canada gauge 08KC001), daily step, ORDERED_SERIES solver, run 1989-01-01 to 2010-12-31 with the first two years discarded as warm-up, so 1991–2010 (7305 days) is evaluated. Five processes are blended: Raven computes every option's flux and uses their weighted sum. Infiltration (M1 INF_HMETS, M2 INF_VIC_ARNO, M3 INF_HBV), quickflow from the topsoil (N1 BASE_LINEAR_ANALYTIC, N2 BASE_VIC, N3 BASE_TOPMODEL), soil evaporation (O1 SOILEVAP_ALL, O2 SOILEVAP_TOPMODEL), baseflow from the phreatic layer (P1 BASE_LINEAR_ANALYTIC, P2 BASE_POWER_LAW), and snow balance (Q1 SNOBAL_HMETS, Q2 SNOBAL_SIMPLE_MELT, Q3 SNOBAL_HBV). The other processes have one option: gamma convolution of surface (R1) and delayed (S1) runoff, POTMELT_HMETS potential melt (T1), linear percolation (U1), RAINSNOW_HBV partitioning (V1), rain and snow correction factors (W1), and Oudin PET, default shortwave, ROUTE_DUMP and no channel routing (X1). With r giving weights of 0 or 1 the deck reduces to one of 3·3·2·2·3 = 108 models; the first option of each process with x35 = 0 is HMETS (Table C1 caption).

**Inputs** (m = 43): every input is independent and uniform on its range (Table C2 caption; Sect. 2.1.2).

| input | Raven parameter (process option) | units | range | source |
|---|---|---|---|---|
| x01 | HMETS_RUNOFF_COEFF (M1) | – | [0, 1] | Table C2, p. 5855 |
| x02 | B_EXP TOPSOIL (M2) | – | [0.1, 3] | Table C2 |
| x03 | HBV_BETA TOPSOIL (M3) | – | [0.5, 3] | Table C2 |
| x04 | BASEFLOW_COEFF TOPSOIL = 10^x04 (N1) | log10(1/d) | [−5, −1] | Table C2 |
| x05 | MAX_BASEFLOW_RATE TOPSOIL (N2, N3) | mm/d | [0, 100] | Table C2 |
| x06 | BASEFLOW_N TOPSOIL (N2, N3) | – | [0.5, 2] | Table C2 |
| x07 | TOPMODEL_LAMBDA (N3) | m | [5, 10] | Table C2 |
| x08 | PET_CORRECTION TOPSOIL (O1, O2) | – | [0, 3] | Table C2 |
| x09 | SAT_WILT TOPSOIL (O2) | frac | [0, 0.05] | Table C2 |
| x10 | FIELD_CAPACITY TOPSOIL = x09 + x10 (O2) | frac | [0, 0.45] | Table C2 |
| x11 | BASEFLOW_COEFF PHREATIC = 10^x11 (P1, P2) | log10(1/d) | [−5, −2] | Table C2 |
| x12 | BASEFLOW_N PHREATIC (P2) | – | [0.5, 2] | Table C2 |
| x13 | SNOW_SWI_MIN (Q1) | frac | [0, 0.1] | Table C2 |
| x14 | SNOW_SWI_MAX = x13 + x14 (Q1) | frac | [0.01, 0.3] | Table C2 |
| x15 | SWI_REDUCT_COEFF (Q1) | 1/mm | [0.005, 0.1] | Table C2 |
| x16 | DD_REFREEZE_TEMP (Q1) | °C | [−5, 2] | Table C2 |
| x17 | REFREEZE_EXP (Q1) | – | [0, 1] | Table C2 |
| x18 | REFREEZE_FACTOR (Q1, Q3) | mm/(d °C) | [0, 5] | Table C2 |
| x19 | SNOW_SWI (Q3) | frac | [0, 0.4] | Table C2 |
| x20, x21 | GAMMA_SHAPE, GAMMA_SCALE (R1) | – | [0.3, 20], [0.01, 5] | Table C2 |
| x22, x23 | GAMMA_SHAPE2, GAMMA_SCALE2 (S1) | – | [0.5, 13], [0.15, 1.5] | Table C2 |
| x24 | MIN_MELT_FACTOR (T1) | mm/(d °C) | [1.5, 3] | Table C2 |
| x25 | MAX_MELT_FACTOR = x24 + x25 (T1) | mm/(d °C) | [0, 5] | Table C2 |
| x26 | DD_MELT_TEMP (T1) | °C | [−1, 1] | Table C2 |
| x27 | DD_AGGRADATION (T1) | 1/mm | [0.01, 0.2] | Table C2 |
| x28 | PERC_COEFF TOPSOIL (U1) | 1/d | [0.00001, 0.02] | Table C2 |
| x29 | thickness TOPSOIL | m | [0, 0.5] | Table C2 |
| x30 | thickness PHREATIC | m | [0, 2] | Table C2 |
| x31, x32 | RAINSNOW_TEMP, RAINSNOW_DELTA (V1) | °C | [−3, 3], [0.5, 4] | Table C2 |
| x33, x34 | RAINCORRECTION, SNOWCORRECTION (W1) | – | [0.8, 1.2] each | Table C2 |
| x35 | PERC_COEFF PHREATIC (U1) | 1/d | [0, 0.02] | Table C2 |
| r1, r2 | weights of M1, M2, M3 | – | [0, 1] | Sect. 2.1.2, Appendix A |
| r3, r4 | weights of N1, N2, N3 | – | [0, 1] | Sect. 2.1.2 |
| r5 | weights of O1, O2 | – | [0, 1] | Sect. 2.1.2 |
| r6 | weights of P1, P2 | – | [0, 1] | Sect. 2.1.2 |
| r7, r8 | weights of Q1, Q2, Q3 | – | [0, 1] | Sect. 2.1.2 |

**Weights** (Appendix A, Eqs. A1–A2, after Moeini et al. 2011). For a process with N options and numbers r_1 … r_{N−1}: w_j = (1 − Σ_{i<j} w_i) · S_{N−j}(r_j) with S_k(r) = 1 − (1 − r)^{1/k}, and w_N = 1 − Σ_{i<N} w_i. For three options w_1 = 1 − √(1 − r_1), w_2 = (1 − w_1) r_2, w_3 = 1 − w_1 − w_2; for two, w_1 = r, w_2 = 1 − r. This puts the weights uniformly on the simplex, each with CDF 1 − (1 − w)^{N−1} (Eq. A3). The paper writes the formula for N + 1 weights from N numbers; the xSSA code (`lib/PieShareDistribution.py`) and Raven's own `CALCULATE_WTS` option use the form above, which is the same. The assignment of r1 … r8 to processes (2 + 2 + 1 + 1 + 2, in the order infiltration, quickflow, evaporation, baseflow, snow balance) is from Sect. 2.1.2 and the order of the xSSA code. `inputs.csv` stores the uniform numbers r; `weights.csv` stores the 13 weights for reference. The notebook analyses the r (the paper's inputs, independent and uniform); the weights are a fixed, smooth, monotone transform inside the model.

**Derived Raven parameters** (Table C2 caption): FIELD_CAPACITY = x09 + x10, SNOW_SWI_MAX = x13 + x14, MAX_MELT_FACTOR = x24 + x25, the two BASEFLOW_COEFF = 10^x04 and 10^x11. The initial soil stores are half of each layer's capacity (x29 · 500 mm and x30 · 500 mm), as in xSSA's `.rvc` template (not stated in the paper).

**Forcing and basin data** (xSSA repository, `examples/data_in/`): basin-mean daily minimum and maximum temperature and precipitation (`data_obs/08KC001/model_basin_mean_forcing_Santa-Clara.rvt`, from 1950; the paper says the forcing is from Natural Resources Canada on a ~10 km grid, aggregated over the basin, available 1954–2010, Sect. 2.1.3), observed daily streamflow at 08KC001 (`model_basin_streamflow.rvt`, m³/s, 1954–2010, missing days flagged −1.2345), and basin constants (`basin_metadata/basin_physical_characteristics.txt`: centroid 54.4565° N, 123.3515° W, area 4230 km², mean elevation 842.93 m, slope 3.358°, forest fraction 0.4764). Note that the forest fraction in this file (0.476) differs from the "83 % evergreen needleleaf" of the paper's text (Sect. 2.1.3); we use the file, as the xSSA scripts do. Likewise the paper describes the temperature forcing as "average daily temperature" (p. 5841), but the file the xSSA template points to holds TEMP_DAILY_MIN and TEMP_DAILY_MAX (header checked); the same xSSA folder also holds `Salmon-River-Near-Prince-George_meteo_daily.rvt`, which no template uses. We use the file the template names.

**Quantities of interest.** The paper's output is the simulated daily streamflow Q(t), 1991–2010; its Sobol' indices are computed per day and aggregated with variance weights (Eqs. 23–24). That is a vector output, so we store scalar reductions of it:

- `mean_Q`: the mean simulated daily flow over 1991–2010 (m³/s), the notebook's main QoI. It is a time-aggregate of the paper's own output, needs no observations, and measures the long-term water balance.
- `NSE`, `KGE`: Nash–Sutcliffe and Kling–Gupta (2009) efficiencies of the simulated against the observed flow on the days of 1991–2010 with observations (5863 days, as Raven counts them), the usual calibration targets. Raven's `Hydrographs.csv` is period-ending: the row dated d holds the flow of the day that ends at 00:00 on d, i.e. the observation of calendar day d − 1 (checked on every row of a saved run: 0 mismatches with a shift of one day). "1991-01-01 … 2010-12-31" by row date therefore covers the daily means of 1990-12-31 … 2010-12-30, which is why Raven counts 5863 observed days and a mean of 29.264 m³/s, against 5862 days and 29.268 m³/s for calendar 1991–2010 in the file; the monthly means are binned by row date and so shifted by one day. This matches Raven's own Diagnostics.csv; the xSSA scripts read the same three metrics from Raven's `Diagnostics.csv`. Our NSE and KGE are computed from `Hydrographs.csv`; `generate_data.py` checks every NSE against Raven's to 1e-5.
- `Q_Jan` … `Q_Dec`: the mean simulated flow of each calendar month over 1991–2010 (m³/s), the analogue of the paper's day-of-year analysis (Fig. 6).

**Gradients.** None. The processes switch on and off (snow present or not, stores at capacity, melt above a threshold temperature, overflow), so the outputs are piecewise smooth, and Raven writes the hydrograph with 6 significant digits, which limits finite-difference steps. The notebook uses QPHD, a gradient-free estimator from a global least-squares quadratic. `fd_gradients_meanQ.csv` holds finite-difference gradients of mean_Q at three design rows, which the notebook uses to check the QPHD estimate (see Accuracy).

**Sampling design.** M = 4096 = 2¹² points of a scrambled Sobol' sequence in 43 dimensions (`scipy.stats.qmc.Sobol(d=43, scramble=True, seed=20201202)`, scipy 1.18.1), mapped linearly to the Table C2 box (the 35 parameters) and [0, 1]⁸ (r1 … r8). QPHD needs at least (m + 1)(m + 2)/2 = 990 points and the brief's rule asks for twice that, 1980; 4096 gives a hold-out test of the quadratic (fit on the first 3072 rows, test on the last 1024) and room for the bootstrap. This is not the paper's design: the paper uses Sobol' sampling for the xSSA estimator with K = 1000 reference sets, (43 + 2) × 1000 = 45 000 runs for the parameter sensitivities (Sect. 2.2, p. 5845; Sect. 4), with no seed stated and its samples not published as files. The design can nevertheless be regenerated: xSSA's `examples/scripts/sa_model_multiple_processes.py` (line 497) draws `sobol.i4_sobol_generate(2·(35 + 8), K, 40000)`, an unscrambled, deterministic Sobol' sequence with skip 40 000, so the paper's points follow from K and the code (we did not regenerate them).

**Samples outside physical meaning.** None: every input lies in its Table C2 range, the weights are in [0, 1] and sum to 1 per process to rounding (checked in the notebook). Two edges are degenerate but valid: 82 samples have a topsoil thinner than 1 cm (x29 < 0.01 m) and 20 a phreatic layer thinner than 1 cm (x30 < 0.01 m). They are kept. Most of the box is far from the observations: 91.6 % of the samples overestimate the observed mean flow (29.26 m³/s on the 5863 days with data), the median NSE is −4.15, and 13 samples have NSE > 0.5 (best 0.708). These are valid model runs, not errors. No run failed and no run wrote a Raven WARNING (every run writes the same three ADVISORY lines on reach length and interception).

**Published results.** The paper does no active-subspace analysis. Its results for the Raven model are Sobol' main and total indices of the 43 inputs, of the 19 process options and of the 11 processes, variance-weighted over the 7305 days (Fig. 5), and normalized process sensitivities per day of the year (Fig. 6). Printed numbers: x24–x27 (potential melt) are the most sensitive parameters, then x5 and x6 (quickflow); x8, x16, x18, x19, x29, x31 and x34 are of medium sensitivity; r3, r4 and r7 are the most influential weight numbers; the 13 parameters x5, x6, x8, x16, x18, x19, x24–x27, x29, x31 and x34 carry 96.5 % of the variability (only parameter indices; 77.2 % with the r's); potential melt T, rain–snow partitioning V and precipitation correction W carry 42.1 %, quickflow N and snow balance Q 38.5 %, infiltration M, evaporation O and percolation U 19.2 %, baseflow P and the convolutions R and S 0.2 % (Sect. 3.3.1–3.3.3, pp. 5847–5849); evaporation carries about 35.4 % of the daily sensitivity in June–October (Sect. 3.3.4).

## Regenerating

```bash
# Raven: build commit cb745e1 of github.com/CSHS-CWRA/RavenHydroFramework (see below)
# xSSA forcing/streamflow: git clone https://github.com/julemai/xSSA
# xSSA deck templates: downloaded by generate_data.py into <cache>/templates on first use (needs network once)
export RAVEN_EXE=/path/to/Raven.exe
export XSSA_OBS_DIR=/path/to/xSSA/examples/data_in/data_obs/08KC001
nice -n 10 python generate_data.py --cache ./_cache --rows 0:1536      # resumable; ~128 rows per cache file
nice -n 10 python generate_data.py --cache ./_cache --rows 1536:2816
nice -n 10 python generate_data.py --cache ./_cache --rows 2816:4096
python generate_data.py --cache ./_cache --assemble                    # writes the three CSVs
python generate_data.py --cache ./_cache --rows 0:8 --check            # rerun 8 rows, compare
nice -n 10 python generate_data.py --cache ./_cache --fd               # fd_gradients_meanQ.csv (258 runs)
```

**Raven build** (2026-11-01, Ubuntu 24.04 container, g++ 13.3.0): `git clone https://github.com/CSHS-CWRA/RavenHydroFramework`, `git checkout cb745e1`, then in `src/`: `touch .depend && make CXXFLAGS="-Wno-deprecated -O2"` (no NetCDF). Commit `cb745e1` (2020-06-17) is the first in which `RavenMain.cpp` sets `Options.version="3.0"`; the string becomes "3.0.1" at `133625e` (2020-10-19). The paper says only "released version 3.0" (pp. 5840, 5853), so the exact revision the authors ran is not known (the xSSA repository holds a 68-byte placeholder in place of `Raven.exe`). Evidence that it may have been a 2.9.x build: the paper was received on 8 May 2020 (p. 5835), before `cb745e1`; on GitHub the version string is "2.9.2" up to `4e1216d` (2020-04-25), "2.9.9" at `ae46a89` (2020-05-05) and first "3.0" at `cb745e1` (2020-06-17); and every xSSA template is headed `:FileType … Raven rev252 (v2.9)`, created June 2019. The authors may have rerun the results during revision (revised 15 October 2020), and "released version 3.0" may refer to the raven.uwaterloo.ca release. The independent review diffed the process sources (infiltration, baseflow, soil evaporation, snow balance, potential melt, percolation, convolution, precipitation partitioning, process groups, solvers, common functions, model, forcing update and others) from `4e1216d` to `cb745e1` to `133625e` (3.0.1): the changes touch only options this deck does not use (BASE_THRESH_STOR, the SOILEVAP_HBV/HYPR snow correction, INF_XINANXIANG, routing diversions and irrigation, gridded forcing, POTMELT_NONE, PRECIP_ICEPT_NONE, freeze/thaw enthalpy helpers). The results are therefore probably the same for this deck, but this has not been verified by running a 2.9.x build.

**The run that wrote the shipped files**: 2026-11-01, Python 3.13.18, numpy 2.5.4, scipy 1.18.1, Raven built as above, one process at `nice -n 10` on a 2-CPU container shared with other jobs, in three invocations (rows 0:1536, 1536:2816, 2816:4096) of 32 cache blocks of 128 runs. The blocks took 1423 s in total (24 min; 25–107 s per block depending on the load), 0.35 s per run on average; a single run measured alone takes 0.55–0.6 s wall, of which about 0.25 s is Raven's CPU time. `--rows 0:8 --check` took 2.7 s. After the deck was switched to the fetched xSSA templates (2026-11-01, see "Provenance of the input deck"), `--rows 0:3 --check` and single-row checks at rows 1443 and 3710 reproduced `outputs.csv` to a relative 3.9e-10, 1.6e-10 and 4.1e-10 (the 10 digits written), and `--fd` (258 runs, 62 s) reproduced the earlier finite-difference gradients to 3e-9 absolute. Inputs carry no randomness beyond the fixed design seed, and Raven is deterministic, so a rerun reproduces the files (see Accuracy).

## Accuracy

- **Reproducibility.** `generate_data.py --rows 0:8 --check` redraws the design (bit for bit equal to `inputs.csv`), reruns Raven at rows 0–7 and agrees with `outputs.csv` to a relative 3.9e-10 over all 15 outputs (the 10 significant digits written). Raven is deterministic; the EMA point run (three repeats) agrees exactly.
- **Own metrics vs Raven's.** For every one of the 4096 runs, `generate_data.py` checked our NSE (from `Hydrographs.csv`) against Raven's `Diagnostics.csv` value (printed with 6 digits) to 1e-5; none failed.
- **Output precision.** Raven writes daily flows with 6 significant digits, so each daily value carries a relative rounding error up to 5e-6; the 20-year mean and the efficiencies average many such values and are accurate to about 1e-6 relative, far below the variation across the design.
- **Smoothness (finite differences; the step-1e-3 gradients of mean_Q are shipped as `fd_gradients_meanQ.csv`, written by `generate_data.py --fd`; the step-1e-2 comparison was a scratch run).** At design rows 0, 1 and 2 we computed central-difference gradients of mean_Q and NSE in normalized coordinates with steps of 1e-2 and 1e-3 of each half-range (86 runs per gradient and step; one-sided at the box edge). For mean_Q the two gradients differ by a relative 0.55 %, 0.15 % and 1.9 % (cosine of the angle ≥ 0.9998); for the NSE by 4.6 %, 0.64 % and 25 % (cosine 0.999, 0.99998 and 0.970). The mean flow is close to smooth at these points; the NSE less so. Gradients are not shipped: at 86 runs (about 30 s) each, 300 or more of them would cost several hours, so the notebook uses QPHD.
- **QPHD fit.** The quadratic behind QPHD explains 86.3 % (mean_Q) and 84.6 % (NSE) of the variance of 1024 held-out samples, and 77–91 % for the monthly flows (printed in the notebook). The leading eigenvectors are well determined (bootstrap mean subspace distance 0.042 and 0.049 for k = 1); the small eigenvalues are partly fitted noise (the bootstrap band lies above them for the NSE).
- **Leading direction robust, eigenvalue split not (mean_Q; printed in the notebook).** w1 agrees with the OLS direction (|cos| = 0.998) and with the leading eigenvector from local-linear gradients (0.997 for 73 and 86 neighbours). The split does not: local-linear gradients give λ1/λ2 = 46.2–54.7 against QPHD's 6.75, while the finite-difference gradients at rows 0, 1, 2 put only 64.4 %, 42.7 % and 23.5 % of ‖∇f‖² on w1 (cosine with the quadratic's gradient 0.90, 0.83, 0.72). Polynomials in the first k QPHD active variables reach held-out R² 0.788 (k = 1), 0.823, 0.858, 0.897 and 0.918 (k = 2, 3, 5, 10): about a fifth of the variance lies in a slowly decaying tail along w2 … w10. λ1/λ2 = 6.75 and λ1/tr = 67.1 % are properties of the fitted quadratic.

## How these files relate to the paper

- **Same model, same inputs, different runs.** The deck is the authors' own xSSA template, filled for our design (it reproduces Tables C1, C2), on the xSSA forcing, with a Raven build of the "3.0" source (possibly not the revision the paper used; see "Raven build"); the design (4096 scrambled Sobol' points) and the QoIs (scalar reductions) are ours. The paper's 45 000 runs, its Sobol' samples and its time series are not published, so the files cannot be compared run by run.
- **Weights at full precision.** xSSA rounds the first N − 1 weights of each process to 6 decimals before writing the deck; we write them at full precision. The difference in a weight is below 5e-7.
- **What the paper's figures can and cannot be compared with.** The paper's sensitivities are of daily flow, variance-weighted, so they are dominated by the spring freshet (Fig. 6, black line). Our mean flow over 20 years is a different QoI: it weights every day equally and measures the water balance. The notebook therefore compares directions and rankings, with the monthly flows as the closest analogue of Fig. 6, and does not claim agreement in numbers.

## Provenance of the input deck

- **Templates (fetched, not redistributed).** The five Raven input files (`.rvi`, `.rvp`, `.rvc`, `.rvt`, `.rvh`) are
  filled from the string constants `RVI`, `RVP`, `RVC`, `RVT`, `RVH` of `examples/scripts/raven_templates.py` in
  github.com/julemai/xSSA at commit `034e7189618b377a8b25ac95fde0ea11656cfa8c` (the master branch on 2026-11-01; last
  change 2021-07-27), SHA-256 `8109faaeb01ed6ad7e3d835e6a08a1e3a328d8911199f2e0c0bfca500aa060d0`, URL
  `https://raw.githubusercontent.com/julemai/xSSA/034e7189618b377a8b25ac95fde0ea11656cfa8c/examples/scripts/raven_templates.py`.
  The file is Copyright 2019–2020 Juliane Mai and licensed under the GNU LGPL, version 3 or later (its header; the
  repository's `LICENSE` is LGPL-3.0). `raven_salmon.fetch_templates` downloads it once into a local cache, refuses it
  if the SHA-256 differs, and `load_templates` reads the five strings with Python's `ast` parser without importing or
  executing the file. No template text is in this repository.
- **Our code.** `raven_salmon.py` (MIT) supplies only the values for the templates' placeholders (`par`: x01 … x35;
  `dpar`: the sums x09 + x10, x13 + x14, x24 + x25, the powers 10^x04 and 10^x11 and the half-full initial stores
  x29 · 500 mm and x30 · 500 mm; `weights`: the Eq. (A1) weights of each process at full precision; `props`: the
  basin constants), runs Raven and reads its output. It was written for this repository and contains no xSSA code.
- **How the shipped files were made.** They were generated with an earlier version of `raven_salmon.py` that wrote
  the same Raven commands from a transcription of these templates; the transcription has been removed (copyleft text
  is not kept in this MIT repository) and the data were not regenerated. The checks under "Raven build" show that the
  fetched templates give the same outputs; the decks differ only in comments and in the templates' `:FileType`,
  `:WrittenBy` and `:CreationDate` header lines.

## License and credit

Code (`raven_salmon.py`, `generate_data.py`, the notebook): MIT (`LICENSE`). Data (the CSVs): CC BY 4.0
(`LICENSE-DATA`). Model and study: J. Mai, J. R. Craig and B. A. Tolson (2020), CC BY 4.0 article.

- **xSSA** (github.com/julemai/xSSA, LGPL-3.0): the Raven deck templates are fetched at run time, not
  redistributed (see above). The basin constants in `raven_salmon.BASIN` are numbers from
  `examples/data_in/basin_metadata/basin_physical_characteristics.txt`. The forcing and observed-streamflow files
  (`examples/data_in/data_obs/08KC001/`) state no licence of their own and are not redistributed here; the streamflow
  is Water Survey of Canada gauge 08KC001 data (HYDAT), which Environment and Climate Change Canada publishes,
  probably under the Open Government Licence – Canada (not checked).
- **Raven** (the Raven Development Team, University of Waterloo; github.com/CSHS-CWRA/RavenHydroFramework): not
  redistributed here. The repository's licence is the Artistic License 2.0, but the file first appears on 2021-03-15
  (`src/LICENSE`, commit `bafa87b`; at the top level on 2023-08-03, `d0aec73`); the 2020 tree we built (`cb745e1`)
  has no licence file, only "Copyright (c) 2008-2020 the Raven Development Team" headers.

Cite J. Mai, J. R. Craig, and B. A. Tolson, Hydrol. Earth Syst. Sci. 24:5835–5858, 2020, doi:10.5194/hess-24-5835-2020,
and J. R. Craig et al., Environ. Model. Softw. 129:104728, 2020, doi:10.1016/j.envsoft.2020.104728.
