# Wind_Turbine_Inflow_Loads: data

Kind: **stored data**. OpenFAST and TurbSim v3.5.5 runs of the NREL 5 MW land-based turbine at a hub-height mean wind speed of 12 m/s, made for this data set by P. G. Constantine in November 2026, following the near-rated case of the wind-inflow study of Robertson et al. (2019). The runs cannot be repeated without TurbSim and OpenFAST; the design and the per-set values can be regenerated here.

## Files

| file | contents | role |
|---|---|---|
| `Wind_Turbine_Inflow_Loads.ipynb` | the case study | executed top to bottom with `SEED = 479`; outputs committed |
| `design.csv` | 300 × 19: `set`, then the 18 inputs in physical units (columns as in `params.json`) | the design (inputs) |
| `params.json` | the 18 inputs: name, lower and upper bound (Table 3 of the paper, near-rated row) | input space |
| `seeds.json` | the 30 TurbSim `RandSeed1` values, used for every set; `RandSeed2` is `"RanLux"` | turbulence seeds |
| `sets.csv` | 300 × 55: one row per set; the inputs, `n_seeds_ok`, 16 ultimate loads `*_ult`, 13 DELs `*_DEL`, `GenPwr_mean` (kW), `RotSpeed_mean` (rpm), `BldPitch1_mean` (deg), `hub_sig{u,v,w}_real` (m/s) | the quantities of interest |
| `runs.csv` | 9000 × 44: one row per (set, seed) run; see below | per-run record |
| `failures.csv` | 90 × 5: one row per failed run: `set`, `seed`, `stop_time_s` (simulation time at which OpenFAST stopped, s), `cause` (`tower strike` or `Mach number > 1`, from OpenFAST's fatal message), `large_deflection_warning` (1 if ElastoDyn first warned that a large blade deflection violated its small-angle assumption) | failure record |
| `openfast_driver.py` | writes the TurbSim inputs and the OpenFAST case, runs both, computes the per-run statistics and the per-set aggregates | drives the outside codes |
| `patch_turbsim.py` | the change made to TurbSim's `Coh2H` before building it | outside-code patch |
| `generate_data.py` | regenerates the design and `sets.csv`; reruns OpenFAST where installed; `--check` compares | regenerates the data |
| `check_accuracy.py` | checks the rainflow counting and the aggregation | accuracy checks |
| `convert_data.py` | writes `runs.csv` from the original per-run file, and `failures.csv` from the failure logs, and checks them | format conversion |

All files are comma-separated text with one header line, read with `csv.DictReader` (as the notebook does) or `np.loadtxt(path, delimiter=',', skiprows=1)` for the all-numeric `design.csv`. Values in `design.csv` and the input columns of `sets.csv` are written with 10 significant digits; the other numbers are Python `repr` of float64 values (full precision). Failed runs have empty statistics fields.

**`runs.csv` columns.** `set`, `seed` (index 0-29 into `seeds.json`), `status` (`ok` or `openfast_fail`), `ts_s` and `of_s` (wall seconds of TurbSim and OpenFAST), `nugget_n` and `nugget_max_eps` (frequencies at which the patched TurbSim scaled the coherence matrix, and the largest $\epsilon$; see Model); for the 16 channels with an ultimate value, `<channel>_maxabs` = max |x| over the 600 s; for the 13 moment channels, `<channel>_dsum` = rainflow damage sum $\sum n_i S_i^m$ over the 600 s, in (channel units)$^m$; and `GenPwr_mean`, `GenPwr_std` (kW), `RotSpeed_mean` (rpm), `BldPitch1_mean` (deg), `Wind1VelX_mean`, `Wind1VelX_std`, `Wind1VelY_std`, `Wind1VelZ_std` (m/s, the hub-point wind). Channel units: moments kN m, deflections m.

**The original per-run file** written on the machines that made the runs has 86 columns (also the per-run mean and standard deviation of every channel and the maxima of the non-load channels), 11.4 MB, SHA-256 `a7a8cbc97976ccc7480f2fb9e977903c29c798b2c05d9e8ac37bed55f1a661a2`. It is over the repository's 10 MB limit for one file, so `runs.csv` keeps 44 of its columns, copied as the same text by `convert_data.py`, which checks that no field differs (0 of 9000 × 44 differ). The original is archived with the failure logs, the build and run logs and the scripts as they ran, on Zenodo, doi:[10.5281/zenodo.24317959](https://doi.org/10.5281/zenodo.24317959).

**Channels** (OpenFAST ElastoDyn and ServoDyn output names; the paper's quantities of interest, Table 1, p. 482): `RootMyc1/2/3` blade-root out-of-plane bending, `RootMxc1` in-plane bending, `RootMzc1` pitching moment (blades 1-3, in the coned frame); `LSSGagMya`, `LSSGagMza` low-speed-shaft bending at the shaft strain gauge, 1.912 m from the rotor apex (the "0°" and "90°" bending moments of the ElastoDyn channel list); `RotTorq` rotor torque; `YawBrMyp`, `YawBrMxp`, `YawBrMzp` tower-top fore-aft, side-side and yaw moments; `TwrBsMyt`, `TwrBsMxt` tower-base fore-aft and side-side moments; `OoPDefl1/2/3` blade-tip out-of-plane deflections.

## Model

**Source.** A. N. Robertson, K. Shaler, L. Sethuraman, J. Jonkman, "Sensitivity analysis of the effect of wind characteristics and turbine properties on wind turbine loads", *Wind Energy Science* 4:479-513, 2019, doi:[10.5194/wes-4-479-2019](https://doi.org/10.5194/wes-4-479-2019): the inflow model (pp. 484-485, Eqs. 7 and 8), the 18 parameters (Table 2, p. 484) and their near-rated ranges (Table 3, p. 486), the wind speed (p. 480), and the quantities of interest (Table 1, p. 482; Eq. 5, p. 483). The codes: OpenFAST and TurbSim, git tag [v3.5.5](https://github.com/OpenFAST/openfast/tree/v3.5.5), commit b48e032303c12df6949c67ebffb7050b888db000, built from source (double precision, release build) with `patch_turbsim.py` applied, and the regression-test inputs `reg_tests/r-test/glue-codes/openfast/5MW_Land_DLL_WTurb` and `5MW_Baseline` of that tag, with the DISCON controller library built from the same tag.

**Solver.** For each (set, seed), `openfast_driver.run_task`:

1. writes a TurbSim input: user-defined spectra (`USRINP`), the Kaimal spectrum of the paper's Eq. 7 for $q = u, v, w$, $S_q(f) = 4\sigma_q^2 (L_q/V_{hub}) / (1 + 6 f L_q/V_{hub})^{5/3}$, on 400 log-spaced frequencies from $10^{-4}$ to 10 Hz (Eq. 7 is printed with $L_1$ in the denominator, read as $L_q$ since the paper gives a separate scale per component); a user profile (`USR`), $U(z) = 12\,(z/90)^\alpha$ m/s at heights 5-200 m, with direction $\beta (z - 90)/126$ degrees counter-clockwise from above for $z$ within the rotor (27-153 m) and held at the edge values outside; the `GENERAL` coherence model with $a_q$, $b_q$ and exponent $\gamma$ (Eq. 8; TurbSim v3.5.5 uses $V_{hub}$ in it, as Eq. 8 does); the hub Reynolds stresses `PC_UW`, `PC_UV`, `PC_VW`; a 15 × 15 grid, 145 m square, hub height 90 m, time step 0.05 s, 660 s; `ScaleIEC = 0`; seed `RandSeed1` from `seeds.json` and `RandSeed2 = "RanLux"`;
2. runs OpenFAST on the regression-test case `5MW_Land_DLL_WTurb` edited by `make_template`: `TMax = 660` s, text output at 0.05 s, the TurbSim file as inflow, and the extra ElastoDyn output channels `RootMxc1`, `RootMyc1`, `RootMzc1`, `RootMyc2`, `RootMyc3`, `OoPDefl2`, `OoPDefl3`. The case uses ElastoDyn, AeroDyn 15 (`WakeMod = 1` BEM, `AFAeroMod = 2` Beddoes-Leishman unsteady aerodynamics, `TwrPotent = 1`, `TwrAero = True`), ServoDyn with the baseline DISCON controller, and InflowWind (full-field TurbSim file, no tower file);
3. drops the first 60 s and, over the remaining 600 s at 20 Hz, computes for each channel the mean, standard deviation and max |x|, and for the moment channels the rainflow damage sum (`rainflow_dsum`: ASTM E1049 three-point counting of the turning points, the residue counted as half cycles), with $m = 10$ for the blade-root moments and $m = 4$ for the others. A run is `ok` only if TurbSim and OpenFAST finish ("terminated normally") and the output has at least 99 % of the expected rows, all finite.

Per set (`openfast_driver.aggregate`, the `ok` runs only): `<channel>_ult` = mean over the seeds of max |x| (Eq. 5 of the paper, p. 483); `<channel>_DEL` = $(\sum_{\text{seeds}} \text{dsum} / (S \cdot 600))^{1/m}$, the damage-equivalent load at 1 Hz over the pooled record of $S$ seeds, about a zero fixed mean and without the Goodman correction (the paper's conventions, p. 483); seed means of `GenPwr_mean`, `RotSpeed_mean`, `BldPitch1_mean`; `hub_sig{u,v,w}_real` = seed means of `Wind1Vel{X,Y,Z}_std`.

**The TurbSim patch.** With $\gamma > 0$ the coherence matrix of the general model need not be positive definite; unmodified TurbSim then stops with a Cholesky error (`Coh2H`, TSsubs.f90). The patched TurbSim retries the factorization with the off-diagonal coherences divided by $1 + \epsilon$, $\epsilon = 10^{-5}, 10^{-4}, \dots, 10$, keeping the diagonal 1, and reports each retry; `nugget_max_eps` is the largest $\epsilon$ in a run. 38 sets used it, the same in all 30 seeds: $\epsilon = 10^{-4}$ (1 set), $10^{-3}$ (1), $10^{-2}$ (7), 0.1 (18) and 1 (11; coherences halved at the affected frequencies). The notebook prints these counts.

**Inputs** (m = 18; independent uniform on Table 3's near-rated min-max ranges, p. 486):

| name | symbol | units | description | range | source |
|---|---|---|---|---|---|
| `alpha` | α | – | power-law shear exponent | U[−0.4, 0.9] | Table 3 |
| `beta` | β | deg | total veer across the rotor | U[−10, 50] | Table 3 |
| `Lu` | L_u | m | integral scale parameter, u | U[8, 1400] | Table 3 |
| `Lv` | L_v | m | integral scale parameter, v | U[2, 1300] | Table 3 |
| `Lw` | L_w | m | integral scale parameter, w | U[2, 450] | Table 3 |
| `sigu` | σ_u | m/s | standard deviation, u | U[0.20, 7.3] | Table 3 |
| `sigv` | σ_v | m/s | standard deviation, v | U[0.05, 8.1] | Table 3 |
| `sigw` | σ_w | m/s | standard deviation, w | U[0.05, 4.3] | Table 3 |
| `au` | a_u | – | coherence decrement, u | U[1.5, 26] | Table 3 |
| `av` | a_v | – | coherence decrement, v | U[1.7, 18] | Table 3 |
| `aw` | a_w | – | coherence decrement, w | U[2, 17] | Table 3 |
| `bu` | b_u | 1/m | coherence offset, u | U[0, 0.08] | Table 3 |
| `bv` | b_v | 1/m | coherence offset, v | U[0, 3.0e-3] | Table 3 |
| `bw` | b_w | 1/m | coherence offset, w | U[0, 6.0e-3] | Table 3 |
| `gamma` | γ | – | coherence exponent | U[0, 1] | Table 3 |
| `PCuw` | PC_uw | m²/s² | hub Reynolds stress u'w' | U[−3.5, 0.50] | Table 3 |
| `PCuv` | PC_uv | m²/s² | hub Reynolds stress u'v' | U[−4.5, 6.0] | Table 3 |
| `PCvw` | PC_vw | m²/s² | hub Reynolds stress v'w' | U[−2.7, 1.0] | Table 3 |

The uniform distributions are this data set's choice; the paper gives ranges only. The inputs are TurbSim's inputs, not the statistics of the simulated wind: in 86 sets a Reynolds stress exceeds the product of the two standard deviations ($|PC_{ab}| > \sigma_a\sigma_b$), TurbSim clips its scaling factors to [−1, 1] and inflates the hub $\sigma_v$ and $\sigma_w$, by up to factors 15.1 and 6.5 (notebook); where all implied correlations are feasible, the realized hub standard deviations are within factors 0.82 to 1.46 of the targets.

**Quantities of interest.** 16 ultimate loads, 13 DELs, the mean power, rotor speed and pitch (columns above). The notebook prints the range of each over the 284 complete sets; e.g. `RootMyc1_ult` 11010 to 21020 kN m, `TwrBsMyt_ult` 67710 to 136600 kN m, `GenPwr_mean` 3845 to 4998 kW.

**Gradients.** None.

**Sampling design.** Monte Carlo: `numpy.random.default_rng(20261112)`, 300 × 18 uniform draws `lo + (hi - lo) * rng.random((300, 18))`, then the 30 seeds `rng.integers(1, 2**31 - 1, 30)` from the same generator (`openfast_driver.design`). Every set was run with all 30 seeds (common random numbers): 9000 runs, 8910 `ok` and 90 `openfast_fail`, the failures in 16 sets (11 to 29 seeds ok there). All 90 failures are OpenFAST aborts, not TurbSim ones: 89 tower strikes (the deflected blade reaching the tower; 79 of them after ElastoDyn warned that a large blade deflection violated its small-angle assumption) and one run (set 102, seed 12) stopped when the flow at a blade section exceeded Mach 1 (`failures.csv`). `n_seeds_ok` gives the number behind each set's values. The notebook analyses the 284 sets with all 30 seeds ok.

## Regenerating

```bash
python generate_data.py           # writes design.csv, seeds.json, params.json here (sets.csv, the record, is left)
python generate_data.py --out DIR # the same and sets.csv recomputed from runs.csv, in DIR
python generate_data.py --check   # regenerates into a temporary directory and compares
python check_accuracy.py          # prints the accuracy numbers below
python convert_data.py ORIGINAL   # rewrites runs.csv from the original 86-column file and checks it
```

Without TurbSim and OpenFAST, `generate_data.py --check` regenerates the design (numpy only) and `sets.csv` (from `runs.csv`) in under a second and compares them, then prints what to install and exits with code 3:

```
runs not rerun: TURBSIM, OPENFAST, OPENFAST_RTEST, DISCON not set or not found. The OpenFAST runs need TurbSim and OpenFAST v3.5.5 (...), with patch_turbsim.py applied to modules/turbsim/src/TSsubs.f90 before building, and the r-test inputs of that tag (...).
```

On macOS 14 (Darwin 23.6), Python 3.11.16, numpy 2.4.6, it printed: `design.csv` difference 0 (and `seeds.json`, `params.json` identical; the regenerated files are byte-identical to the shipped ones), `sets.csv` largest relative difference 5.5e-16 (the seeds are summed in a different order than on the machines that made the runs). With the four environment variables set (`TURBSIM`, `OPENFAST`, `OPENFAST_RTEST`, `DISCON`), `--check` also reruns set 0 with seeds 0 and 1 (about 160 s per run: TurbSim 34 s and OpenFAST 126 s on average on the machines below) and compares the per-run statistics with `runs.csv`; `--check --full` reruns all 9000 runs, about 400 CPU-hours. Neither rerun has been done since the original runs, so the reproducibility of the OpenFAST statistics on another machine or compiler is not known.

The original runs: four AWS EC2 c7a.4xlarge machines (16 vCPU each, Ubuntu), 2026-11-12 to 13, 16 runs in parallel per machine, each machine a quarter of the sets, 6 to 6.5 hours each; Ubuntu 24.04.5 LTS; OpenFAST built there from source (CMake release build, double precision) with the GNU 13.3.0 compilers and the Ubuntu reference BLAS and LAPACK (`libblas`, `liblapack`); the r-test submodule at commit d9691df0339bbc718c63e1de8a1aaa642e978398. The build logs of the four machines agree, and the four machines produced the same design. Their per-machine result files, merged and sorted by set and seed, are exactly the original per-run file (the archive has the logs).

## Accuracy

* `runs.csv` against the original per-run file: 0 of 9000 × 44 fields differ (text comparison) — `convert_data.py`, needs the original file.
* `sets.csv` against the aggregation of `runs.csv` (ultimate loads, DELs, seed means): largest relative difference 5.5e-16 — `check_accuracy.py` (check 2), `generate_data.py --check`; the notebook recomputes the ultimate loads and DELs independently (4e-16).
* The rainflow damage sum (`openfast_driver.rainflow_dsum`) against hand counts for the turning points 0, 5, 1, 4, 2, 6, 0 (m = 0, 1, 3, 10; exact) and for a sampled sine over 50 periods (m = 4, 10; relative difference ≤ 1.1e-16) — `check_accuracy.py` (checks 1a, 1b). The time series of the runs were not kept, so the damage sums themselves cannot be recomputed.
* The design regenerated from its seed: difference 0 — `generate_data.py --check`.
* `failures.csv` against the 90 failure logs in the archive: identical, and the logs are exactly the runs whose status in `runs.csv` is not `ok` — `convert_data.py --fail-logs`, needs the archive.
* Seed noise of the per-set values (from the 30 seeds): 0.1 % to 76 % of the variance across sets, depending on the quantity (0.3 % to 2.3 % for the four loads the notebook studies in detail) — notebook.
* Realized hub turbulence against the target: median ratio 0.933 ($\sigma_u$), 0.966 ($\sigma_v$), 0.997 ($\sigma_w$); hub mean wind speed 11.863 to 12.063 m/s — notebook.
* No comparison with the paper's runs is possible (they were not kept, paper p. 504). The OpenFAST and TurbSim statistics are an external run, not reproducible from the repository without the outside codes.

## How these files relate to the paper

These are new runs that use the paper's model; they are not a reproduction of the paper's simulations, and were sized to the computing available for them (a limited number of cloud machine-hours). They follow the paper's near-rated case (12 m/s; the 18 inputs and Table 3 ranges; the inflow model of Eqs. 7 and 8; 30 seeds; Eq. 5 ultimate loads and MLife-convention DELs) and differ from it in:

* **design**: 300 independent uniform draws instead of the paper's radial elementary-effects design (30 Sobol starting points, ±10 % steps, 19 points each; pp. 483 and 487); the same 30 seeds for every set (the paper does not say);
* **wind grid** 15 × 15 points, 145 m square, instead of 25 × 25 (p. 480);
* **transient** 60 s dropped instead of 30 s (p. 480), 600 s analysed in both;
* **tower drag**: the regression-test case computes AeroDyn tower loads (`TwrAero = True`); the paper did not consider tower drag (p. 480);
* **ultimate bending moments** per component; the paper takes the largest vector sum of the first two components (p. 481), which cannot be formed from the stored per-channel maxima;
* **S-N exponents** m = 10 (blade root) and 4 (others), and a 1 Hz equivalent frequency: the paper does not give them;
* **non-positive-definite coherence matrices**: modified by the TurbSim patch in 38 sets; the paper does not say how it handled them;
* **code versions**: OpenFAST v3.5.5; the paper does not give its version (its reference is the documentation of November 2017).

The paper is used to define the model (inputs, ranges, inflow model, quantities of interest); the notebook's results are not expected to reproduce the paper's and may or may not agree with its conclusions. The notebook relates them briefly to the paper's headline conclusion (σ_u and α the most influential parameters, pp. 479 and 490): σ_u leads here too, while α ranks lower. One possible reason is that this data set has only the near-rated bin, whereas the paper's shear sensitivity comes mostly from the below-rated bin with its wider α range (pp. 486-487); whether the other differences in settings contribute cannot be determined from the paper's text and these runs.

Equation 7 is printed with $L_1$ in the denominator (p. 485); the runs use $L_q$, since the paper gives a separate integral scale for each component.

## Unknowns

* The realized Reynolds stresses of the simulated wind: no cross-component statistics were stored.
* Whether the TurbSim patch matters for the paper's comparison: the paper does not say whether its coherence matrices were ever not positive definite or how that was handled.
* The paper's OpenFAST and TurbSim versions, wind-grid extent, time step, S-N exponents and DEL equivalent frequency, and whether it used the land-based or another variant of the NREL 5 MW turbine.
* Whether the OpenFAST statistics are reproducible on another platform: `--check` with the outside codes has not been run since the original runs.

## License and credit

Code (`*.py`, the notebook): MIT (`LICENSE`). Data (`*.csv`, `*.json`): CC BY 4.0 (`LICENSE-DATA`); the runs were made for this data set. TurbSim and OpenFAST (Apache License 2.0) are not included; `openfast_driver.py` copies and edits the OpenFAST regression-test inputs at run time from a local checkout, and `patch_turbsim.py` edits a local TurbSim source file. Credit the runs to P. G. Constantine and cite the study they follow:

- A. N. Robertson, K. Shaler, L. Sethuraman, and J. Jonkman, "Sensitivity analysis of the effect of wind characteristics and turbine properties on wind turbine loads", *Wind Energy Science* 4:479-513, 2019, doi:[10.5194/wes-4-479-2019](https://doi.org/10.5194/wes-4-479-2019).
- OpenFAST v3.5.5, https://github.com/OpenFAST/openfast/tree/v3.5.5.
- J. Jonkman, S. Butterfield, W. Musial, and G. Scott, "Definition of a 5-MW Reference Wind Turbine for Offshore System Development", NREL/TP-500-38060, 2009, doi:[10.2172/947422](https://doi.org/10.2172/947422).
