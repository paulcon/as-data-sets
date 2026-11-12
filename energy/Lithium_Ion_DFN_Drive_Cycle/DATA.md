# Lithium_Ion_DFN_Drive_Cycle: data

Kind: **stored data, generated for this repository with an outside code (PyBaMM 24.9.0); regeneration script included**. The runs come from PyBaMM, driven by `dfn_us06_driver.py` and `generate_data.py` in this directory. PyBaMM is not a numpy/scipy model and the DFN solver was not reimplemented, so the CSVs are the record; their values are never edited. **Regenerating the data, or rerunning the PyBaMM part of the accuracy checks (`check_accuracy.py`), needs PyBaMM 24.9.0 with casadi 3.6.7** (`pip install "pybamm==24.9.0" "casadi==3.6.7"`, Python 3.9–3.12); both scripts stop with an error that says what to install when PyBaMM is missing. The repo's `.venv` has no PyBaMM. The notebook reads the files and the driver's constants and needs only numpy, matplotlib and active_subspaces; in `.venv` (Python 3.11) it runs in about 55 s (`jupyter nbconvert --execute`, 2026-11-02).

## Files

| file | contents | role |
|---|---|---|
| `Lithium_Ion_DFN_Drive_Cycle.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with `SEED = 2026`; outputs committed |
| `inputs.csv` | 650 × 24 inputs in physical units, columns in the order of `dfn_us06_driver.NAMES` (Table A.2 order) | stored data |
| `outputs.csv` | 650 × 4: status (1 ok, 0 failed), Ū [V] (time-averaged voltage), U(600 s) [V], run time [s] | stored data |
| `voltage.csv` | 650 × 601: terminal voltage [V] at t = 0, 1, …, 600 s (9 decimals) | stored data |
| `dfn_us06_driver.py` | the PyBaMM set-up: `INPUTS`, `NAMES`, `LOWER`, `UPPER`, `LOG`, `SCALE`, `parameter_values(x)`, `voltage(x)`, `theoretical_capacity`, `nominal()` | generating code (needs PyBaMM; its constants import with numpy alone) |
| `generate_data.py` | the seeded Monte Carlo design and the loop that writes the three CSVs; `--resume` (chunked runs), `--check --rows`, `--oat` (the one-at-a-time runs under Accuracy) | generating code (needs PyBaMM 24.9.0) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" and the binned indices and emulation table of "How these files relate to the paper"; `--data-only` runs the two numpy checks alone | accuracy check (checks 3–7 need PyBaMM 24.9.0) |
| `US06.csv` | the US06 drive-cycle current [A] at 1 Hz, t = 0…600 s (unchanged copy from the TDGSA repository, md5 e2109257f9ce806c083293166766adbe; BSD-3-Clause, not CC BY 4.0, see "License and credit") | model input |
| `US06_LICENSE.txt` | BSD-3-Clause licence of the TDGSA repository, from which `US06.csv` was copied | licence |

The CSVs have two `#` header lines (description; column names with units) and are read with `np.loadtxt(path, delimiter=",")`. Total size 5.0 MB (`voltage.csv` 4.7 MB).

## Model

**Citation.** E. Zonta, I. Jovanovic Buha, M. Spinola, C. Weißinger, H.-J. Bungartz, A. Jossen, "Time-dependent global sensitivity analysis of the Doyle–Fuller–Newman model", *Journal of Energy Storage* 158 (2026) 121751, doi:10.1016/j.est.2026.121751 (CC BY 4.0). The version read for this data set is the preprint arXiv:2502.17275v2 (31 Mar 2026); section, table and page references below are to it.

**Model.** The DFN (P2D) model of a lithium-ion cell (paper Section 2.2, Eqs. (23)–(42)) as implemented in PyBaMM (`pybamm.lithium_ion.DFN()`, default options: isothermal, Fickian particles, no SEI). Base parameter set: `Marquis2019` (Kokam SLPB78205130H; Marquis et al. 2019), which supplies everything not sampled: open-circuit potentials (Dualfoil LiCoO₂ and graphite MCMB 2528 fits), electrode area 0.137 m × 0.207 m, one electrode pair, temperature 298.15 K, current-collector properties, and the electrode-phase Bruggeman exponents (1.5). Default mesh: 20 points in each of the negative electrode, separator and positive electrode in x and 20 in r of each particle (paper Section 3.1). Solver: `pybamm.CasadiSolver(mode="fast")`, rtol = atol = 1e-6 (the paper says "CasADi solver"; "fast" mode, which detects no events and so applies no voltage cut-off, follows the authors' TDGSA battery example). Output on the 601-point grid t = 0, 1, …, 600 s.

**How the 24 inputs enter** (`dfn_us06_driver.parameter_values`): each overwrites the PyBaMM entry named in the code, as a scalar (the paper treats every parameter as independent of concentration, Section 2.2). Concentration-dependent Marquis functions for D_s, D_l and κ are replaced by these constants. In addition, following the paper:

- active material volume fraction ε_s = 1 − ε_l in each electrode (Section 3.3.1; Marquis has inactive material);
- initial particle concentrations c_s,0 = c_s^max / 2 in both electrodes, so U(0) is the same for every sample (Eq. (44)); here U(0) = 3.9906 V;
- exchange current density j₀ = F k₀ c_l^½ c_s^½ (c_s^max − c_s)^½, dimensional concentrations, α = ½, no Arrhenius factor. Eq. (31) prints (1 − c_s)^½, as for a normalized concentration; the units of k₀ in Table A.2, m^2.5 mol^-0.5 s^-1, fix the dimensional form used here;
- the Bruggeman coefficients b_pos, b_neg, b_sep set PyBaMM's "(electrolyte)" exponents (Eqs. (24), (27)); the "(electrode)" exponents stay at Marquis' 1.5.

**Inputs** (m = 24; Table A.2, p. 18; ranges from LiionDB, Section 3.3.1). Independent; "U" uniform, "LogU" log-uniform.

| # | name | symbol | units | range | dist. | source |
|---|---|---|---|---|---|---|
| 0 | `cs_max_pos` | c_s,pos^max | mol m⁻³ | 23900 – 51765 | U | Table A.2, p. 18 |
| 1 | `L_pos` | L_pos | m | 6.00e-6 – 6.60e-5 | U | Table A.2, p. 18 |
| 2 | `eps_pos` | ε_l,pos | – | 0.171 – 0.648 | U | Table A.2, p. 18 |
| 3 | `R_pos` | R_pos | m | 5.00e-7 – 1.00e-5 | U | Table A.2, p. 18 |
| 4 | `k0_pos` | k₀,pos | m^2.5 mol^-0.5 s⁻¹ | 2.10e-12 – 2.41e-5 | LogU | Table A.2, p. 18 |
| 5 | `Ds_pos` | D_s,pos | m² s⁻¹ | 9.59e-19 – 2.51e-12 | LogU | Table A.2, p. 18 |
| 6 | `sigma_pos` | σ_pos | S m⁻¹ | 5.20e-6 – 1.00e3 | LogU | Table A.2, p. 18 |
| 7 | `b_pos` | b_pos | – | 1.44 – 2.44 | U | Table A.2, p. 18 |
| 8 | `cs_max_neg` | c_s,neg^max | mol m⁻³ | 16100 – 31920 | U | Table A.2, p. 18 |
| 9 | `L_neg` | L_neg | m | 4.60e-5 – 7.20e-5 | U | Table A.2, p. 18 |
| 10 | `eps_neg` | ε_l,neg | – | 0.26 – 0.50 | U | Table A.2, p. 18 |
| 11 | `R_neg` | R_neg | m | 1.00e-5 – 1.26e-5 | U | Table A.2, p. 18 |
| 12 | `k0_neg` | k₀,neg | m^2.5 mol^-0.5 s⁻¹ | 1.00e-11 – 3.00e-3 | LogU | Table A.2, p. 18 |
| 13 | `Ds_neg` | D_s,neg | m² s⁻¹ | 2.00e-16 – 9.07e-12 | LogU | Table A.2, p. 18 |
| 14 | `sigma_neg` | σ_neg | S m⁻¹ | 1.11e-1 – 2.20e3 | LogU | Table A.2, p. 18 |
| 15 | `b_neg` | b_neg | – | 1.50 – 4.10 | U | Table A.2, p. 18 |
| 16 | `L_sep` | L_sep | m | 1.60e-5 – 5.00e-5 | U | Table A.2, p. 18 |
| 17 | `eps_sep` | ε_l,sep | – | 0.37 – 0.60 | U | Table A.2, p. 18 |
| 18 | `b_sep` | b_sep | – | 1.50 – 2.57 | U | Table A.2, p. 18 |
| 19 | `Theta` | Θ | – | 1.00 – 1.86 | U | Table A.2, p. 18 |
| 20 | `c_l0` | c_l,0 | mol m⁻³ | 500 – 1500 | U | Table A.2, p. 18 |
| 21 | `D_l` | D_l | m² s⁻¹ | 3.60e-11 – 1.09e-9 | LogU | Table A.2, p. 18 |
| 22 | `kappa` | κ | S m⁻¹ | 4.45e-3 – 2.45 | U | Table A.2, p. 18 |
| 23 | `t_plus` | t⁺ | – | −0.37 – 0.51 | U | Table A.2, p. 18 |

The paper's Fig. 5 (p. 9) shows the same ranges. The negative transference numbers are the paper's choice (Section 3.3.1).

**Current (drive cycle).** The US06 current profile in `US06.csv` (PyBaMM's, "delivered with PyBaMM up until v24.1", Section 3.3.2), applied as `pybamm.Interpolant` (linear) and multiplied by one factor for all samples. The paper scales it "such that the peak C_theo-rate does not exceed 2 h⁻¹" with C_theo = min_k F c_s,k^max L_k ε_k (Eqs. (45)–(46)). **Choice made here:** the factor uses the smallest C_theo in the Table A.2 box, F · 23900 · 6.00e-6 · (1 − 0.648) / 3600 = 1.353 A h m⁻², so the peak current density is 2 h⁻¹ × 1.353 = 2.706 A m⁻² and the factor is `SCALE` = 0.0094730 (US06 peak 8.1 A over 0.028359 m²). Eq. (46) as printed has ε_l, but the paper's Fig. 6a (C_theo,neg from about 11 to 44 A h m⁻², read from the figure) matches ε_s = 1 − ε_l, which is used here: over the box C_theo,neg spans 9.9–45.6 A h m⁻² with ε_s and 5.2–30.8 A h m⁻² with ε_l (notebook, "The model"). The paper's Fig. 6e shows a peak of about 2.85 A m⁻² (read from the figure), so its factor was about 5 % larger (2.85 / 2.706 = 1.053); it does not state the value. The paper's value equals 2 h⁻¹ × the smallest C_theo among its 2000 Fig. 6 samples (Fig. 6c minimum ≈ 1.4 A h m⁻²), which is probably its rule (Unknowns, question 1). Over the 650 samples the largest peak C_theo-rate is 1.37 h⁻¹, and the positive electrode is limiting in 61 % (paper: "around 60 %", Fig. 6c). To first order, both the overpotentials and the charge drawn scale linearly with the current, so a 5 % difference in the factor cannot reorder the inputs; it does not explain the main difference from the paper (below).

**Quantities of interest.** The paper analyses U(t) on the 601-point grid with generalized Sobol' indices (no scalar QoI). Here: Ū = (1/600) ∫₀⁶⁰⁰ U dt by the trapezoidal rule on the 1-s grid (`generate_data.qois`; the paper also uses a composite trapezoidal rule for its time integrals, p. 7), and U(600 s). `voltage.csv` keeps the full curve, so other scalars can be formed.

**Gradients.** None. The paper has none; finite differences would cost 25 runs per gradient (about 2 min each at the median 5.2 s per run), and PyBaMM's solver sensitivities were not tried. The notebook estimates the subspace from values with a global quadratic (QPHD; a full quadratic in 24 inputs has 325 coefficients, and the 650 runs give twice as many points), cross-checked with global linear and local linear estimates.

**Sampling design.** i.i.d. Monte Carlo, like the paper's (Section 3.1): row i of `numpy.random.default_rng(20261031).random((2000, 24))`, mapped linearly to each uniform range and linearly in log₁₀ to each log-uniform range. The shipped files are the first 650 rows, run in four resumable chunks of at most 20 minutes each, one process at a time: rows 0–115 (2026-10-31, `--n 400 --budget 1200`, stopped by the budget), then `nice -n 10 python generate_data.py --n 650 --budget 1200 --resume` three times (rows 116–249, 250–501, 502–649). `--resume` checks that the inputs already in the files are the first rows of the design (to a relative tolerance of 1e-15, see below) and carries their inputs and outputs over unchanged. `--n N` for N > 650 extends the same design. The paper used 10,000 samples (full PC order 2) and 70,000 (sparse PC order 5), Fig. 8.

**Failures.** 0 of 650 runs failed (no solver error, every solution reached t = 600 s, all voltages finite). The generator keeps a failed run as a row with status 0 and NaN outputs; the notebook drops rows with status 0 (none here). The paper drops non-converged samples and reports at most 0.003 % failures at N = 100,000 (p. 7). No sample left the physical domain: the voltage stayed between 3.362 V and 4.125 V (the maximum during regenerative, charging pulses). The paper's Fig. 7b shows some runs near 2 V, which it describes as "a few even being deeply discharged to voltages of around 2 V for short time spans" (p. 11). Under the paper's stated current scaling these dips cannot be capacity-limited: the scaled cycle draws a net 0.0469 A h m⁻² (largest running total 0.0493 A h m⁻²), at most 3.6 % of the smallest C_theo anywhere in the box (1.353 A h m⁻²), 2.5 % of the smallest among the 650 samples (1.97 A h m⁻²), and 7.3 % of the half of the box minimum that lies above the half-charge start (all printed in the notebook's comparison cell). Such dips must be transient overpotentials; here the eight lowest minimum voltages (3.36–3.65 V) all have log₁₀ σ_pos ≤ −4.09 and log₁₀ D_s,pos ≤ −15.37. Why none of these 650 runs goes below 3.36 V is open (Unknowns, question 4).

**Published results to compare.** No active subspaces in the paper. Generalized total-order Sobol' indices (Fig. 8a, read from the figure): L_pos ≈ 0.39, ε_l,pos ≈ 0.33, R_pos ≈ 0.12, c_s,pos^max ≈ 0.11, all others small; D_s,pos and σ_pos ≈ 0. Subgroup analysis (Section 4.2, Fig. 9b, p. 13: the paper's group 2, the five non-capacity inputs of the positive electrode, with the other groups held at 30 Latin-hypercube points and the indices averaged; read from the figure): total-order R_pos ≈ 0.75, b_pos ≈ 0.22, k₀,pos ≈ 0.16, σ_pos ≈ 0.12, D_s,pos ≈ 0.02. Mean voltage ≈ 3.95 V for most of the cycle (Fig. 7a). Fig. 8 (p. 11) and Fig. B.12 (p. 17) label their 9th row "Θ", between b_pos and L_neg, where Table A.2 has c_s,neg^max; Θ appears again as the 20th row, in its Table A.2 place.

## Regenerating

```bash
# a separate venv for PyBaMM (the notebook's .venv does not need it)
python3.12 -m venv venv && venv/bin/pip install "pybamm==24.9.0" "casadi==3.6.7"
OMP_NUM_THREADS=1 nice -n 10 venv/bin/python generate_data.py --n 650 --budget 1200 --resume   # repeat until 650 rows
OMP_NUM_THREADS=1 venv/bin/python generate_data.py --check --rows 0,300,649                       # rerun 3 rows, compare; writes nothing
OMP_NUM_THREADS=1 venv/bin/python generate_data.py --oat                                          # one-at-a-time runs (Accuracy), about 1 min
OMP_NUM_THREADS=1 venv/bin/python check_accuracy.py                                               # the Accuracy numbers
```

`generate_data.py` stops if the installed PyBaMM is not 24.9.0 (override with `--any-pybamm`). casadi 3.6.7 is pinned because pip pulls casadi 3.8 with PyBaMM 24.9.0, whose IDAKLU solver then fails to load casadi functions (observed when the files were written); the CasADi solver used here ran with either. The paper used PyBaMM v24.9.0 (Section 3.1).

`inputs.csv` is the source of truth for the inputs. `10.0 ** z` in `generate_data.design` can differ in the last bit between numpy builds: under the `.venv`'s numpy 2.4.6, 1231 of the 15,600 entries differ from `inputs.csv` (written with numpy 1.26.4), by at most 2.2e-16 relative. `--resume` and `--check` therefore compare the design with the stored inputs to a relative tolerance of 1e-15 and then run or carry over the stored values.

**Provenance and run time of the shipped files.** Written on 2026-10-31 in a Linux container (Python 3.12.3, PyBaMM 24.9.0, casadi 3.6.7, numpy 1.26.4, scipy 1.17.1), one process, on 2 CPUs shared with other jobs, in four resumable chunks of at most 20 minutes: 1201 s for rows 0–115 and 1204 + 1203 + 639 s for rows 116–649, **4247 s (71 min) in all**. Median 5.2 s per run, range 1.6–27.2 s; `outputs.csv` has each run's time. An unloaded run took about 4–5 s (about 1 s to build and 3.4 s to solve). In that environment, `--check --rows 0,300,649` reproduced `voltage.csv` to 5.0e-10 V, the rounding of its 9 decimals, with the same (ok) status, and a `--resume` on a copy of the 650-row files ran no rows and rewrote all three files byte for byte (original run, not repeated here). `check_accuracy.py` check 3 repeats the reproduction test on another platform (below).

## Accuracy

`check_accuracy.py` recomputes every number in this section and the binned indices and emulation table of the next one; the one-at-a-time effects come from `generate_data.py --oat`. Checks 1–2 need only numpy and were run in `.venv` (Python 3.11, numpy 2.4.6; about 10 s). Checks 3–7 need PyBaMM: they were run with PyBaMM 24.9.0, casadi 3.6.7, numpy 1.26.4 and scipy 1.17.1 from PyPI (Python 3.11, macOS x86-64, 2026-11-02), about 4 minutes of run time on one core of a heavily shared machine, and printed the values below.

- **Reproduction** (check 3). Rows 0, 300 and 649 rerun with the shipped settings reproduce `voltage.csv` to 2.3e-5, 2.6e-5 and 7.4e-5 V on this platform (in the Linux container that wrote the files, to the 9-decimal rounding, 5.0e-10 V). The platform difference is smaller than the effect of the solver tolerance itself (next item), so it is numerical noise of the adaptive solver at rtol = atol = 1e-6.
- **Solver tolerance** (check 4). Re-solving with rtol = atol = 1e-9 changes U(t) by at most 2.8e-5, 1.3e-4 and 1.9e-4 V (rows 0, 1, 2) and 3.4e-4 V at row 62 (the lowest Ū among the first 116 rows); Ū by at most 3.2e-5 V and U(600) by at most 1.1e-4 V (rows 0–2 and 62). These are below 1 % of the standard deviations of Ū (0.046 V) and U(600) (0.053 V) over the 650 runs.
- **Solver choice** (check 5). At the box centre PyBaMM's IDAKLU solver (default settings) agrees with the CasADi solution to 3.1e-5 V over the cycle.
- **One-at-a-time effects at the box centre** (check 7, which calls `generate_data.py --oat`; `dfn_us06_driver.nominal()`, Ū = 3.9846 V), each input at its lower / upper bound: σ_pos 3.8503 / 3.9847 V (difference 0.1345 V); D_s,pos 3.9752 / 3.9872 V (0.0120 V); L_pos 3.9568 / 3.9860 V (0.0292 V); R_pos 3.9871 / 3.9829 V (−0.0042 V). In the container that wrote the files three of the lower-bound values were 1e-4 V different (σ_pos 3.8502, D_s,pos 3.9751, L_pos 3.9569 V); the differences agreed to the 4 decimals printed. These confirm that σ_pos and D_s,pos act strongly in this implementation (see the next section).

## How these files relate to the paper

- **Not the paper's runs.** The authors' 24-input DFN/US06 script is not in their repository (https://github.com/ezonta/TDGSA, commit 6f5e78e, 2026-06-09: package, a 4-input battery example and `US06.csv`), and no run data are published. These data are a rebuild from the paper's text with the same tool and version.
- **The main difference in results.** Here log σ_pos has the largest weight in the active direction of both QoIs, in every estimator and at each of the five times examined, and log D_s,pos the second largest except at t = 60 s, where k₀,pos (+0.25) is just ahead of it (+0.22) (QPHD first eigenvector of Ū: +0.89 and +0.37; normalized activity scores 0.55 and 0.14). In the paper their total-order indices are about 0 and R_pos is third (here 7th). The paper's subgroup analysis sharpens the disagreement: within its group 2 (R_pos, b_pos, k₀,pos, σ_pos, D_s,pos), with the inputs of the other groups held at each of 30 Latin-hypercube points and the indices averaged, Fig. 9b gives R_pos ≈ 0.75 and σ_pos ≈ 0.12, D_s,pos ≈ 0.02, while here the activity scores within the same five inputs put σ_pos and D_s,pos first (notebook, "Comparison with the paper"). The capacity-related inputs L_pos, ε_l,pos and c_s,pos^max matter in both, with the expected signs. The disagreement does not depend on the design size: the global linear direction from the first 116 runs and from all 650 agree to |w·w| = 0.95, with σ_pos and D_s,pos first in both. It does not come from the estimator either: binned first-order indices computed directly from the data (`check_accuracy.py` check 1; 10 equal-width bins in the sampling coordinates) rank σ_pos (0.41) and D_s,pos (0.16) first (16 bins: 0.46 and 0.16); the estimator's bias for an input without effect is about 0.01–0.02. **The disagreement is unresolved; it is an open question for the authors** (Unknowns, questions 1–5).
- **Two open choices that do not explain it** (numbers printed in the notebook's comparison cell).
  - *The current factor.* The paper's is about 5 % larger (above). Overpotentials and charge throughput both scale, to first order, linearly with current, so this cannot reorder the inputs.
  - *The electrode Bruggeman factor on σ.* PyBaMM's (1 − ε_l,pos)^1.5 (kept here) lies in 0.21–0.75 over the box, so it shifts log₁₀ σ_eff by at most 0.68 of the 8.28 decades that σ_pos spans. The effect of σ_pos on Ū sits in the lowest two to three decades (binned mean Ū 3.880, 3.947, 3.972 V in the first three of eight equal bins of log₁₀ σ_pos, then 3.969–3.974 V), so a shift of ≤ 0.7 decade leaves it inside the box.
- **The paper's dips to about 2 V** (Fig. 7b; "deeply discharged", p. 11) cannot be capacity-limited under the stated scaling (the cycle draws ≤ 3.6 % of the smallest C_theo in the box; see Failures above). They must be transient overpotentials, i.e. transport, ohmic or kinetic limits. In our data such limits come from low σ_pos and D_s,pos, the inputs the paper rates ≈ 0. So the paper's simulations probably contain strong effects of these inputs, and the gap would lie in how the paper attributes variance.
- **Two possible causes in the authors' code, unconfirmed.** Both come from reading TDGSA (https://github.com/ezonta/TDGSA, commit 6f5e78e, 2026-06-09), and both act specifically on the log-uniform inputs. In Fig. 8a, D_s,pos, k₀,neg and D_s,neg have no visible bar and k₀,pos and σ_pos only slivers, while uniform inputs with little expected effect (e.g. Θ (20th row), L_neg, ε_l,neg, L_sep) show small bars; σ_neg, also log-uniform, shows a small bar. Neither can be confirmed without the authors' 24-input script, which is not in the repository.
  1. *Log-uniform bounds passed to chaospy.* `TDGSA/utils.py`, lines 44–45, passes the `[lo, hi]` of a `"loguniform"` entry straight to `chaospy.LogUniform(lo, hi)`. chaospy's `log_uniform` has `_ppf = e**(q*(up − lo) + lo)`, `_lower = e**lo` and `_upper = e**up` (`chaospy/distributions/collection/log_uniform.py`, lines 19–26, identical in chaospy 4.3.13, which TDGSA's `requirements.txt` pins, and in 4.3.21), so the support is [e^lo, e^hi] and the bounds are read as natural logarithms. Table A.2's values cannot have been passed as they stand (σ_pos's upper bound would become e^1000). If log₁₀ bounds were passed, the effective ranges would be σ_pos 5.1e-3–20 S m⁻¹, D_s,pos 1.5e-8–9.2e-6 m² s⁻¹, k₀,pos 8.5e-6–9.9e-3, k₀,neg 1.7e-5–0.080, D_s,neg 1.5e-7–1.6e-5, σ_neg 0.38–28, D_l 2.9e-5–1.3e-4 (SI units as in the table), all at the fast end. With PyBaMM (`check_accuracy.py` check 6): at the box centre, σ_pos = 5.07e-3 S m⁻¹ gives Ū = 3.9834 V against 3.9846 V at the centre value, a 1.2 mV change, where σ_pos moves Ū by 134 mV over its true range. Row 503 (the lowest Ū, 3.6997 V), with its seven log-uniform inputs remapped x → e^(log₁₀ x), gives Ū = 3.9854 V and a minimum of 3.981 V. This would make σ_pos and D_s,pos inert, as in Fig. 8a, but it would also shrink the voltage spread to a few mV: our 102 samples with σ_pos ≥ 5.1e-3 S m⁻¹ and D_s,pos ≥ 1e-13 m² s⁻¹ have a std of Ū of 3.0 mV, against 46.0 mV overall (notebook). That conflicts with the spread (std band ≈ 0.05 V) and the 2 V dips of the paper's Fig. 7, so this is at most part of the story.
  2. *A PC basis in physical rather than log units.* `TDGSA/core.py`, lines 347–354 (KL method) and 490–497 (PC method), call `cp.generate_expansion(PCE_order, joint_dist, normed=True, graded=False, retall=True, cross_truncation=...)`. Its default rule, the three-term recurrence (`chaospy/expansion/frontend.py`), builds polynomials in the physical variable that are orthonormal under the `LogUniform` density. Over 6–8 decades such polynomials cannot resolve effects confined to the lower decades. An emulation on our 650 runs (`check_accuracy.py` check 2: orthonormal PC bases fitted by least squares, total-order indices from the coefficients; for the log-uniform inputs in physical coordinates, polynomials orthonormal under the log-uniform density, built by a discretized Stieltjes procedure) gives:

     | basis for the log-uniform inputs | terms | R² (in-sample) | total index σ_pos | D_s,pos | L_pos |
     |---|---|---|---|---|---|
     | log₁₀ coordinates, full degree 2 | 325 | 0.87 | 0.50 | 0.24 | 0.08 |
     | physical coordinates, full degree 2 | 325 | 0.58 | 0.14 | 0.12 | 0.10 |
     | log₁₀ coordinates, additive degree 3 | 73 | 0.76 | 0.60 | 0.19 | 0.08 |
     | physical coordinates, additive degree 3 | 73 | 0.26 | 0.07 | 0.21 | 0.19 |

     In physical coordinates σ_pos falls, the surrogate captures much less of the variance, and L_pos, c_s,pos^max and ε_l,pos rise relative to σ_pos; but D_s,pos stays at 0.12–0.21, so this too explains only part of the gap. The paper's remark (p. 8) that the KL eigenvalue variance and the PC variance can differ by more than 10 % fits a surrogate that misses variance; the indices are then normalized by the captured part only.
- **Overlap with `energy/Lithium_Ion_Battery`.** Same model family and the same idea (time-dependent active subspaces of a Newman model); different code (PyBaMM vs. Hadigol et al.'s solver), cell (Kokam/Marquis vs. LiC₆/LiCoO₂), inputs (24 from LiionDB, seven of them log-uniform, with the electrode thicknesses and particle radii among the uniform ones, vs. 19), load (dynamic US06 from half charge vs. constant-current discharge at 0.25C, 1C, 4C) and kind of output (voltage vs. capacity curves).

## Unknowns

- Why σ_pos and D_s,pos dominate here while the paper's indices put them at about 0: unresolved (see "The main difference in results").
- **Questions for the authors** (open):
  1. **The current-scaling factor.** Over which samples is the minimum C_theo taken? Was it 2 h⁻¹ × min C_theo over the 2000 samples of Fig. 6 (≈ 2.85 A m⁻², matching Fig. 6e)? Low stakes: a 5 % factor cannot reorder the inputs.
  2. **Which bounds were passed for the log-uniform inputs** (natural-log, log₁₀ or physical values), given that `chaospy.LogUniform(lo, hi)` reads them as natural-log bounds?
  3. **Was the PC expansion built in physical rather than log coordinates** for the log-uniform inputs, and how much variance did it capture (the > 10 % KL/PC mismatch of p. 8)?
  4. **How do the dips to about 2 V in Fig. 7b arise** when the cycle moves the state of charge by ≤ 3.6 % of the smallest capacity, and which inputs do those runs have? None of our 650 runs goes below 3.36 V.
  5. Other choices the paper does not fix: the solver mode and tolerances; whether the electrode-phase Bruggeman exponent was kept at 1.5 (the paper's Eq. (25) has no Bruggeman factor on σ; shown above not to explain the difference); the exact form of j₀ (Eq. (31) prints c_s^½(1 − c_s)^½, read here as dimensional c_s^max − c_s); the voltage cut-offs (none here; the paper's Fig. 7b goes below Marquis' 3.105 V cut-off, so none were enforced there either); ε_l vs ε_s in Eq. (46) (Fig. 6a supports ε_s); and whether the US06 file was the one in TDGSA (identical md5 to the copy here).

## License and credit

Notebook and code (`Lithium_Ion_DFN_Drive_Cycle.ipynb`, `dfn_us06_driver.py`, `generate_data.py`, `check_accuracy.py`): MIT (`LICENSE`), written for this repository from the paper; no TDGSA or PyBaMM code was copied. Data (`inputs.csv`, `outputs.csv`, `voltage.csv`): CC BY 4.0 (`LICENSE-DATA`). `US06.csv` is not under CC BY 4.0: it is an unchanged copy from the TDGSA repository (© 2024 TUM Chair of Electrical Energy Storage Technology, BSD-3-Clause, `US06_LICENSE.txt`, kept beside it), which took it from PyBaMM (BSD-3-Clause). It is hosted with the source credited and is listed in the exceptions of `LICENSE-DATA`. Model setup and parameter ranges: Zonta et al. (2026); simulator: PyBaMM (Sulzer et al., J. Open Res. Softw. 9(1):14, 2021, doi:10.5334/jors.309); base parameters: Marquis et al., J. Electrochem. Soc. 166(15):A3693–A3706, 2019, doi:10.1149/2.0341915jes. Runs, data and notebook: Claude (Anthropic) for P. Constantine, 2026.
