# HBV_SASK_Oldman_Streamflow: data

Kind: **Python model**. The model is `hbv_sask_model.py` (numpy only), a reimplementation of HBV-SASK written for this data set, and `generate_data.py` writes `inputs.csv`, `outputs.csv`, `gradients.csv` and `tt_tangent.csv` from it with a saved seed. The basin forcing (`oldman_daily.csv`, `oldman_monthly.csv`, `oldman_basin.csv`) is stored data from VARS-TOOL, converted losslessly by `convert_forcing.py`.

## Files

| file | contents | role |
|---|---|---|
| `HBV_SASK_Oldman_Streamflow.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 689`; outputs committed |
| `hbv_sask_model.py` | `qoi(theta, grad=False)`, `simulate(theta)`, `uh_weights(ubas)`; `NAMES`, `UNITS`, `LB`, `UB`, `TUTORIAL`, `QOI_NAMES`, `WINDOW`; reads the three forcing CSVs relative to its own path | the model |
| `generate_data.py` | writes the four data CSVs to `--out DIR` (required); `--check` (with `DIR` outside this directory) compares with the shipped ones | regenerates the data |
| `convert_forcing.py` | converts VARS-TOOL's `tutorials/Oldman Basin/*.inp` files into the three forcing CSVs (`--out DIR` required, outside this directory); `--check` compares | regenerates the forcing |
| `check_accuracy.py` | the accuracy numbers of this file that the notebook and `generate_data.py` do not print; checks 3–4 need a VARS-TOOL clone | check |
| `oldman_daily.csv` | 10958 × 6: year, month, day, P (mm/day), T (°C), observed Q (m³/s), 1979-01-01 to 2008-12-31 | stored forcing |
| `oldman_monthly.csv` | 12 × 3: month, long-term mean T (°C), long-term mean PET (mm/day) | stored forcing |
| `oldman_basin.csv` | 1 × 5: area (km²), initial SWE, SM, S1, S2 (mm) on 1979-01-01 | stored forcing |
| `inputs.csv` | 2000 × 12: TT, C0, ETF, LP, FC, beta, FRAC, K1, alpha, K2, UBAS, PM (physical units) | model inputs |
| `outputs.csv` | 2000 × 3: NSE (-), Qmean (m³/s), Qpeak (m³/s) | model output |
| `gradients.csv` | 2000 × 36: column 12 k + j is ∂(QoI k)/∂(parameter j), physical units; TT columns by central difference (below) | model output |
| `tt_tangent.csv` | 2000 × 3: the tangent ∂(QoI k)/∂TT, which omits the rain/snow jumps | model output, for the notebook's check |

The CSVs are comma-separated numbers after two `#` lines (a description and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. `gradients.csv` reshapes to (2000, 3, 12).

## Model

**Citation.** S. Razavi, R. Sheikholeslami, H. V. Gupta, and A. Haghnegahdar, "VARS-TOOL: A toolbox for comprehensive, efficient, and robust sensitivity and uncertainty analysis", *Environmental Modelling & Software* 112:95–107, 2019, [doi:10.1016/j.envsoft.2018.10.005](https://doi.org/10.1016/j.envsoft.2018.10.005) (open access; publisher PDF read). The paper presents VARS-TOOL and the HBV-SASK model shipped with it (Sect. 9, pp. 104–105; Fig. 9, p. 104; Table 2, p. 105), with ready-to-run set-ups for the Bow River at Banff (2178.53 km²) and the Oldman River at Waldron's Corner (1434.73 km², gauge 05AA023). For the Oldman basin it gives an average annual precipitation of 611 mm, an average annual streamflow of 11.7 m³/s and a runoff ratio of about 0.42 (p. 105); the shipped forcing reproduces all three (611 mm, 11.71 m³/s, 0.42; notebook, input cell).

**HBV-SASK.** "An interpretation of Hydrologiska Byråns Vattenbalansavdelning model (Lindström et al., 1997), developed by the first author for educational and research purposes" (p. 104): a lumped daily model with a snow store (rain/snow split at TT, degree-day melt C0·(T − TT)), a soil store (field capacity FC, release R = (RF + MS)(SM/FC)^β, actual evapotranspiration AET = PET·(SM/LP) below the limit LP·FC), a nonlinear fast reservoir (Q1 = K1·S1^α) receiving the fraction FRAC of R, a linear slow reservoir (Q2 = K2·S2) receiving the rest, and a triangular unit hydrograph of base UBAS days for Q1 (Fig. 9). Potential evapotranspiration is PET = (1 + ETF·(T − T_mth))·PET_mth with long-term monthly means T_mth, PET_mth (p. 105). Fig. 9 prints these equations without caps or update order. The rest of the model — the caps (melt ≤ SWE; AET ≤ SM and the ratio SM/(LP·FC) capped at 1; SM/FC capped at 1 in R; Q1 ≤ S1; PET ≥ 0), the updates within a day from start-of-day states (snow and melt first, then AET, R, Q1 and Q2 from the start-of-day SM, S1, S2), the precipitation multiplier applied to P before the snow module, and the unit-hydrograph weights (the area of a unit-area symmetric triangle on [0, UBAS] over each day; at most 3 nonzero weights for UBAS ≤ 3) — was determined by running the module functions of the VARS-TOOL Python implementation (`tutorials/hbv.py`, GPL-3.0) as a black box on test inputs, and checked against complete runs of that implementation (Accuracy). `hbv_sask_model.py` was written fresh from the paper and those observations, vectorized over samples, with its own forward-mode derivative recursion; no VARS-TOOL code or comments were copied. One behaviour of that implementation is omitted: `hbv.py` sets simulated flows below 1e-5 m³/s to 1e-5 m³/s, and the port has no such floor. At the 2000 samples of this data set one sample has a daily flow below 1e-5 m³/s (minimum 5.1e-6 m³/s), and applying the floor would change its NSE by 1.2e-10 and its Qmean by 8.5e-9 m³/s (`check_accuracy.py`, check 5). Under pandas 3 (Copy-on-Write) that line of `hbv.py` is a chained assignment that has no effect, so check 4 compares the port with an original run in which the floor did not act either. The equations are written out in the notebook and in the module docstring.

**Forcing and run.** The VARS-TOOL Oldman set-up: daily basin precipitation and temperature and observed streamflow for 1979-01-01 to 2008-12-31 (10958 days), the 12 long-term monthly means, and initial states SWE = 0, SM = 5, S1 = 3, S2 = 10 mm (`tutorials/Oldman Basin/initial_condition.inp`). The paper does not describe the data source beyond the period (p. 105). Every run covers the whole record.

**Inputs** (m = 12). Independent uniform on the bounds of Table 2 (p. 105). The paper gives bounds only; uniform sampling on them is this data set's choice (it is also what the VARS-TOOL tutorials do). The bounds are the same in VARS-TOOL's `tutorials/Oldman Basin/par_bounds.inp`.

| # | name | description (Table 2) | units | range | source |
|---|---|---|---|---|---|
| 1 | TT | air temperature threshold for melting/freezing and rain/snow | °C | U[−4, 4] | Table 2, p. 105 |
| 2 | C0 | base melt factor | mm/°C/day | U[0, 10] | Table 2, p. 105 |
| 3 | ETF | temperature anomaly correction of PET | 1/°C | U[0, 1] | Table 2, p. 105 |
| 4 | LP | limit for PET as a multiplier of FC | - | U[0, 1] | Table 2, p. 105 |
| 5 | FC | field capacity of the soil | mm | U[50, 500] | Table 2, p. 105 |
| 6 | beta | exponent of the soil release equation | - | U[1, 3] | Table 2, p. 105 |
| 7 | FRAC | fraction of soil release entering the fast reservoir | - | U[0.1, 0.9] | Table 2, p. 105 |
| 8 | K1 | fast reservoir coefficient | 1/day (Table 2: unitless proportion per day) | U[0.05, 1] | Table 2, p. 105 |
| 9 | alpha | exponent of the fast reservoir equation | - | U[1, 3] | Table 2, p. 105 |
| 10 | K2 | slow reservoir coefficient | 1/day (as K1) | U[0, 0.05] | Table 2, p. 105 |
| 11 | UBAS | base of the unit hydrograph | day | U[1, 3] | Table 2, p. 105 |
| 12 | PM | precipitation multiplier | - | U[0.5, 2] | Table 2, p. 105 |

No nominal point is given in the paper (Table 2 calls UBAS = 1 and PM = 1 defaults). `TUTORIAL` in the module is the parameter set of VARS-TOOL's `tutorials/Oldman Basin/pars.inp` (TT 0, C0 1, ETF 0.1, LP 0.3, FC 250, beta 2, FRAC 0.7, K1 0.05, alpha 1.5, K2 0.01, UBAS 1, PM 1), used only for illustration.

**Quantities of interest** (over 1982-01-01 to 1996-12-31, 5479 days; the window and NSE definition of the VARS-TOOL tutorials "2 - SA of a real-world model" and "9 - SA with given-data"; the paper defines no scalar output for the Oldman case):

- `NSE`: Nash-Sutcliffe efficiency of simulated vs observed daily flow (m³/s), dimensionless, ≤ 1 (primary).
- `Qmean`: mean simulated daily flow, m³/s (this data set's choice).
- `Qpeak`: mean over the 15 calendar years of the annual maximum daily simulated flow, m³/s (this data set's choice).

**Gradients and their coordinates.** `gradients.csv` holds gradients with respect to the physical parameters (the notebook passes them to `ac.Uniform`, which maps to [−1, 1]). For the 11 parameters other than TT they are forward (tangent) sensitivities of the daily recursion, exact wherever the derivative exists; the min/max caps make the outputs only piecewise smooth (kinks), which matters on a set of measure zero. TT is different: when TT crosses the temperature of a day with precipitation, that day's precipitation changes between rain and snow, and every output jumps. The temperatures are recorded to 0.01 °C, so the outputs are piecewise smooth in TT with jumps wherever TT passes a recorded wet-day temperature, and the tangent misses them. The TT column of `gradients.csv` is therefore the central difference TT ± 0.1 °C, (f(TT + 0.1) − f(TT − 0.1))/0.2, over an interval 0.2 °C wide that contains many recorded wet-day temperatures (the other 11 parameters held fixed); this is the data set's choice. The tangent TT derivative is stored separately in `tt_tangent.csv`. At single samples the difference depends on the half-width (`check_accuracy.py`, check 2, in Accuracy). The notebook shows that the choice barely moves the NSE subspace (|w1 · w1(tangent)| = 0.9967), but that the two-dimensional Qpeak subspace moves by 0.157 when the tangent replaces the difference: Qpeak's n = 2 subspace depends on how the TT derivative is defined.

**Sampling design.** M = 2000 independent uniform (Monte Carlo) samples on the Table 2 box, numpy `default_rng(689)`. All samples are inside the box and every run completes; there are no samples outside the model's meaning (a negative NSE is a poor fit, at 1381 of the 2000 samples).

**Published results.** The paper does not compute active subspaces, and it reports no sensitivity result for the Oldman case. Its only HBV-SASK sensitivity result is Fig. 6 (p. 102): a factor-grouping dendrogram for the Bow River at Banff based on the mean squared error of log-transformed flows, sorting the parameters from most to least influential as K2, PM, TT, ETF, FRAC, LP, beta, FC, K1, alpha, C0, UBAS. The notebook compares with it, with Sobol' total effects computed with the model, and with the VARS-TO values printed in VARS-TOOL's tutorial 2 (same output and window; not a result of the paper).

## Regenerating

```bash
python generate_data.py --out /tmp/hbv --check     # about 1 minute, one process
python convert_forcing.py --src "vars-tool/tutorials/Oldman Basin" --out /tmp/hbv_forcing --check
```

`generate_data.py --check` reproduces all four files to the tolerance 1e-9 (maximum difference divided by the column maximum). The shipped files were written in a Linux container (Python 3.11.18), where a rerun agreed with them below the printed resolution. In the repository's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-10) the check prints `check OK`: `inputs.csv` is reproduced with difference 0, `outputs.csv` and `tt_tangent.csv` to rounding level (of order 1e-16), and `gradients.csv` to an order of 1e-12. Run time in `.venv`: about 50 s (one process, other jobs running), of which about 47 s are the model runs. It also prints the tangent check against central differences (below). `convert_forcing.py` needs a clone of <https://github.com/vars-tool/vars-tool> (commit df72ad2a3b773ca9f0fc285a0b7d1bc9e043a86d, 2022-08-16, was used); its check tolerance is 1e-12 absolute, and the conversion is lossless (P and T have two decimals in the source and are written with two; Q has at most three and is written with three). The source files and their sha256: `Precipitation_Temperature.inp` 4c79a91ff60581aae304da9b7b755a91bb3b9ab8023a1480e72a02a6d9814b64, `streamflow.inp` f4f0da403f0e8a6a930c66e491164d30ad9d923978a71b95349029189d115966, `monthly_data.inp` 4b6119a2b641eb063dcbac5921f96dfe949aaf0792ed2a2382dae66d56f1e175, `initial_condition.inp` 5549b41433b08f742723a9f5db5b569a822bfb42b76fa0902d86bd1c82391867 (the `tutorials/Oldman Basin` and `src/varstool/example_models/oldman` copies are identical).

## Accuracy

| check | result | printed by |
|---|---|---|
| port vs VARS-TOOL's stored sample `tutorials/9b - sample_HBV.csv` (400 uniform random parameter sets with their NSE over 1982–1996; NSE from −14.93 to 0.6335) | max absolute difference 6.6e-14, max relative 3.7e-13 | `check_accuracy.py --vars-tool DIR`, check 3 |
| port vs the VARS-TOOL Python model (`tutorials/hbv.py`, run as an outside tool with pandas 3.1.1), daily Q for all 10958 days at the tutorial point and 4 uniform samples | max difference / max Q of order 1e-15 (9.8e-16 in this run) | `check_accuracy.py --vars-tool DIR`, check 4 |
| port at the tutorial point vs the tutorial's printed NSE (0.5771895069928394) | NSE = 0.577190 | notebook, input cell |
| tangent gradients vs central differences (step 1e-7 of each range, all 12 parameters including the TT tangent), first 20 samples, row 2-norm in [−1, 1] coordinates | max relative error of order 1e-8 for NSE and Qmean and 1e-9 for Qpeak (3.0e-8, 4.1e-8, 1.9e-9 in `.venv`) | `generate_data.py` |
| share of the change in each output over TT ± 0.25 °C carried by the rain/snow jumps (2001-point TT grid, 8 uniform samples) | median NSE 0.26, Qmean 0.13, Qpeak 0.32 (ranges 0.04–0.92, 0.05–0.82, 0.13–0.71) | `check_accuracy.py`, check 1 |
| TT difference step: half-widths 0.05 and 0.2 °C vs 0.1 °C (TT ± h), |difference| × 4 / gradient 2-norm in [−1, 1] coordinates, 200 uniform samples | NSE median 0.029 and 0.023 (90 % 0.097, 0.084); Qmean 0.001 and 0.001; Qpeak 0.064 and 0.068 (90 % 0.236, 0.239) | `check_accuracy.py`, check 2 |
| TT share of the normalized gradient, tangent vs 0.1 °C difference (median, 200 samples) | NSE 0.080 vs 0.101; Qmean 0.005 vs 0.008; Qpeak 0.178 vs 0.161 | `check_accuracy.py`, check 2 |
| VARS-TOOL's flow floor (1e-5 m³/s), omitted in the port, at the 2000 shipped samples | 1 sample has a daily flow below it (min 5.1e-6 m³/s); with the floor its NSE changes by 1.2e-10, Qmean by 8.5e-9 m³/s; Qpeak unchanged (printed 0.0e+00) | `check_accuracy.py`, check 5 |

Checks 1, 2 and 5 need only numpy; checks 3–4 need a clone of VARS-TOOL (`git clone https://github.com/vars-tool/vars-tool`, then `python check_accuracy.py --vars-tool vars-tool/tutorials`), and check 4 also pandas, scipy, matplotlib and IPython, which VARS-TOOL's `hbv.py` imports. pandas 3 removed the `read_csv` keyword `delim_whitespace` (deprecated in 2.2) that `hbv.py` uses; `check_accuracy.py` maps it to `sep=r'\s+'` for that import only. The numbers of checks 3–4 are from a run of `check_accuracy.py --vars-tool` in a Linux container with Python 3.13.18, numpy 2.5.4 and pandas 3.1.1; VARS-TOOL and pandas are not in the repository's `.venv`, so these two checks are not rerun there. Checks 1, 2 and 5 were rerun in `.venv` (Python 3.11.16, numpy 2.4.6; 2026-11-10) and print the numbers given above; that run took about 3.5 minutes (other jobs running), mostly check 1. Without `--vars-tool` the script runs checks 1, 2 and 5 and exits with an error that says how to get VARS-TOOL.

## How these files relate to the paper

The model, the Oldman forcing and the parameter ranges are those of the paper's HBV-SASK case study as shipped in VARS-TOOL. The quantities of interest, the window 1982–1996 (VARS-TOOL tutorials, not the paper), the uniform Monte Carlo design, the sample size and the gradient treatment of TT are this data set's choices. The paper's own HBV-SASK sensitivity result (Fig. 6) is for the other basin and another metric; the notebook compares with it qualitatively and states why the orders differ.

**Related studies with the same model.** Two later papers use HBV-SASK with the Oldman forcing and are covered by this data set rather than given their own: S. Razavi and H. V. Gupta, "A multi-method Generalized Global Sensitivity Matrix approach to accounting for the dynamical nature of earth and environmental systems models", *Environmental Modelling & Software* 114:1–11, 2019, doi:10.1016/j.envsoft.2018.12.002; and I. Jovanović Buha, T. Neckel, F. Künzner, and H.-J. Bungartz, "UQEF-Dynamic: A scalable open-source tool for forward uncertainty quantification and global sensitivity analysis of time-varying model outputs", *Environmental Modelling & Software* 204:107088, 2026, doi:10.1016/j.envsoft.2026.107088. Neither was read on the page for this data set, and their results are not compared here. The UQEF-Dynamic entry was checked on Crossref; the Razavi & Gupta (2019) entry has not been checked on Crossref.

## License and credit

`hbv_sask_model.py`, `generate_data.py`, `convert_forcing.py`, `check_accuracy.py` and the notebook: MIT (repository code licence). `inputs.csv`, `outputs.csv`, `gradients.csv`, `tt_tangent.csv`: CC BY 4.0 (made here with `hbv_sask_model.py`). `oldman_daily.csv`, `oldman_monthly.csv`, `oldman_basin.csv`: converted from the VARS-TOOL repository (<https://github.com/vars-tool/vars-tool>, © Saman Razavi and contributors), which is distributed under GPL-3.0; no separate licence is stated for its data files, and the original sources of the forcing and of the gauge record are not named there. They are hosted here with credit and listed in the repository's `LICENSE-DATA` exceptions (they come from a GPL-3.0 repository with no separate data licence). If they are not shipped, the data set can be rebuilt with `convert_forcing.py` from a VARS-TOOL clone (Regenerating). A lead on the licence of the observed flow, not verified here: gauge 05AA023 is a Water Survey of Canada station, so if the observed flow (column 6 of `oldman_daily.csv`) is HYDAT data, its own licence (an open-government licence, to be checked) may be the relevant one for that column. Credit: Razavi et al. (2019), doi:10.1016/j.envsoft.2018.10.005. No VARS-TOOL code is included in this directory.

## Unknowns

- The original data sources of the Oldman forcing (station or gridded product behind the basin-average P and T, and how PET_mth was computed) are not stated in the paper or the repository.
- The update order and caps of HBV-SASK are not in the paper; they are those of the VARS-TOOL Python implementation (commit df72ad2a). The MATLAB HBV-SASK of the 2019 VARS-TOOL release was not compared; it may differ in such details.
- The window 1982–1996 is the VARS-TOOL tutorials' choice; other windows (for example the whole record after spin-up) were not studied. The 3-year spin-up from fixed initial states is short against the slow reservoir's time constant 1/K2 (20 days to unbounded on K2 ∈ [0, 0.05]), so outputs at small K2 carry a start-up transient; this is part of why ∂Qmean/∂K2 is so large near K2 = 0.
- Uniform independent sampling on the Table 2 bounds is assumed; the paper gives bounds only.

## Open questions

- Which HBV-SASK version (MATLAB or Python, and which release) produced Fig. 6, and with which spin-up and record period for the Bow River?
- What are the original sources of the Oldman P/T forcing and of the observed flow, and under what licence can they be redistributed (for the flow, possibly the Water Survey of Canada's HYDAT licence)?
