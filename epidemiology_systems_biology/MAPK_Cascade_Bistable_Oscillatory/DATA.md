# MAPK_Cascade_Bistable_Oscillatory: data

Kind: **Python model**. The model is `MAPK_model.py` (numpy/scipy), and `generate_data.py` writes the nine CSVs from it with fixed seeds; in the repo's `.venv` a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `MAPK_Cascade_Bistable_Oscillatory.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with `SEED = 570`; outputs committed |
| `MAPK_model.py` | `bistable(theta, x0)`, `oscillatory(theta)`, `solve`, `limit_cycle_features`; `NAMES`, nominal values, ranges, initial conditions, held settings | the model |
| `generate_data.py` | writes the CSVs; `--study`, `--rows`, `--check`, `--fd-check` | regenerates the data |
| `check_accuracy.py` | recomputes the Accuracy numbers that the notebook and `generate_data.py` do not print | check |
| `bistable_inputs.csv` | 512 × 7: k1, k2, k3, k4, k5, k6, alpha (physical units) | Study 1 inputs (the paper's bistable design; the notebook's headline) |
| `bistable_outputs.csv` | 512 × 4: x2_high, x3_high, x2_low, x3_low (nM) at t = 1800 s | Study 1 outputs |
| `bistable_gradients.csv` | 512 × 28: column 7 i + j is ∂(output i)/∂(input j), physical units | Study 1 gradients |
| `oscillatory_inputs.csv` | 512 × 7, as above | Study 2 inputs (the paper's oscillatory design) |
| `oscillatory_outputs.csv` | 512 × 4: mean_x3 (nM), amplitude (nM), period (min), oscillates (0/1) | Study 2 outputs |
| `oscillatory_gradients.csv` | 512 × 7: ∂ mean_x3/∂(input j), physical units | Study 2 gradients |
| `bistable_log_inputs.csv` | 512 × 7, physical units | Study 3 inputs (a log-uniform design inside the paper's box; this data set's own choice, secondary) |
| `bistable_log_outputs.csv` | 512 × 4, as Study 1 | Study 3 outputs |
| `bistable_log_gradients.csv` | 512 × 28, as Study 1 (physical; the notebook converts to ln θ) | Study 3 gradients |

The CSVs are comma-separated numbers (15 significant digits) after two `#` lines (a description, then the column names), so `np.loadtxt(path, delimiter=',')` reads them. Total size about 0.9 MB.

## Model

**Citation.** N. J. Linden, B. Kramer, P. Rangamani, "Bayesian parameter estimation for dynamical models in systems biology", *PLOS Computational Biology* 18(10): e1010651, 2022, doi:10.1371/journal.pcbi.1010651 (CC BY 4.0), with its S2 Table (supplementary PDF). The model is the simplified MAPK cascade of L. K. Nguyen, A. Degasperi, P. Cotter, B. N. Kholodenko, *Sci. Rep.* 5: 12569, 2015 (the paper's ref. 56), Eqs. 32a–c, p. 28:

    dx1/dt = k1 (S1t − x1) K1^n1 / (K1^n1 + x3^n1) − k2 x1
    dx2/dt = k3 (S2t − x2) x1 (1 + α x3^n2 / (K2^n2 + x3^n2)) − k4 x2
    dx3/dt = k5 (S3t − x3) x2 − k6 x3

with x1, x2, x3 the phosphorylated RAF, MEK and MAPK/ERK (nM; the paper's Fig. 4B axes and p. 31 text) and 14 parameters. Time is in seconds, as in the authors' code (`tend = 1800` for the paper's "30 min"; the period is divided by 60 to give minutes).

**Sources of the numbers.** The nominal values and ranges are in the paper's S2 Table and in the authors' code, github.com/RangamaniLabUCSD/CIUKF-MCMC, commit 11a4f1e: `MAPK/sensitivity_analysis_RUN.m` (`ptrueBistable`, `ptrueOSC`, `paramBounds`, `x0`, `tend`, `DT`, `startIdx = 50`), `tutorial/MAPK_cascade.m` (parameter order), `utils/limitCycleFeatures.m` (thresholds), and `MAPK/BISTABLE_*SS_RUN.m` (initial conditions). The S2 Table and the code give the same nominal values for both regimes and the same ranges for k2–k6 and α. They differ in the range of k1: the S2 Table gives [0, 0.05], the code's `paramBounds` [0, 0.1]. The data use the code's [0, 0.1]; the oscillatory nominal value k1 = 0.1 (S2 Table and code) lies outside the S2 Table's range, and 256 of the 512 Study 1 samples have k1 > 0.05 (`check_accuracy.py`). The S2 Table lists every rate constant in nM·s⁻¹, which does not fit Eqs. 32a–c (units below). The paper confirms the initial conditions (Fig. 4 caption), the k5 range for the oscillatory regime (S2 Table caption, p. 43) and the 30 min time (p. 28: "the values at t = 30 min"). The window of the limit-cycle features (980–5400 s) and the 20 s output grid are from the code only; the paper (Sec. 1.13, pp. 23–24) gives no window.

**Free parameters: 7, not 9.** The text (p. 28) fixes n1, n2 (exponents), K1, K2 (only locally identifiable) and S1t, S2t, S3t (cell-type totals) and then says this "narrowed the free parameters down to a set of 9 parameters, from a set of 14 originally". Fixing those 7 leaves 7: k1–k6 and α, which is also the Fig. 4 caption's list and the code's (`fixedParamIndex = [1,2,3,10,11,12,13]`). We use m = 7.

**Inputs.** The units are inferred from Eqs. 32a–c with time in seconds: k1, k2, k4, k6 in 1/s; k3, k5 in 1/(nM s); α dimensionless. (The S2 Table prints nM·s⁻¹ for all six rate constants.)

| name | bistable nominal | Study 1 range | Study 3 range | oscillatory nominal | Study 2 range | source |
|---|---|---|---|---|---|---|
| k1 | 0.0012 | U[0, 0.1] | logU[0.001, 0.1] | 0.1 | U[0, 0.1] | code `paramBounds`; the S2 Table gives [0, 0.05] |
| k2 | 0.006 | U[0, 0.1] | logU[0.001, 0.1] | 0.01 | U[0, 0.1] | code and S2 Table |
| k3 | 0.049 | U[0, 0.05] | logU[0.0005, 0.05] | 0.01 | U[0, 0.05] | same |
| k4 | 0.084 | U[0, 0.1] | logU[0.001, 0.1] | 0.01 | U[0, 0.1] | same |
| k5 | 0.043 | U[0, 0.05] | logU[0.0005, 0.05] | 0.01 | U[1e-5, 0.05] | same; S2 Table caption (k5 ≥ 1e-5, oscillatory) |
| k6 | 0.066 | U[0, 0.1] | logU[0.001, 0.1] | 0.01 | U[0, 0.1] | same |
| α | 95 | U[0, 100] | logU[1, 100] | 10 | U[0, 100] | same |

Held: bistable S1t = 0.22, S2t = 10, S3t = 53 nM, n1 = 5, K1 = 9.5 nM, n2 = 10, K2 = 15 nM; oscillatory S1t = S2t = S3t = 100 nM, n1 = 10, K1 = 1 nM, n2 = 15, K2 = 8 nM (S2 Table and code). Initial conditions: high steady state x0 = (0.1245, 2.4870, 31.2623), low steady state (0.0015, 3.6678, 28.7307) (Fig. 4 caption), oscillatory (10, 80, 80) (code).

**The three studies.** Studies 1 and 2 are the paper's designs: every free parameter uniform on its range ("All parameters except … are varied uniformly over the identified ranges (see S2 Table)", Fig. 4 caption, p. 27; "we allowed the biological model parameters to vary uniformly over the ranges listed in S2 Table", p. 28). Study 1 is the notebook's headline. Study 3 is this data set's own choice, not the paper's: ln θ uniform on [ln(U/100), ln U], U = the upper bound of the Study 1 range, so the box lies inside the paper's box and contains the bistable nominal point (k1* = 0.0012 ≥ 0.001). The paper uses uniform inputs throughout, also for its third model ("All free parameters were uniformly varied over two orders of magnitude centered around the nominal values, … θi ∼ U(0.1·θ*i, 10·θ*i)", p. 35). We add the log design because the paper's ranges start at 0, where derivatives with respect to a deactivation rate are very large (see Accuracy, "Heavy tails"), and a log scale removes that bound. A log design centred on the nominal values, [θ*/10, 10 θ*], would leave the paper's box: its upper ends are 9.8, 8.4, 8.6, 6.6 and 9.5 times the paper's upper bounds for k3, k4, k5, k6 and α. `generate_data.py` writes it if `LOWER_LOG, UPPER_LOG` are set to θ*/10, 10 θ*.

**Quantities of interest.**

- Studies 1 and 3: x2 and x3 at t = 1800 s from each of the two initial conditions (the paper's "steady state values … at t = 30 min", p. 28). States at a fixed finite time are smooth (C^∞) functions of the parameters; near the separatrix between the two basins they change steeply. The notebook's main QoI is x3_high.
- Study 2: `mean_x3`, the mean of x3 on the 20 s grid over 980 ≤ t ≤ 5400 s (MATLAB `startIdx = 50`), which the authors' script computes as a third QoI (`meanQOI`) but the paper does not show; smooth, but in oscillating samples it depends on how many periods fit in the window. `amplitude` and `period`, the paper's QoIs (Sec. 1.13, Eqs. 29–30), by the paper's rule with the code's thresholds: peak-height spread < 17, mean amplitude > 1 nM, amplitude spread < 5 nM, at least two maxima or two minima. For the peak heights the code (`utils/limitCycleFeatures.m`, lines 22–23: `peakDiff = range(lMax); if peakDiff < peakThreshold`, with `peakThreshold = 17` in `sensitivity_analysis_RUN.m`) tests the full spread, and so do the data (`MAPK_model.py`, `np.ptp(lmax) >= PEAK_THRESHOLD` discards); the paper's text says "A threshold on one half of this difference, called for the helpPeakThreshold, is set to 17.0" (p. 23). 0 means no limit cycle, as the code's `nullValue = 0`. The features are read on the code's 20 s output grid, which under-resolves the shortest cycles (Accuracy, "Output grid"). They are discontinuous at the edge of the oscillating region and piecewise constant in places (peak times on a 20 s grid), so no gradients are given for them. `oscillates` is 1 where amplitude > 0.

**Gradients.** Forward sensitivities: the states and the 21 sensitivities dx/dθ_j are solved together with `scipy.integrate.solve_ivp`, LSODA, rtol = atol = 1e-10 (bistable) and 1e-8 (oscillatory), with a block-diagonal Newton matrix kron(I8, ∂f/∂x) (it affects only the Newton iteration, not the error control). The gradient of `mean_x3` is the mean of the sensitivities on the window. Gradients are with respect to the physical parameters for all studies; Study 3 is analyzed in z = ln θ with ∂f/∂z = θ ∂f/∂θ, converted in one notebook cell.

**Sampling design.** One scrambled Sobol' sequence of 2^9 = 512 points per study (`scipy.stats.qmc.Sobol(d=7, scramble=True, seed=s)`, s = 570, 571, 572), mapped linearly to the ranges (to the log ranges for Study 3). The paper used Sobol' sequences in UQLab with 5,000 (bistable) and 15,000 (oscillatory) samples; its seed (`rng(100,'twister')` in the script) and UQLab's scrambling are not reproduced.

**Samples outside physical meaning.** None: no output is negative or non-finite; the states stay in [0, S_it] (the solver raises on a state below −1e-6). Samples at which the two initial conditions reach different x3 at 30 min (|difference| > 1e-3 relative): 43 of 512 in Study 1, 255 of 512 in Study 3. Samples with a detected limit cycle in Study 2: 171 of 512. All are kept.

**Published results.** The paper has no active subspace. It reports first- and total-order Sobol' indices as bar charts without values (Fig. 4C, bistable x2 and x3 from both initial conditions; Fig. 4D, amplitude and period) and selects k2, k4, k5, k6 (bistable) and k2, k4, k6, α (oscillatory). At the oscillatory nominal point it gives amplitude 26.12 and period 13.04 min (p. 30).

## Regenerating

```bash
python generate_data.py                                   # all three studies, writes here
python generate_data.py --study bistable                  # one study
python generate_data.py --out DIR --check                 # writes to DIR (not this directory) and compares
python generate_data.py --out DIR --rows 0:16 --check     # a subset of rows
python generate_data.py --fd-check 5                      # gradients vs. central differences
python check_accuracy.py                                  # the Accuracy numbers below
```

`--check` and `--rows` refuse to write into this directory. There is no randomness beyond the seeded designs; the run is serial. The shipped files were written on 2026-11-01 in a Linux container (Python 3.13.18, numpy 2.5.4, scipy 1.18.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-02), `python generate_data.py --out DIR --check` took 1030 s (17 min) of wall time while other jobs shared the machine (101 s Study 1, 850 s Study 2, 77 s Study 3), about 0.2 s per bistable sample (both initial conditions) and 1.7 s per oscillatory sample. It reproduced the files to these maximum relative differences (each row scaled by its largest entry): inputs 0 (Study 3: 3e-17, rounding of the log map); outputs 6.8e-11, 5.1e-7 and 2.5e-10 and gradients 2.8e-9, 1.8e-4 and 2.2e-9 for Studies 1, 2 and 3. The tolerance to expect is therefore 1e-8 relative for the bistable studies, 1e-6 for the oscillatory outputs and 1e-3 for the oscillatory gradients; the oscillatory differences are largest where x3 spikes, because the phase error of a 5400 s oscillation accumulates (see Accuracy). The inputs are reproduced exactly.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name (in `.venv`, 2026-11-02: 330 s in all, of which 217 s for the output-grid check that `--skip-slow` leaves out).

- **Outputs.** The notebook compares four rows per study with solves without sensitivities at rtol = atol = 1e-12: in `.venv` the largest relative difference is of order 1e-13 (Study 1), 1e-11 (Study 3) and 1e-8 (Study 2, written at 1e-8). For Study 2, a state-only solve at 1e-8 differs from the stored values by up to 3.5e-4 (`mean_x3`) and 6.1e-4 (amplitude) relative (`check_accuracy.py`, four rows; spiking oscillations, phase error accumulates): the stored values come from the coupled solve, whose error control includes the sensitivities and is tighter. Use rtol ≤ 1e-10 for state-only oscillatory solves.
- **Gradients.** `generate_data.py --fd-check 5` compares five rows per study (rows 0, 127, 255, 383, 511) with central differences (relative step 1e-5) of state-only solves at 1e-12; in `.venv` (116 s) the largest relative error (2-norm of the 7-vector) is 4.5e-7 for Study 1 (row 255, x3_low), 1.1e-5 for Study 3 (row 0) and 4.0e-5 for Study 2 (`mean_x3`, row 0). The notebook's four-row check gives 4.2e-7, 1.1e-5 and 1.0e-4. In Study 3 the error at row 0 is that of the differences themselves: relative steps of 1e-5, 1e-6 and 1e-7 give 1.1e-5, 1.1e-4 and 4.7e-4 (`check_accuracy.py`). The largest Study 3 gradient (row 306, 84 % of tr Ĉ) agrees with central differences to 4.8e-6 with a relative step of 1e-6 (`check_accuracy.py`); the step 1e-5 gives 1.0e-3, too coarse for its curvature.
- **Difference steps near the edge of the oscillating region.** At Study 2 row 312 (oscillating, amplitude 1.3 nM, near the Hopf edge) relative steps of 1e-4, 1e-5 and 1e-6 give relative errors of 0.17, 5.4e-3 and 4.4e-5 against the stored gradient of `mean_x3` (`check_accuracy.py`): use steps of 1e-6 or smaller there. Coupled solves at 1e-9 instead of 1e-8 change the oscillatory gradients of rows 0–15 by at most 3.1e-6 relative and the amplitudes by at most 4.1e-6 nM (`check_accuracy.py`).
- **Output grid.** The amplitude and period are read on the 20 s grid of the authors' code, which under-resolves short cycles (`check_accuracy.py`: one state-only solve per Study 2 sample at rtol = atol = 1e-10, read on the 20 s grid and on a 1 s grid). On the 20 s grid 171 samples oscillate, as in the files; on the 1 s grid 215 do. 44 samples with periods of 43–171 s (amplitudes 1.0–72.9 nM on the 1 s grid) oscillate only on the 1 s grid, and one (row 312, amplitude 1.3 nM) only on the 20 s grid. For the 22 samples that oscillate on both grids with a period under 2 min, the amplitude on the 1 s grid is 5 % to 35 % larger (median 14 %); for the 148 with longer periods the amplitude changes by a median 1.2 % (at most 7.1 %) and the period by at most 0.12 min. The data keep the 20 s grid, the authors' choice, for comparability with the paper's Fig. 4D (author, 2026-11-03); the set of "oscillating" samples, and so the analysis of Study 2's amplitude and period, depends on it.
- **Heavy tails, not errors.** In the paper's designs the five largest gradients carry 90 % (Study 1) and 66 % (Study 2) of tr Ĉ; three of Study 1's five and all of Study 2's five are samples with k6 or k4 within 0.7 % of the range above 0, where, for example, ∂x3/∂k6 = −S3t/(k5 x2) at steady state with k6 = 0 is large. They are correct derivatives (checked), and they make the gradient-based matrix a statement about that corner: without the 5 largest, λ1/λ2 falls from 10.37 to 1.59 (Study 1) and from 2.21 to 1.20 (Study 2), and w1 turns from the k6 axis to mostly k4 and to k2 (notebook). This is the caveat on the headline Study 1 result: on the paper's ranges, x3 at 30 min has no robust low-dimensional active subspace in the gradient sense. In the log design (Study 3) one interior sample near the separatrix (row 306) carries 84 % of tr Ĉ and the 5 largest 89 %; the ratio to quote there is λ1/λ2 ≈ 12 (12.8 without row 306, 10.5 without the 5 largest, 12.0 on rows 0–255; w1 stable, |w1ᵀw1′| ≥ 0.955); with all samples it is 74.4, only because of that one sample. See the notebook.

## How these files relate to the paper

- **Not the paper's computation.** The paper's Sobol' indices come from MATLAB `ode15s` and UQLab (5,000 and 15,000 samples). The data here are an independent Python computation with the same equations, nominal values, ranges (k1: the code's), initial conditions and QoI definitions, on smaller Sobol' designs; Study 3 is this data set's own addition. No outside code or data were copied (the code repository states no licence).
- **Nominal checks** (notebook). x3(30 min) = 17.92 nM (high) and 5.99 nM (low), as in Fig. 4B (about 18 and 6 nM). At the oscillatory nominal point the period on the data window is 13.04 min (paper 13.04). The amplitude depends on the window and on the reading of the minima:
  - Data window (980–5400 s, the authors' GSA window `startIdx = 50`), true minima: 26.96 nM; this is the shipped QoI. The window holds a transient first cycle (maximum at 32 min, amplitude 29.09 nM against 26.60, 26.39, 26.37, 26.36 for the next four), which passes the decay test (spread 2.73 < 5 nM) and raises the mean by 0.53 nM.
  - The code's `limitCycleFeatures` reads each minimum one grid point late (`lMin = trajectory(delay+minLocs)` with 1-based `minLocs` into `trajectory(delay:end)`): 26.72 nM on the data window.
  - Settled cycle (run to 20 000 s, cycles after 10 000 s, 20 s grid): 26.33 nM with true minima, **26.06 nM** with the late minima (period 12.99 min). This is 0.06 nM from the paper's 26.12 "true value" (p. 30), which most likely belongs to the settled cycle read the authors' way (not proven).
  - So the 0.90 nM between 26.96 and 26.06 is the transient first cycle (0.53), slower settling (0.10) and the off-by-one (0.27), not the model. `MAPK_model.limit_cycle_features(..., late_minima=True)` reproduces the authors' reading; the data use true minima (the paper's Sec. 1.13 definition) on the authors' GSA window.
- **Observations on the authors' code** (no effect on these files): the sensitivity script names the paper's high-steady-state initial condition `x0Low` and the low one `x0High` (the run scripts and Fig. 4 caption use the opposite, as here), so Fig. 4C's "low"/"high" panels may be swapped; `MAPK_Jacobian.m` uses x3^n1 where x3^n2 belongs (J21, J22), S1 for S2 (J23) and x3 for x2 (J33), and the parameter order of `MAPK.m` (K1, n1, K2, n2) while the script passes `MAPK_cascade.m`'s order (n1, K1, n2, K2). `ode15s` uses the Jacobian only in its Newton iteration, so this should affect cost, not results. The committed script sets `SobolSensOpts.Sobol.SampleSize = 5`, not the paper's 5,000/15,000.
- **Open questions (for the authors).** (1) The peak-height threshold: the code tests the full spread of the peak heights, the text (p. 23) one half of it; were the Fig. 4D indices computed with the code as committed? The data follow the code. (2) The range of k1: [0, 0.05] (S2 Table) or [0, 0.1] (`paramBounds`, used here), and the oscillatory nominal k1 = 0.1 outside the S2 range. (3) The units of the rate constants: the S2 Table's nM·s⁻¹ for all six, or the s⁻¹ and nM⁻¹ s⁻¹ that the equations imply (used here). (4) The "9 of 14" free parameters (p. 28), where the stated fixes leave 7.

## License and credit

Code (`MAPK_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they were generated here from the published equations. Model: Nguyen et al. (2015); study design, nominal values, ranges and QoIs: Linden, Kramer and Rangamani (2022), whose article and S2 Table are CC BY 4.0; the code repository github.com/RangamaniLabUCSD/CIUKF-MCMC states no licence and was used only as a source of numbers. Notebook, Python model and data: Claude (Anthropic) for P. Constantine, 2026. Cite N. J. Linden, B. Kramer and P. Rangamani, PLOS Comput. Biol. 18(10): e1010651, 2022, doi:10.1371/journal.pcbi.1010651.
