# Karst_Aquifer_LuKARS_Calibration: data

Kind: **Python model**. The model is `lukars_model.py`, a numpy port of the authors' LuKARS R script. `generate_data.py` writes every data file from it with saved seeds. The forcing files (`forcing.csv`, `monthly_mean_temp.csv`) are measured data from the authors' HydroShare resource, converted losslessly.

## Files

| file | contents | role |
|---|---|---|
| `Karst_Aquifer_LuKARS_Calibration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2019`; outputs committed |
| `lukars_model.py` | `to_physical`, `to_calibration` (paper Eqs. 28–33), `forcing`, `simulate`, `discharge`, `misfit`, `misfit_and_grad`, `observed`; `NAMES`, `CAL_NAMES`, `LB`, `UB` | the model |
| `generate_data.py` | writes the files below; `--check` recomputes them and compares; `--forcing DIR` converts the HydroShare forcing; `fd_gradients`, `misfit_rounded`, `round_like_percent_f` (imported by the notebook) | regenerates the data |
| `forcing.csv` | 5844 days (2001-01-01 to 2016-12-31): day, precipitation (mm/day), spring discharge (m³/day), air temperature at Mitterlug (°C) | model input (measured) |
| `monthly_mean_temp.csv` | 192 months 2001–2016: year, month, mean air temperature (°C) | model input (Thornthwaite ET) |
| `inputs.csv` | 1000 × 21 physical parameters, Table 1 order and units (`NAMES`) | sampled inputs |
| `outputs.csv` | 1000 misfits f_d (dimensionless) | model output |
| `gradients.csv` | 1000 × 21 exact ∂f_d/∂x, with x the normalized calibration parameters of Eq. (33) (`CAL_NAMES`) | model output |
| `gradients_fd.csv` | 1000 × 21 central finite differences of f_d in x, untruncated parameters, step h = 1e-4 (the step in the authors' 2020 `model.py`) | for comparison with the paper only |
| `gradients_fd_rounded.csv` | 1000 × 21 central finite differences, h = 1e-5, with the physical parameters rounded to 6 decimals before each run (emulating the `%f` interface of the authors' 2020 `model.py`), Table 1 bounds | for comparison with the paper only; not gradients of the model |
| `gradients_fd_rounded_lb70.csv` | the same with the e_min,3 lower bound of the authors' settings file (70 mm) in the transform | for comparison with the paper only; not gradients of the model |
| `validation.csv` | 5000 × 22: physical parameters and f_d of an independent Monte Carlo set | surrogate validation |
| `discharge.npz` | `discharge` (1000 × 1096, l/s, float32): simulated daily discharge 2006-01-01 to 2008-12-31 for the rows of `inputs.csv`; `observed` (1096, l/s); `days`; `description` | model output (the discharge series) |

The CSVs are comma-separated numbers after two `#` lines (a description, then the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are stored in physical units. The notebook converts them once to x with `lukars_model.to_calibration`, the exact inverse of the paper's transform (round-trip error 4e-15). The gradients are with respect to x, because only in x is the prior a product of independent uniforms.

## Model

**Citation.** M. Teixeira Parente, D. Bittner, S. A. Mattis, G. Chiogna, B. Wohlmuth, "Bayesian Calibration and Sensitivity Analysis for a Karst Aquifer Model Using Active Subspaces", *Water Resources Research* 55(8):7086–7107, 2019, doi:10.1029/2019WR024739 (preprint arXiv:1901.03283). The publisher PDF (CC BY-NC-ND) was read.

**LuKARS** (Bittner et al. 2018; paper Sec. 2.2 and Appendix A, Eqs. A1–A8). It is a lumped model of the Kerschbaum spring recharge area (2.5 km², Waidhofen a.d. Ybbs, Austria) with three calibrated hydrotopes: Hyd 1 (13 % of the area), Hyd 2 (56 %) and Hyd 3 (27 %). A dolomite quarry (4 %) has no quickflow and no recharge, so it never reaches the spring and is omitted. Each hydrotope is a bucket with quickflow Q_hyd (a hysteresis switch: on at e ≥ e_max, off below e_min), recharge Q_is to a shared linear baseflow store, and secondary-spring loss Q_sec. Spring discharge is the sum of Q_hyd over the hydrotopes plus Q_b. The source term is precipitation minus interception (DVWK 1996 beech values, capped at 5 mm/day), with degree-day snow (factor 4, threshold 0.5 °C), minus Thornthwaite evapotranspiration. The time step is daily, with spin-up from 2001-01-01. Fixed settings from the script: k_b = 0.00043 day⁻¹, l_hyd = 1600, 900, 960 m, initial e_b = 2900 mm, all other states 0.

The port follows `main_exe.R` line by line, including three quirks of that script that are kept on purpose:
- The heat index is summed as `rowSums(matrix(h, nrow = 16))`, which sums every 16th month rather than each year's 12 months.
- On a cold day the snow store adds the *next* day's precipitation.
- The flows of day n+1 are computed from the storages after day n's update, so the window value for 2006-01-01 is the state after the forcing of 2005-12-31.

The baseflow store is not clipped at 0 in the script (Eq. A4 has a max); it stays positive here.

**Inputs (m = 21).** Table 1, p. 7090:

| No. | name | units (Table 1) | range | calibration parameter (Eq. 33) |
|---|---|---|---|---|
| 1 | khyd_1 | m²/day | [9, 900] | ln khyd_1, uniform |
| 2 | emin_1 | mm | [10, 50] | emin_1, uniform |
| 3 | emax_1 | mm | [15, 75] | Δe_1 = emax_1 − emin_1 ∈ [5, 25] (Eqs. 31–32) |
| 4 | alpha_1 | – | [0.7, 1.6] | alpha_1 |
| 5 | kis_1 | m mm⁻¹ day | [0.002, 0.2] | ln kis_1 |
| 6 | ksec_1 | m mm⁻¹ day | [0.0095, 0.95] | ln ksec_1 |
| 7 | esec_1 | mm | [25, 70] | esec_1 |
| 8–14 | khyd_2, emin_2, emax_2, alpha_2, kis_2, ksec_2, esec_2 | as above | [8.5, 850], [40, 80], [80, 160], [0.5, 1.3], [0.00055, 0.055], [0.0023, 0.23], [130, 220] | fractions △p_(1,2) ∈ [0, 1] (Eqs. 29–30); Δe_2 ∈ [40, 80] |
| 15–21 | khyd_3, …, esec_3 | as above | [7.7, 770], [75, 120], [160, 255], [0.2, 0.7], [0.00025, 0.025], [0.0015, 0.15], [320, 450] | fractions △p_(2,3) ∈ [0, 1]; Δe_3 ∈ [85, 135] |

The ordering constraints of Eq. (1) are built into the transform. k_hyd, alpha, k_is and k_sec decrease from Hyd 1 to Hyd 3: p_i = lb_i + △·(min(ub_i, p_{i−1}) − lb_i), on a log scale for the k's. e_min and e_sec increase: p_i = max(p_{i−1}, lb_i) + △·(ub_i − max(…)). e_max_i = e_min_i + Δe_i. Every calibration parameter is scaled to [−1, 1], and the prior is x ~ U[−1, 1]²¹ (Sec. 4), so the physical prior is not uniform. All 1000 + 5000 samples lie inside the Table 1 bounds and satisfy Eq. (1) (counted in the notebook: 0 violations).

**Quantity of interest.** The Bayesian data misfit f_d(x) = ½ Σ_t ((d_t − G_t(x))/(0.05 d_t))² (Eq. 20; Γ_ij = (0.05 d_i)² δ_ij, Sec. 4). It runs over the 1096 days 2006-01-01 to 2008-12-31 (the script's `cal_p <- c(1827, 2922)`), where d is the measured and G the simulated discharge in l/s. On the shipped samples f_d ranges from 185.8 to 1.282 × 10⁵ (median 1.43 × 10⁴).

**Gradients.** Exact: a forward-mode (tangent-linear) derivative of the time stepping with respect to the 21 physical parameters, chained through the analytic Jacobian of the transform. This is the derivative on each branch of the model's switches (hysteresis, e ≥ e_sec, e ≥ 0, the min/max in the transform). f_d is discontinuous where a parameter change moves a quickflow switch by one day, so the exact gradient is the pathwise derivative, the derivative between jumps. Its C = E[∇f_pw ∇f_pwᵀ] is the matrix reported. With jumps, a finite-difference matrix C_h gains a diagonal term of order Σ_j ρ_j E[J²]/(2h) (ρ_j the jump density along x_j), which grows like 1/h and is sampled as rare outliers; a C that includes the jumps is infinite. Because the jumps are small (below), the pathwise C is the meaningful one. The Poincaré-type error bound for active subspaces assumes a differentiable f, so it does not strictly apply. The paper used "central finite differences" (p. 7097), with no step stated. `gradients_fd.csv` gives untruncated central differences at h = 1e-4 in x, the step in the authors' 2020 `model.py` (which also computes (G(x−h) − G(x+h))/(2h), a sign flip that does not affect C). The two `gradients_fd_rounded*.csv` files emulate the authors' parameter interface (see "How these files relate to the paper").

**Sampling design.** Monte Carlo: x ~ U[−1, 1]²¹ from `numpy.random.default_rng(2019)`, N = 1000 (the paper's N). The validation set is 5000 Monte Carlo samples from `default_rng(2020)`. The paper's seed is not reported.

**Published results.**
- Fig. 2: eigenvalues with a min–max bootstrap band. Read off the figure: λ₁ ≈ 1.2e10 (band ≈ 3.5e9 to 3.5e10), λ₂ ≈ 2.3e9, λ₃ ≈ 1.7e9, λ₄ ≈ 1.4e9, λ₅ ≈ 5e8; the text reports gaps after λ₁ and λ₄. The gradients came from 1000 × (21 × 2 + 1) runs of central differences (p. 7097).
- Fig. 3a–d: w₁ dominated by No. 12 (△k_is,(1,2), ≈ 0.95), then No. 5 (≈ 0.27).
- Fig. 3e: sensitivities s = (W∘W)λ (Eq. 17) with No. 12 the largest, then 5 and 19; max/min ratio 3.6 × 10⁶.
- A 4-D quartic surrogate with r² ≈ 0.77 (p. 7098).

**Results here** (notebook):
- λ₁ = 1.410e9, λ₁/λ₂ = 1.85, λ₂/λ₃ = 6.73, λ₄/λ₅ = 1.18; the gap rule chooses n = 2.
- The subspace is spanned by the k_hyd and k_is inputs of Hyd 1–2 (Nos. 1, 8, 5, 12).
- Quartic surrogates: 2-D validation R² = 0.787, 4-D validation R² = 0.851.
- Central differences at h = 1e-5 through 6-decimal parameter rounding: λ₁ = 1.055e10, λ₁/λ₂ = 5.21, λ₄/λ₅ = 2.24, w₁ led by Nos. 12, 5, 19 (70 mm bound); λ₁ = 7.269e9, λ₁/λ₂ = 3.8, λ₄/λ₅ = 2.25 (75 mm bound).

## Regenerating

```bash
python generate_data.py           # all files, about 140 s
python generate_data.py --check   # recompute and compare with the files here, about 140 s
python generate_data.py --forcing <HydroShare>/as_dwt/examples/as_wavelets/scens/scen_main   # once
```

The run that wrote the shipped files (2026-10-31; Python 3.13.18, numpy 2.5.4, 2-core Linux container) took 140.1 s: 10.7 s for the 1000 misfits with exact gradients and discharge series, 8.6 s for the 5000 validation misfits, 39.9 s for the untruncated finite-difference gradients (42 runs per sample) and 81.0 s for the two rounded finite-difference sets. (Adding the rounded sets left every earlier file byte-identical except the reworded header line of `gradients_fd.csv`.) The model is vectorized over samples, at about 2 ms per run in batches and 0.19 s for a single run. `--check` (139 s) reproduces every file exactly (all maximum differences 0; round trip 4.4e-15).

## Accuracy

Checked with scratch scripts (not committed):

- **Port vs. the authors' R script.** `main_exe.R` from HydroShare ran under R 4.3.3 with dplyr, with only `setwd`, the output path, 17-digit output, and the unused `library(hydroGOF)` line changed. Its discharge for 5 parameter vectors (x = 0 and 4 random) agrees with `lukars_model.simulate` to a maximum relative difference of 2.8e-16.
- **Transform vs. the authors' `env.py`.** With their lower bound e_min,3 = 70 mm, `to_physical` agrees with `fromCalibrationToPhysicalParameters` to 1.9e-15 on 2000 samples. The shipped files use the paper's Table 1 value, 75 mm (see open questions); with it only e_min,3 and e_max,3 differ (by up to 6.7 % and 3.1 %). The analytic Jacobian of the transform agrees with central differences (h = 1e-7) to 1.4e-9.
- **Gradients vs. finite differences**, as the relative error of the 21-vector per sample:
  - h = 1e-6 on all 1000 samples: median 8.9e-9, 99th percentile 7.6e-8, maximum 3.3e-5.
  - h = 1e-7 on 20 samples: maximum 1.9e-7.
  - Larger steps sometimes straddle a jump of f_d. At h = 1e-4 (`gradients_fd.csv`), 13 of 1000 samples have an error above 0.1, and the largest is 29 (row 338). Row 338 alone causes the whole change in the FD eigenvalues: its x₁₂ stencil straddles a jump of f_d of −137.4 (0.43 % of f_d = 3.232e4; a fine scan locates it within 1e-10 in x₁₂, where the smooth change over that interval is 5e-7), it carries 27 % of trace(C_FD), and without it the FD eigenvalues λ₁–λ₃ equal the exact ones to within 0.6 %. The other 12 samples have no visible effect on the spectrum. At h = 1e-5, the 99th percentile is 2.6e-4 and the maximum is 2.5.
  - On 300 fresh samples (`default_rng(7)`, from the independent review's scratch files, exact gradients recomputed here identically), untruncated FD gives λ₁ = 2.006e9 (h = 1e-4), 1.956e9 (1e-3), 2.003e9 (1e-5) and 2.003e9 (1e-6), against the exact 2.003e9. Jump straddling is thus a rare, one-sample effect.
  - The review's 90 one-dimensional scans (15 base points × Nos. 1, 5, 8, 12, 13, 19, spacing 5e-5) found jumps along every one of these coordinates, with median |J| between 5e-5 and 1.1 and a maximum of 14.7, on f_d of 1e3–4e4. Their estimated contribution to C_jj at h = 1e-4 is at most 1.4e6, against λ₁ = 1.4e9.
- **Rounded finite differences.** `generate_data.round_like_percent_f` applies Python's `'%f'` and reads the value back, which equals `np.round(P, 6)` on all shipped inputs. The notebook recomputes rows 0–99 of `gradients_fd_rounded.csv` and rows 0–19 of `gradients_fd_rounded_lb70.csv` exactly. The independent review's arrays for the 70 mm case (h = 1e-5) equal ours exactly.
- The notebook's check cell re-runs five samples: the misfits are identical and the gradients agree to 3.1e-16.
- `discharge.npz` stores the series as float32 (relative rounding ≤ 6e-8) to keep the file at 3.8 MB; the misfits and gradients are computed in float64.

## How these files relate to the paper

- **The same model, inputs, prior, window and noise model; not the paper's samples.** The paper's code and data (https://bitbucket.org/m_parente/2019_karst_unmix_data/) could not be reached from this session. The model code is from the HydroShare resource of the follow-up study (Bittner et al. 2021), which provides the LuKARS R script, the Python AS code and the forcing files, and no stored samples.
- **The spectrum differs, and the likely reason is the gradient approximation (a hypothesis, not confirmed).** The paper's λ₁ (read off Fig. 2, ≈ 1.2e10) is about 8 times the exact-gradient value here (1.410e9), its gaps are after λ₁ and λ₄ rather than after λ₂, and its w₁ is led by No. 12 rather than shared by the k_hyd and k_is of Hyd 1–2.
  - Jump straddling does not explain this: it changes the h = 1e-4 FD spectrum here only through one sample (row 338, see Accuracy), and not at all on fresh samples.
  - The authors' 2020 interface, `as_dwt/examples/as_wavelets/model.py` in the HydroShare resource (line 35), passes the physical parameters to `Rscript` with the format `%f`, which rounds them to 6 decimals; `main_exe.R` reads them with `as.numeric` (line 3). The k_is are the smallest parameters (k_is,3 ≥ 2.5e-4). At h = 1e-4 the step moves the parameter by less than 1e-6 for 16 %, 68 % and 94 % of samples in Nos. 5, 12 and 19, and at h = 1e-5 for 72 %, 97 % and 100 %.
  - Emulating that rounding with h = 1e-5 on the shipped samples largely reproduces Fig. 2 and Figs. 3a, 3e: with the settings-file bound (70 mm), λ₁ = 1.055e10, λ₁/λ₂ = 5.21, λ₄/λ₅ = 2.24, bootstrap min–max of λ₁ 3.67e9 to 2.59e10, w₁ led by Nos. 12 (+0.74), 5 (+0.62), 19 (+0.27), sensitivities ordered 12, 5, 19, max/min 7.03e6 (paper 3.6e6). With Table 1's 75 mm bound: λ₁ = 7.269e9, λ₁/λ₂ = 3.8, λ₄/λ₅ = 2.25, w₁ led by Nos. 12 and 5.
  - The signature is noise: λ₁ depends on an insensitive bound (7.3e9 vs 1.06e10), on the step (70 mm bound, all 1000 samples, from the review's arrays: 1.347e10 at h = 3e-5, 1.055e10 at 1e-5, 1.206e11 at 1e-6), and on single samples (the largest carries 25 % of trace(C) with the 70 mm bound; exact gradients: 6 %). At h = 1e-6 (70 mm), the eigenvectors are almost coordinate axes and take the paper's Fig. 3a–d form: w₁ No. 12 (+0.98), w₂ No. 5 (−0.97), w₃ No. 19 (−1.00), w₄ the k_sec Nos. 13 and 6 (review arrays, checked here).
  - What does not reproduce: the 2020 code's own step, h = 1e-4, with rounding gives λ₁ = 2.13e9 on 300 fresh samples (review arrays) and 1.517e9 on the first 100 shipped samples (notebook); and no step without rounding gives λ₁ ≈ 1e10.
  - So the paper's spectrum is *likely* a finite-difference artifact of parameter rounding, and the exact-gradient 2-D subspace is the meaningful result. This stays a hypothesis: the step and code version used for the 2019 figure are not stated, and the 2019 code (Bitbucket) was not available. It is offered as a reproducibility note; the model, prior, window and misfit agree with the paper.
- **Misfit scaling.** No further scaling of f_d is needed to reach λ₁ ≈ 1e10: the rounded FD runs reach it with the same ½ and 5 % noise.
- **Surrogate.** The paper's 4-D quartic r² ≈ 0.77 is close to the 2-D quartic here (validation 0.787). The 4-D quartic here reaches 0.851.

## Open questions (for the authors)

1. Which central-difference step, and which code version, produced Figs. 2–3? Were the parameters passed to R through `%f` (6 decimals), as in the 2020 `model.py`? If so, do the figures change with untruncated parameters or exact gradients? (The 2020 code uses h = 1e-4, and `uq_tools` defaults to 1e-3.)
2. e_min,3 lower bound: Table 1 says 75 mm, the HydroShare settings file says 70 mm. Which did the 2019 runs use? (75 is used here; it is consistent with Table 1's e_max,3 range [160, 255] = [75 + 85, 120 + 135]. It does not matter for the exact-gradient results, but it changes the rounded-FD λ₁ from 7.3e9 to 1.06e10.)
3. Is the 2019 LuKARS script (Bitbucket) identical to the 2020 HydroShare `main_exe.R`? This includes the heat-index summation, the next-day snow input, and the one-day offset between simulated and observed discharge noted above.
4. Did the 2019 runs include the 2001–2005 spin-up of the 2020 script, or did they start on 2006-01-01 (the paper's "model period from 2006 to 2008", p. 7088) with e_b = 2900 mm? With k_b⁻¹ ≈ 6.4 years, the slow baseflow store makes this matter for the k_is sensitivities.

## License and credit

Code (`lukars_model.py`, `generate_data.py`, notebook): MIT (`LICENSE`). Data (CSVs, `discharge.npz`): CC BY 4.0 (`LICENSE-DATA`). The forcing and discharge series and the LuKARS R code that `lukars_model.py` reimplements come from D. Bittner, M. Engel, B. Wohlmuth, D. Labat, G. Chiogna, "Discrete wavelet transform coupled with the active subspace method", HydroShare (2020), doi:10.4211/hs.4901a0d654334c259f4ff9b49dc0a74e, shared under CC BY 4.0. The resource states no separate code licence. The measurements are by the water works Waidhofen a.d. Ybbs. The model is from Bittner et al., *J. Hydrol.* 567:267–279 (2018), doi:10.1016/j.jhydrol.2018.10.017. The analysis follows Teixeira Parente et al. (2019), doi:10.1029/2019WR024739. Cite both papers and the HydroShare resource when using these files.
