# COVID19_SEIR_Epidemic_Peak: data

Kind: **Python model**. The model is `seir_model.py` (numpy and scipy), written from the paper's equations. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; in the repo's `.venv` a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `COVID19_SEIR_Epidemic_Peak.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with seed 2022; outputs committed |
| `seir_model.py` | `peak(x, grad=False)`, `peaks(X, grad=False)`, `trajectory(x, t)`, `rates(x)`, `gauss_hermite_moments(npts)`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `MEAN`, `SD`, `MU`, `SIGMA`, `N_POP`, `Y0`, `T_END`, `RTOL`, `ATOL`, `QOI_NAMES` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files; checks gradients by finite differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `inputs.csv` | 1000 × 3 physical inputs R0 (–), tau_inc (days), tau_inf (days) | model input |
| `outputs.csv` | 1000 × 2: t_peak (days), I_peak (individuals) | model output |
| `gradients.csv` | 1000 × 6: ∂t_peak/∂(R0, tau_inc, tau_inf) in columns 0–2, ∂I_peak/∂(R0, tau_inc, tau_inf) in columns 3–5, with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits. The notebook converts to log coordinates in one stated cell: z = ln x and ∂f/∂z_i = x_i ∂f/∂x_i.

## Model

**Source.** B. C. S. Jensen, A. P. Engsig-Karup, K. Knudsen, "Efficient uncertainty quantification and variance-based sensitivity analysis in epidemic modelling using polynomial chaos", *Math. Model. Nat. Phenom.* 17:8, 2022, doi:10.1051/mmnp/2022014 (publisher PDF read; arXiv:2109.08066 is the preprint). Case 1, "Epidemic peak", Sec. 4.1, pp. 7–10. Case 2 (superspreaders, Sec. 4.2) is not built: its compartment equations are given only as a diagram (Fig. 7), and its fit uses Danish hospital data that the paper says are only in the authors' GitLab repository.

**Equations** (Eq. (4.1), p. 7): S' = −β I S/N, E' = β I S/N − σE, I' = σE − γI, R' = γI, with N = 5.8 × 10⁶ (p. 7). Inputs: R0 = β/γ, τ_inc = 1/σ, τ_inf = 1/γ (p. 8), so β = R0/τ_inf, σ = 1/τ_inc, γ = 1/τ_inf. Outputs: the map F: (R0, τ_inc, τ_inf) ↦ (t_peak, I_peak), I_peak = I(t_peak) (Eq. (4.2), p. 8).

**Inputs** (m = 3), independent lognormal (table, p. 10):

| name | symbol | units | mean | variance (sd) | μ = E[ln x] | s = sd(ln x) | source |
|---|---|---|---|---|---|---|---|
| `R0` | R₀ | – | 1.4 | 0.025² (0.025) | 0.336313 | 0.017856 | table, p. 10 |
| `tau_inc` | τ_inc = 1/σ | days | 4.2 | 0.25² (0.25) | 1.433316 | 0.059471 | table, p. 10 |
| `tau_inf` | τ_inf = 1/γ | days | 3.3 | 0.25² (0.25) | 1.191061 | 0.075649 | table, p. 10 |

The PDF's text layer reads the variances as "0.0252" and "0.252"; the page image shows 0.025² and 0.25². The paper does not say whether the mean and variance are those of x or of ln x. **Chosen: those of x**, so s² = ln(1 + sd²/mean²) and μ = ln(mean) − s²/2. We tested the readings with the paper's own 27-run Gauss–Hermite design (3 points per input) and `seir_model.gauss_hermite_moments(3)` (`check_accuracy.py`, check 4):

| reading | t_peak mean, sd (days) | I_peak mean, sd (×10³) | first-order Sobol t_peak; I_peak |
|---|---|---|---|
| **paper (table p. 10; Fig. 5A)** | **284.78, 19.89** | **115.63, 12.56** | **0.541, 0.217, 0.240; 0.753, 0.094, 0.152** |
| mean, variance of x (chosen) | 284.875, 19.864 | 115.641, 12.556 | 0.538, 0.216, 0.244; 0.753, 0.094, 0.151 |
| median = 1.4, 4.2, 3.3; s from the variance of x | 285.391, 19.896 | 115.809, 12.567 | 0.538, 0.216, 0.245; 0.753, 0.094, 0.151 |
| s = 0.025, 0.25, 0.25 for ln x; mean of x = 1.4, 4.2, 3.3 | 285.051, 55.358 | 116.231, 27.185 | 0.139, 0.506, 0.351; 0.317, 0.335, 0.335 |
| s = 0.025, 0.25, 0.25 for ln x; median = 1.4, 4.2, 3.3 | 293.835, 57.057 | 116.422, 27.220 | 0.138, 0.506, 0.351; 0.317, 0.336, 0.336 |

The chosen reading matches the paper's 27-run I_peak mean to within one unit of the last printed digit (0.01 × 10³: 115.641 rounds to 115.64, the paper prints 115.63) and its sd exactly, and the 27-run t_peak to 0.095 day (mean) and 0.026 day (sd). The log-sd readings contradict the paper's Fig. 5 (R0 dominant). The median reading is close but worse in both means. The paper's Fig. 3 (p. 8) supports the chosen reading too: its 27 quadrature points (Fig. 3A) lie at peak times of about 232–350 days and peak heights of about 0.85–1.52 × 10⁵ (read by eye), and the 27 nodes of the chosen reading give 232.1–349.0 days and 0.85–1.53 × 10⁵, while the log-sd reading (mean of x fixed) gives 159.0–484.3 days and 0.51–2.12 × 10⁵; the 1000 points of Fig. 3B span about 165–520 days and 0.45–2.3 × 10⁵, and the chosen reading's 1000 nodes give 164.1–519.7 days and 0.44–2.36 × 10⁵ (`check_accuracy.py`, checks 4 and 7).

**Initial state** (not stated in the paper): (S, E, I, R)(0) = (N − 1, 0, 1, 0), one infectious individual at t = 0, the same for every sample. With (N − 1, 1, 0, 0) instead (`check_accuracy.py`, check 5), t_peak at the input means is 287.911 days instead of 284.091, and the 27-run mean of t_peak is 288.695, 3.9 days from the paper's 284.78; I_peak is unchanged (it does not depend on where the single seed starts, to the digits shown). With S(0) = N rather than N − 1, the 27-run means are 284.875 days and 115.641 × 10³ (unchanged to these digits). The residual 0.1-day difference in the mean of t_peak is not explained by the initial state. We infer (it cannot be checked without the authors' MATLAB code) that it is within the size of the numerical error in the authors' own t_peak: from 27 to 1000 quadrature nodes their mean moves by +0.06 day (284.78 → 284.84) and their first-order Sobol indices by up to 0.003, while ours, with the peak located as an event root of the smooth function I', move by 3 × 10⁻⁵ day and 4 × 10⁻⁵ (notebook). A smooth output has converged at 27 nodes, so the paper's 27 → 1000 change suggests coarse numerics (solver tolerances or peak location) in the authors' t_peak.

**Quantities of interest.** t_peak (days), the first time at which I' = σE − γI crosses zero from above; I_peak (individuals). At the input means: t_peak = 284.091 days, I_peak = 115 582.6 (1.993 % of N).

**Solver.** `scipy.integrate.solve_ivp`, LSODA, rtol = 1e-11, atol = 1e-9, up to 2000 days, with the peak as a terminal event of g = σE − γI (direction −1). With I(0) = 1 and E(0) = 0, g < 0 at first (I falls while E builds up), so the first downward crossing is the peak. `peak` raises RuntimeError if there is no crossing before day 2000 or if I at the crossing does not exceed I(0) (no outbreak). Near R0 = 1 the peak moves out of the window (with the periods at their means, R0 = 1.05 peaks at day 1548 and R0 = 1.02 has no peak before day 2000; `check_accuracy.py`, check 6), but the study's R0 lies in [1.3284, 1.4861].

**Gradients.** Exact up to the ODE tolerance: forward sensitivities of (S, E, I) with respect to θ = (β, σ, γ) are integrated with the states (12 ODEs). At the peak, dt_peak/dθ = −(∂g/∂θ + ∇_y g · s_θ)/(dg/dt) with dg/dt = σE' (because I' = 0 there), and dI_peak/dθ = s_I(t_peak). The chain rule through β = R0/τ_inf, σ = 1/τ_inc, γ = 1/τ_inf gives the gradient with respect to the physical inputs, which is what `gradients.csv` holds.

**Exact structure.** Scaling τ_inc and τ_inf by c with R0 fixed only rescales time, because the initial state does not depend on the rates. So I_peak is invariant (∂I_peak/∂ln τ_inc + ∂I_peak/∂ln τ_inf = 0) and t_peak scales by c (∂t_peak/∂ln τ_inc + ∂t_peak/∂ln τ_inf = t_peak). Both hold in the shipped gradients at every sample, to 7.6 × 10⁻¹¹ and 1.6 × 10⁻¹⁰ (relative; printed in the notebook). In normalized log coordinates the first makes λ₃ = 0 for I_peak, with eigenvector (0, 1/s_inc, 1/s_inf)/‖·‖.

**Sampling design.** M = 1000 independent Monte Carlo draws, x = exp(μ + s ⊙ ξ), ξ ~ N(0, I₃), `numpy.random.default_rng(2022).standard_normal((1000, 3))`. Every sample is within the model's meaning: R0 ∈ [1.3284, 1.4861] (all > 1), τ_inc ∈ [3.481, 4.981], τ_inf ∈ [2.569, 4.311] days; t_peak ∈ [225.56, 348.68] days, I_peak ∈ [82 333, 162 965].

**Published results** (no active subspace in the paper). Table p. 10: t_peak mean, sd 284.78, 19.89 (gPC, 27 runs), 284.84, 19.90 (gPC, 10³ runs), 285.01 ± 1.37, 20.28 ± 0.82 (MCMC, 8 chains of 10 000 Metropolis–Hastings iterations, ~2660 evaluations each); I_peak (×10³) 115.63, 12.56; 115.64, 12.56; 115.73 ± 0.67, 12.64 ± 0.27. Fig. 5 (p. 9, values on the bars), Sobol indices R0, τ_inc, τ_inf, then the pairwise and triple interactions: t_peak 0.541, 0.217, 0.240, 0.001, 0.000, 0.000, 0.000 (27 runs) and 0.539, 0.216, 0.243, 0.001, 0.001, 0.000, 0.000 (1000 runs); I_peak 0.753, 0.094, 0.152, 0.001, 0.001, 0.000, 0.000 (both).

**What the notebook finds.** n = 1 for both outputs: λ₁/λ₂ = 1179 (t_peak) and 1065 (I_peak); for I_peak λ₃ is numerically zero (of order 10⁻¹³ in the committed run, below the round-off level m ε λ₁ ≈ 10⁻⁷; the computed value changes with the eigensolver and platform, so λ₂/λ₃ is not reported and the eigenvalue plot draws only λ₁ and λ₂). w₁ = (−0.7345, 0.4651, 0.4942) for t_peak and (0.8684, −0.3065, 0.3899) for I_peak in normalized log coordinates; w₁² agrees with the 1000-run first-order Sobol indices (ours) within 0.0014 (largest gap 0.00133) and with the paper's Fig. 5B within 0.0013 (largest gap 0.00122). A quadratic in y₁ has leave-one-out R² = 0.99912 (t_peak) and 0.99947 (I_peak).

## Regenerating

```bash
python generate_data.py                                 # writes the three CSVs here
python generate_data.py --out DIR --check               # writes to DIR (not this directory) and compares
python check_accuracy.py                                # the numbers marked below
```

One process, no multiprocessing. `--check` refuses to write into this directory. The shipped files were written on 2026-10-31 in a Linux container (Python 3.13.18, numpy 2.5.4, scipy 1.18.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-02), `python generate_data.py --out DIR --check` took 52 s: 40 s for the 1000 solves with sensitivities (about 40 ms each) and 11 s for the finite-difference check on the first 100 samples. Its files match the shipped ones to a maximum relative difference of 5.1e-16 (inputs, rounding of exp), 2.5e-11 (outputs) and 5.6e-10 (gradients, elementwise): the stated tolerance is 1e-15 for the inputs and 1e-9 for the outputs and gradients, the size of the solver's error control (rtol 1e-11) amplified by the event location and the sensitivities.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name; in `.venv` it took 43 s (2026-11-02).

- **Outputs** (`check_accuracy.py`, check 1). At rows 0–19 the shipped outputs agree with LSODA at rtol = 1e-13, atol = 1e-11 to a relative 4.8 × 10⁻¹² (t_peak) and 1.2 × 10⁻¹¹ (I_peak). The notebook's check re-solves 5 samples without sensitivities (3 ODEs instead of 12) and agrees with the files to about 10⁻¹⁰.
- **Gradients.** At rows 0–19 the forward-sensitivity gradients agree with the same sensitivities solved at rtol = 1e-13 to a relative 9.0 × 10⁻¹¹ (t_peak) and 1.3 × 10⁻¹⁰ (I_peak) (row 2-norm; `check_accuracy.py`, check 2). Against central differences in z = ln x: step 1e-5 at rtol 1e-11, first 100 samples (`generate_data.py`), largest relative error 4.4 × 10⁻⁷ (t_peak) and 1.1 × 10⁻⁶ (I_peak); step 1e-4 at rtol 1e-13, first 100 samples (check 3), 1.0 × 10⁻⁷ and 1.0 × 10⁻⁸ (median 7.5 × 10⁻⁸ and 8.6 × 10⁻⁹), the size of the differences' truncation error. At the input means, step 1e-5 with tight solves gives 3 × 10⁻⁹ and 8 × 10⁻⁹ (check 2).
- **Moments.** The gPC moments are converged in the quadrature: from 27 to 1000 runs the mean of t_peak changes by 3 × 10⁻⁵ day and the first-order Sobol indices by at most 4 × 10⁻⁵ (notebook).

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from the authors' MATLAB toolbox (https://gitlab.gbar.dtu.dk/bcsj/covid-19-ctrl-public-code, cited on p. 3; not used for this data set, licence not checked). The data here are an independent Python computation of the paper's Eq. (4.1) with its stated inputs, plus the two choices above (the reading of the input table and the initial state). The paper uses tensor Gauss–Hermite quadrature (27 and 1000 nodes) and MCMC; the 1000 Monte Carlo samples here are new. The notebook repeats the paper's 27- and 1000-node quadrature with `seir_model.gauss_hermite_moments` for the comparison.
- **Agreement.** I_peak: the mean to within one unit of the last printed digit (0.01 × 10³) for the 27-run mean (115.64 against 115.63); exactly for the 1000-run row (115.64); the sd (12.56) exactly in both rows. t_peak: our mean is 0.095 day above the paper's 27-run value and 0.035 day above its 1000-run value; our sd is 0.026 and 0.035 day below. First-order Sobol indices agree with Fig. 5B within 0.0014 (largest gap 0.00135, τ_inf for t_peak).
- **Open questions.** The initial state, the reading of the input table and the authors' t_peak numerics are open (Unknowns).

## Unknowns

Open questions (only the authors can settle):

- The initial state of Case 1 and the time window. The paper does not state the initial state; (N − 1, 0, 1, 0) is this data set's choice ("Model").
- Whether the table's mean and variance are those of x (our reading) or of ln x.
- How t_peak is located (event, grid, or interpolation) and which MATLAB solver and tolerances were used, which may explain the 0.1-day difference in the mean of t_peak. The paper's own 27 → 1000-node change in t_peak (+0.06 day in the mean, up to 0.003 in the Sobol indices, against 3 × 10⁻⁵ day and 4 × 10⁻⁵ for our smooth event-located t_peak) suggests numerical error of that size in the authors' t_peak (an inference, not checked against their code).

## License and credit

Code (`seir_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations; no data or code were copied. The paper is open access under CC BY 4.0. Cite B. C. S. Jensen, A. P. Engsig-Karup, K. Knudsen, "Efficient uncertainty quantification and variance-based sensitivity analysis in epidemic modelling using polynomial chaos", *Math. Model. Nat. Phenom.* 17:8, 2022, doi:10.1051/mmnp/2022014. The paper's work was supported by the Novo Nordisk Fonden (NNF20SA0063089) and the Academy of Finland (320022).
