# Undamped_Oscillator_Limit_State: data

Kind: **Python model**. The model is `oscillator_model.py` (numpy only): the limit state g = 3r − |z_max| of Rajashekhar and Ellingwood (1993), Eq. (12), with z_max the largest displacement of the yielding oscillator of their Fig. 10, solved exactly event by event, and exact gradients. The closed-form *linear* version of the benchmark literature (UQTestFuns 0.7.0) is kept in the same module as a secondary study. `generate_data.py` draws the inputs with a saved seed and writes all four CSVs; a rerun reproduces them to the tolerance under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Undamped_Oscillator_Limit_State.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 1993`; outputs committed |
| `oscillator_model.py` | `response(X, grad=False, full=None)` (yielding oscillator: z_max, g, exact gradient, yielding count, peak diagnostics, distances to switching sets), `limit_state(X, grad=False)`, `limit_state_linear(X, grad=False)`, `parts_linear(X)`, `sample_lognormal`, `sample_normal`, `failure_probability`, `wilson`; `NAMES`, `DESCRIPTIONS`, `UNITS`, `MEAN`, `SD`, `MU`, `SIGMA` | the model |
| `generate_data.py` | draws the samples and writes the four CSVs; checks every gradient of both models by central differences; `--check` compares a rerun with the shipped files | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `inputs.csv` | 1000 × 6 physical inputs m, c1, c2, r, F1, t1 (units not stated in the paper; consistent units) | model input |
| `outputs.csv` | 1000 × 5: g and z_max of the yielding oscillator (failure when g ≤ 0), `yielded` (1 if spring 2 yields), g_lin and z_lin of the linear closed form | model output |
| `gradients.csv` | 1000 × 6: ∂g/∂(m, c1, c2, r, F1, t1) of the yielding oscillator, with respect to the **physical** inputs | model output |
| `gradients_linear.csv` | 1000 × 6: ∂g_lin/∂(m, c1, c2, r, F1, t1) of the linear closed form, physical inputs | model output (secondary study) |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits. The notebook converts to log coordinates in one stated cell: z = ln x and ∂g/∂z_i = x_i ∂g/∂x_i.

## Model

**Source.** M. R. Rajashekhar and B. R. Ellingwood, "A new look at the response surface approach for reliability analysis", *Structural Safety* 12(3):205–220, 1993, doi:10.1016/0167-4730(93)90003-J (publisher PDF read, a scan with a text layer). Example 2, Sec. 4.2 "Dynamic response of a nonlinear oscillator", pp. 217–219, Fig. 10, Eq. (12), Tables 7–9. The paper takes the example from Bucher and Bourgund (1990), doi:10.1016/0167-4730(90)90012-E, and Bucher, Chen and Schuëller (1989) (its refs. [15], [17]; not read).

**Limit state** (Eq. (12), p. 217): g(x) = 3r − |z_max|, "in which z_max is the maximum displacement response of the system and r is the displacement at which one of the springs yields"; g ≤ 0 is failure (p. 205).

**The oscillator (Fig. 10, p. 217).** A mass m on rollers held by two springs c1 and c2 in parallel, loaded by a rectangular pulse F(t) of height F1 and duration t1; undamped ("non-linear undamped single degree of freedom system", p. 217). The label r is drawn beside spring c2, and the restoring-force sketch is bilinear: slope up to z = r, a shallower slope after, unloading parallel to the first slope. This data set reads it as c1 linear and c2 elastic–perfectly plastic with yield displacement r (yield force c2 r) and elastic unloading, starting at rest. With p the plastic offset of spring 2 and s = z − p (|s| ≤ r): m z̈ + c1 z + c2 s = F(t); elastic phases (|s| < r) have frequency ω0 = √((c1 + c2)/m), yielding phases (s = ±r, while the velocity keeps its sign) ω1 = √(c1/m). Two points are this data set's reading, not the paper's statement: spring 2 is perfectly plastic after yield (the post-yield slope is c1 alone), and it yields at |s| = r in compression too. The second does not affect z_max here, because the largest |z| is the first peak in every sample, reached before any compressive yielding (notebook; and in 10⁶ further draws, `check_accuracy.py`, check 4).

**Solution.** Between switching events (pulse end t = t1; s reaching ±r; velocity zero) the motion is harmonic about a fixed equilibrium, so `response` solves it exactly, event by event, with each event time in closed form, until the pulse has ended and the motion is elastic for good (periodic). z_max is the largest |z(t)| over all t ≥ 0; no time step and no response window. After the pulse the energy m v²/2 + c1 z²/2 + c2 s²/2 never increases, so every later |z| is at most √(2E/c1); without gradients a sample stops once that bound is at most the largest |z| found, which leaves z_max unchanged (`check_accuracy.py`, check 4, difference 0 on 10⁴ draws).

**Linear closed form (secondary study).** z_lin = 2F1/(m ω0²) sin(ω0 t1/2): the peak of the *linear* undamped oscillator of stiffness c1 + c2 under a rectangular pulse, valid for ω0 t1/2 ≤ π/2 (true for every sample here, ω0 t1/2 ≤ 1.03). This is the benchmark as UQTestFuns 0.7.0 (`UndampedOscillator`, MIT, D. Wicaksono and M. Hecht, JOSS 8:5671, 2023, doi:10.21105/joss.05671) implements it; UQTestFuns cites Gayton, Bourinet and Lemaire (2003), doi:10.1016/S0167-4730(02)00045-0, Sec. 6.4.1 (not read). No UQTestFuns code is copied; `limit_state_linear` is written from the formula and agrees with `uqtestfuns.UndampedOscillator` to rounding level (below 10⁻¹⁵) at the shipped inputs and at 10⁵ draws of its input (`check_accuracy.py`, check 5, run in the Linux container). It does not model the yielding, so r enters only through the threshold 3r.

**Inputs** (m = 6), independent lognormal ("All variables are assumed to be independent and lognormally distributed", p. 217), mean and standard deviation from Table 7, p. 218. Chosen reading (the paper does not say): the mean and sd are those of x, so ln x ~ N(μ, s²), s² = ln(1 + sd²/mean²), μ = ln mean − s²/2.

| name | description | mean | sd | μ = E[ln x] | s = sd(ln x) | source |
|---|---|---|---|---|---|---|
| `m` | mass | 1.00 | 0.05 | −0.001248 | 0.049969 | Table 7, p. 218 |
| `c1` | stiffness of spring 1 (linear) | 1.00 | 0.10 | −0.004975 | 0.099751 | Table 7, p. 218; Fig. 10 |
| `c2` | stiffness of spring 2 (yields) | 0.10 | 0.01 | −2.307560 | 0.099751 | Table 7, p. 218; Fig. 10 |
| `r` | yield displacement of spring 2 | 0.50 | 0.05 | −0.698122 | 0.099751 | Table 7, p. 218; Eq. (12), Fig. 10 |
| `F1` | pulse load | 1.00 | 0.20 | −0.019610 | 0.198042 | Table 7, p. 218; Fig. 10 |
| `t1` | pulse duration | 1.00 | 0.20 | −0.019610 | 0.198042 | Table 7, p. 218; Fig. 10 |

UQTestFuns's default input ("Gayton2003", from Gayton et al. 2003 Table 9 as UQTestFuns states) has the same means and sds with **normal** marginals; the notebook gives failure probabilities with it for both models. UQTestFuns also lists two reduced pulse loads (F1 ~ N(0.6, 0.1) and N(0.45, 0.075), Echard et al. 2013, Table 4), checked for the linear closed form in `check_accuracy.py`.

**Quantity of interest.** g = 3r − z_max (dimensionless in consistent units); at the input means z_max = 0.91909 and g = 0.58091 (spring 2 yields), against z_lin = 0.91036, g_lin = 0.58964 (notebook).

**Gradients.** Yielding oscillator: forward-mode differentiation (dual numbers) of the event-by-event solution, including the dependence of every event time on the inputs, with respect to the physical inputs; exact wherever z_max is differentiable. z_max is only piecewise smooth. Where the pulse ends at the same time as a yield event or the end of yielding, and where a peak just touches the yield level, the restoring force is continuous (only its slope changes), so z_max stays differentiable and only its curvature can jump; where the largest |z| moves from one peak to another, z_max has a kink. In the shipped design: 10 samples within 10⁻² (3 within 10⁻³) of a pulse-end switching set and 2 within 10⁻² (none within 10⁻³) of a yield onset, in first-order distance in the standardized log inputs; the largest |z| is the first, positive, peak in all 1000 samples, with relative gap ≥ 2.6 × 10⁻⁴ to the next-largest peak, so no kink is in the design (notebook). Linear closed form: exact closed-form gradients as in the module (∂z/∂c1 = ∂z/∂c2).

**Sampling design.** M = 1000 independent Monte Carlo draws of the lognormal inputs, x = exp(μ + s ⊙ ξ), ξ = `numpy.random.default_rng(1993).standard_normal((1000, 6))`. All inputs are positive. Spring 2 yields in 974 samples (exactly those in which z_lin > r, and exactly those in which z_max > z_lin, by up to 3.28 %); 33 samples fail (g ≤ 0), against 29 for the linear closed form.

**Published results** (no active subspace or gradients in the paper). P_f = 3.99864 × 10⁻² from "Importance sampling with 1000 simulations of nonlinear dynamic response" (p. 218; also in Tables 8 and 9), no interval given. In the text of p. 218, from Bucher et al.: 3.81 × 10⁻² (adaptive sampling, 1500 simulations), 3.01 × 10⁻² and 3.67 × 10⁻² (response surfaces without and with cross terms). The paper's Approach A-0: 2.61898, 4.09679, 3.93365 × 10⁻² over three iterations (Table 8, p. 218); first-iteration variants 3.22473, 3.42005, 3.40402 × 10⁻² (Table 9, p. 219).

**What the notebook finds** (yielding oscillator). Scaling each input through its CDF (y = (ln x − μ)/s): λ₁/λ₂ = 65.4, λ₂/λ₃ = 26.3, λ₃/λ₄ = 108.7; λ₅ and λ₆ at round-off level, the exact zeros of a four-dimensional span (g/r depends only on c2/c1, F1/(c1 r) and t1√(c1/m)); w₁ = (0.0714, 0.1656, 0.0127, 0.4905, −0.6372, −0.5663), oriented so that g increases along it. Scaling by the sd instead: λ₁/λ₂ = 57.6, λ₂/λ₃ = 3.67; w₁ agrees (|w₁·w₁| = 0.99993), w₂ does not (0.1050). Chosen n = 2 (CDF scaling): quadratic leave-one-out R² 0.98599 (n = 1) and 0.99933 (n = 2); on 10⁶ lognormal draws the n = 2 quadratic's P_f is 0.6 % below the model's and the n = 1 quadratic's 5.0 % below. Failure probability (Monte Carlo 4 × 10⁶, 95 % Wilson interval): 3.8287 × 10⁻² [3.8099, 3.8475] × 10⁻². Linear closed form: λ₁/λ₂ = 65.9, λ₂/λ₃ = 25.2; w₁ within |w₁·w₁| = 0.99993 of the yielding oscillator's; P_f = 3.2310 × 10⁻² [3.2201, 3.2420] × 10⁻² (Monte Carlo 10⁷), 15.6 % below.

## Regenerating

```bash
python generate_data.py --out DIR --check     # writes to DIR (not this directory) and compares
python generate_data.py --out .               # rewrites the shipped CSVs (from this directory)
python check_accuracy.py                      # the numbers marked below
```

`--out` is required, so the script never writes here unless asked; `--check` refuses this directory and any directory inside it, and exits with status 1 on a mismatch. One process. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-10) `generate_data.py --out DIR --check` takes about 3 s, including the finite-difference check of all 1000 gradients of both models, and passes: the largest relative differences from the shipped files are at the rounding level, of order 1e-16 to 1e-15 for `inputs.csv` and the two gradient files and of order 1e-14 for `outputs.csv`. The stated tolerance is 10⁻¹⁰ relative, elementwise (10⁻¹² absolute near zero), allowing for rounding differences of `exp`, `sqrt`, `arccos`, `arctan2` and the trigonometric functions between platforms, propagated through the event sequence. The notebook takes about 1 minute in `.venv` (45 s and 91 s in two runs with other jobs running; its last cell prints the run time).

## Accuracy

- **Yielding oscillator, solution.** With yielding switched off (r = 10⁶) the event-by-event solution equals the linear closed form to rounding level, below 10⁻¹⁵ (relative, 1000 shipped inputs; `check_accuracy.py`, check 1). An independent velocity-Verlet integrator with a return map for spring 2, written in `check_accuracy.py` (check 2), converges to it at rows 0–199: largest relative difference in z_max 1.9 × 10⁻⁵ at 100 steps per pulse duration, 1.0 × 10⁻⁶ at 400, 5.3 × 10⁻⁸ at 1600 and 2.9 × 10⁻⁸ at 3200 (roughly second order, limited at the end by sampling the peak on the time grid).
- **Yielding oscillator, gradients.** Against central differences in the standardized log inputs with step 10⁻⁵ at all 1000 samples: largest relative error 2.3 × 10⁻¹⁰, median 1.0 × 10⁻¹⁰ (row 2-norm; `generate_data.py` and notebook). With steps 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁷: 1.5 × 10⁻⁸, 1.5 × 10⁻¹⁰, 2.5 × 10⁻⁹, 2.8 × 10⁻⁸ (`check_accuracy.py`, check 3); at step 10⁻³ the stencils of 3 samples cross a pulse-end switching set, and their errors stay within the same bound, consistent with a curvature jump only.
- **Yielding oscillator, peaks.** In 10⁶ lognormal draws (seed 21) the largest |z| is the first, positive, peak in every draw; on 10⁴ draws the energy-bound stop changes z_max by 0.0 (`check_accuracy.py`, check 4).
- **Yielding oscillator, failure probability** with 10⁷ other lognormal draws (seed 31): 3.8226 × 10⁻² [3.8107, 3.8345] × 10⁻² (95 % Wilson; `check_accuracy.py`, check 7, about 70 s); the linear closed form on the same draws 3.2279 × 10⁻² [3.2170, 3.2389] × 10⁻².
- **Linear closed form, gradients.** Against central differences (step 10⁻⁵ in the standardized log inputs) at all 1000 samples: largest relative error 1.3 × 10⁻¹⁰, median 5.8 × 10⁻¹¹ (`generate_data.py`).
- **Linear closed form, implementation.** Against UQTestFuns 0.7.0: rounding level, below 10⁻¹⁵, at the shipped inputs and at 10⁵ draws of its input; its printed input model matches the normal marginals above (`check_accuracy.py`, check 5; needs `pip install uqtestfuns==0.7.0`; run on 2026-11-09 in the Linux container, `.venv` does not have the package).
- **Linear closed form, normal inputs** (notebook, Monte Carlo 10⁷, 95 % Wilson interval): 2.8531 × 10⁻² [2.8428, 2.8634] × 10⁻², against 2.834 × 10⁻² (Monte Carlo 7 × 10⁴, CoV 2.2 %; Echard et al. 2011, Table 6, as listed by UQTestFuns).
- **Linear closed form, reduced pulse loads** (`check_accuracy.py`, check 6; references from Echard et al. 2013, Table 5, as listed by UQTestFuns). F1 ~ N(0.6, 0.1): FORM (HL-RF with exact gradients) β = 4.2703, Φ(−β) = 9.759 × 10⁻⁶ (reference FORM 9.76 × 10⁻⁶); importance sampling about the design point, 10⁶ draws, 9.112 × 10⁻⁶ ± 4.1 × 10⁻⁸ (95 %); plain Monte Carlo 10⁸ draws, 891 failures, 8.910 × 10⁻⁶ [8.344, 9.515] × 10⁻⁶ (reference MCS 9.09 × 10⁻⁶, CoV 2.47 %). F1 ~ N(0.45, 0.075): β = 5.5344, Φ(−β) = 1.562 × 10⁻⁸ (reference FORM 1.56 × 10⁻⁸); importance sampling 1.510 × 10⁻⁸ ± 8.0 × 10⁻¹¹ (reference MCS 1.55 × 10⁻⁸, CoV 2.68 %; reference IS 1.53 × 10⁻⁸).
- Run times. In `.venv` (no UQTestFuns; 2026-11-10) `python check_accuracy.py --skip-uqtestfuns` prints the values of checks 1–4, 6 and 7 above in about 105 s (about 70 s of it the 10⁷-draw run of the yielding oscillator and 17 s the 10⁸-draw run of the linear closed form); without the flag the script runs checks 1–4 (about 20 s) and exits with an error that names the package to install. The full run with UQTestFuns 0.7.0 took 176 s in the Linux container.

## How these files relate to the paper

- **The paper's model, solved here; not the paper's computation.** The paper's failure probabilities come from simulations of the nonlinear dynamic response of the Fig. 10 oscillator, by importance sampling (its own value) and adaptive sampling (Bucher et al.); its integrator and importance-sampling density are not given, but for an undamped piecewise-linear system started at rest these are numerical choices, and the event-by-event solution here has no time step. The data here are that model, with the reading of Fig. 10 stated above, evaluated at new Monte Carlo samples of the paper's input distributions (Table 7, lognormal).
- **Agreement.** With the paper's lognormal inputs the yielding oscillator gives P_f = 3.8287 × 10⁻² [3.8099, 3.8475] × 10⁻² (notebook, 4 × 10⁶ draws) and 3.8226 × 10⁻² [3.8107, 3.8345] × 10⁻² (`check_accuracy.py`, 10⁷ draws). Bucher et al.'s 3.81 × 10⁻² lies inside both intervals. The paper's 3.99864 × 10⁻² lies outside the notebook's interval (the notebook's estimate is 4.2 % below it); the difference (0.0017) is 0.28 of the standard deviation a 1000-run plain Monte Carlo estimate would have at this P_f (the paper's importance-sampling error is not stated). The linear closed form gives 3.2310 × 10⁻², 15.6 % below the yielding oscillator: the larger P_f of the paper's model is a computed result, because yielding raises z_max in 974 of the 1000 samples. With the normal inputs the linear closed form agrees with the benchmark literature as listed by UQTestFuns (above); those reference values belong to the linear closed form, not to the paper's model.
- **Attribution.** The input values, the limit state and the oscillator are cited to the paper (Table 7, Eq. (12), Fig. 10); the reading of the restoring force (perfectly plastic c2, symmetric compressive yield) is this data set's. The closed-form z_lin and the normal input model are cited to UQTestFuns and, through it, to Gayton et al. (2003); that paper and the Echard et al. papers were not read for this data set, and the reference values attributed to them are as UQTestFuns lists them (documentation source `docs/test-functions/undamped-oscillator.md` of github.com/damar-wicaksono/uqtestfuns, read 2026-11-09, sha256 65609be57cfa9167cbd14783d7a45bb617b68f21bf6d88aec38f35db83cf4aae).

## Open questions

Only the authors (or the original Bucher papers) can settle these:
1. Whether spring 2 is perfectly plastic after yield (this data set's reading; the sketch shows a positive post-yield slope, consistent with c1 alone) or has a small post-yield stiffness of its own, and whether compressive yield is symmetric (it does not affect z_max here).
2. Whether Table 7's mean and sd are those of x (our reading) or of ln x.
3. The importance-sampling density and the sampling error of the 1000-simulation reference 3.99864 × 10⁻².
4. Units of the inputs.

## License and credit

Code (`oscillator_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`), generated here from the paper's model and the published formula; no data or code were copied. The article is © Elsevier and is not distributed here; cite M. R. Rajashekhar and B. R. Ellingwood, *Structural Safety* 12(3):205–220, 1993, doi:10.1016/0167-4730(93)90003-J, and for the closed-form benchmark UQTestFuns (Wicaksono and Hecht 2023, doi:10.21105/joss.05671). The paper acknowledges the Council for International Exchange of Scholars, Washington D.C., for sponsoring the first author's research (pp. 219–220).

## Unknowns

- The post-yield behaviour of spring 2 and its compressive yield (open question 1); the data use a perfectly plastic c2 with symmetric yield.
- The reading of Table 7 as moments of x (open question 2).
- Units (open question 4).
- Gayton et al. (2003) and Echard et al. (2011, 2013) were not read; the closed form and the normal-input reference values are taken from UQTestFuns.
