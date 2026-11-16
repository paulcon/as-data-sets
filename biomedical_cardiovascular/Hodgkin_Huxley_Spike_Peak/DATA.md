# Hodgkin_Huxley_Spike_Peak: data

Kind: **Python model**. The model is `hh_model.py` (numpy and scipy). `generate_data.py` draws the 1000 input samples with a fixed seed (2018) and writes `inputs.csv`, `outputs.csv` and `gradients.csv`. All three files can be regenerated here.

## Files

| file | contents | role |
|---|---|---|
| `Hodgkin_Huxley_Spike_Peak.ipynb` | the case study (Python 3.11, active_subspaces v1) | executed top to bottom with `SEED = 2018`; outputs committed |
| `hh_model.py` | `qoi_and_grad(p)`, `qoi(p)`, `trajectory(p, times)`, `extrema(p)`, `sample(M, rng)`; `NAMES`, `UNITS`, `NOMINAL`, `LOWER`, `UPPER`, `I_STIM`, `T_END`, `T_WINDOW`, `TIMES`, `QOI_NAMES`, `QOI_UNITS` | the model |
| `generate_data.py` | draws the inputs (`numpy.random.default_rng(2018)`) and writes the three CSVs; `--check` compares a rerun with them | regenerates the data |
| `check_accuracy.py` | the accuracy numbers below (checks 1–4) | checks |
| `inputs.csv` | 1000 × 11 physical inputs, columns in the order of `NAMES`: `V0` (mV), `Cm` (µF/cm²), `gNa`, `gK`, `gL` (mS/cm²), `ENa`, `EK`, `EL` (mV), `n0`, `m0`, `h0` (dimensionless) | model input |
| `outputs.csv` | 1000 × 25: `vpeak` (mV), `tpeak` (ms), `vmin` (mV), `tmin` (ms), then v (mV) at t = 5, 5.5, …, 15 ms | model output |
| `gradients.csv` | 1000 × 275: column 11 i + j is ∂(output i)/∂(input j), with respect to the **physical** inputs | model output |

Each CSV has two `#` header lines (a description with the provenance, then the column names with units) followed by comma-separated numbers, so `np.loadtxt(path, delimiter=',')` reads it. Inputs are written with 17 significant digits (exact), outputs with 13, gradients with 11. `gradients.csv` reshapes to (1000, 25, 11).

## Model

**Source.** S. Tennøe, G. Halnes and G. T. Einevoll, "Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience", *Frontiers in Neuroinformatics* 12:49, 2018, doi:10.3389/fninf.2018.00049, §4.2 ("Hodgkin-Huxley model", pp. 18–19), Table 2 (p. 19) and Fig. 4 (p. 20). Open access, CC BY 4.0. Page numbers refer to the published PDF.

**Equations.** The paper names "the original Hodgkin-Huxley model (Hodgkin and Huxley, 1952)" (§4.2, p. 18) but does not print its equations. The model here is the space-clamped Hodgkin–Huxley model, with the potential v (mV) measured from rest and depolarization positive. This is the convention of Fig. 4 of the paper (caption, p. 20: "parameterized so it has a resting potential of 0 mV"):

- Cm dv/dt = I − gNa m³ h (v − ENa) − gK n⁴ (v − EK) − gL (v − EL)
- dx/dt = αx(v)(1 − x) − βx(v) x for x = m, n, h
- αn = 0.01(10 − v)/(exp((10 − v)/10) − 1), βn = 0.125 exp(−v/80)
- αm = 0.1(25 − v)/(exp((25 − v)/10) − 1), βm = 4 exp(−v/18)
- αh = 0.07 exp(−v/20), βh = 1/(exp((30 − v)/10) + 1)

t is in ms. The rate functions are as printed in Eq. (1) of Torres Valderrama et al. (2015), doi:10.1186/2190-8567-5-3, the study the paper repeats. Their removable singularities at v = 10 and v = 25 mV are evaluated through u/expm1(u) with a series near u = 0. Eq. (1) of Torres Valderrama et al. adds I without dividing it by C. We use the Hodgkin–Huxley form above, in which Cm multiplies dv/dt.

**Inputs** (m = 11), nominal values from Table 2 of the paper (p. 19). "[W]e assume each of these parameters has a uniform distribution in the range ±10% around their original value" (§4.2, p. 19):

| name | symbol | units | description | range or distribution | source |
|---|---|---|---|---|---|
| `V0` | V₀ | mV | initial potential | U[−11, −9] (nominal −10) | Table 2 and §4.2, p. 19 |
| `Cm` | C_m | µF/cm² | membrane capacitance | U[0.9, 1.1] (1) | Table 2 and §4.2, p. 19 |
| `gNa` | ḡ_Na | mS/cm² | maximum Na conductance | U[108, 132] (120) | Table 2 and §4.2, p. 19 |
| `gK` | ḡ_K | mS/cm² | maximum K conductance | U[32.4, 39.6] (36) | Table 2 and §4.2, p. 19 |
| `gL` | ḡ_L | mS/cm² | maximum leak conductance | U[0.27, 0.33] (0.3) | Table 2 and §4.2, p. 19 |
| `ENa` | E_Na | mV | Na reversal potential | U[100.8, 123.2] (112) | Table 2 and §4.2, p. 19 |
| `EK` | E_K | mV | K reversal potential | U[−13.2, −10.8] (−12) | Table 2 and §4.2, p. 19 |
| `EL` | E_L | mV | leak reversal potential | U[9.5517, 11.674] (10.613) | Table 2 and §4.2, p. 19 |
| `n0` | n₀ | – | initial K activation | U[0.00099, 0.00121] (0.0011) | Table 2 and §4.2, p. 19 |
| `m0` | m₀ | – | initial Na activation | U[0.00027, 0.00033] (0.0003) | Table 2 and §4.2, p. 19 |
| `h0` | h₀ | – | initial Na inactivation | U[0.89982, 1.0998] (0.9998) | Table 2 and §4.2, p. 19 |

The bounds are nominal ± 0.1 |nominal| (printed by the notebook). For h0 the upper bound exceeds 1, the largest value of a gating variable; 473 of the 1000 samples have h0 > 1 (`generate_data.py`). We keep the paper's stated ranges. The equations stay well defined, since h relaxes toward h∞(v) < 1.

**Stimulus, time window, solver.** I = 140 µA/cm², constant from t = 0, and the window 5 ≤ t ≤ 15 ms are from the caption of Fig. 4 of the paper (p. 20). The following are this data set's choices: the end time T = 15 ms, and the solver, `scipy.integrate.solve_ivp` with DOP853, rtol 1e-10 and atol 1e-12 on the state and the sensitivities together.

**Quantities of interest.** The paper studies v(t) and its first-order Sobol' indices over the window and defines no scalar output. Its Fig. 4 marks "the peak of the first action potential" with a yellow line and "the minimum after the first action potential" with a cyan line (caption, p. 20). The lines sit at about 8.67 and 10.85 ms (our approximate read-off of the figure), inside the window, so the paper's "first action potential" is the first one in the window, not the spike at 0.62 ms. Our outputs, our choice:

- `vpeak`, `tpeak`: value and time of the first local maximum of v after 5 ms. Range 32.123–59.075 mV, 8.333–9.048 ms.
- `vmin`, `tmin`: value and time of the first local minimum after it. Range 4.187–14.171 mV, 10.499–11.220 ms.
- v(t) at t = 5, 5.5, …, 15 ms (21 times).

The ranges are printed by the notebook and `generate_data.py`. Extrema are found by a `solve_ivp` event on dv/dt = 0 and classified by the sign of d²v/dt². In every sample, v has exactly one maximum before 5 ms, the first spike. After the peak in the window, 592 samples have a further maximum before 15 ms, the peak of the next action potential; it is not used. Both counts are from `generate_data.py`.

**Gradients.** These are exact up to the solver tolerance, from the forward sensitivity equations (4 states × 11 parameters) solved with the state. For v(t), the gradient is the sensitivity of v at t. At an extremum t*, dv/dt = 0, so ∂v(t*)/∂p = s_v(t*), and ∂t*/∂p = −(∇ₓF_v · s(t*) + ∂F_v/∂p)/(∇ₓF_v · F) at t*. They are with respect to the **physical** inputs.

**Sampling design.** Monte Carlo: M = 1000 independent uniform samples, `LOWER + (UPPER − LOWER) * numpy.random.default_rng(2018).random((1000, 11))`.

## Regenerating

```bash
python generate_data.py           # writes the data files here
python generate_data.py --check   # reruns into a temporary directory and compares with the shipped files
python check_accuracy.py          # prints the accuracy numbers below
```

Observed on macOS (Darwin 23.6.0, Intel Core i7-1068NG7, 8 logical CPUs), Python 3.11.16, numpy 2.4.6, scipy 1.17.1:

- `python generate_data.py` takes 200 s with 4 worker processes (the default, `--workers` changes it). Each worker uses one BLAS thread.
- `python generate_data.py --check` regenerates all three files, with no subset, and took 405 s in one run and 299 s in another on the same machine. The run time depends on the machine's load. It printed:
  - `inputs.csv` 0.0e+00
  - `outputs.csv` 4.4e-11 (9.7e-13 in a second run)
  - `gradients.csv` 3.3e-08
  - "largest difference from the shipped files: 3.3e-08 (relative, gradients.csv)"

  These are relative, elementwise. They are at the level of the solver tolerance (rtol 1e-10) and of the 11 digits written for the gradients.
- `python check_accuracy.py` takes 110 s.
- The notebook takes about 45 s.

## Accuracy

- **Gradients against central differences.** `check_accuracy.py`, check 1: first 10 samples, step 1e-5 of each input's range, state-only solves at rtol 1e-12. The largest relative error of an output's gradient (2-norm, normalized coordinates) is 3.1e-09 for `vpeak`, 1.6e-08 for `tpeak`, 4.6e-09 for `vmin`, 4.5e-08 for `tmin`, and 4.5e-07 over the v(t) outputs. The notebook repeats this at 3 samples (largest 4.5e-07).
- **Solver tolerance.** `check_accuracy.py`, check 2: shipped outputs against a solve at rtol 1e-12, atol 1e-14, first 20 samples. The largest absolute differences are 1.1e-09 mV (`vpeak`), 1.1e-10 ms (`tpeak`), 8.5e-11 mV (`vmin`), 4.6e-11 ms (`tmin`) and 1.3e-09 mV (v(t)).
- **Extrema.** `check_accuracy.py`, check 3: event-located peak and trough against the largest and smallest v on a 0.001 ms grid within 0.1 ms, first 20 samples. The largest difference in v is 1.1e-09 mV.
- **Files against the model.** The notebook recomputes the first 5 samples; the largest difference is of order 1e-12 (5.0e-12 or 4.9e-12 between runs).

## How these files relate to the paper

These are new runs of our own implementation, not the paper's runs. The paper used polynomial chaos with point collocation through Uncertainpy (§4.2, p. 19) and does not publish its samples or its numerical Sobol' indices. We used the paper's inputs, ranges, stimulus and window, and the rate functions of Torres Valderrama et al. (2015). The notebook compares with the statements of §4.2 (pp. 18–19) and with approximate read-offs of Fig. 4 (p. 20). The read-offs were measured on the published figure against its axis labels. In Fig. 4B only the E_Na panel has tick labels; we assume that every panel shares its 0–1 axis. We estimate the read-offs good to about 0.05 ms in time, 0.5 mV for the mean, 0.1 mV for the standard deviation and 0.02 for an index. The notebook prints each read-off, this data set's value and the difference. The comparisons:

- the variance of v is largest at the peak: 6.10 mV at 8.45 ms on a 0.01 ms grid, recomputed in the notebook. The paper says between 8 and 9 ms, and Fig. 4A shows about 6.1 mV at about 8.45 ms;
- the mean and standard deviation of v(t) match the read-offs of Fig. 4A, and the yellow and cyan lines, to within the read-off precision. The largest difference is 0.31 mV, in the peak of the mean;
- ḡ_K and E_Na lead;
- the leading input over time follows the sequence the paper describes;
- V₀, m₀ and n₀ are negligible.

The paper finds that the model "maintains a sensitivity to the initial Na+ inactivation state (h0) through most of the simulation" (§4.2, p. 19). Here h₀'s share of the trace of C, C_jj/tr C, is moderate up to 10 ms, lower around the peak, and at most 0.054 from 10.5 ms on. That share equals the first-order Sobol' index for a function linear in the inputs. Fig. 4B shows the same shape for the Sobol' index of h₀: about 0.02 at 11 ms, 0.03–0.05 from 11.5 to 13.5 ms and at most about 0.02 from 14 ms on (approximate read-offs). The index and the share agree to within 0.01 at the five times read. So h₀ is moderate up to about 10 ms and small after it, here and in Fig. 4B. The paper's sentence describes a longer persistence than either shows.

The paper's Table 2 differs from Table 2 of Torres Valderrama et al. (2015) in two places. The paper has E_Na = 112 mV, n₀ = 0.0011 and m₀ = 0.0003; Torres Valderrama et al. have E_Na = 115 mV, n₀ = 0.0003 and m₀ = 0.0011. We use the paper's values as printed. `check_accuracy.py`, check 4, shows the effect at the nominal point:
- exchanging m₀ and n₀ changes v(t) by at most 0.00695 mV;
- E_Na = 115 mV moves the peak from 45.8692 mV at 8.6814 ms to 48.1284 mV at 8.6495 ms, and changes v(t) by up to 2.51 mV.

The paper points to the code of the case study, `uncertainpy/examples/valderrama/` in https://github.com/simetenn/uncertainpy. At commit ffb2400 (master as read in 2026; the file was last changed on 2018-05-25), that code uses E_Na = 112, m₀ = 0.0011, n₀ = 0.0003, `odeint` with outputs every 0.025 ms, and the end time 15 ms. The code is GPL-3.0; it was read for these settings and not copied.

## Unknowns

- The paper does not print the model equations. The rate functions here are from Torres Valderrama et al. (2015), Eq. (1). Hodgkin and Huxley (1952) could not be read for this data set, so the equations were not checked against its pages.
- Whether the paper's n₀ = 0.0011 and m₀ = 0.0003 (Table 2) are an exchange of the earlier values, and whether E_Na = 112 mV (rather than 115 mV) is intended. The effects are quantified above.
- The paper's simulation end time, time step and solver. Its §4.5 (p. 25) mentions "the total simulation time in the model, disregarding the first 5 ms" without giving it. The example code uses 15 ms and `odeint`, but we cannot confirm that this is the code the published figures came from.
- The paper's polynomial chaos settings for Fig. 4. The example code's `uq_valderrama.py` (commit ffb2400) calls `quantify(seed=10)` and does not set the polynomial order or the number of collocation nodes, so the order and number of nodes behind Fig. 4 are not stated. Its Sobol' indices are published only as a figure; the comparisons use approximate read-offs of it.
- Whether the ±10 % range was meant to allow h₀ > 1.
- Which quantity the yellow and cyan lines of Fig. 4 mark: the nominal peak and minimum, the means of their times over the samples, or the extrema of the mean. All three lie within 0.04 ms of the lines, closer than the read-off can separate.

## License and credit

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`); the data are new runs of the model here. Cite:

- S. Tennøe, G. Halnes and G. T. Einevoll, "Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience", *Frontiers in Neuroinformatics* 12:49, 2018, doi:10.3389/fninf.2018.00049 (the uncertainty study, inputs and setting).
- A. L. Hodgkin and A. F. Huxley, "A quantitative description of membrane current and its application to conduction and excitation in nerve", *The Journal of Physiology* 117(4):500–544, 1952, doi:10.1113/jphysiol.1952.sp004764 (the model).
- A. Torres Valderrama, J. Witteveen, M. Navarro and J. Blom, "Uncertainty propagation in nerve impulses through the action potential mechanism", *The Journal of Mathematical Neuroscience* 5:3, 2015, doi:10.1186/2190-8567-5-3 (the printed rate functions, Eq. 1).
