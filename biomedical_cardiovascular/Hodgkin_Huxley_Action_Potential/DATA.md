# Hodgkin_Huxley_Action_Potential: data

Kind: **Python model**. The model is `HH_model.py`, and `generate_data.py` draws the 1000 input samples with a saved seed (2015) and writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it.

## Files

| file | contents | role |
|---|---|---|
| `Hodgkin_Huxley_Action_Potential.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2015`; outputs committed |
| `HH_model.py` | `qoi(p)`, `qoi_and_grad(p)`, `trajectory(p, times)`; `NAMES`, `UNITS`, `NOMINAL`, `LOWER`, `UPPER`, `I_INJ`, `FORM`, `TIMES`, `QOI_NAMES` | the model |
| `generate_data.py` | draws the inputs (`numpy.random.default_rng(2015)`) and writes the three CSVs | regenerates the data |
| `inputs.csv` | 1000 × 11 physical inputs, columns in the order of `NAMES` | model input |
| `outputs.csv` | 1000 × 21: `vpeak` (mV), `tpeak` (ms), v (mV) at t = 5, 5.5, …, 14 ms | model output |
| `gradients.csv` | 1000 × 231: column 11 i + j is ∂(output i)/∂(input j), with respect to the **physical** inputs | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are written with 17 significant digits (exact), outputs with 13, gradients with 11.

## Model

**Citation.** A. Torres Valderrama, J. Witteveen, M. Navarro and J. Blom, "Uncertainty propagation in nerve impulses through the action potential mechanism", The Journal of Mathematical Neuroscience 5:3, 2015, doi:10.1186/2190-8567-5-3. Open access, CC BY 2.0. The PDF read is the publisher's version.

**Equations.** The Hodgkin–Huxley (1952) model, Eq. (1) of the paper, for the membrane potential v (mV, from rest, depolarization positive) and the gating variables m, n, h:

- C dv/dt = −gNa m³ h (v − ENa) − gK n⁴ (v − EK) − gL (v − EL) + I
- dn/dt = 0.01(10 − v)/(exp((10 − v)/10) − 1) (1 − n) − 0.125 exp(−v/80) n
- dm/dt = 0.1(25 − v)/(exp((25 − v)/10) − 1) (1 − m) − 4 exp(−v/18) m
- dh/dt = 0.07 exp(−v/20) (1 − h) − h/(exp((30 − v)/10) + 1)

with (v, m, n, h)(0) = (v0, m0, n0, h0) and t in ms. The rate functions are as printed; their removable singularities at v = 10 and v = 25 are evaluated with `expm1` and a series.

**Inputs (m = 11)**, the vector ξ of Eq. (2), nominal values from Table 2 (p. 6), independent uniform on nominal ± 20 % (p. 3: "assuming 20 % of variability in the nominal values"):

| name | symbol | description | units | nominal | lower | upper | source |
|---|---|---|---|---|---|---|---|
| `n0` | n0 | initial K activation | – | 0.0003 | 0.00024 | 0.00036 | Table 2, p. 6 |
| `m0` | m0 | initial Na activation | – | 0.0011 | 0.00088 | 0.00132 | Table 2, p. 6 |
| `h0` | h0 | initial Na inactivation | – | 0.9998 | 0.79984 | 1.19976 | Table 2, p. 6 |
| `v0` | v0 | initial potential | mV | −10 | −12 | −8 | Table 2, p. 6 |
| `gK` | gK | max. K conductance | mS/cm² | 36 | 28.8 | 43.2 | Table 2, p. 6 |
| `gNa` | gNa | max. Na conductance | mS/cm² | 120 | 96 | 144 | Table 2, p. 6 |
| `gL` | gL | leak conductance | mS/cm² | 0.3 | 0.24 | 0.36 | Table 2, p. 6 |
| `EK` | EK | K Nernst potential | mV | −12 | −14.4 | −9.6 | Table 2, p. 6 |
| `ENa` | ENa | Na Nernst potential | mV | 115 | 92 | 138 | Table 2, p. 6 |
| `EL` | EL | leak reversal potential | mV | 10.613 | 8.4904 | 12.7356 | Table 2, p. 6 |
| `C` | C | membrane capacitance | µF/cm² | 1 | 0.8 | 1.2 | Table 2, p. 6 |

Ranges: p. 3 of the paper. Table 2 prints the conductances in µS/cm²; with C in µF/cm², t in ms and I in µA/cm² the H-H values are mS/cm². Units do not affect the normalized inputs. Table 2 also lists other initial values (n0 = 0.297, m0 = 0.00616, h0 = 0.12, v0 = 0.001), used only for the paper's Fig. 2; they are not used here.

**Settings the paper does not state, and our choices.**

1. **"20 % of variability"** is read as nominal ± 20 % (a 40 %-wide interval). For the negative nominals (v0, EK) the bounds are 1.2 and 0.8 times nominal.
2. **h0 > 1.** Under ± 20 %, h0 reaches 1.19976, outside the physical range [0, 1], in about half the samples. We keep the paper's stated ranges. The equations stay well defined (h relaxes toward h∞(v) < 1), and h0 has a small effect on the outputs.
3. **Injected current I = 150 µA/cm²**, constant from t = 0, for Fig. 1. The paper does not give it. We chose it by matching the nominal trajectory to the deterministic curve of Fig. 1(a), read from the figure: v ≈ 5 mV at 5 ms, peak ≈ 45 mV at ≈ 8.7 ms, trough ≈ 10 mV at ≈ 10.8 ms, v ≈ 30 mV at 14 ms. Nominal trajectories (scratch scan, DOP853 at rtol 1e-10):

   | I (µA/cm²) | v(5) (mV) | second peak (mV at ms) | trough (mV at ms) | v(14) (mV) |
   |---|---|---|---|---|
   | 130 | 4.6 | 50.5 at 8.79 | 6.4 at 10.90 | 24.0 |
   | 140 | 5.6 | 48.1 at 8.65 | 7.9 at 10.78 | 27.5 |
   | **150** | **6.5** | **46.0 at 8.52** | **9.3 at 10.67** | **30.2** |
   | 160 | 7.4 | 44.0 at 8.40 | 10.8 at 10.58 | 31.4 |
   | 170 | 8.3 | 42.3 at 8.29 | 12.3 at 10.50 | 31.3 |

   At I = 50 the nominal v(5 ms) is about −5 mV and the second maximum is a full spike of about 78 mV near 11 ms; at I ≤ 20 the second spike comes at or after 14 ms. Neither looks like Fig. 1(a). So Fig. 1(a) fixes I to roughly 140–160, and 150 is a round value in that range.
4. **Time window.** Solve on [0, 14] ms; Fig. 1(a) shows 5–14 ms. Outputs at t = 5, 5.5, …, 14 ms (19 times). The markers of the Sobol curves in Fig. 1(a) appear at about this spacing.
5. **The current term.** Eq. (1) as printed is dv/dt = −(gNa/C) m³h(v − ENa) − (gK/C) n⁴(v − EK) − (gL/C)(v − EL) + I, with I not divided by C. We use the H-H form C dv/dt = … + I (`FORM = 'standard'`). At the nominal C = 1 the two are identical, so the I fit above holds for both. They differ in how C matters. We computed both on the first 300 samples (scratch script; printed form by central differences with h = 1e-3 in normalized inputs):
   - **H-H form** (shipped): time-averaged normalized activity scores of v(t), scaled to max 1, are gK 1.000, ENa 0.993, gNa 0.322, h0 0.292, C 0.216, EK 0.199.
   - **Printed form**: the same scores are C 1.000, ENa 0.691, gK 0.643, gNa 0.216, h0 0.185, EK 0.126. vpeak has λ1/λ2 = 64 (121 for the H-H form on the same 300 samples), and C has weight −0.31 in w1 (−0.01 for the H-H form).

   The paper's Table 2 mode 1 is gK 1.000, ENa 0.533, EK 0.232, gNa 0.172, h0 0.141, C 0.100 (Fig. 1(b): gK and ENa dominate, C minor). Only the H-H form reproduces that ranking, so we take the missing division as a typesetting slip. `HH_model.py` keeps the printed form available as `form='printed'`.

**Quantities of interest.** The paper defines no scalar output: it studies time courses and time-averaged first-order Sobol indices. With I = 150, every sample fires one spike before 1 ms (first maximum at 0.46–0.77 ms) and then oscillates with decreasing amplitude. Fig. 1(a) shows the second, smaller peak, its "acme" (p. 3). The outputs are:

- `vpeak`: the value of the second local maximum of v(t) (mV);
- `tpeak`: its time (ms);
- `v(t)` at the 19 times above (mV).

Maxima are found by a `solve_ivp` event on dv/dt = 0 (decreasing). `HH_model.qoi` raises an error unless the first maximum is before 2 ms and the second lies in (5, 12) ms. Some samples have a third maximum after 13.2 ms, which is not used.

**Smoothness and regime check: no threshold crossing.** These were scratch checks with the shipped model:
- 2000 uniform random points and all 2048 corners of the box: every point has its first maximum at 0.46–0.77 ms (94–141 mV) and its second at 7.76–9.32 ms (20.7–77.2 mV). Each has 2 or 3 maxima (624 corners and 181 random points have a third, at ≥ 13.29 ms), and `qoi` succeeds at every point.
- The 1000 shipped samples: vpeak 21.7–73.6 mV, tpeak 7.89–9.16 ms.
- In the notebook, a line through the center of the box along w1 at 81 points: vpeak varies smoothly (largest second difference 0.014 mV).

The second peak is a graded, damped oscillation, not an all-or-none spike, so vpeak, tpeak and v(t) are smooth on the whole box. The paper's Fig. 2 shows thresholds at small current pulses; that regime is not used here.

**Gradients.** These come from the forward sensitivity equations S' = J_y S + J_p, S(0) = ∂y0/∂ξ, solved together with the model in one 48-dimensional `solve_ivp` (DOP853, rtol = 1e-10, atol = 1e-12) with analytic Jacobians:
- ∂v(t_i)/∂ξ = S_v(t_i);
- ∂vpeak/∂ξ = S_v(tpeak), since dv/dt = 0 there;
- ∂tpeak/∂ξ = −(dS_v/dt)(tpeak)/v''(tpeak), by the implicit function theorem.

They are with respect to the physical inputs. Multiply column j by (upper_j − lower_j)/2 for normalized inputs.

**Sampling design.** N = 1000 independent uniform samples, `numpy.random.default_rng(2015).random((1000, 11))` mapped to the box. That is about 90 per input; the brief's minimum for the gradient estimator is max(10 m, 300) = 300. The paper used level-3/4 Gauss–Patterson sparse grids (17249 and 18591 runs) and 1000 Monte Carlo samples per time point from a piecewise-linear sparse-grid interpolant, with no seed reported.

**Published results.** The paper has no active subspace. It reports:
- time-resolved first-order Sobol indices of v (Fig. 1(a), middle);
- time-averaged Sobol indices for v, n, m, h (Fig. 1(b));
- the SVD of that 11 × 4 matrix (Table 2): mode-1 energy 0.944690, with normalized mode 1 gK 1.000000, ENa 0.532895, EK 0.232445, gNa 0.172271, h0 0.140569, C 0.099635, and n0, m0, v0, gL, EL ≤ 0.0018;
- a two-parameter model (gK, ENa) that reproduces the standard deviation of v to within 10 % at all times (Fig. 1(a), bottom).

In the notebook:
- vpeak is one-dimensional (λ1/λ2 = 114, w1 mostly ENa +0.87 and gNa +0.44, cubic LOO R² = 0.9889).
- tpeak is two-dimensional (λ1/λ2 = 4.5, λ2/λ3 = 25.3).
- The time-averaged normalized activity scores of v(t) rank gK and ENa first (1.000, 0.993), with the same five negligible inputs as Table 2.
- The time-averaged matrix C̄ has two dominant eigenvalues (0.953 of the trace).

## Regenerating

```bash
python generate_data.py                                    # 1000 samples, 2 processes
python generate_data.py --n 20 --out /tmp/hh --check       # the first 20 rows, compared with the files here
```

The seed is fixed and the samples are the first N rows of one stream, so a run reproduces the files. `--workers` changes only the run time. The run that wrote the shipped files (2026-10-30; Python 3.13.18, numpy 2.5.4, scipy 1.18.1; Linux x86_64, 2 cores) took 120 s on two processes, about 0.2 s of CPU per sample. `--n 20 --check` printed: inputs identical; maximum relative difference, outputs 4.3e-13 and gradients 3.4e-11 (the digits written). The notebook runs in about 15 s.

## Accuracy

These checks were made with scratch scripts, not committed:

- **Outputs.** At 5 samples, the shipped values (DOP853, rtol 1e-10, atol 1e-12) agree with Radau at rtol = 1e-13, atol = 1e-14 to 2.8e-10 mV (vpeak), 2.7e-11 ms (tpeak) and 5.2e-10 mV (v(t)).
- **Gradients against tighter sensitivities.** At 4 samples, the shipped gradients agree with the same sensitivity solve at rtol = 1e-13 to a relative 5.2e-10 (row norm, worst output).
- **Gradients against independent central differences.** Central differences of `qoi` with step h in normalized inputs (22 extra plain solves per sample):
  - at rtol = 1e-13, h = 1e-3, 4 samples: worst relative error 1.3e-6 over all 21 outputs;
  - at the shipped rtol = 1e-10, h = 1e-3, 10 samples (rows 0, 100, …, 900): vpeak 9.7e-7, tpeak 3.0e-6, v(t) 8.4e-5. The larger v(t) values are solver noise in the differences near the trough (they change irregularly with h and disappear at rtol = 1e-13), not errors in the sensitivities.

  At the nominal point (rtol 1e-10), the worst error of central differences against the sensitivities falls from 2.9e-6 at h = 3e-3 to 5.5e-7 at h = 1e-3, about as O(h²), and then levels off at solver noise (4.0e-7 at h = 3e-4).
- The notebook re-solves 3 samples with sensitivities and prints the largest relative differences from the files: 3.2e-13 (outputs) and 2.7e-11 (gradients).

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from stochastic collocation on Gauss–Patterson sparse grids, with Sobol indices from sparse-grid quadrature. Its code and samples are not available, and it reports no scalar output or gradient. The data here are an independent Python computation of the same model and input ranges. They use three settings the paper does not state: I = 150 µA/cm², the window [0, 14] ms, and the H-H form of the current term. Changing I changes the outputs and, to a smaller degree, the ranking.
- **What is compared.** Two things are compared: the nominal trajectory, against Fig. 1(a) read by eye, and the ranking of the inputs, against Fig. 1(b) and Table 2. Our scores are derivative-based activity scores of v only; the paper's are first-order Sobol indices of v, n, m, h. They agree on the two dominant inputs (gK, ENa) and the five negligible ones. ENa's relative weight is larger here (0.993 against 0.533).
- No data were copied from the paper or from any outside source.

## Open questions (for the authors)

- What injected current I (and its time course) and which time interval produced Fig. 1?
- Does "20 % of variability" mean nominal ± 20 % or ± 10 %, and was h0 allowed above 1?
- Is the current in the code I or I/C (Eq. (1) prints I)?
- At which time points were the Sobol indices of Fig. 1 computed and averaged?

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model and uncertainty study: A. Torres Valderrama, J. Witteveen, M. Navarro and J. Blom (CC BY 2.0 article); equations of A. L. Hodgkin and A. F. Huxley, J. Physiol. 117(4):500–544, 1952, doi:10.1113/jphysiol.1952.sp004764. Notebook, model code and data: Claude (Anthropic) for P. Constantine, 2026. Cite A. Torres Valderrama, J. Witteveen, M. Navarro and J. Blom, "Uncertainty propagation in nerve impulses through the action potential mechanism", J. Math. Neurosci. 5:3, 2015, doi:10.1186/2190-8567-5-3.
