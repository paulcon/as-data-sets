# Two_DOF_Equipment_Structure_Oscillator: data

Kind: **Python model**. The model is `oscillator_model.py`, a closed form (the paper's Eq. 37) plus an exact Lyapunov solve used only for checking. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun reproduces them bit for bit.

## Files

| file | contents | role |
|---|---|---|
| `Two_DOF_Equipment_Structure_Oscillator.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1985; outputs committed |
| `oscillator_model.py` | `mean_square`, `mean_square_as_printed`, `rms`, `limit_state`, `grad_log_mean_square`, `grad_log_rms`, `grad_log_limit_state`, `exact_mean_squares`, `sample`; `NAMES`, `DESCRIPTIONS`, `MEAN`, `COV`, `MU`, `SIGMA`, `M_RMS` | the model (numpy; scipy only for the Lyapunov check) |
| `generate_data.py` | draws the samples and writes the three CSVs; prints a gradient check; `--check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 1000 × 8 physical inputs `mp, ms, kp, ks, zeta_p, zeta_s, S0, Fs` | model input |
| `outputs.csv` | 1000 × 3: `rms` (Eq. 37), `G` = Fs − 3 ks rms, `rms_exact` (Lyapunov solve, accuracy check only) | model output |
| `gradients.csv` | 1000 × 15: columns 0–6 ∂rms/∂(ln x_j) for j = mp … S0; columns 7–14 ∂G/∂(ln x_j) for j = mp … Fs. Gradients with respect to the **log** inputs | model output |

The CSVs have two `#` lines (a description and the column names) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly. Units are not given by the benchmark: any consistent set (the mean ω_s is 1 rad per unit time); `rms` has the units of length, `G` and `Fs` of force.

## Model

**Source.** T. Igusa and A. Der Kiureghian, "Dynamic characterization of two-degree-of-freedom equipment-structure systems", *J. Eng. Mech.* 111(1):1–19, 1985, doi:10.1061/(ASCE)0733-9399(1985)111:1(1) (checked on Crossref). The PDF read is the ASCE journal copy with journal pagination (the file is named `_oa`, but it is not a preprint). The equations were read from page images (the text layer garbles them): Eq. 1 and Eqs. 2a–2d, 3a on p. 3, Eqs. 35–37 on p. 12, Fig. 5 on p. 13.

**Equations.** A light secondary oscillator (equipment: m_s, ω_s, ζ_s) on a primary oscillator (structure: m_p, ω_p, ζ_p), base excited by stationary white-noise acceleration with one-sided PSD G_0. With β = (ω_p − ω_s)/ω_a (tuning, Eq. 2a), γ = m_s/m_p (interaction, Eq. 2b), ω_a = (ω_p + ω_s)/2 (Eq. 2d), ζ_a = (ζ_p + ζ_s)/2 (Eq. 3a), the general ("matched") mean-square response of the equipment is (Eq. 37, p. 12)

  E[x_s²] = π G_0/(4 ζ_s ω_s³) · [ ζ_a ζ_s / (ζ_p ζ_s (4 ζ_a² + β²) + γ ζ_a²) · (ζ_p ω_p³ + ζ_s ω_s³) ω_p / (4 ζ_a ω_a⁴) ].

The benchmark (Der Kiureghian and De Stefano 1991, as given by OpenTURNS and UQTestFuns) uses this formula unchanged, with ω_p = √(k_p/m_p), ω_s = √(k_s/m_s), S_0 for G_0 and θ for β; we checked it term by term against the page image. `mean_square` uses the algebraically equal form π S_0 N ω_p/(16 ω_s³ ω_a⁴ D), N = ζ_p ω_p³ + ζ_s ω_s³, D = ζ_p ζ_s (4ζ_a² + β²) + γ ζ_a² (ζ_a cancels); `mean_square_as_printed` is the printed form, and the two agree to 7e-16 at the 1000 samples.

**Which displacement.** The paper's Eq. 1 writes the mass, damping and stiffness matrices in displacements relative to the ground, with q = [0 1]ᵀ selecting x_s (p. 12). But Eq. 37 is the mean square of the equipment's displacement relative to the primary mass (the deformation of the secondary spring): an exact Lyapunov solve of the Eq. 1 system agrees with Eq. 37 for x_s − x_p (median relative error 1.4 % over the 1000 samples) and not for x_s relative to the ground (median 37.7 %). The paper's text (p. 2) defines x_s as the displacement "relative to the base of each oscillator", which fits the spring deformation. The benchmark uses it the same way: k_s·RMS is the RMS force in the secondary spring.

**Quantities of interest.**
- `rms` = √E[x_s²] (Eq. 37), RMS deformation of the secondary spring; 7 inputs (all but F_s).
- `G` = F_s − 3 k_s · rms, the limit state of the reliability benchmark (failure if G ≤ 0); 8 inputs. The factor 3 and F_s come from the benchmark (OpenTURNS), not from the 1985 paper.
- `rms_exact`: √ of the exact stationary mean square of x_s − x_p from `exact_mean_squares` (Lyapunov equation A P + P Aᵀ + π S_0 b bᵀ = 0, one-sided PSD S_0 ↔ intensity π S_0). For the accuracy check only; not used in the analysis.

**Inputs** (m = 7 for rms, 8 for G). The 1985 paper is deterministic: it has no input distributions. The distributions are those of the reliability benchmark built on this model by Der Kiureghian and De Stefano (1991), as given on the OpenTURNS oscillator use-case page (which cites Der Kiureghian and De Stefano 1991 and Dubourg et al. 2013); UQTestFuns' "damped oscillator" page gives the same seven for the RMS (its lognormal μ, σ equal ours to the digits it prints) and attributes them to Der Kiureghian and De Stefano 1991. We did not read the 1991 or 2013 papers. All independent lognormal, given by mean and coefficient of variation; ln x ~ N(μ, σ²) with σ = √ln(1 + CoV²), μ = ln(mean) − σ²/2.

| name | symbol | description | mean | CoV | μ | σ | source |
|---|---|---|---|---|---|---|---|
| `mp` | m_p | primary (structure) mass | 1.5 | 0.1 | 0.40049 | 0.099751 | OpenTURNS; UQTestFuns |
| `ms` | m_s | secondary (equipment) mass | 0.01 | 0.1 | −4.61015 | 0.099751 | OpenTURNS; UQTestFuns |
| `kp` | k_p | primary spring stiffness | 1 | 0.2 | −0.01961 | 0.198042 | OpenTURNS; UQTestFuns |
| `ks` | k_s | secondary spring stiffness | 0.01 | 0.2 | −4.62478 | 0.198042 | OpenTURNS; UQTestFuns |
| `zeta_p` | ζ_p | primary damping ratio | 0.05 | 0.4 | −3.06994 | 0.385253 | OpenTURNS; UQTestFuns |
| `zeta_s` | ζ_s | secondary damping ratio | 0.02 | 0.5 | −4.02359 | 0.472381 | OpenTURNS; UQTestFuns |
| `S0` | S_0 (paper: G_0) | white-noise base-acceleration intensity, one-sided PSD | 100 | 0.1 | 4.60020 | 0.099751 | OpenTURNS; UQTestFuns |
| `Fs` | F_s | force capacity of the secondary spring | 27.5 | 0.1 | 3.30921 | 0.099751 | OpenTURNS |

At the input means ω_p = 0.8165, ω_s = 1, β = −0.202, γ = 0.00667, E[x_s²] = 20643.07, RMS = 143.677, G = 23.190.

**Samples and physical meaning.** Every input is positive by construction. No damping ratio reaches 1 (largest ζ_p 0.224, ζ_s 0.068), so every sample is a physically meaningful underdamped system; none is dropped. The perturbation theory behind Eq. 37 assumes small damping and small γ (paper Eqs. 4–5); the sample has γ ∈ [0.0046, 0.0110] and β ∈ [−0.686, 0.260], and its error grows with ζ_p (see Accuracy). No sample fails (G ≥ 7.293), as expected for P_f ≈ 3.8e-7. No duplicate rows.

**Transform and scaling.** active_subspaces v1 has uniform and Gaussian spaces only, so the analysis is in the log inputs z = ln x, which are exactly Gaussian: `ac.Gaussian(MU, SIGMA**2)`, normalized coordinates (z − μ)/σ for **every** input (one scaling rule for all eight). `inputs.csv` holds physical x; the notebook converts once (Z = ln X) in its load cell.

**Gradients.** Exact: forward-mode chain rule through ω_p, ω_s, ω_a, ζ_a, β, γ, N, D, hand-coded in `grad_log_mean_square`; ∂rms/∂z = ½ rms ∂ln E/∂z; ∂G/∂z_j = −3 k_s ∂rms/∂z_j, plus −3 k_s rms for k_s and F_s for F_s. Physical gradients are these divided by x_j.

**Structure.** E[x_s²] depends on (m_p, m_s, k_p, k_s) only through ω_p, ω_s, γ, and is S_0 times a function of the frequencies homogeneous of degree −3, so in z it has two exact null directions (all four of m_p, m_s, k_p, k_s scaled together; frequencies scaled by c with S_0 by c³). For G two combinations are null (masses by c⁻², S_0 by c³; stiffnesses by c², S_0 by c⁻¹). These show as eigenvalues at rounding level (1e-28 to 1e-32).

**Sampling design.** M = 1000 independent Monte Carlo draws (not Latin hypercube), x = exp(μ + σ ⊙ ξ), ξ = `numpy.random.default_rng(1985).standard_normal((1000, 8))`. M ≥ 10 m and ≥ 300. The sample kurtoses of the normalized log inputs are 2.73–3.64. The largest, for ζ_p, is 4.1 standard errors above 3: it is a rare draw (0.46 % of simulated normal samples of this size reach it), driven by one draw at −4.21 standard deviations, and it is 3.43 without that draw. It is a property of this sample, not of the transform, and we keep it; see the notebook's data-check cell.

**Published results.** The 1985 paper gives curves only: Fig. 5 (amplification factor vs β from Eqs. 35–37 and "exact", ζ_p = 0.04, ζ_s = 0.02, γ = 0.01) and Figs. 6–8 (error criteria). No sampling study, no subspace. Benchmark reference values: P_f = 3.78 × 10⁻⁷ (OpenTURNS, method not stated); RMS mean 188.48 and variance 14 628 (UQTestFuns, Monte Carlo with 10⁷ samples, seed 42; 10⁶: 188.33, 14 596).

## Regenerating

```bash
python generate_data.py                           # writes the three CSVs here
python generate_data.py --out /tmp/osc --check    # writes elsewhere and compares
```

The run takes 0.3 s (0.4 s with Python start-up; 0.25 s of it is the 1000 Lyapunov solves) and prints the gradient check. `--check` on 2026-10-30 (numpy 2.5.4, scipy 1.18.1, Python 3.13.18) reported the three files identical to the shipped ones. The notebook runs in 5–8 s (12.6 s under `nbconvert`).

## Accuracy

- **Values.** Closed form in double precision; `oscillator_model` reproduces `outputs.csv` exactly (printed in the notebook).
- **Gradients.** Exact. Against central finite differences in z with step 10⁻⁶ at all 1000 samples, the largest relative error (row 2-norm) is 4.3 × 10⁻¹⁰ for rms and 2.7 × 10⁻¹⁰ for G, the size of the differencing error.
- **Eq. 37 against the exact 2-DOF response** (a property of the published model, not a numerical error): relative error in the mean-square spring deformation, median 1.4 %, 1st/99th percentiles −8.1 % / +3.8 %, largest 18.5 % at the sample with the largest ζ_p (0.224). The data set uses Eq. 37, as the benchmark does; `rms_exact` is shipped so users can compare.
- **Benchmark moments and P_f** (notebook, 10⁶ fresh samples, seed 1985): RMS mean 188.41 ± 0.24 and variance 14 575 ± 50 (95 %), against UQTestFuns' 188.48 and 14 628 (10⁷ samples; the variance differs by about two standard errors; their 10⁶-sample 14 596 is inside our interval). P_f = 3.81 × 10⁻⁷, 95 % interval [3.77, 3.86] × 10⁻⁷, by importance sampling (10⁶ samples) at the FORM design point, which contains OpenTURNS' 3.78 × 10⁻⁷. FORM alone gives 3.89 × 10⁻⁶ (10.2 times too high): the failure domain wraps around perfect tuning (design point β = −0.0012).

## How these files relate to the paper

- **Not the paper's computation.** The 1985 paper evaluates Eq. 37 deterministically (Fig. 5). The distributions, the capacity F_s and the limit state come from the later benchmark (Der Kiureghian and De Stefano 1991, via OpenTURNS/UQTestFuns); the samples are new.
- **Fig. 5 reproduced from the model.** The notebook recomputes Fig. 5 and compares it with values read from the page image by pixel position (about ±0.2; the paper prints no numbers). Eq. 37 gives 25.25, 22.75, 16.76 at β = 0, 0.05, 0.1, against 25.2, 22.9, 16.75 read from the "matched" curve. The figure's "exact" curve (from the paper's Ref. 3, Crandall and Mark) is within 0.58 of our exact spring-deformation solution at the six points read and agrees at β = 0 (25.85 vs 25.7), where the ground-relative x_s is 2.05 too high (27.75). Between β = 0.1 and 0.25 the figure's curve lies between our two exact curves. A variant with ω_p² = k_p/(m_p + m_s) is within 0.36 at all six points; that is our guess at how the exact curve was parameterized, not something the paper states.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equation; no data were copied, and no outside code was used. The input distributions are parameter values from the OpenTURNS oscillator use-case page (https://openturns.github.io/openturns/latest/usecases/use_case_oscillator.html, "© Copyright 2005-2026 Airbus-EDF-IMACS-ONERA-Phimeca", no separate licence stated on the page) and the UQTestFuns damped-oscillator page (D. Wicaksono and M. Hecht, *J. Open Source Softw.* 8(90):5671, 2023, doi:10.21105/joss.05671; no licence stated on the page). Cite:

- T. Igusa and A. Der Kiureghian, "Dynamic characterization of two-degree-of-freedom equipment-structure systems", *J. Eng. Mech.* 111(1):1–19, 1985, doi:10.1061/(ASCE)0733-9399(1985)111:1(1) (model; NSF Grant CEE-8217800).
- A. Der Kiureghian and M. De Stefano, "Efficient algorithm for second-order reliability analysis", *J. Eng. Mech.* 117(12):2904–2923, 1991, doi:10.1061/(ASCE)0733-9399(1991)117:12(2904) (reliability benchmark and input distributions, as attributed by OpenTURNS and UQTestFuns; not read).
- V. Dubourg, B. Sudret and F. Deheeger, "Metamodel-based importance sampling for structural reliability analysis", *Probab. Eng. Mech.* 33:47–57, 2013, doi:10.1016/j.probengmech.2013.02.002 (cited by OpenTURNS for the benchmark; not read).
