# Fire_Satellite_Coupled_System: data

Kind: **Python model**. The model is `fire_satellite_model.py`, a numpy/scipy reimplementation of the three-discipline fire detection satellite of Sankararaman and Mahadevan (2012), and `generate_data.py` writes the three CSVs from it with a saved seed.

## Files

| file | contents | role |
|---|---|---|
| `Fire_Satellite_Coupled_System.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2012`; outputs committed; about 11 s under nbconvert |
| `fire_satellite_model.py` | `solve`, `qoi`, `grad`, `sample`, `orbit`, `attitude`, `power`; `NAMES`, `UNITS`, `MEAN`, `SD`, `LOWER`, `UPPER`, the Table 1 constants | the model |
| `generate_data.py` | writes the three CSVs; `--out DIR --check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 2000 × 9 physical inputs, columns in the order of `NAMES` | model input |
| `outputs.csv` | 2000 × 3: τ_tot (N m), P_tot (W), A_sa (m²) | model output |
| `gradients.csv` | 2000 × 27: column 9 i + j is ∂(output i)/∂(input j), with respect to the **physical** inputs | model output |

The CSVs are comma-separated numbers (17 significant digits) after two `#` lines (a description, then the column names with units), so `np.loadtxt(path, delimiter=',')` reads them.

## Model

**Citation.** S. Sankararaman and S. Mahadevan, "Likelihood-Based Approach to Multidisciplinary Analysis Under Uncertainty", *Journal of Mechanical Design* 134(3):031008, 2012, doi:[10.1115/1.4005619](https://doi.org/10.1115/1.4005619). Section 6, "Three-Discipline Fire Detection Satellite Model" (pp. 031008-7 to -11). The problem is from Wertz and Larson, *Space Mission Analysis and Design* (Microcosm, 1999), in the three-subsystem form of Ferson et al. (NASA SBIR report, 2009) and Zaman (Vanderbilt PhD thesis, 2010).

**Equations.** Orbit, Eqs. (11)–(14); attitude control, Eqs. (15)–(22); power, Eqs. (23)–(33). Deterministic constants: Table 1, p. 031008-10. The feedback coupling P_ACS → (I_max, I_min) → P_ACS is solved by fixed-point iteration on P_ACS, started at P_ACS = n P_hold = 60 W and stopped when successive iterates agree to a relative 1e-13 (real and, for the complex step, imaginary parts separately). Every sample converges in 3 sweeps. The model was written from the paper; no code was copied (the OpenTURNS implementation, LGPL, was only run as a check, below).

Choices where the printed paper is ambiguous or inconsistent:

1. **ω_max = 6000 rpm is converted to rad/s** (628.3 rad/s). With it, P_ACS at the mean inputs is 67.33 W and its sample mean 67.32 W, matching the peak of the paper's Fig. 11 (about 67 W) and P_tot (mean 1066 W) matching Fig. 12 (peak about 1070 W). The OpenTURNS use case uses 6000 as rad/s, which gives P_ACS ≈ 130 W and P_tot ≈ 1130 W at the mean.
2. **Eq. (27)** prints L = sqrt(A_sa r_lw / m_sa) and W = sqrt(A_sa / (r_lw m_sa)) with m_sa, the array mass, which Eq. (27) itself defines from L and W. We read it as n_sa = 3, the number of solar arrays (Table 1), as OpenTURNS does.
3. **ε_deg = 0.0375 "% per year"** (Table 1) is used as the fraction 0.0375 per year in P_EOL = P_BOL (1 − ε_deg)^LT (as in OpenTURNS); read as 0.0375 %, A_sa would be about 6.3 m², far from Fig. 13.
4. θ and i are in degrees and converted to radians; θ_slew from Eq. (14) is in radians.
5. μ = 3.986e14 m³ s⁻² (Table 1; OpenTURNS uses 3.986004418e14).

**Inputs** (m = 9). Table 2, p. 031008-10, gives independent normals; the paper does not truncate them (Sec. 6.2: "All the stochastic quantities are treated to be normally distributed"). We use the paper's means and standard deviations, truncated to the bounds of the OpenTURNS use case so that every input stays physical (e.g. C_d > 0, q in [0, 1]):

| name | symbol | unit | mean | sd | lower | upper | source |
|---|---|---|---|---|---|---|---|
| `H` | altitude H | m | 18,000,000 | 1,000,000 | 15e6 | 21e6 | Table 2, p. 10; bounds OpenTURNS |
| `P_other` | power other than ACS | W | 1000 | 50 | 850 | 1150 | Table 2; OpenTURNS |
| `F_s` | average solar flux | W/m² | 1400 | 20 | 1340 | 1460 | Table 2; OpenTURNS |
| `theta` | deviation of moment axis θ | deg | 15 | 1 | 12 | 18 | Table 2; OpenTURNS |
| `L_sp` | moment arm, radiation torque | m | 2 | 0.4 | 0.8 | 3.2 | Table 2; OpenTURNS |
| `q` | reflectance factor | – | 0.5 | **1** | 0.2 | 0.8 | Table 2 (sd as printed; OpenTURNS has 0.1); OpenTURNS bounds |
| `R_D` | residual dipole | A m² | 5 | 1 | 2 | 8 | Table 2; OpenTURNS |
| `L_a` | moment arm, aerodynamic torque | m | 2 | 0.4 | 0.8 | 3.2 | Table 2; OpenTURNS |
| `C_d` | drag coefficient | – | 1 | 0.3 | 0.1 | 1.9 | Table 2; OpenTURNS |

The bounds are mean ± 3 sd for the eight inputs other than q. For q, the paper's sd of 1 (probably a typo for 0.1) with the bounds [0.2, 0.8] makes q close to uniform on [0.2, 0.8]. The choice does not matter: q enters only the solar radiation torque, and the notebook shows that sd 0.1 gives the same first eigenvectors (|w1 · w1'| = 1.00000 for all three outputs).

**Quantities of interest** (Sec. 6.3): total torque τ_tot (N m), total power P_tot (W), solar array area A_sa (m²). At all 2000 samples τ_tot = τ_dist (the disturbance torque exceeds the slewing torque), and the aerodynamic torque is at least 99.1 % of τ_dist.

**Gradients.** Complex step (Martins, Sturdza and Alonso, ACM TOMS 29(3), 2003, doi:10.1145/838250.838251): input j is perturbed by i · 1e-30 · sd_j and the whole fixed-point iteration runs in complex arithmetic until both parts of P_ACS converge, so the result is the derivative of the converged fixed point with no subtractive cancellation. `max`, `min` and `|·|` select by the real part. Gradients are with respect to the physical inputs; the notebook passes them with `ac.Gaussian(MEAN, SD**2)`, which maps them to the normalized coordinates x_j = (ξ_j − μ_j)/σ_j.

**Sampling design.** N = 2000 independent draws from the truncated normals (`scipy.stats.truncnorm`, `numpy.random.default_rng(2012)`, one column after another). This is ≥ 10 m and ≥ 300, as needed for the gradient estimator. The notebook normalizes with the paper's untruncated means and standard deviations (`ac.Gaussian`); since the samples come from the truncated density, Ĉ estimates C under the truncated density in those coordinates. The approximation is only that v1's space object describes an untruncated normal; nothing in the notebook samples from it.

**Published results.** The paper reports no active subspace, gradient or sensitivity results. It gives the distributions of P_ACS (Fig. 11) and of the three outputs (Figs. 12–14) from a fixed-point Monte Carlo benchmark (SOFPI, about 200,000 evaluations of each subsystem, no seed) and from its SOMDA and LAMDA methods (10 integration points). Peaks read by eye from the figures: P_ACS ≈ 67 W, P_tot ≈ 1070 W, A_sa ≈ 12.2 m², τ_tot ≈ 0.011 N m.

## Regenerating

```bash
python generate_data.py                                   # writes the three CSVs here
python generate_data.py --out /tmp/fs --check             # writes elsewhere and compares
```

The shipped files were written on 2026-10-30 in a Linux container (numpy 2.5.4, scipy 1.18.1, Python 3.13). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) a rerun takes 0.09 s for the 2000 samples (1.6 s including start-up), and `--check` prints maximum relative differences of 1.0e-15 (inputs), 1.0e-15 (outputs) and 8.6e-16 (gradients): rounding error.

## Accuracy

- **Fixed point.** The contraction factor of one sweep, dP_ACS^(k+1)/dP_ACS^(k) at the mean inputs, is −1.7e-10 (complex step, printed in the notebook): the coupling through the moments of inertia is very weak. So the outputs at the OpenTURNS tolerance of 1e-3 (2 sweeps) differ from those at 1e-13 by at most a relative 2e-16 (rounding error), and the fixed-point-tolerance concern raised for this model does not arise. The tight tolerance is kept anyway.
- **Gradients.** Against central differences of the real solve (tolerance 1e-15) with step 1e-4 sd_j, on 20 samples (in the notebook), the largest relative errors of the gradient vectors are 1.1e-11 (τ_tot), 5.4e-10 (P_tot) and 9.0e-10 (A_sa). In a scratch check on 50 samples, steps 1e-4, 1e-5 and 1e-6 sd gave at most 1.0e-9, 1.2e-8 and 1.0e-7 (growing as the step shrinks, i.e. rounding error in the differences, not in the complex step).
- **Against OpenTURNS.** OpenTURNS 1.27 (`openturns.usecases.fire_satellite.FireSatelliteModel`, installed in a scratch environment and run, not copied) gives at its mean inputs τ_tot = 0.0116597, P_tot = 1129.96, A_sa = 11.8626. Our model with OpenTURNS's conventions (ω_max = 6000 rad/s, μ = 3.986004418e14) gives 0.0116597, 1129.96, 11.8626. The OpenTURNS code also has (2 R_E + H)³ in place of 2 (R_E + H)³ in the gravity-gradient torque; τ_g is 0.26 % of τ_dist at the mean inputs (at most 2 % over our samples), and squared in τ_dist, so this does not show at these digits. With 5000 samples (q sd 0.1) OpenTURNS gave means 0.01159, 1129.2, 11.862.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results are density estimates from its own Monte Carlo samples (untruncated normals, q sd as printed, unknown code and seed); the data here are an independent computation of the same equations, means and standard deviations, with truncation and the choices above.
- **Agreement.** P_ACS (mean 67.32 W), P_tot (mean 1066 W) and τ_tot (KDE mode 0.0103 N m, mean 0.0117 N m) agree with the peaks of Figs. 11, 12 and 14.
- **Disagreement in A_sa.** Our A_sa has mean 11.19 m² (sd 0.54); Fig. 13 peaks near 12.2 m², about 9 % higher, with a similar relative spread (the peak density of about 0.64 m⁻² corresponds to an sd of about 0.6 m² for a normal). Since A_sa depends on the inputs only through P_tot (which matches Fig. 12), H and F_s, the published Fig. 13 is not consistent with Eqs. (24)–(26) and Table 1 as printed. We tried the plausible alternative readings (ε_deg as 0.0375 %: 6.3 m²; T_e and T_d swapped in Eq. (24): 14.2 m²; ω_max used as rad/s: 11.86 m² at the mean) and none gives 12.2; Eq. (27) does not enter A_sa. The active subspace of A_sa is unaffected by a constant factor, but would change if the difference is in the eclipse/daylight terms.

## License and credit

Code: MIT. Data (the CSVs): CC BY 4.0. Model: S. Sankararaman and S. Mahadevan (2012), after J. Wertz and W. Larson (1999) and S. Ferson et al. (2009); the truncation bounds follow the OpenTURNS use case (https://openturns.github.io/openturns/latest/usecases/use_case_fire_satellite.html). The paper acknowledges NASA Langley Research Center (Contract No. NNX08AF56A1) and NASA Jet Propulsion Laboratory (Contract No. RSA 1400821). Cite Sankararaman and Mahadevan, J. Mech. Des. 134(3):031008, 2012, doi:10.1115/1.4005619.
