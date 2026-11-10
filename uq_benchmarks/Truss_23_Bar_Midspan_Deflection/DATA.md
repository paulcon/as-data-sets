# Truss_23_Bar_Midspan_Deflection: data

Kind: **Python model**. The model is `truss_model.py` (numpy and the standard library), a bar finite-element model of the 23-bar truss written from the paper's description and Fig. 8. `generate_data.py` draws the inputs with a saved seed and writes all four CSVs; a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Truss_23_Bar_Midspan_Deflection.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2011; outputs committed |
| `truss_model.py` | `deflection(Z, grad=False)`, `deflection_xi`, `physical`, `dphysical_dxi`, `sample_xi`, `displacements`, `bar_forces`, `virtual_work`, `gumbel_cdf`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `DISTRIBUTIONS`, `MEAN`, `SD`, `LN_MU`, `LN_S`, `GUM_U`, `GUM_BETA`, `NODES`, `BARS` | the model |
| `generate_data.py` | draws the samples, writes the four CSVs, checks the gradients by central differences; `--check` compares a rerun with the shipped files | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below marked with its name | check |
| `inputs.csv` | 1000 × 10 physical inputs E1, E2 (Pa), A1, A2 (m²), P1–P6 (N) | model input |
| `inputs_standard.csv` | 1000 × 10: the same samples in standard-normal coordinates ξ | model input |
| `outputs.csv` | 1000 × 1: midspan deflection V1 (m, positive downward) | model output |
| `gradients.csv` | 1000 × 10: ∂V1/∂(E1, E2, A1, A2, P1, …, P6) with respect to the **physical** inputs (m/Pa, m/m², m/N) | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits. The notebook maps the gradients to ξ in one stated cell: ∂V1/∂ξ_i = (∂V1/∂Z_i) dZ_i/dξ_i (`truss_model.dphysical_dxi`).

## Model

**Source.** G. Blatman and B. Sudret, "Adaptive sparse polynomial chaos expansion based on least angle regression", *Journal of Computational Physics* 230(6) (2011) 2345–2367, doi:10.1016/j.jcp.2010.12.021 (publisher version read). Example #2, Sec. 6.3 "Maximum deflection of a truss structure", pp. 2357–2359, Fig. 8, Tables 1 and 2. The paper cites S. H. Lee and B. M. Kwak, *Structural Safety* 28(3) (2006) 261–272, doi:10.1016/j.strusafe.2005.08.003 (its ref. [56]) for the example; that paper was not opened.

**Structure** (p. 2357 and Fig. 8). 23 members, 11 horizontal and 12 oblique bars, modelled with 23 bar elements; span 24 m, height 2 m; six vertical loads on the upper chord; response the midspan deflection V1. From Fig. 8 (no table of nodes in the paper; this data set's reading): bottom nodes at x = 0, 4, …, 24 m, top nodes at x = 2, 6, …, 22 m, y = 2 m; each top node joined to the two bottom nodes below it; pin at x = 0, roller at x = 24 m; P1…P6 downward at the top nodes from left to right; V1 the downward displacement of the bottom node at x = 12 m. Horizontal bars E1, A1; oblique bars E2, A2 (p. 2358).

**Equations.** Linear-elastic bar elements, small displacements: K u = f with K = E1A1 K_h + E2A2 K_d (unit-property assemblies of c cᵀ/L_e over the horizontal and oblique bars), V1 = −qᵀu. The truss is statically determinate, so by the unit-load method V1 = Σ_j P_j (a_j/(E1A1) + b_j/(E2A2)) with a = (36, 100, 140, 140, 100, 36) m² and b = (2.828427, 8.485281, 14.142136, 14.142136, 8.485281, 2.828427) m² (`check_accuracy.py`, check 1). The horizontal bars contribute between 0.7439 and 0.9081 of V1 over the 1000 samples (same check).

**Inputs** (m = 10, independent, Table 1 p. 2358, order of Eq. (54)):

| name | units | distribution | mean | sd | parameters used here | source |
|---|---|---|---|---|---|---|
| `E1`, `E2` | Pa | lognormal | 2.10e11 | 2.10e10 | ln Z ~ N(μ, s²), s = √ln(1 + 0.1²) = 0.099751, μ = ln(mean) − s²/2 | Table 1, p. 2358 |
| `A1` | m² | lognormal | 2.0e-3 | 2.0e-4 | same s | Table 1, p. 2358 |
| `A2` | m² | lognormal | 1.0e-3 | 1.0e-4 | same s | Table 1, p. 2358 |
| `P1`–`P6` | N | Gumbel | 5.0e4 | 7.5e3 | largest-value Gumbel, β = sd √6/π = 5847.73 N, u = mean − γ_E β = 46624.60 N | Table 1, p. 2358 |

The paper gives mean and standard deviation only; reading them as the moments of the variable itself and the Gumbel as the largest-value type is this data set's choice (the standard one for loads). At the input means V1 = 0.077836 m.

**Coordinates and scaling.** As in the paper's Eq. (55), every input is mapped through its CDF to a standard normal, ξ_i = Φ⁻¹(F_i(Z_i)); the data set stores both Z and ξ and the analysis is done in ξ (space N(0, I₁₀)). All inputs are scaled the same way (through the CDF). In ξ, ∂V1/∂ξ_E1 = ∂V1/∂ξ_A1 and ∂V1/∂ξ_E2 = ∂V1/∂ξ_A2 exactly (same s, product dependence), so two eigenvalues are zero in exact arithmetic.

**Quantity of interest.** V1 (m), positive downward.

**Gradients.** Exact, by one adjoint solve K λ = q: ∂V1/∂E1 = A1 λᵀK_h u, ∂V1/∂A1 = E1 λᵀK_h u (E2, A2 with K_d), ∂V1/∂P_j = λ at the vertical dof of top node j. `gradients.csv` holds the physical gradients.

**Sampling design.** M = 1000 independent Monte Carlo draws ξ ~ N(0, I₁₀), `numpy.random.default_rng(2011).standard_normal((1000, 10))`, mapped to Z by the inverse CDFs. None is outside physical meaning: every input positive, loads in [33 429, 92 169] N; none dropped; no duplicates.

**Published results** (no active subspace in the paper). Table 2, p. 2358, total Sobol' indices (reference by crude Monte Carlo, 5 500 000 FE runs; stepwise sparse PC, 60 runs; LAR sparse PC, 70 runs): A1 0.388/0.374/0.374, E1 0.367/0.378/0.373, P3 0.075/0.073/0.073, P4 0.079/0.069/0.079, P5 0.035/0.036/0.034, P2 0.031/0.037/0.037, A2 0.014/0.013/0.013, E2 0.010/0.014/0.012, P6 0.005/0.005/0.004, P1 0.004/0.005/0.005; relative L2 error 6e-3 (stepwise) and 5e-3 (LAR); PC degree 3. Fig. 9 (p. 2359): convergence of full and sparse PC (not reproduced here: no PC is fitted). No mean or standard deviation of V1 is reported.

**What the notebook finds.** n = 1: λ1/λ2 = 105.49, next ratios 1.24–2.65, λ1 holds 97.558 % of the trace, bootstrap subspace distance 0.0051 at k = 1 (≥ 0.0476 for k = 2…7); λ9, λ10 at the round-off level (of order 10⁻³³ and 10⁻³⁶ of λ1 in the committed run; the values change between platforms). w1 = (−0.6109, −0.1110, −0.6109, −0.1110, +0.0661, +0.1899, +0.2713, +0.2718, +0.1908, +0.0676), signs set so that y1 has positive covariance with V1. A quadratic in y1 has leave-one-out R² 0.9873 (0.9926 with two active variables). The normalized activity scores λ1 w_i1²/Σ agree with the notebook's Jansen total Sobol' indices within 0.0039 and with the paper's reference column within 0.0148 (largest for A1).

## Regenerating

```bash
python generate_data.py --out DIR --check     # writes to DIR (must be outside this directory) and compares
python generate_data.py --out .               # rewrites the shipped CSVs (explicit only)
python check_accuracy.py                      # the numbers marked below, about 2 minutes
```

`--out` is required, and `--check` refuses this directory and any directory inside it. One process. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-10) `--check` takes under 1 s (the script reports 0.3 s, of it 0.03 s for the 1000 solves with adjoints) and passes: `inputs_standard.csv` is reproduced exactly, and the largest relative differences from the other shipped files are at the rounding level, of order 1e-16 (`inputs.csv`), 1e-14 (`outputs.csv`) and 1e-13 (`gradients.csv`). Stated tolerances: 1e-15 (ξ), 1e-13 (Z, rounding of exp and log), 1e-12 (V1), 1e-11 (gradients), for other platforms' linear-algebra libraries. The notebook runs in about 4–8 s in `.venv` by its own timer.

## Accuracy

- **Model vs file and the unit-load method.** The stiffness solver reproduces `outputs.csv` to rounding level, of order 1e-14 (relative, notebook). The unit-load method (`truss_model.virtual_work`, equilibrium only, no stiffness matrix) agrees to the same order at the first 100 samples (notebook); the closed form above agrees with `outputs.csv`, and its gradients with `gradients.csv` (in ξ, row 2-norm), to rounding level, below 1e-13, at all 1000 rows (`check_accuracy.py`, check 1).
- **Gradients vs finite differences.** Central differences in ξ, step 1e-5, all 1000 samples: largest relative error 1.4e-8 (row 2-norm), the size of the differencing error (`generate_data.py` and notebook).
- **Transform.** The Gumbel CDF of the stored P1 equals Φ(ξ) to rounding level, of order 1e-16 (notebook).
- **Moments and total Sobol' indices** (`check_accuracy.py`, check 2; Jansen estimator, N = 500 000 base samples in ξ, 5 500 000 FE runs, seed 737, about 60 s): mean V1 = 0.079417 ± 0.000031 m (95 %), sd 0.011065 m (variance ± 0.43 %), coefficient of variation 0.1393. Total indices (95 % half-width of the numerator): E1 0.3710 ± 1.6e-3, A1 0.3711 ± 1.5e-3, E2 0.0127, A2 0.0127 (± 5.3e-5), P1 0.0048, P6 0.0048 (± 2.5e-5), P2 0.0378, P5 0.0377 (± 2.0e-4), P3 0.0781, P4 0.0777 (± 4.0e-4); sum 1.0084. The notebook repeats this with N = 20 000.
- **Sampling spread of total-index estimators at N = 500 000** (`check_accuracy.py`, check 3; 100 replicates at N = 20 000 with the closed form, sd scaled by √(20 000/500 000)). Jansen (1999), T_i = mean((f_A − f_i)²)/(2V): sd 0.0010 for E1 and A1, 0.00013 for P2. Homma and Saltelli (1996), T_i = 1 − (mean(f_A f_i) − f0²)/V, and the Sobol' (2007) form mean(f_A (f_A − f_i))/V, which coincides with it when f0 and V come from f_A: sd 0.010 (E1), 0.0092 (A1), 0.0027 (P2), 0.0031 (P5), large because E[V1]²/Var V1 = 51.5. Measured against the check-2 values, the Table 2 reference entries are within 2.5 sd of the truth under the Homma–Saltelli form (A1 +1.8, E1 −0.4, P2 −2.5, P5 −0.9, E2 −1.6, A2 +0.8) but up to 68.7 sd away under Jansen (A1 +16.1).
- In `.venv` (2026-11-10) `check_accuracy.py` takes about 106 s (about 60 s for check 2 and 47 s for check 3) and prints the values above.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from the authors' finite-element runs (code not named). These data are an independent numpy computation from the paper's description, Fig. 8 and Table 1, with the choices stated above (node layout and supports read from the figure, the moment and Gumbel parameterization). The 1000 Monte Carlo samples are new; the paper used Sobol' quasi-random designs for its PC fits (p. 2359) and crude Monte Carlo for its reference.
- **Agreement.** The total Sobol' indices from `check_accuracy.py` are close to both of the paper's columns; the largest gap from the reference column is for A1 (0.3711 against 0.388), and from the LAR column for P3 (0.0781 against 0.073). The paper notes (p. 2358) that V1 depends on E1, A1 only through E1A1 and that they have the same distribution type and coefficient of variation, which makes their total indices exactly equal; the gap between its reference values 0.388 and 0.367 is therefore estimation error. It is consistent with the sampling error of a Homma–Saltelli-type pick-freeze estimator at N = 5 × 10⁵ (sd 0.0092 for A1, 0.010 for E1, 0.0027 for P2), not with a Jansen estimator of that size (sd 0.0010; Accuracy, check 3). By the symmetry of the truss P2 and P5 also have equal indices (here 0.0378 and 0.0377); the reference gives 0.031 and 0.035, the sparse expansions 0.037 and 0.036/0.034.
- **The paper's 5 500 000 runs** equal N(m + 1) with N = 500 000, which suggests a pick-freeze estimator of that size; the spreads of check 3 point to a Homma–Saltelli-type estimator rather than Jansen's (an inference; the paper does not name its estimator). `check_accuracy.py` uses that size.

## License and credit

Code (`truss_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published description; no data or code were copied. Cite G. Blatman and B. Sudret, *J. Comput. Phys.* 230 (2011) 2345–2367, doi:10.1016/j.jcp.2010.12.021, and, for the original example, S. H. Lee and B. M. Kwak, "Response surface augmented moment method for efficient reliability analysis", *Structural Safety* 28(3) (2006) 261–272, doi:10.1016/j.strusafe.2005.08.003. The total-index estimators are those of M. J. W. Jansen, *Comput. Phys. Commun.* 117(1–2) (1999) 35–43, doi:10.1016/S0010-4655(98)00154-4, and T. Homma and A. Saltelli, *Reliab. Eng. Syst. Saf.* 52(1) (1996) 1–17, doi:10.1016/0951-8320(96)00002-6. The paper states no funding.

## Unknowns

- Node coordinates, connectivity and supports are read from Fig. 8; the paper has no table. The symmetric Sobol' indices and the agreement above support the reading.
- Whether the Gumbel is the largest-value type, and whether Table 1 gives moments of the variables (assumed for both).
- The reference estimator behind Table 2 (5 500 000 runs); check 3 narrows it to a high-variance pick-freeze form such as Homma–Saltelli.

## Open questions

Only the authors could settle the three points above: the geometry read from Fig. 8, the Gumbel type and moment reading, and which estimator and N produced the 5 500 000-run reference column of Table 2 (its pattern of deviations fits a Homma–Saltelli/Sobol'-type pick-freeze estimator with N = 5 × 10⁵). Table 2 states no sampling error for the reference column. None of these changes the active-subspace analysis.
