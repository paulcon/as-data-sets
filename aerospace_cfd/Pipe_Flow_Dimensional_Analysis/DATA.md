# Pipe_Flow_Dimensional_Analysis: data

Kind: **Python model**. The model is `pipe_model.py` (numpy only), written from the paper's equations. `generate_data.py` draws the inputs with a saved seed and writes the three CSVs; a rerun reproduces them to the tolerance under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Pipe_Flow_Dimensional_Analysis.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1708; outputs committed |
| `pipe_model.py` | `friction_factor(Q, law, grad)`, `pressure_gradient(Q, law, grad)`, `dimensionless_loss(Q, law, grad)`, `colebrook(Re, rr)`, `colebrook_residual`, `reynolds`, `relative_roughness`, `null_basis()`, `sample(regime, M, rng)`, `gauss_legendre(regime, npts)`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `BOUNDS`, `SOURCES`, `REGIMES`, `DIM_MATRIX`, `DIM_DPDX`, `W_PI`, `W_RE`, `W_RR`, `RE_CRIT` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs to `--out DIR`; `--check` compares a rerun with the shipped files; checks gradients by central differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `laminar.csv` | 1000 × 11, laminar case (Table 2) | model input and output |
| `turbulent.csv` | 1000 × 11, turbulent case (Table 3) | model input and output |
| `high_re.csv` | 1000 × 11, high Reynolds number case (Table 4) | model input and output |

Each CSV has two `#` lines (a description and the column names with units) and is read by `np.loadtxt(path, delimiter=',')`. Columns 0–4: ρ (kg/m³), μ (kg/(m s)), D (m), ε (m), V (m/s); column 5: dp/dx (kg m⁻² s⁻², i.e. Pa/m); columns 6–10: ∂(dp/dx)/∂ln q_i for q = (ρ, μ, D, ε, V), in the units of dp/dx (gradients with respect to the **log** inputs, q_i ∂(dp/dx)/∂q_i). Values are written with 17 significant digits. The notebook derives the dimensionless pressure loss π = (dp/dx) D/(ρV²) and its gradient in one stated cell.

## Model

**Source.** P. G. Constantine, Z. del Rosario, G. Iaccarino, "Data-driven dimensional analysis: algorithms for unique and relevant dimensionless groups", arXiv:1708.04303, 2017, [doi:10.48550/arXiv.1708.04303](https://doi.org/10.48550/arXiv.1708.04303) (preprint read; we found no journal version: the arXiv DOI is not in Crossref and no published version is known to us). Sec. 6, "Example: viscous pipe flow", pp. 27–41.

**Equations.** Re = ρVD/μ. The paper's computational model solves the Colebrook equation (Eq. (60), p. 30), 1/√λ = −2.0 log₁₀(ε/(3.7D) + 2.51/(Re√λ)), for the friction factor λ with a Newton method, "valid over a wide range of flow conditions, i.e., laminar and turbulent regimes" (p. 30), and computes dp/dx = λρV²/(2D) (Eq. (61), p. 30). The Poiseuille relation λ = 64/Re (Eq. (59)) is stated for Re < Re_c ≈ 3000 but is not the paper's model; `law="poiseuille"` exists only for the comparison in the notebook, which shows that it does not reproduce Table 5. `pipe_model.colebrook` iterates on s = 1/√λ from s = 8 until the relative Newton step is below 10⁻¹⁵ (at most 50 iterations; it raises RuntimeError otherwise).

**Dimensional analysis** (Sec. 6.1, p. 32). Dimension matrix D (Eq. (62); rows kg, m, s; columns ρ, μ, D, ε, V) = [[1, 1, 0, 0, 0], [−3, −1, 1, 1, 1], [0, −1, 0, 0, −1]]; dp/dx has dimension vector (1, −2, −2); w = (1, 0, −1, 0, 2); the null-space basis W is the last two right singular vectors of D. Dimensionless pressure loss π = (dp/dx) exp(−wᵀ ln q) = (dp/dx) D/(ρV²) = λ/2 (Eq. (7)).

**Inputs** (m = 5; Table 1, p. 30; bounds in Tables 2–4, pp. 31–32), uniform and independent on the box of the physical variables (Sec. 4.3, Eq. (34); Sec. 6.2, p. 32). No nominal values are given in the paper.

| name | symbol | units | laminar (Table 2, p. 31) | turbulent (Table 3, p. 31) | high Re (Table 4, p. 32) |
|---|---|---|---|---|---|
| `rho` | ρ, fluid density | kg/m³ | 0.10 – 0.14 | 0.10 – 0.14 | 0.10 – 0.14 |
| `mu` | μ, fluid viscosity | kg/(m s) | 1.0e-6 – 1.0e-5 | 1.0e-6 – 1.0e-5 | 1.0e-6 – 1.0e-5 |
| `D` | D, pipe diameter | m | 0.5 – 0.8 | 0.5 – 1.0 | 0.5 – 1.0 |
| `eps` | ε, pipe roughness | m | 3.0e-5 – 8.0e-5 | 5.0e-4 – 2.0e-3 | 1.0e-2 – 4.0e-2 |
| `V` | V, fluid velocity | m/s | 2.5e-2 – 3.0e-2 | 2.0 – 4.0 | 5.0e2 – 7.0e2 |

Over the boxes Re ranges over [125, 3360] (laminar), [1.0e4, 5.6e5] (turbulent) and [2.5e6, 9.8e7] (high Re); ε/D over [3.75e-5, 1.6e-4], [5e-4, 4e-3] and [0.01, 0.08] (notebook).

**Quantity of interest.** dp/dx (kg m⁻² s⁻²) on disk; the active subspace analysis uses π = (dp/dx) D/(ρV²), as the paper's Algorithms 1 and 2 do (pp. 26, 28). The paper's unique and relevant groups are π̂_i = exp(z_iᵀ ln q), z_i = W u_i, where u_i are the eigenvectors of C_γ = ∫∇_γ g ∇_γ gᵀ σ dγ, γ = Wᵀ ln q (Eqs. (35)–(40)). The notebook computes C_x = ∫∇_x π ∇_x πᵀ ρ dq in x = ln q, which equals W C_γ Wᵀ, so its nonzero eigenpairs are the paper's (λ_i, z_i).

**Gradients.** Exact: implicit differentiation of the Colebrook equation, ∂s/∂ln a = −(2/ln 10)(a/t)/F_s and ∂s/∂ln b = −(2/ln 10)(bs/t)/F_s with a = ε/(3.7D), b = 2.51/Re, t = a + bs, F_s = 1 + (2/ln 10) b/t, then the chain rule through ln a = ln ε − ln D − ln 3.7 and ln b = ln 2.51 − ln Re. The CSVs hold ∂(dp/dx)/∂ln q. The paper itself uses first-order finite differences with step h = 10⁻⁶ (p. 35).

**Sampling design.** M = 1000 independent Monte Carlo draws per case, uniform on the box of q, `numpy.random.default_rng([1708, k]).random((1000, 5))` with k = 0, 1, 2 for laminar, turbulent, high Re. This design is this data set's choice; the paper uses a tensor Gauss–Legendre rule with 11 points per variable (161051 points), which the notebook repeats with the model (`pipe_model.gauss_legendre`) for the comparison. Every sample lies inside its box. The largest laminar sample Reynolds number is 2551 (none above Re_c ≈ 3000), but the laminar box itself reaches 3360 at one corner; the Colebrook equation is used for laminar flow because the paper's model does so. The laminar case is therefore not laminar physics: on the 1000 laminar samples the Poiseuille λ = 64/Re is 0.55 to 3.26 times the Colebrook λ (notebook), so its results describe Colebrook's extrapolation to low Re.

**Published results** (Algorithm 2 / Algorithm 1). Table 5 (laminar, p. 36): z₁ = (0.500, −0.500, 0.500, −0.001, 0.500) / (0.500, −0.500, 0.500, 0.000, 0.500), z₂ = (−0.189, 0.189, 0.567, −0.756, −0.189) / (0.189, −0.189, −0.567, 0.756, 0.189), λ = 2.39e-2, 3.39e-9 / 2.19e-2, 2.65e-8. Table 6 (turbulent, p. 38): z₁ = (0.309, −0.309, 0.732, −0.423, 0.309) / (0.304, −0.304, 0.734, −0.429, 0.304), z₂ = (0.436, −0.436, −0.190, 0.627, 0.436) / (0.440, −0.440, −0.183, 0.622, 0.440), λ = 3.58e-4, 1.70e-5 / 3.34e-4, 2.09e-5; Eqs. (64)–(65) (p. 37): π̂₁ = Re^0.309 (ε/D)^−0.423, π̂₂ = Re^0.436 (ε/D)^0.627 (Alg 2). Table 7 (high Re, p. 38): z₁ = (0, 0, 0.707, −0.707, 0), z₂ = (−0.535, 0.535, −0.267, −0.267, −0.535) (both algorithms), λ = 5.71e-3, 6.97e-11 / 5.30e-3, 1.31e-10. Fig. 4 (p. 34): eigenvalues of C for dp/dx against the finite difference step, two going to zero. Fig. 5 (p. 36): convergence of z with the step. Fig. 6 (p. 40): ln π in colour against (ln Re, ln ε/D) and (ln π̂₁, ln π̂₂), turbulent case; "approximately 127 degrees" of rotation (p. 39).

**What the notebook finds.** Estimator `as` (exact gradients), 500 bootstrap replicates, in the coordinates ln q (the library's box is given a common half-width L = 1.2 in every ln q_i, so its normalization is a shift and a common scale; eigenvalues are divided by L²). In every case λ₃, λ₄, λ₅ are at rounding level (dimensional analysis leaves two groups). Laminar: λ₁ = 1.511e-3, λ₂ = 1.887e-10, λ₁/λ₂ = 8.0e6, n = 1, z₁ ≈ Re exponents / 2. Turbulent: λ₁ = 2.232e-5, λ₂ = 1.123e-6, λ₁/λ₂ = 19.87 (bootstrap 18.5–21.4), n = 2 (the gap rule says 1, but the leave-one-out R² of a cubic is 0.95913 in one active variable against 0.99999 in two; n = 2 is the whole dimensionless space, so here the active subspace ranks and rotates the two groups but reduces nothing beyond dimensional analysis), π̂₁ = Re^0.3073 (ε/D)^−0.4250. High Re: λ₁ = 3.509e-4, λ₂ = 4.707e-12, λ₁/λ₂ = 7.5e7, n = 1, z₁ ≈ −0.7071 × (ε/D exponents). The paper's quadrature with exact gradients gives the exponent vectors of Tables 5–7 (Alg 2) to the printed three decimals and eigenvalues that are 1/16 of the paper's (ratios 15.97 to 16.02); see "How these files relate to the paper".

## Regenerating

```bash
python generate_data.py --out DIR               # writes the three CSVs to DIR (never to this directory)
python generate_data.py --out DIR --check       # also compares them with the shipped files (exit status 1 on failure)
python check_accuracy.py                        # the numbers marked below
```

One process. `generate_data.py` refuses an `--out` inside this directory (symbolic links resolved); to replace the shipped files, write to a scratch directory and copy. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-10), `python generate_data.py --out DIR --check` took about 0.2 s with start-up (0.07 s printed for the model runs, the finite-difference checks and the files) and passed: its files matched the shipped ones to rounding level (maximum relative differences of order 10⁻¹⁶). The stated tolerance of `--check` is 10⁻¹² (relative, elementwise): the model is closed form apart from a Newton solve converged to a relative step of 10⁻¹⁵, so a rerun on another platform differs only by rounding, amplified by the Newton solve and the gradient formulas. The notebook runs in about 7 s in `.venv` (6.7 s printed by its last cell).

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name (about 1.3 s in `.venv`, 2026-11-10; the values below are from that run).

- **Gradients.** Against complex-step derivatives in ln q (step 10⁻³⁰), all 3000 shipped gradient rows agree to rounding level (largest relative error of order 10⁻¹⁵ in each case; row 2-norm; `check_accuracy.py`, check 1). Against central differences in ln q (step 10⁻⁶), first 100 rows of each file: 2.8 × 10⁻¹⁰, 2.1 × 10⁻¹⁰, 2.8 × 10⁻¹⁰ for dp/dx (`generate_data.py`), and 5.0 × 10⁻¹⁰, 7.1 × 10⁻¹⁰, 4.3 × 10⁻¹⁰ for π at five rows (notebook), the truncation and rounding error of the differences (values from `.venv`; they change in the leading digits between platforms).
- **Outputs.** The friction factor recovered from the shipped dp/dx satisfies the Colebrook equation (60) to rounding level (largest absolute residual of order 10⁻¹⁵ or below in each case; `check_accuracy.py`, check 2). Dimensional homogeneity, D ∇_x π = 0 and D ∇_x ln(dp/dx) = (1, −2, −2), holds at every sample to rounding level (of order 10⁻¹⁶ to 10⁻¹⁵; notebook).
- **Quadrature.** The 11-point Gauss–Legendre rule (the paper's) is converged: 15 points per variable change λ₁ by a relative 1.6 × 10⁻¹⁰ (laminar) and by rounding-level amounts, of order 10⁻¹⁶ and 10⁻¹⁵ (turbulent, high Re); λ₂ by 1.5 × 10⁻⁶, 5.8 × 10⁻¹¹ and 1.8 × 10⁻⁷; and z₁, z₂ by at most 2.9 × 10⁻⁸, 2.9 × 10⁻¹³ and 2.6 × 10⁻⁷ (laminar, turbulent, high Re; `check_accuracy.py`, check 3). The high-Re λ₂ is about 10⁻⁸ of λ₁, so the high-Re changes of λ₂ and z are at the level of the rounding error of C and vary between platforms.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from the authors' Matlab code (posted at https://bitbucket.org/paulcon/pipe-code, p. 29; not used, its licence not checked). The data here are an independent numpy computation of Eqs. (60)–(61) with the paper's input ranges. The Monte Carlo samples are new; the notebook repeats the paper's 161051-point quadrature with the model.
- **Agreement.** Exponent vectors z₁, z₂: agree with Tables 5–7 (Algorithm 2) to the three printed decimals in every case (largest difference 0.0005). Laminar z₁ has −0.001 for ε, as printed, which the Colebrook model gives and the Poiseuille relation does not. Rotation angle arccos(U₁₁) = 126.6° with numpy's SVD basis W (paper: "approximately 127 degrees"; the value depends on the SVD's column signs).
- **The factor of 16 in the eigenvalues.** All six Algorithm 2 eigenvalues of Tables 5–7 are 16 times ours (ratios 15.97 to 16.02), i.e. the function the paper differentiated is 4 times ours. (a) Fig. 4 (p. 34) places the factor in dp/dx itself: its nine nonzero dp/dx eigenvalues, read from the page against the gridlines (log10 −8.47, −11.53, −16.64; −1.36, −4.51, −5.73; 8.64, 5.15, −0.92), equal 16 times ours (−8.45, −11.50, −16.64; −1.32, −4.51, −5.71; 8.66, 5.13, −0.93) within 0.04 decade, the reading precision, high-Re λ₃ included; without the factor they differ by 1.17 decades or more (notebook). (b) Fig. 6 (p. 40) rules out the alternatives: its colour axis (ln π about −3.3 to −2.7) matches ln(4π) of our turbulent samples (−3.299 to −2.702; −3.34 to −2.63 on the quadrature grid), not ln π (−4.685 to −4.088). A quadrature-weight error would scale the eigenvalues without shifting ln π (our weights sum to 1), and taking π to be the friction factor λ would give ln(2π), −4.03 to −3.32 on the grid. (c) A mechanism in the paper: p. 29 calls λ in Eq. (58) the "Fanning friction factor", but Eqs. (58), (59) (64/Re) and (60) define the Darcy factor (Fanning = λ/4). A Darcy λ from Colebrook used in the Fanning form dp/dx = 2fρV²/D gives 2λρV²/D, exactly 4 × Eq. (61), hence π = 2λ instead of λ/2 and eigenvalues 16 times ours. This is an inference (the authors' code was not available). The factor scales all eigenvalues equally and does not change the exponent vectors, the eigenvalue ratios or the ranking. These files follow Eq. (61) as printed, with the Darcy factor (`pipe_model.py`).
- **Signs.** Eigenvectors are oriented to the Algorithm 2 columns of Tables 5–7. Fig. 6b's horizontal axis (−8 to −5) corresponds to −z₁ for the turbulent case (our ln π̂₁ with Table 6's sign lies in [5.37, 7.05]).
- **Algorithm 1.** The paper's Gaussian-process values (Alg 1) differ from its Algorithm 2 values (for example laminar λ₂ 2.65e-8 against 3.39e-9); this data set has exact gradients and is compared with Algorithm 2.

## License and credit

Code (`pipe_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations; no data or code were copied. Cite P. G. Constantine, Z. del Rosario, G. Iaccarino, "Data-driven dimensional analysis: algorithms for unique and relevant dimensionless groups", arXiv:1708.04303, 2017, doi:10.48550/arXiv.1708.04303, and, for Eq. (60), C. F. Colebrook and C. M. White, "Experiments with fluid friction in roughened pipes", *Proc. R. Soc. Lond. A* 161(906):367–381, 1937, doi:10.1098/rspa.1937.0150. The paper's work was supported by DARPA's program Enabling Quantification of Uncertainty in Physical Systems, the U.S. Department of Energy Office of Science, ASCR Applied Mathematics program (DE-SC-0011077), and an NSF Graduate Research Fellowship (DGE-114747).

## Unknowns

- The origin of the factor 16 between the paper's eigenvalues and those of Eqs. (58), (61) and (7) as printed. Figs. 4 and 6 show that the authors' pressure loss is 4 times Eq. (61); the likely cause, a Darcy factor (mislabelled "Fanning" on p. 29) used in the Fanning-form pressure drop, cannot be confirmed without their code.
- The paper gives no nominal values of the inputs and no random seed (its quadrature design needs none).
- The paper's Newton initial guess and stopping tolerance are not stated (ours: s = 8, relative step below 10⁻¹⁵).

## Open questions (only the authors can settle)

1. Did the code compute dp/dx with the Fanning-form formula 2λρV²/D (that is, 4 × Eq. (61)) using the Darcy λ of Eq. (60)? Fig. 4 (dp/dx) and Fig. 6 (ln π offset by ln 4) both point to this.
2. Which column signs of W (Matlab SVD) were used for the 127° angle?
3. Which sign of z₁ was used in Fig. 6b (the axis range suggests −z₁ of Table 6)?
