# Steel_Column_Reliability: data

Kind: **Python model**. The model is `steel_column_model.py` (numpy and scipy), written from the paper's equations (pp. 326–327). `generate_data.py` draws the inputs with saved seeds and writes all six CSVs; a rerun reproduces them to the tolerance under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Steel_Column_Reliability.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 736`; outputs committed |
| `steel_column_model.py` | `limit_state(X, grad=False)`, `limit_state_u(U, config, grad=False)`, `euler_load(X)`, `means(config)`, class `Marginals(config)` with `x_from_u`, `u_from_x`, `dx_du`, `sample`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `DISTRIBUTIONS`, `SD`, `CONFIGS`, `L_COLUMN` | the model |
| `generate_data.py` | draws the samples and writes the six CSVs (`--out DIR` required); `--check` compares a rerun with the shipped files; checks the gradients by central differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `benchmark_inputs.csv` | 2000 × 9 physical inputs Fs (MPa), P1, P2, P3 (N), B, D, H, F0 (mm), E (MPa); configuration `benchmark` | model input |
| `benchmark_outputs.csv` | 2000 × 1: G (MPa) | model output |
| `benchmark_gradients.csv` | 2000 × 9: ∂G/∂x with respect to the **physical** inputs, same column order | model output |
| `paper_inputs.csv`, `paper_outputs.csv`, `paper_gradients.csv` | the same for configuration `paper` | model input / output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits. The notebook maps inputs to standard normal coordinates u = Φ⁻¹(F(x)) with `Marginals.u_from_x` and gradients by ∂G/∂u_i = (∂G/∂x_i)(dx_i/du_i) with `Marginals.dx_du`.

## Model

**Source.** N. Kuschel and R. Rackwitz, "Two basic problems in reliability-based structural optimization", *Mathematical Methods of Operations Research* 46(3):309–333, 1997, [doi:10.1007/BF01194859](https://doi.org/10.1007/BF01194859) (publisher scan read; Technische Universität München). Section 6, second numerical example, pp. 326–327. The first example (rectangular section, pp. 325–326) and the third (reinforced concrete beam, pp. 328–) are not built.

**Limit state** (p. 327, as printed): G(x, p) = F_s − 𝒫 (1/𝒜_s + F_0/ℳ_s · ℰ_b/(ℰ_b − 𝒫)), with 𝒫 = P_1 + P_2 + P_3, 𝒜_s = 2BD (area of section), ℳ_s = BDH (modulus of section), ℳ_i = ½BDH² (moment of inertia; printed "iteria"), ℰ_b = π²Eℳ_i/s² (Euler buckling load). The symbol s is not defined on the page; we read it as the column length, "a constant length of 7500 mm" (p. 326). G ≤ 0 is failure. The quantity of interest is G (MPa). The formulas are those of two flanges of area BD at distance H with the web neglected (our reading; the paper calls the profile "T-shaped", p. 327).

**Inputs** (m = 9). "The independent uncertain vector X = (F_s, P_1, P_2, P_3, B, D, H, F_0, E)" (p. 326), with this table (p. 326, "Mean/Standard derivation"):

| name | symbol | distribution | mean / sd as printed | units | used here | source |
|---|---|---|---|---|---|---|
| `Fs` | yield stress F_s | lognormal | 400 / 35 | MPa | as printed | table, p. 326 |
| `P1` | dead weight load P_1 | normal | 500000 / 50000 | N | as printed | table, p. 326 |
| `P2` | variable load P_2 | Gumbel | 600000 / 90000 | N | largest-value Gumbel | table, p. 326 |
| `P3` | variable load P_3 | Gumbel | 600000 / 90000 | N | largest-value Gumbel | table, p. 326 |
| `B` | flange breadth B | lognormal | b / 3 | printed "N" | mm (misprint) | table, p. 326 |
| `D` | flange thickness D | lognormal | d / 2 | mm | as printed | table, p. 326 |
| `H` | height of profile H | lognormal | h / 5 | mm | as printed | table, p. 326 |
| `F0` | initial deflection F_0 | normal | 30 / 10 | mm | as printed | table, p. 326 |
| `E` | Young's modulus E | Weibull | 21000 / 4200 | MPa | 210000 / 4200 (`benchmark`), 21000 / 4200 (`paper`) | table, p. 326 |

The paper does not give the Gumbel and Weibull parameterizations. Chosen: largest-value Gumbel, scale = sd·√6/π, location = mean − γ·scale; two-parameter Weibull with shape k and scale λ solved from the mean and sd (k = 63.4086, λ = 211877.37 MPa for mean 210000; k = 5.7974, λ = 22679.48 MPa for mean 21000; `check_accuracy.py`, check 2). Lognormals: σ_ln² = ln(1 + sd²/mean²), μ_ln = ln(mean) − σ_ln²/2.

**Design variables and fixed values.** The cost parameters p = (b, d, h) are the means of B, D, H (table, p. 326); they are deterministic and the paper optimizes them within 200 ≤ b ≤ 400, 10 ≤ d ≤ 30, 100 ≤ h ≤ 500 mm (p. 327) with the cost (bd + 5 mm · h) CU/mm² (p. 326) at most C_max (table, p. 327). The length s = 7500 mm is constant. So the example has 9 random inputs, 3 design variables and 1 constant; this data set fixes the design and studies the 9 random inputs.

**Configurations** (this data set's choices; `steel_column_model.CONFIGS`):

- `benchmark`: (b, d, h) = (300, 20, 300) mm, mean of E = 210000 MPa. This is the configuration of the steel-column page of Surjanovic and Bingham (https://www.sfu.ca/~ssurjano/steelcol.html), which attributes the design to Eldred, Webster and Constantine 2008 (doi:10.2514/6.2008-1892; the citation is confirmed on Crossref, the design values in that paper were not checked here) and uses E mean 210000 MPa. The column is below its Euler load at every sample (ℰ_b/𝒫 ∈ [3.7168, 9.3322]; `generate_data.py`).
- `paper`: (b, d, h) = (200, 17.5, 100) mm, the paper's optimum for C_max = 4000 (table, p. 327), with E mean 21000 MPa as printed. Every sample is above its Euler load (ℰ_b/𝒫 ∈ [0.0090, 0.0816]), so the amplification factor ℰ_b/(ℰ_b − 𝒫) is small and negative and G is close to the axial-stress margin F_s − 𝒫/(2BD); this regime is outside the formula's physical meaning (a column above its Euler load buckles).

**Why both.** A mean of 21000 MPa is a tenth of the modulus of steel. With the table as printed (E 21000 / 4200 MPa, s = 7500 mm), FORM reproduces all ten reliability indices of the table on p. 327 to within 0.0042 (notebook; `check_accuracy.py`, check 3). With E 210000 / 4200 MPa it does not: SLSQP converges at 8 of the 10 designs and the indices there differ from the paper's by 0.699 to 10.402 (3.9109 against 3.132 at cost 4000, 4.4822 against 6.369 at cost 6000, −0.7975 against 9.605 at cost 10000; no convergence at costs 8000 and 13000; notebook and check 3). With E mean 210000 MPa the first six designs are above their Euler load at the means (ℰ_b/𝒫 = 0.379 to 0.921) and those with C_max ≥ 10000 below it (1.030 to 5.202; check 3), so the limit state has its pole inside the input distribution. E itself is not identifiable from the indices: E enters G only through ℰ_b = π²Eℳ_i/s² and the Weibull is a scale family, so only the distribution of E/s² matters. E with mean 210000 MPa and sd 42000 MPa together with s = 7500√10 ≈ 23717 mm gives the same ℰ_b in distribution and the same β at every design (the two Weibulls have the same shape, 5.7974, and scales in the ratio 10; notebook and check 3); a factor of 10 in the units of ℳ_i would act the same way. The defensible statement is that the printed table (E 21000 / 4200 MPa, s = 7500 mm) reproduces β*, and E 210000 / 4200 MPa with s = 7500 mm does not.

**Why h* stays at 100 mm.** The table on p. 327 has h* = 100 mm up to C_max = 12000; the text says "up to a maximum cost of 12 500 CU" (p. 328). Along the cost constraint with the printed table, β falls as h grows: at C_max = 4000 (b = 200, d = (4000 − 5h)/200) β = 3.1321, 3.0376, 2.8453, 2.1431 for h = 100, 110, 130, 200 mm (ℰ_b/𝒫 at the means 0.038 to 0.130); at C_max = 12000 (d = 30, b = (12000 − 5h)/30) β = 10.7048, 10.6909, 10.6627, 10.5605 (ℰ_b/𝒫 0.125 to 0.477) (notebook; `check_accuracy.py`, check 4). So money spent on h buys less reliability than money spent on b or d. The pole at ℰ_b = 𝒫 is not the reason: below it the amplification factor is negative, and a more negative factor raises G for F_0 > 0.

**Gradients.** Closed form with respect to the physical inputs (`limit_state(..., grad=True)`), stored as such; the notebook maps them to u with the diagonal Jacobian dx/du. G depends on B and D only through BD, so B ∂G/∂B = D ∂G/∂D and the gradient matrix in u has the exact null direction s_D e_B − s_B e_D (s = log-sd).

**Sampling design.** For each configuration, M = 2000 independent Monte Carlo draws u ~ N(0, I₉), `numpy.random.default_rng(736)` (`benchmark`) and `default_rng(1997)` (`paper`), `.standard_normal((2000, 9))`, mapped by x = F⁻¹(Φ(u)). Samples outside physical meaning: 5 per configuration have F_0 < 0 (the printed formula uses F_0, not |F_0|; kept); none has P_1 ≤ 0; all 2000 `paper` samples are above the Euler load (see above). Samples with G ≤ 0: 0 (`benchmark`), 2 (`paper`).

**Published results.** The table on p. 327: optimal (b*, d*, h*) and reliability index β* for C_max = 4000, …, 13000 (β* = 3.132 at (200, 17.5, 100), …, 11.065 at (400, 30, 200)); Figs. 6.1 and 6.2 (p. 328) plot the same. The paper reports no failure probabilities, no sensitivity indices and no active subspace.

**What the notebook finds.** Inputs scaled through their CDFs (u-space). n = 1 in both configurations: λ₁/λ₂ = 88.99 (`benchmark`) and 88.10 (`paper`), next ratio 1.79 and 1.22; bootstrap subspace distance 0.0029 and 0.0033 for n = 1 (0.0434, 0.1209 for n = 2). w₁ = (0.7984, −0.1237, −0.2199, −0.2198, 0.0423, 0.4224, 0.0190, −0.2645, 0.0034) (`benchmark`); in `paper` F_0 drops out (0.0385, positive). |w₁·α_FORM| = 0.8786 and 0.9623. A linear function of y₁ has leave-one-out R² = 0.98772 and 0.98508. FORM β = 4.5958 and 3.1321. Monte Carlo P_f: 3.70 × 10⁻⁶ (37 failures in 10⁷; 95 % Clopper–Pearson [2.61, 5.10] × 10⁻⁶) and 1.2435 × 10⁻³ (2487 in 2 × 10⁶; [1.1951, 1.2933] × 10⁻³), 1.717 and 1.433 times the FORM values.

## Regenerating

```bash
python generate_data.py --out DIR --check     # writes to DIR (not this directory) and compares
python generate_data.py --out .               # rewrites the shipped files (from this directory)
python check_accuracy.py                      # the numbers marked below
```

`--out` is required; `--check` refuses this directory and any directory inside it. One process. The shipped files were written on 2026-11-09 in a Linux container (x86_64; Python 3.11.18, numpy 2.4.6, scipy 1.17.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-10) `python generate_data.py --out DIR --check` takes under 1 s for both configurations plus the finite-difference check (the script reports 0.1 s) and passes: the largest relative differences from the shipped files are at the rounding level, of order 1e-16 to 1e-15 for five of the six files and of order 1e-14 for `paper_outputs.csv`. The stated tolerance is 1e-12 (relative, elementwise), which allows for platform differences in exp, log and the special functions of the inverse CDFs.

## Accuracy

- **Outputs and gradients** (`check_accuracy.py`, check 1). A fresh evaluation reproduces the shipped G and ∂G/∂x to round-off, and the shipped ∂G/∂x agree with complex-step derivatives in x (step 10⁻³⁰, no cancellation) to a relative error below 10⁻¹⁵ in both configurations (row 2-norm). Against central differences in u (step 10⁻⁶): 4.4 × 10⁻⁹ and 3.8 × 10⁻⁹ on the first 100 rows (`generate_data.py`), 5.1 × 10⁻⁹ and 4.7 × 10⁻⁹ on all rows (notebook).
- **Input maps** (`check_accuracy.py`, check 2). x(u(x)) = x to a relative error of about 10⁻¹⁵ (rounding level) on the shipped samples; 10⁶ draws (seed 0) reproduce the tabulated means and sds to 2.1 × 10⁻³ and 1.3 × 10⁻³ standard deviations, the Monte Carlo error.
- **Paper's table** (`check_accuracy.py`, check 3; also printed in the notebook). FORM (SLSQP in u, exact gradients) at the ten designs of p. 327 with E mean 21000 MPa: differences from β* between −0.0042 (C_max = 12000) and +0.0004; all others within 0.0008. With E mean 210000 MPa (sd 4200): 8 of 10 converge, differences 0.699 to 10.402. Weibull shape 5.7974000657 for both (21000, 4200) and (210000, 42000), scale ratio 10.000000.
- **Cost line** (`check_accuracy.py`, check 4; also printed in the notebook). β along bd + 5h = C_max for h = 100 to 200 mm, as quoted under "Why h* stays at 100 mm".

## How these files relate to the paper

- **Not the paper's computation.** The paper's numbers come from the authors' FORM-based one-level optimization code (NLPQL is named for the first example, p. 326), which is not available. The data here are an independent Python computation of the printed limit state and input table, plus the choices above (distribution parameterizations, the reading of s, the two configurations and the value of E mean in `benchmark`). The 2000-sample Monte Carlo designs are new.
- **Agreement.** With the table as printed, FORM here reproduces all ten β* of p. 327 to within 0.0042, which supports the limit state, the input table and the Gumbel/Weibull parameterizations as read here; E 210000 / 4200 MPa does not reproduce them. Only the distribution of E/s² is identified (see "Why both"), so the indices cannot tell E = 21000 MPa from, for example, E 210000 / 42000 MPa with s = 7500√10 mm. The paper's index for the cost-4000 design, 3.132, corresponds to Φ(−3.132) = 8.6810 × 10⁻⁴; Monte Carlo gives 1.2435 × 10⁻³ (95 % [1.1951, 1.2933] × 10⁻³), so the first-order value underestimates the failure probability by a factor of about 1.43.
- **Other implementations.** The Surjanovic–Bingham page gives the same input table with E mean 210000 MPa and the design (300, 20, 300) mm; its MATLAB code was not used (the code file https://www.sfu.ca/~ssurjano/Code/steelcolm.html carries a GNU GPL version 2.0 notice; the page itself states no licence). UQTestFuns 0.7.0 (PyPI, and its GitHub master of 2026-10-22) does not include this function, so no cross-check against it was possible.
- **Open questions (only the authors can settle).** (1) Whether E mean 21000 MPa was intended or a slip that entered the computation, or whether a different s (or different units of ℳ_i) was used: only E/s² is identifiable from the published indices. (2) The Gumbel and Weibull parameterizations (our choices reproduce β* to 0.0042). (3) Whether |F_0| was meant in the limit state.

## License and credit

Code (`steel_column_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations; no data or code were copied. Cite N. Kuschel and R. Rackwitz, *Math. Methods Oper. Res.* 46:309–333, 1997, doi:10.1007/BF01194859, and, for the `benchmark` configuration, S. Surjanovic and D. Bingham, Virtual Library of Simulation Experiments, https://www.sfu.ca/~ssurjano/steelcol.html. The paper states no funding.

## Unknowns

- The paper's Gumbel (largest or smallest value) and Weibull (two- or three-parameter) forms: not stated; chosen as above.
- s in ℰ_b: not defined; read as the column length 7500 mm.
- E mean: printed 21000 MPa (used in `paper`, reproduces the paper's indices with s = 7500 mm); 210000 MPa in `benchmark` (the steel value, from the Surjanovic–Bingham page).
- E and s separately: only the distribution of E/s² enters G, so the indices of p. 327 do not identify E, s or the units of ℳ_i separately (E 210000 / 42000 MPa with s = 7500√10 mm gives the same β as the printed table).
- The design (300, 20, 300) mm in Eldred, Webster and Constantine 2008: the attribution is from the Surjanovic–Bingham page; the citation is confirmed on Crossref, the values were not checked in that paper.
- The unit of B: printed "N"; read as mm.
- F_0 < 0 (normal input): the printed formula is used as is.
