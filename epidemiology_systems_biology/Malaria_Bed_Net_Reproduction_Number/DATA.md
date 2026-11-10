# Malaria_Bed_Net_Reproduction_Number: data

Kind: **Python model**. The model is `malaria_model.py` (numpy only), written from the paper's equations: the closed-form effective reproduction number of a two-strain malaria model with bed nets and treatment, and its exact gradient. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; a rerun reproduces them to the tolerance under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Malaria_Bed_Net_Reproduction_Number.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2019; outputs committed |
| `malaria_model.py` | `reff(x, grad=False)`, `reffs(X, grad=False)`, `strain_parts(x)`, `ngm(x, alpha, alpha_s, alpha_r)`, `sample(M, rng, lhs=True)`, `ppf(U)`, `prcc(X, y)`; `TABLE3` (all 23 rows), `NAMES`, `SYMBOLS`, `UNITS`, `BASELINE`, `LOWER`, `UPPER`, `DIST`, `PEAK`, `NOT_IN_REFF` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files; checks the gradients by central differences and R_eff against the next-generation matrix | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below marked with its name | check |
| `inputs.csv` | 1000 × 20 physical inputs, in the order of the inputs table below | model input |
| `outputs.csv` | 1000 × 3: R_eff, R_r, R_s (all dimensionless; R_eff = R_r + R_s) | model output |
| `gradients.csv` | 1000 × 20: ∂R_eff/∂x_i with respect to the **physical** inputs (units: the reciprocal of each input's units) | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits.

## Model

**Source.** S. Bala and B. Gimba, "Global sensitivity analysis to study the impacts of bed-nets, drug treatment, and their efficacies on a two-strain malaria model", *Mathematical and Computational Applications* 24(1):32, 2019, doi:10.3390/mca24010032 (publisher PDF, open access under CC BY 4.0). No code or data were released with the paper.

**Equations.** Biting rate (Eq. 1, p. 4): a = β_max β_min / ((β_max − β_min) b γ + β_min). The eight-compartment model (Eq. 3, p. 6) is scaled to six equations (Eq. 15, p. 10) with ξ1 = a b1 N_v/N_h = a b1 k_v ρ1/k_h, ρ1 = N_v*/N_h*, and k1 = a b2 (p. 10), carrying capacities k_v = (ψ_v − μ_v − μ_v3 b γ)/μ_v2 and k_h = (ψ_h − μ_h)/μ_h2 (proof of Lemma 1, p. 8), and p1 = r_s + δ_h + T_s + ψ_h, p2 = r_r + δ_h + ψ_h, p3 = T_s(1 − e) (p. 10). The next-generation matrices F, V (Eqs. 16–17, pp. 10–11) give Eq. (19), p. 11:

R_eff = ξ1 k1 (1 − β)/(p2 ψ_v) + k1 ξ1 β (p2 + p3)/(p2 p1 ψ_v) = R_r + R_s (Eq. 20).

Two notes on the printed equations, both checked in the notebook: (i) Eq. (19) is the **square** of the spectral radius ρ(FV⁻¹) (one human and one mosquito generation), not the radius; the threshold at 1 is the same. (ii) The radicand of Eq. (18) as printed, −p2 p1 ψ_v ξ1 k1 (β p1 + β − p3 − β p2 − p1), is negative at the baselines and at 956 of the 1000 samples (notebook; `check_accuracy.py`, check 3); with −β p3 in place of "+ β − p3" its square root divided by p2 p1 ψ_v is the spectral radius, and its square equals Eq. (19) to rounding level (of order 10⁻¹⁶, relative; `check_accuracy.py`, check 3).

**Which inputs (20, not 23).** Table 3 (p. 19) lists 23 parameters; the text says that R_eff "was regulated by 20 various malaria-related epidemiological parameters" (p. 18) and that the Latin hypercube matrix was "1000 by 20" (p. 19). The 20 are the parameters of Eq. (19). Three Table 3 rows do not enter R_eff and are **not inputs here**: α_s and α_r (immunity loss) appear only in c1 = α_s + ψ_h, c2 = α_r + ψ_h, the removal rates of the recovered classes, which are not infected compartments; α (resistant fraction of new mosquito infections) splits infections between two mosquito classes with the same removal rate ψ_v and cancels from the spectral radius. Varying all three over their Table 3 ranges changes ρ(FV⁻¹) only at rounding level (by about 10⁻¹⁵ at most, relative; `check_accuracy.py`, check 2). The paper's PRCC figure for R_eff (Fig. 5, p. 21) shows exactly the other 20. Five of the 20 (μ_h, μ_h2, μ_v, μ_v2, μ_v3) enter only through k_v/k_h in ξ1; the reading ξ1 = a b1 ρ1 would leave 15 inputs, and it does not reproduce the paper (under "How these files relate to the paper"). This data set follows the printed definition with ρ1 set to Table 3's ρ ("mosquitoes per human", 2–8). That counts the mosquito-to-human ratio twice: ρ1 is defined on p. 10 as the ratio of the scaled populations N_v*/N_h* (1 at the scaled equilibrium), and N_v/N_h = ρ1 k_v/k_h, so at the baselines (k_v/k_h = 2.372) the ratio actually used is 16.6 mosquitoes per human (notebook). See "Open questions".

**Inputs** (m = 20), independent, Table 3, p. 19 (meanings in Table 2, p. 7):

| name | symbol | units | baseline | range | distribution | source |
|---|---|---|---|---|---|---|
| `psi_h` | ψ_h, human recruitment | 1/day | 9.3614e-5 | [6.0849, 12.17] × 10⁻⁵ | uniform | Table 3, p. 19 |
| `psi_v` | ψ_v, mosquito recruitment | 1/day | 0.4478 | [0.2911, 0.7] | uniform | Table 3, p. 19 |
| `T_s` | T_s, treatment rate (sensitive strain) | 1/day | 0.35 | [0.2275, 0.455] | uniform | Table 3, p. 19 |
| `b1` | b1, transmission probability mosquito → human | – | 0.75 | [0.1, 0.8] | uniform | Table 3, p. 19 |
| `b2` | b2, transmission probability human → mosquito | – | 0.5342 | [0.072, 0.64] | uniform | Table 3, p. 19 |
| `rho` | ρ1, mosquitoes per human | – | 7 | [2, 8] | uniform | Table 3, p. 19 |
| `b` | b, proportion using ITNs | – | 0.53 | [0.1325, 0.6625] | triangular, peak 0.5 | Table 3, p. 19 |
| `beta_max` | β_max, maximum biting rate | 1/day | 0.6334 | [0.1, 1] | uniform | Table 3, p. 19 |
| `beta_min` | β_min, minimum biting rate | 1/day | 0.0696 | [0, 0.1] | uniform | Table 3, p. 19 |
| `e` | e, treatment efficacy | – | 0.4 | [0.01, 0.61] | uniform | Table 3, p. 19 |
| `gamma` | γ, ITN efficacy | – | 0.5 | [0.2, 1] | uniform | Table 3, p. 19 |
| `r_r` | r_r, immunity acquisition (resistant) | 1/day | 0.0078 | [0.00507, 0.01014] | triangular, peak 0.0078 | Table 3, p. 19 |
| `r_s` | r_s, immunity acquisition (sensitive) | 1/day | 0.0078 | [0.00507, 0.01014] | uniform | Table 3, p. 19 |
| `beta` | β, proportion infected with the sensitive strain | – | 0.7 | [0.455, 0.91] | uniform | Table 3, p. 19 |
| `delta_h` | δ_h, disease-induced death rate | 1/day | 1e-3 | [0.00065, 0.0013] | uniform | Table 3, p. 19 |
| `mu_h2` | μ_h2, density-dependent human death | 1/(human day) | 1e-7 | [6.5, 13] × 10⁻⁸ | uniform | Table 3, p. 19 |
| `mu_h` | μ_h, density-independent human death | 1/day | 4.212e-5 | [2.74, 5.48] × 10⁻⁵ | uniform | Table 3, p. 19 |
| `mu_v` | μ_v, density-independent mosquito death | 1/day | 0.1429 | [0.092885, 0.18577] | uniform | Table 3, p. 19 |
| `mu_v2` | μ_v2, density-dependent mosquito death | 1/(mosquito day) | 2.28e-4 | [1.48, 2.96] × 10⁻⁴ | uniform | Table 3, p. 19 |
| `mu_v3` | μ_v3, ITN-induced mosquito death | 1/day | 0.0995 | [0.064675, 0.12935] | uniform | Table 3, p. 19 |

Not inputs (Table 3, p. 19): α_s 0.0017, U[0.001105, 0.00221]; α_r 0.0017, triangular on [0.001105, 0.00221] with peak 0.0017; α 0.3, U[0.195, 0.39].

Units: the units above are those the equations imply. Table 2 (p. 7) prints ψ_h, μ_h, μ_h2 as "human × day⁻¹", ψ_v, μ_v as "mosquitoes × day⁻¹", μ_v2 as "day⁻¹", and the efficacies e, γ as "day⁻¹"; but ψ_h and ψ_v multiply a population to give a rate (Eqs. 3–5), μ_h2 and μ_v2 multiply a population to give a per-capita rate, and e, γ are proportions in [0, 1] (Eq. 1). The baseline of b (0.53) is not the peak of its triangular distribution (0.5); both are as printed. The paper is not consistent about the lower end of β_min: p. 4 says the transmission "can only be reduced to a minimum value β_min > 0", while Table 3 (p. 19) gives U[0, 0.1]. This data set uses Table 3's range (lower bound 0); the statement below about ln R_eff rests on that bound.

**Quantity of interest.** R_eff (dimensionless), Eq. (19); outputs.csv also holds R_r and R_s (Eq. 20). At the Table 3 baselines R_eff = 49.2087 (R_r = 20.3079, R_s = 28.9008; notebook). Every point of the input box has k_v > 0 and k_h > 0 (the smallest values in the sample are ψ_v − μ_v − μ_v3 b γ = 0.0453 /day and ψ_h − μ_h = 7.03 × 10⁻⁶ /day; notebook), R_eff is defined for β_min ≥ 0 (at β_min = 0, a = 0 and R_eff = 0), and its gradient for β_min > 0. `reff` raises ValueError outside these conditions (k_v ≤ 0, k_h ≤ 0, β_min < 0, or a gradient at β_min = 0).

**Gradients.** Exact: R_eff = a² b1 b2 ρ1 (k_v/k_h) H/ψ_v with H = (1 − β)/p2 + β(p2 + p3)/(p1 p2), so ∇R_eff = R_eff ∇ln R_eff, with ∇ln R_eff written out term by term in `malaria_model.reff`. `gradients.csv` holds the gradient with respect to the physical inputs; the notebook normalizes each input through its bounds to [−1, 1] and the library applies the chain rule.

**Sampling design.** M = 1000 Latin hypercube samples (one point per stratum per input, a uniform point inside each stratum, independent random permutations), mapped through the inverse CDFs of the Table 3 distributions; `numpy.random.default_rng(2019)` in `malaria_model.sample`. The design is this data set's: the paper used a 1000-point Latin hypercube from SaSAT (p. 18–19) and gives no seed. Every sample is inside the model's meaning. R_eff ranges from 1.44 × 10⁻⁶ to 589.4 (median 5.694, mean 18.27); 199 of the 1000 samples have R_eff < 1 (notebook).

**Published results** (no active subspace in the paper). Sec. 5.2.1, pp. 18–21: PRCC tornado plots for R_r (Fig. 3, p. 20, 17 parameters), R_s (Fig. 4, p. 20, 20) and R_eff (Fig. 5, p. 21, 20); no values are printed. The five leading parameters are β_min, γ, b1, b2, β for R_r and β_min, γ, b1, b2, b for R_s (p. 19), and β_min, γ, b2, b1, ψ_h for R_eff (p. 21). Sec. 5.2.2, p. 21: 210 of the 1000 samples gave R_eff < 1. Figs. 6–7 (pp. 22–23): intervention scenarios with (e, γ, b, T_s) fixed at 0.75 or 0.95 on the 790 samples with R_eff > 1 (not reproduced here). No Sobol' indices.

**What the notebook finds.** n = 1: λ1/λ2 = 10.59 (bootstrap 6.58–17.74), λ2/λ3 = 1.59; the bootstrap mean subspace distance is 0.071 for n = 1 and 0.360 for n = 2. The direction is robust, the ratio is not: the 10 samples (1 %) with the largest squared gradient norm hold 0.468 of the trace of the estimated C (the 50 largest, 0.788); without those 10, λ1/λ2 = 16.13 (without the 50, 13.41), while w1 hardly moves (|w1 · w1′| = 0.992 and 0.977). These 10 are among the 20 largest R_eff values, 9 of them with γ < 0.4; R_eff has skewness 6.7 and kurtosis 71.6. So λ1/λ2 = 10.59 rests on a few samples; another design could give a noticeably different ratio with the same w1. w1 (normalized inputs) is led by γ (−0.472), ψ_h (−0.425), β_min (+0.328), b (−0.317), β_max (+0.290), b1 (+0.275), b2 (+0.260), ρ1 (+0.224); the treatment parameters T_s and r_s are below 0.003 in magnitude. The eigenvalues after the first hold 0.185 of the sum, and R_eff is heavy-tailed: a radial-basis surrogate in y1 has leave-one-out R² = 0.824 (quartic 0.814; two variables 0.881). For √R_eff = ρ(FV⁻¹), λ1/λ2 = 6.99 with nearly the same direction (|w1 · w1′| = 0.980) and a quadratic in y1 has LOO R² = 0.8725. For ln R_eff, C is not finite under Table 3's density, because ∂ln R_eff/∂β_min ~ 2/β_min and β_min ~ U[0, 0.1], so E[(∂ln R_eff/∂β_min)²] is infinite (with p. 4's β_min > 0 it would be finite, but no positive bound is given). The sample estimate exists and is a trivial one-dimensional subspace along β_min: λ1/λ2 = 1205 (bootstrap 53–3436), β_min holds 0.9988 of the trace and the 10 rows with the largest gradients 0.987 (notebook).

## Regenerating

```bash
python generate_data.py --out DIR --check     # writes to DIR (outside this directory) and compares with the shipped files
python generate_data.py --here                # rewrites the shipped files in this directory
python check_accuracy.py                      # the numbers marked below
```

One process, no multiprocessing. Without `--here` the script never writes into this directory or any directory below it; `--out` must be outside it (a path inside it, subdirectories included, is rejected), and `--check` needs `--out DIR`. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-10), `python generate_data.py --out DIR --check` took about 0.2 s (0.3 s with start-up) and passed: `inputs.csv` and `outputs.csv` were reproduced with a maximum relative difference printed as 0.0e+00 and `gradients.csv` to a maximum relative difference of order 10⁻¹⁴ (elementwise). The stated tolerance is 1e-12 (relative, elementwise), which allows for a different libm or BLAS on another platform. The notebook runs in about 12 s in `.venv` (11.6 s printed by its last cell; most of it the 50 extra PRCC designs and the bootstrap). `check_accuracy.py` took about 3 s there.

## Accuracy

- **Outputs.** R_eff is closed form; there is no discretization or solver error. R_r + R_s = R_eff and ρ(FV⁻¹)² (eigenvalues of the 6 × 6 matrix of Eqs. 16–17) = R_eff to rounding level (relative differences of order 10⁻¹⁶ and 10⁻¹⁵, all samples; `generate_data.py` and notebook).
- **Gradients.** Against central differences with relative step 10⁻⁶, first 100 samples: largest relative error (row 2-norm) 1.5 × 10⁻¹⁰ (`generate_data.py`). At all 1000 samples (`check_accuracy.py`, check 1): step 10⁻⁵, max 1.4 × 10⁻¹⁰ (median 6.6 × 10⁻¹²); step 10⁻⁶, max 3.1 × 10⁻¹⁰; step 10⁻⁷, max 2.4 × 10⁻⁹. The error grows as the step shrinks, so it is the differences' rounding error, not an error in the gradient.

## How these files relate to the paper

- **Not the paper's computation.** The paper's values come from the authors' computation with SaSAT (p. 18), which was not released. The data here are an independent computation of Eq. (19) with Table 3's distributions; the 1000 samples are new.
- **Agreement.** On the shipped design the PRCC ranking (`malaria_model.prcc`) has the paper's five leading parameters in the paper's order (β_min, γ, b2, b1, ψ_h), and 199 samples have R_eff < 1 (95 % interval 174–224; 180–223 over 50 further designs) against the paper's 210 (notebook). The paper's Fig. 5 values, read from the bar ends on a 300 dpi rendering of p. 21 (reading precision about 0.005; read from the figure, not reproducible from the repository), lie inside the 2.5–97.5 % range of PRCC over 50 further 1000-point designs for 18 of the 20 inputs; γ (−0.657) and b1 (0.624) lie 0.013 and 0.006 outside (notebook). The largest difference on the shipped design is 0.105 (μ_h2), where the shipped design's own value (0.186) is below the range of the other 50 designs.
- **Figs. 3 and 4 (R_r, R_s).** Fig. 3 has 17 bars, the 20 inputs without e, T_s and r_s, which do not enter R_r = ξ1 k1 (1 − β)/(p2 ψ_v); the five density-dependent parameters are among them, so the paper's R_r also contains k_v/k_h (ξ1 = a b1 ρ1 would give 12 bars). The bar values, read in the same way from a 300 dpi rendering of p. 20 (not reproducible from the repository), agree in sign with the PRCC here for every bar of magnitude ≥ 0.1 (15 for R_r, 16 for R_s) and lie inside the 50-design range for 16 of 17 inputs of R_r (outside: μ_h) and 19 of 20 of R_s (outside: b). Examples (figure, here): R_s: e −0.371, −0.336; β +0.248, +0.304; ψ_h −0.441, −0.545; b −0.502, −0.450; R_r: β −0.531, −0.509. The order within the leading group differs: top five here β_min, b2, γ, b1, ψ_h for R_r (β sixth; paper β_min, γ, b1, b2, β) and β_min, γ, b2, b1, ψ_h for R_s (paper β_min, γ, b1, b2, b, with ψ_h eighth in Fig. 4). The largest differences on the shipped design, 0.088 (μ_h2, R_r) and 0.104 (ψ_h, R_s), are where this design's value lies outside the range of the other 50 designs (notebook). The agreement is of the same kind as for Fig. 5: same parameters and signs, similar magnitudes, not the same order.
- **The reading of ξ1.** With ξ1 = a b1 ρ1 (no k_v/k_h; 15 inputs), ψ_h falls out of the five leading parameters (PRCC −0.094), the largest difference from Fig. 5 is 0.527 and 353 of the 1000 samples have R_eff < 1 (338–382 over 50 designs) (notebook; `check_accuracy.py`, check 4). So the paper's R_eff contains k_v/k_h, as its printed definition of ξ1 says.
- **Open questions (only the authors can settle).** (1) The source of the remaining PRCC differences, up to about 0.1 (SaSAT's LHS and PRCC settings, the handling of the triangular rows). (2) The intended radicand of Eq. (18). (3) Whether the units printed in Table 2 or those implied by the equations are meant (they do not change R_eff, which is computed from the printed values). (4) Whether ρ1 in ξ1 = a b1 k_v ρ1/k_h (p. 10) is the scaled ratio N_v*/N_h* (1 at the scaled equilibrium) or Table 3's ρ = 7 mosquitoes per human; this data set uses 7, which with k_v/k_h = 2.37 at the baselines gives 16.6 mosquitoes per human, and the PRCC comparison supports keeping k_v/k_h. (5) Whether β_min's lower bound is 0 (Table 3, p. 19) or strictly positive (p. 4).

## License and credit

Code (`malaria_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations and parameter table; no data or code were copied. The paper is open access under CC BY 4.0. Cite S. Bala and B. Gimba, "Global sensitivity analysis to study the impacts of bed-nets, drug treatment, and their efficacies on a two-strain malaria model", *Math. Comput. Appl.* 24(1):32, 2019, doi:10.3390/mca24010032. The paper states no funding.

## Unknowns

- The Latin hypercube details and seed of the paper's sample (SaSAT, p. 18); the design here is new.
- The paper's PRCC values are not printed; the values used for the comparison were read from Fig. 5.
- Eq. (18) as printed is not consistent with Eq. (19); Eq. (19) is used, and it agrees with the next-generation matrix of Eqs. (16)–(17).
- The paper's Fig. 3 and Fig. 4 PRCC values are not printed either; they were read from p. 20 in the same way as Fig. 5.
- ρ1 in ξ1 (p. 10) versus Table 3's ρ, and the lower bound of β_min (p. 4 versus Table 3): see the open questions (4) and (5) above.
- The estimated C of R_eff is dominated by about 1 % of the samples; the leading direction is stable under removing them, the eigenvalue ratio is not (10.59 with all rows, 16.13 without the 10 largest).
