# River_Flood_Dyke_Overflow: data

Kind: **Python model**. The model is `flood_model.py`, a closed form. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun reproduces them bit for bit.

## Files

| file | contents | role |
|---|---|---|
| `River_Flood_Dyke_Overflow.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2013; outputs committed |
| `flood_model.py` | `overflow`, `river_height`, `grad_overflow`, `cost`, `grad_cost`, `sample`, `ppf`, `cdf_sf_pdf`, `to_gaussian`, `dx_dz`, `mean_and_sd`, `exact_moments`, `poincare_constants`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `DISTRIBUTIONS`, `LOWER`, `UPPER`, `NOMINAL` | the model and its input distributions (numpy, scipy) |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 10000 × 8 physical inputs, columns in the order of `NAMES` (units in the header) | model input |
| `outputs.csv` | 10000 × 2: overflow S in m, cost Cp in million euros | model output |
| `gradients.csv` | 10000 × 16: columns 0–7 ∂S/∂x_j, columns 8–15 ∂Cp/∂x_j, with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly.

## Model

**Source.** M. Lamboni, B. Iooss, A.-L. Popelin, F. Gamboa, "Derivative-based global sensitivity measures: General links with Sobol' indices and numerical tests", *Math. Comput. Simul.* 87:45–54, 2013, doi:10.1016/j.matcom.2013.02.002, Section 4.3. The PDF read is the authors' manuscript (file `_oa`; its text is arXiv:1202.0943v2, 2 July 2012, recompiled), not the journal version; pages and tables below are the manuscript's. The model is the pedagogical flood model of B. Iooss, *J. Soc. Fr. Stat.* 152(1):1–23, 2011 (not read).

**Equations** (Eq. (4.20), p. 14). A crude simplification of the 1-D Saint-Venant equations for uniform, constant flow in a wide rectangular channel with the Strickler law:

  H = (Q / (B K_s √((Z_m − Z_v)/L)))^0.6,  S = Z_v + H − H_d − C_b,

with H the maximal annual river height and S the maximal annual overflow above the dyke, both in m. The cost (Eq. (4.21), p. 14), in million euros, is

  C_p = 1{S > 0} + [0.2 + 0.8 (1 − exp(−1000/S⁴))] 1{S ≤ 0} + (H_d 1{H_d > 8} + 8·1{H_d ≤ 8}) / 20.

**Inputs** (m = 8, independent; Table 4, p. 14):

| name | symbol | units | description | distribution (Table 4) | exact mean | exact std | source |
|---|---|---|---|---|---|---|---|
| `Q` | Q | m³/s | maximal annual flowrate | truncated Gumbel G(1013, 558) on [500, 3000] | 1356.88 | 561.147 | Table 4, p. 14 |
| `Ks` | K_s | m^(1/3)/s | Strickler coefficient | truncated normal N(30, 8) on [15, +∞) | 30.5675 | 7.42730 | Table 4, p. 14 |
| `Zv` | Z_v | m | river downstream level | triangular T(49, 50, 51) | 50 | 0.408248 | Table 4, p. 14 |
| `Zm` | Z_m | m | river upstream level | triangular T(54, 55, 56) | 55 | 0.408248 | Table 4, p. 14 |
| `Hd` | H_d | m | dyke height (design parameter) | uniform U[7, 9] | 8 | 0.577350 | Table 4, p. 14 |
| `Cb` | C_b | m | bank level | triangular T(55, 55.5, 56) | 55.5 | 0.204124 | Table 4, p. 14 |
| `L` | L | m | length of the river stretch | triangular T(4990, 5000, 5010) | 5000 | 4.08248 | Table 4, p. 14 |
| `B` | B | m | river width | triangular T(295, 300, 305) | 300 | 2.04124 | Table 4, p. 14 |

Readings of Table 4 that the paper leaves implicit (our choices):

- G(μ, β) is the Gumbel law for maxima, CDF exp(−exp(−(x − μ)/β)), location 1013, scale 558 (Table 1, p. 8 gives its v(x), median and Cheeger constant in this parameterization).
- N(30, 8) has **standard deviation** 8. Table 1 writes N(μ, σ²), but the paper's Morris test (Table 3, p. 12) gives C = 0.01 for N(0.5, 0.1), which is σ² only if the second argument is σ; Iooss (2011) gives K_s a standard deviation. A variance of 8 would make Table 5's Υ_Ks (0.198) impossible to match with any of the constants below.
- T(a, c, b) is triangular with minimum a, mode c, maximum b (all five are symmetric).
- K_s has no unit in Table 4; in the Strickler law it is m^(1/3)/s.
- The exact means and standard deviations in the table are computed by quadrature (`flood_model.mean_and_sd`); the paper does not give them.

`NOMINAL` in `flood_model.py` (Q = 1013, K_s = 30, the modes, H_d = 8) is a reference point for EMA, not a quantity of the paper.

**Quantity of interest.** S in m, smooth everywhere on the support (Z_m > Z_v always). On the support, S ≤ −3.663 m (at Q = 3000, K_s = 15, Z_v = 51, Z_m = 54, H_d = 7, C_b = 55, L = 5010, B = 295), so flooding (S > 0) never occurs under Table 4.

**The cost and its kink.** Cp is also stored. Its first two terms join continuously at S = 0 (0.8(1 − e^(−1000/S⁴)) → 0.8 as S → 0⁻), but since S < 0 everywhere only the maintenance branch is used. The investment term max(H_d, 8)/20 has a **kink at H_d = 8 m**, the middle of the H_d range, where ∂Cp/∂H_d jumps by 1/20. `grad_cost` returns the derivative from the left there (no sample lies on it). The notebook's main analysis uses the smooth S, as the paper does for its main table.

**Gradients.** Exact, with respect to the physical inputs: ∂S/∂Q = 0.6H/Q, ∂S/∂K_s = −0.6H/K_s, ∂S/∂Z_v = 1 + 0.3H/(Z_m − Z_v), ∂S/∂Z_m = −0.3H/(Z_m − Z_v), ∂S/∂H_d = ∂S/∂C_b = −1, ∂S/∂L = 0.3H/L, ∂S/∂B = −0.6H/B. For Cp, ∂Cp/∂S = −3200 S⁻⁵ exp(−1000/S⁴) for S < 0, times ∇S, plus 1/20 in the H_d component for H_d > 8. S depends on Z_v, H_d, C_b only through Z_v − H_d − C_b and Z_m − Z_v, so two directions are exactly null for every gradient.

**Sampling design.** M = 10 000 independent draws by the inverse CDF of `numpy.random.default_rng(2013).random((10000, 8))` (`flood_model.sample`); the truncated laws are sampled exactly (inverse CDF restricted to the truncation interval). M ≥ 10 m and ≥ 300; the paper's DGSM also used 10⁴ points (a Sobol' sequence, p. 15).

**Input spaces in the notebook** (active_subspaces v1 has uniform and Gaussian spaces only):

1. Main analysis, physical inputs: `ac.Product([ac.Gaussian(mean, var) for Q, K_s, Z_v, Z_m; ac.Uniform(7, 9) for H_d; ac.Gaussian(mean, var) for C_b, L, B])`, with the exact marginal means and variances. The Gaussian blocks are used only for their affine map (x − μ)/σ; the samples keep their true distributions. H_d is mapped from [7, 9] to [−1, 1], a scale of 1 m (√3 standard deviations).
2. Second analysis, transformed inputs: z_j = Φ⁻¹(F_j(x_j)) (`flood_model.to_gaussian`), exactly standard normal, in `ac.Gaussian.reference` blocks, with H_d kept in `ac.Uniform(7, 9)`; gradients ∂S/∂z_j = ∂S/∂x_j · φ(z_j)/ρ_j(x_j) (`flood_model.dx_dz`).

**Published results** (Table 5, p. 15, for S; Table 6, p. 16, for Cp): first-order and total Sobol' indices S_j, ST_j (two Monte Carlo samples of 10⁵, 20 replicates, 2 × 10⁷ runs), DGSM ν_j = E[(∂f/∂x_j)²] and weighted τ_j from 10⁴ Sobol'-sequence points with finite differences, and the bound Υ_j = C_j ν_j / D (Eq. (4.18)). For S: S_j / ST_j = 0.343/0.353 (Q), 0.130/0.139 (K_s), 0.185/0.186 (Z_v), 0.003/0.003 (Z_m), 0.276/0.276 (H_d), 0.036/0.036 (C_b), 0/0 (L, B); ν_j = 1.296e-06, 3.286e-03, 1.123, 2.279e-02, 0.8389, 0.8389, 2.147e-08, 2.386e-05; Υ_j = 2.807, 0.198, 0.561, 0.011, 0.340, 0.105, 0.000, 0.000. No active subspace.

**What the notebook finds** (seed 2013): a one-dimensional active subspace of S, λ₁/λ₂ = 47.0 (λ₁ = 1.915, 97.0 % of the trace), w₁ ≈ (Q +0.496, K_s −0.309, Z_v +0.341, Z_m −0.047, H_d −0.720, C_b −0.147, L 0.000, B −0.008); a line in y₁ has leave-one-out R² = 0.9783. In the transformed inputs, λ₁/λ₂ = 55.7 and |w₁ᵀw₁| = 0.9996. For Cp, λ₁/λ₂ = 2.05 and λ₂/λ₃ = 15.38 (two-dimensional).

## Regenerating

```bash
python generate_data.py                           # writes the three CSVs here
python generate_data.py --out /tmp/flood --check  # writes elsewhere and compares
```

The run takes 0.2 s (0.8 s with Python start-up) and also prints the gradient check. `--check` on 2026-10-30 (numpy 2.5.4, scipy 1.18.1, Python 3.13.18) reported the three files identical to the shipped ones. The notebook runs in about 5 s (9 s with kernel start-up under nbconvert).

## Accuracy

- **Values.** Closed form in double precision; `flood_model.overflow` reproduces `outputs.csv` exactly (difference 0, printed in the notebook).
- **Gradients.** Exact. Checked against central finite differences with step 10⁻⁶ standard deviations of each input, at all 10 000 samples: largest relative error (row 2-norm) 1.4 × 10⁻⁸ for S and 2.9 × 10⁻⁸ for Cp (no sample within 10⁻⁵ m of the kink), the size expected from rounding in the differences.
- **Quadrature.** `exact_moments` uses adaptive quadrature (relative tolerance 10⁻¹³) for the moments of Q and K_s, and 40-point Gauss–Legendre rules on each side of the mode for the triangular inputs (exact for the polynomial densities times the smooth integrands to machine precision). Against 400 000 Monte Carlo samples (seed 0, scratch check): Var(S) 1.1745 vs 1.1704, ν_j within 0.2 % for every input. On the shipped 10 000 samples: mean of S −10.9342 (exact −10.9445), variance 1.2087 (exact 1.1745).

## How these files relate to the paper

- **Not the paper's computation.** The paper's indices come from its own Monte Carlo and Sobol'-sequence runs with finite differences (perturbation not stated for this model; 0.01 % for the Morris test, p. 11); no code or data were published with it. The samples here are new, and the exact Sobol' indices, DGSM and variance come from quadrature.
- **Sobol' indices agree.** The exact values (S_Q 0.3448, ST_Q 0.3536, S_Ks 0.1338, ST_Ks 0.1423, S_Zv 0.1896, ST_Zv 0.1899, S_Zm 0.0035, ST_Zm 0.0038, S_Hd = ST_Hd 0.2838, S_Cb = ST_Cb 0.0355) are within 0.008 of Table 5.
- **Table 5's DGSM carry a common factor.** Every ν_j in Table 5 is 0.8381–0.8392 times the exact value, including ν_Hd = ν_Cb = 0.8389, where the exact value is 1 (∂S/∂H_d = ∂S/∂C_b = −1 at every point). We report the exact values (1.5447e-06, 3.9210e-03, 1.3394, 2.7156e-02, 1, 1, 2.5601e-08, 2.8453e-05) and do not copy the table's. The cause is not in the paper; a sum over fewer than 10⁴ valid points divided by 10⁴ would produce it.
- **τ_j.** Table 5's τ_j are reproduced, up to the same factor (ratio 0.829–0.840), by E[(∂S/∂x_j)² w(x_j)] with the Sobol'–Kucherenko weight w(x) = (1 − 3x + 3x²)/6, which is defined for U[0, 1], applied to the unnormalized inputs.
- **Υ_j.** For Q and H_d, whose constants are unambiguous, Table 5's Υ_j equal C_j ν_j with its ν_j and no division by D (or D = 1; the exact D is 1.1745). Read this way, the implied constants are C_Q = 2.1659 × 10⁶ (the Cheeger constant 1/ρ(m)² of the truncated Gumbel at its median is 2.1656 × 10⁶) and C_Hd = 0.40529 = 4/π²; for the triangular inputs about half of 1/ρ(m)² (0.4996, 0.4827, 0.1252 against 1, 1, 0.25); for K_s 60.3, neither σ² = 64 nor 1/ρ(m)² = 378.6. `flood_model.poincare_constants` returns 1/ρ(m)² for the log-concave inputs and (b − a)²/π² for H_d; with them and exact ν_j and D, Υ = 2.848, 1.264, 1.140, 0.0231, 0.345, 0.213, 2.2e-06, 6.1e-04, which bound ST_j for every input.
- **Table 6 (cost).** The ν_j of Table 6 are 1391–1475 times the sample values from Eq. (4.21) as printed (for every input, including H_d). We could not reconcile this; the formula may differ from the one used in the computations. The Sobol' indices of Cp were not recomputed.
- **The OpenTURNS flood use case differs** (per the paper card, not checked here): K_s N(30, 7.5) truncated at 0, Z_v U(49, 51), Z_m U(54, 56), Q truncated at 0. These files use Table 4 of the paper.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations and distributions; no data or code were copied. Model and analysis: M. Lamboni, B. Iooss, A.-L. Popelin, F. Gamboa (EDF R&D, Université Paris Descartes, IMT); the model is from B. Iooss (2011). Cite:

- M. Lamboni, B. Iooss, A.-L. Popelin, F. Gamboa, "Derivative-based global sensitivity measures: General links with Sobol' indices and numerical tests", *Math. Comput. Simul.* 87:45–54, 2013, doi:10.1016/j.matcom.2013.02.002.
- B. Iooss, "Revue sur l'analyse de sensibilité globale de modèles numériques", *J. Soc. Fr. Stat.* 152(1):1–23, 2011.
