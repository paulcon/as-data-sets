# Hat_Stiffened_Panel_Shear_Buckling: data

Kind: **Python model**. The model is `panel_model.py`, a closed form. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun reproduces them bit for bit.

**Thickness (author decision, 2026-10-31).** The shipped data use a thickness t uniform on [8.02e-4, 8.181e-4] m (0.0316–0.0322 in), **ten times the range coded in the OpenTURNS benchmark** (8.02e-5 to 8.181e-5 m). The OpenTURNS benchmark's thickness range appears to be a factor-10 slip: with it the formula gives 1.13 lb/in at the midpoint, 797 times below the memorandum's 900 lb/in, while the range used here gives 1107 lb/in, slightly above 900 lb/in, as the memorandum says of its closed-form prediction (evidence under "The thickness" below). The data here follow the memorandum's panel. The OpenTURNS-as-coded thickness is kept as a documented variant: `panel_model` takes `variant='openturns'`, the notebook recomputes that variant from the same draws (check (b)), and `python generate_data.py --variant openturns --out <dir>` writes its CSVs (not shipped).

## Files

| file | contents | role |
|---|---|---|
| `Hat_Stiffened_Panel_Shear_Buckling.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1994; outputs committed |
| `panel_model.py` | `buckling_load`, `grad_buckling_load`, `intermediates`, `sample`, `cdf`, `ppf`, `pdf`, `to_normal`, `dx_dz`, `mean_and_cov`, `copula_correlation`, `spearman_of_copula`, `bounds`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `DISTRIBUTIONS`, `LOWER`, `UPPER`, `NOMINAL`, `SPEARMAN_F1F2`, `R_COPULA`, `T_OPENTURNS`, `LOWER_OPENTURNS`, `UPPER_OPENTURNS`, `NOMINAL_OPENTURNS` | the model and its input distributions (numpy, scipy); the distribution functions take `variant='tm'` (default, shipped) or `'openturns'` |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files; `--variant openturns --out <dir>` writes the OpenTURNS-thickness variant elsewhere | regenerates the data |
| `inputs.csv` | 5000 × 10 physical inputs, columns in the order of `NAMES` (units in the header) | model input |
| `outputs.csv` | 5000 × 1: critical shear buckling load (N_xy)_cr in N/m | model output |
| `gradients.csv` | 5000 × 10: ∂(N_xy)_cr/∂x_j with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly.

## Model

**Source.** W. L. Ko and R. H. Jackson (NASA Dryden Flight Research Center, Edwards, California), "Shear Buckling Analysis of a Hat-Stiffened Panel", NASA Technical Memorandum 4644 (also H-2019), November 1994. No DOI; NASA Technical Reports Server record 19950011075 (https://ntrs.nasa.gov/citations/19950011075). **The formula and the input distributions were taken from the OpenTURNS use case** "The stiffened panel model" (OpenTURNS 1.25 documentation, https://openturns.github.io/openturns/1.25/usecases/use_case_stiffened_panel.html, read 2026-10-30, and its code `python/src/usecases/stiffened_panel.py` at tag v1.25.1, https://github.com/openturns/openturns/blob/v1.25.1/python/src/usecases/stiffened_panel.py, read 2026-10-31 and identical to master that day), which cites the memorandum as [ko1994], **except the thickness range, which is ten times OpenTURNS's** (see "The thickness"). We located the memorandum on NTRS, but the only text of it we could read was a machine extraction of the 1994 scan, read through a summarizing web fetch, in which the equations and dimensions were garbled or missing. So **the formula was not checked against the memorandum**, and the page and table references below are to the OpenTURNS page and code. OpenTURNS is LGPL; its use-case code was read (for the copula convention and the distribution definitions) but none was copied, and `panel_model.py` was written from the equations. An independent transcription of the OpenTURNS formula string reproduces the model to 6.7 × 10⁻¹⁶ relative (review, 2026-10-31).

**Equations** (OpenTURNS page, equal-thickness, equal-material form). Local shear buckling of the flat face-sheet strip between two stiffener feet, with a correction for the shear flow carried by the hat:

  (N_xy)_cr = (5.35 + 4 (b₀/a)²) (π²/b₀²) E t³ / (12 (1 − ν²)) · Q,
  Q = 1 + ¼ [2p (h − 2h₀) − h_c (f₁ − f₂)] / (h₀ ℓ),
  A = ℓ t, Ā = A + t (p + (f₁ − f₂)/2), h₀ = [A (h_c + 2t) + t² (f₁ − f₂)] / (2Ā), h = h_c + t.

The page also gives the general form with different face-sheet and hat materials and thicknesses (G_c t_c / G_s t_s multiplying the correction); with t_c = t_s = t and the same material it reduces to the form above (we checked the reduction of h, Ā and h₀ by hand). k_xy = 5.35 + 4(b/a)² is the memorandum's Eq. (2) as quoted by OpenTURNS and in the NTRS text (5.35 for an infinitely long strip, 9.34 for a square plate).

**Inputs** (m = 10; OpenTURNS use-case table and distribution definitions, `stiffened_panel.py` v1.25.1 lines 117–144, except t):

| name | symbol | units | description (OpenTURNS) | distribution | exact mean | exact std | source |
|---|---|---|---|---|---|---|---|
| `E` | E | Pa | Young modulus | normal N(110e9, 55e9) (mean, sd) truncated to [99e9, 121e9] | 1.1e11 | 6.3339e9 | OpenTURNS table |
| `nu` | ν | – | Poisson coefficient | U[0.3675, 0.3825] | 0.375 | 0.0043301 | OpenTURNS table |
| `h_c` | h_c | m | distance between the mean surfaces of the hat and the foot of the stiffener | U[0.0285, 0.0315] | 0.03 | 8.6603e-4 | OpenTURNS table |
| `ell` | ℓ | m | length of the stiffener side (flank) | U[0.04655, 0.05145] | 0.049 | 1.4145e-3 | OpenTURNS table |
| `f_1` | f₁ | m | width of the stiffener foot | U[0.0266, 0.0294] | 0.028 | 8.0829e-4 | OpenTURNS table |
| `f_2` | f₂ | m | width of the stiffener hat | U[0.00627, 0.00693] | 0.0066 | 1.9053e-4 | OpenTURNS table |
| `t` | t | m | thickness of the panel and the stiffener | U[8.02e-4, 8.181e-4] (10 × OpenTURNS's U[8.02e-5, 8.181e-5]) | 8.1005e-4 | 4.6477e-6 | OpenTURNS table × 10, to match the memorandum (see "The thickness") |
| `a` | a | m | width of the panel | U[0.6039, 0.6161] | 0.61 | 3.5218e-3 | OpenTURNS table |
| `b_0` | b₀ | m | distance between two stiffeners (feet) | U[0.04455, 0.04545] | 0.045 | 2.5981e-4 | OpenTURNS table |
| `p` | p | m | half-width of the stiffener | U[0.03762, 0.03838] | 0.038 | 2.1939e-4 | OpenTURNS table |

**Dependence.** f₁ and f₂ are dependent through a normal copula; every other pair is independent. How we read the sources, and our choices:

- **The copula parameter.** The OpenTURNS code sets `correlation_matrix[4, 5] = -0.8` and builds the copula as `ot.NormalCopula(ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(correlation_matrix))` (`stiffened_panel.py`, v1.25.1, lines 148–154; the same on master), so −0.8 is the **Spearman** correlation of f₁ and f₂ and the normal-copula parameter is 2 sin(−0.8π/6) = −0.8135 (`R_COPULA`, computed from `SPEARMAN_F1F2 = -0.8`). We do the same. (The first build read the documentation's matrix description and used −0.8 as the copula parameter, Spearman −0.7859; the review of 2026-10-31 corrected this.) For uniform marginals the Spearman correlation is also the Pearson correlation, so the f₁–f₂ correlation in the exact covariance is −0.8. The shipped samples have Spearman −0.7971. The analysis hardly depends on the dependence: scaling without any decorrelation changes w₁ by an inner product of 0.999603 (notebook, check (a)).
- **Truncated normal E.** N(110e9, 55e9) is read as mean and standard deviation of the parent normal (OpenTURNS's `TruncatedNormal(mu, sigma, a, b)` convention). The parent standard deviation is five times the half-width of the interval, so the truncated law is nearly uniform (sd 6.3339e9 Pa against 6.3509e9 Pa for U[99e9, 121e9]). It is sampled exactly by the inverse CDF on the truncation interval.
- **Units of the QoI.** (N_xy)_cr is a force per unit length (shear flow), N/m with the SI inputs (dimensionally E t³/b₀²); the OpenTURNS page labels it N. The memorandum gives loads in lb/in (1 lb/in = 175.127 N/m).
- **The thickness.** OpenTURNS codes t ~ U(8.02e-5, 8.181e-5) m (`stiffened_panel.py` v1.25.1, line 135), a mean of 8.1005e-5 m = 0.00319 in, a 0.08 mm foil. **This appears to be a factor-10 slip, and the data here use ten times that range**, t ~ U(8.02e-4, 8.181e-4) m (0.0316–0.0322 in). The evidence, computed in the notebook at the midpoint of every range:
  - with the OpenTURNS thickness the load is 197.762 N/m = 1.1292 lb/in, 797 times below the memorandum's finite-element local load of 900 lb/in;
  - with ten times that thickness it is 193859 N/m = 1107 lb/in, 1.230 times 900 lb/in (the flat strip alone, Q = 1, gives 848 lb/in, 0.942 times). The memorandum says its closed-form local prediction is "slightly higher" than the finite-element 900 lb/in (seen in the NTRS text by the reviewer, 2026-10-31), which this is consistent with;
  - a thickness of 7.5569e-4 m (0.0298 in) gives exactly 900 lb/in;
  - 0.032 in is a standard gauge of titanium sheet, and E = 110 GPa is a titanium alloy (Ti-6Al-4V).

  We could not read the memorandum's dimensions, so the exact thickness, and which closed form (with or without Q) the memorandum used, remain open; the evidence settles the factor of ten, not the last 20 %. The active subspace does not depend on the choice (with the OpenTURNS thickness, λ₁/λ₂ = 1153 and |w₁·w₁| = 0.999904; notebook, check (b)); the level of the load does, by a factor of 980.2 on average.
- `NOMINAL` in `panel_model.py` is the midpoint of every range (also the mean of E). It is a reference point for EMA, not a value from either source.

**Quantity of interest.** (N_xy)_cr in N/m, smooth on the whole support. Over the 5000 samples it ranges from 160849 to 230534 N/m (918 to 1316 lb/in; mean 194195 N/m, sd 12803 N/m); the hat correction Q is between 1.2368 and 1.3841.

**Samples outside physical meaning.** None: every input is a positive length or modulus in a narrow interval, f₂ < f₁ always (f₁ − f₂ between 0.01967 and 0.02313 m), and `generate_data.py` counts 0 samples with a non-positive input, ν ≥ 0.5, f₂ ≥ f₁, 2t ≥ h_c, h₀ ≤ 0 or Q ≤ 0.

**Gradients.** Exact, with respect to the physical inputs (`panel_model.grad_buckling_load`): ∂N/∂E = N/E, ∂N/∂ν = 2νN/(1 − ν²), ∂N/∂a = −8π² P Q / a³, ∂N/∂b₀ = −10.7 π² P Q / b₀³ (P = E t³/(12(1 − ν²))), ∂N/∂t = 3N/t + K P ∂Q/∂t (K = π²(5.35/b₀² + 4/a²)), and the derivatives of Q with respect to h_c, ℓ, t, p and f₁ − f₂ through h₀ = [ℓ(h_c + 2t) + t(f₁ − f₂)] / (2ℓ + 2p + f₁ − f₂) by the quotient rule. ∂N/∂f₂ = −∂N/∂f₁ exactly, so the direction of equal steps in f₁ and f₂ is null for every gradient.

**Sampling design.** Monte Carlo: M = 5000 draws by the inverse CDF of `numpy.random.default_rng(1994).random((5000, 10))` (`panel_model.sample`). For the copula (Rosenblatt order f₁, then f₂; r = −0.8135): z₁ = Φ⁻¹(V_f1), z₂ = r z₁ + √(1 − r²) Φ⁻¹(V_f2), f₂ = F⁻¹(Φ(z₂)). M ≥ 10 m and ≥ 300. The OpenTURNS-thickness variant uses the same draws, so it differs only in the t column (exactly a factor 10).

**Input spaces in the notebook** (every input scaled by its own standard deviation; consistent across inputs):

1. Main analysis: u = L⁻¹(x − μ) with LLᵀ = Σ the exact covariance of the physical inputs (`mean_and_cov`), from one `ac.Gaussian(mean, cov)` block used only for its affine map. Eight inputs are simply (x_j − μ_j)/σ_j; u_f1 is the scaled f₁ and u_f2 the part of the scaled f₂ uncorrelated with f₁. No input is scaled by a half-range.
2. Second analysis (Nataf): z_j = Φ⁻¹(F_j(x_j)) (`to_normal`), exactly N(0, R) with R the copula matrix, whitened by `ac.Gaussian(0, R)`; gradients ∂N/∂z_j = ∂N/∂x_j · φ(z_j)/ρ_j(x_j) (`dx_dz`).
3. Checks: (a) diagonal scaling by σ_j without decorrelation; (b) the OpenTURNS-as-coded thickness (variant), with its own exact moments.

**Published results.** Neither source reports a sensitivity analysis or an active subspace, and the OpenTURNS page gives no numerical results. The memorandum (as far as the NTRS text could be read) gives a local shear buckling load of 900 lb/in from a finite-element analysis (credited to a personal communication), says that its closed-form local prediction is slightly higher than that, gives a global shear buckling load of 4296 lb/in (its Eq. 43, with k_xy = 1.89), and states in the abstract that flat-plate theory predicts the local load well and that the global load is several times the local one.

**What the notebook finds** (seed 1994, main analysis): a one-dimensional active subspace, λ₁/λ₂ = 1188 (λ₁ = 1.596 × 10⁸ (N/m)², 99.895 % of the trace; λ₂/λ₃ = 4.40), w₁ ≈ (E +0.885, ν +0.058, h_c +0.009, ℓ −0.285, u_f1 −0.160, u_f2 +0.019, t +0.263, a −0.0007, b₀ −0.177, p +0.077); a line in y₁ has leave-one-out R² = 0.999013 and a quadratic 0.999564. In the Nataf variables, λ₁/λ₂ = 34.8 and |w₁ᵀw₁| = 0.9992; the linearized model gives 35.6 there, so the lower ratio comes from the transform of uniforms to normals, not from the model. With the OpenTURNS-as-coded thickness: λ₁/λ₂ = 1153, |w₁ᵀw₁| = 0.999904.

## Regenerating

```bash
python generate_data.py                           # writes the three CSVs here
python generate_data.py --out /tmp/panel --check  # writes elsewhere and compares
```

```bash
python generate_data.py --variant openturns --out /tmp/panel_ot   # OpenTURNS-as-coded thickness (not shipped)
```

The run takes 0.08 s (under 1 s with Python start-up) and also prints the gradient checks and the count of unphysical samples. `--check` on 2026-10-31 (numpy 2.5.4, scipy 1.18.1, Python 3.13.18) reported the three files identical to the shipped ones. The notebook runs in about 3 s (7 s with kernel start-up under nbconvert).

## Accuracy

- **Values.** Closed form in double precision; `panel_model.buckling_load` reproduces `outputs.csv` exactly (difference 0, printed in the notebook).
- **Gradients.** Exact. Checked at all 5000 samples against central finite differences with step 10⁻⁶ standard deviations of each input (largest relative error, row 2-norm, 3.4 × 10⁻⁸, the size expected from rounding in the differences) and against a complex step of 10⁻³⁰ times each input (largest relative error 1.0 × 10⁻¹⁵).
- **Moments.** The mean and covariance in `mean_and_cov` are closed form (truncated-normal moments; Pearson correlation (6/π) arcsin(r/2) = −0.8 of uniform marginals under a normal copula with r = −0.8135). The shipped samples agree: their sample Spearman correlation is −0.7971 against −0.8.

## How these files relate to the paper

- **Not the paper's computation.** The memorandum is a deterministic buckling analysis; the probabilistic inputs (ranges, the truncated normal and the copula) are OpenTURNS's, not the memorandum's (the thickness range scaled by 10), and we do not know how OpenTURNS chose them. No code or data were published with either source. The samples here are new.
- **Formula from OpenTURNS, not verified against the memorandum** (see Source). We could not check that the hat correction Q, as OpenTURNS writes it, is the memorandum's, nor which formula gives the memorandum's closed-form local load.
- **Thickness.** See "The thickness" under Inputs: the OpenTURNS t gives loads about 800 times below the memorandum's value, so the shipped data use ten times that thickness, which gives 1.23 times the memorandum's finite-element load. The level of the load, not the active subspace, depends on it.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the equations and distributions published in the OpenTURNS documentation and code (thickness range × 10); no OpenTURNS code (LGPL) was copied. Model: W. L. Ko and R. H. Jackson, NASA Dryden Flight Research Center. Probabilistic benchmark: the OpenTURNS developers. Cite:

- W. L. Ko and R. H. Jackson, "Shear Buckling Analysis of a Hat-Stiffened Panel", NASA Technical Memorandum 4644, 1994 (NTRS 19950011075).
- OpenTURNS 1.25 documentation, "The stiffened panel model", https://openturns.github.io/openturns/1.25/usecases/use_case_stiffened_panel.html, and `python/src/usecases/stiffened_panel.py` at tag v1.25.1.
- M. Baudin, A. Dutfoy, B. Iooss, A.-L. Popelin, "OpenTURNS: An Industrial Software for Uncertainty Quantification in Simulation", in *Handbook of Uncertainty Quantification*, Springer, 2017, pp. 2001–2038, doi:10.1007/978-3-319-12385-1_64.
