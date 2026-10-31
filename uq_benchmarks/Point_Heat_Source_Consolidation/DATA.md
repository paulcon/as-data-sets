# Point_Heat_Source_Consolidation: data

Kind: **Python model**. The model is `heat_source_model.py` (numpy and scipy only), and `generate_data.py` writes all four CSV files from it with the saved seed 1985.

## Files

| file | contents | role |
|---|---|---|
| `Point_Heat_Source_Consolidation.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 1985`; outputs committed |
| `heat_source_model.py` | `fields(X, r, t)`, `fields_and_grads(X, r, t)`, `headline(X)`, `derived(X)`, `sample`, `ppf`, `mean_and_sd`, `bounds`, `to_gaussian`, `dx_dz`, checks `u_r_by_quadrature`, `pde_residuals`; constants `NAMES`, `UNITS`, `DISTRIBUTIONS`, `NOMINAL`, `Q`, `A_W`, `MU_W`, `R_GRID`, `T_GRID`, `R_STAR`, `T_STAR` | the model |
| `generate_data.py` | writes the four CSVs | regenerates the data |
| `inputs.csv` | 1000 × 8 physical inputs K, C, rho, E, a_s, k_s, nu, phi (units in the header) | model input |
| `outputs.csv` | 1000 × 3: ΔT (K), p (Pa), u_r (m) at r = 6 m, t = 365 d | model output |
| `gradients.csv` | 1000 × 24: column 8 k + j is ∂(QoI k)/∂x_j with respect to the **physical** inputs (k = 0 ΔT, 1 p, 2 u_r) | model output |
| `fields.csv` | 1000 × 300: column 100 k + 10 i + j is field k at r = `R_GRID[i]`, t = `T_GRID[j]` | model output (6.1 MB) |

The CSVs are comma-separated numbers after two `#` lines (a description; column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Values are written with 17 significant digits, so the notebook's check that the model reproduces the files is exact (`np.array_equal`).

Grid (UQTestFuns parameter set `Chaudhry2021`, commit `82d5c8b`, see Model): r = 2, 4, 6, 8, 10, 12, 15, 20, 25, 30 m; t = 1, 7, 30, 90, 180, 365, 730, 1095, 1460, 1825 days (1 y = 365 d).

## Model

**Source.** J. R. Booker and C. Savvidou, "Consolidation around a point heat source", *Int. J. Numer. Anal. Methods Geomech.* 9(2):173–184, 1985, doi:10.1002/nag.1610090206. The PDF read is a scan of the published article (file tagged `_oa`; OCR text with partly garbled equations), not a preprint. Closed-form solution for a constant point heat source buried deep in a saturated poro-thermo-elastic soil, found with Fourier and Laplace transforms (pp. 176–179). The temperature, excess pore pressure and displacement are the paper's Eq. (23) (p. 178–179; written with functions f and g plotted in Fig. 1, pp. 179–180). A. A. Chaudhry, J. Buchwald, O. Kolditz and T. Nagel (2019, doi:10.1002/nag.2998) re-derived the solution: by their abstract (read via Crossref) they confirmed the temperature, pressure and displacement expressions and corrected only the normal and shear stresses, which this data set does not use. So the spreadsheet's risk note ("must use the corrected equations") is satisfied: the three fields are the same in both papers. The module uses the notation of Chaudhry et al. (2019, Eqs. 18–25) as reproduced in the UQTestFuns documentation:

    dT  = Q/(4 pi K r) f(kappa)
    p   = X/(1 - c/kappa) * Q/(4 pi K r) * [f(kappa) - f(c)]
    u_r = Q/(4 pi K M) * { b' g(kappa) + X/(1 - c/kappa) * [g(kappa) - g(c)] }
    f(A) = erfc(z),  g(A) = tau + (1/2 - tau) erfc(z) - sqrt(tau/pi) exp(-z^2),  tau = A t/r^2,  z = 1/(2 sqrt(tau))
    kappa = K/(C rho),  lam = E nu/((1+nu)(1-2nu)),  G = E/(2(1+nu)),  M = lam + 2G,
    c = k_s M/mu_w,  a_u = phi a_w + (1-phi) a_s,  b' = (lam + 2G/3) a_s,  X = a_u M - b'

**Inputs (m = 8, independent).** The 1985 paper has no parameter study and gives no values. The distributions are those of Chaudhry, Buchwald & Nagel (2021, *Int. J. Rock Mech. Min. Sci.* 139:104662, Table 1), as listed in the UQTestFuns parameter file `src/uqtestfuns/test_functions/saturated_point_heat_source.yaml` (input set `Chaudhry2021`). We have not read the 2021 paper.

**Provenance of the UQTestFuns values.** The test function `SaturatedPointHeatSource` is **not in the UQTestFuns 0.7.0 release on PyPI** (that wheel has no `saturated_point_heat_source` file). It is in the unreleased GitHub source: https://github.com/damar-wicaksono/uqtestfuns, branch `dev` (the default branch; `main` is still at 0.7.0), commit `82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f` of 2026-10-22, downloaded 2026-10-30. The function's files were last changed in commit `1531e92282c3e8f67fa825b8f6c790864762d2aa` (2026-10-15). The downloaded tree had no `.git` directory; the commit was identified by comparing it file by file with a clone of the repository (every text file identical; only image files were missing from the download). Values in an unreleased tree may change before a release; this data set pins that commit. UQTestFuns is MIT-licensed (© 2022 Damar Wicaksono); its notice is reproduced at the end of `heat_source_model.py`. TruncNormal(μ, σ; [a, b]) is a normal with parameters μ, σ truncated to [a, b]; Triangular[a, b] has mode 0.3.

| name | symbol | units | distribution | mean | sd | nominal (`NOMINAL`) | source |
|---|---|---|---|---|---|---|---|
| `K` | K, total thermal conductivity | W/(m K) | TruncNormal(1.79, 0.34; [1.29, 2.45]) | 1.818 | 0.2725 | 1.79 | UQTestFuns / Chaudhry 2021 Table 1 |
| `C` | C, total specific heat capacity | J/(kg K) | Normal(978, 68) | 978 | 68 | 978 | same |
| `rho` | ρ, total density | kg/m³ | TruncNormal(2480, 30; [2420, 2540]) | 2480 | 26.39 | 2480 | same |
| `E` | E, Young's modulus | Pa | TruncNormal(1.28e10, 3.7e9; [5.5e9, 2.01e10]) | 1.28e10 | 3.234e9 | 1.28e10 | same |
| `a_s` | α_s, volumetric thermal expansion of the solid | 1/K | Uniform[3e-5, 7.5e-5] | 5.25e-5 | 1.299e-5 | 5.25e-5 | same |
| `k_s` | k_s, intrinsic permeability | m² | TruncNormal(5.6e-20, 5.5e-20; [7.8e-21, 2.2e-19]) | 7.417e-20 | 4.185e-20 | 5.6e-20 | same |
| `nu` | ν, Poisson's ratio | – | Triangular[0.2, 0.4], mode 0.3 | 0.3 | 0.04082 | 0.3 | same |
| `phi` | φ, porosity | – | TruncNormal(0.15, 0.0276; [0.097, 0.185]) | 0.1463 | 0.02107 | 0.15 | same |

Held: Q = 3000 W (source power), α_w = 4e-4 1/K (water expansion), μ_w = 1e-3 Pa s (water viscosity) (UQTestFuns parameter set `Chaudhry2021`, same commit). `NOMINAL` is the reference point of the UQTestFuns documentation (μ of the normals, mid-range of the uniform, mode of the triangular). Means and standard deviations are exact moments of the marginals (scipy).

**Quantities of interest.** ΔT, p and u_r at r = 6 m, t = 1 y (`outputs.csv`), chosen because the pore pressure at 6 m peaks between 180 d and 1 y at the nominal point (5.66 and 5.63 MPa); and all 300 field values (`fields.csv`). The spreadsheet row lists the fields on the 10 × 10 grid; the grid values are not listed on the UQTestFuns web page but are in its parameter file (above).

**Gradients.** Exact, by the chain rule: the partial derivatives of each field with respect to (Q/(4πK), κ, c, M, b', X), times the Jacobian of these with respect to the eight inputs. The quotients [f(κ) − f(c)]/(1 − c/κ) and [g(κ) − g(c)]/(1 − c/κ) have a removable singularity at c = κ (72 of the 1000 samples have |1 − c/κ| < 0.1; c/κ ranges from 0.083 to 7.03). They are written as κ (h(κ) − h(c))/(κ − c); where |κ − c|/min(κ, c) · max(1, z²) < 1 the divided difference is the integral of h′(A) over [c, κ] by a 24-point Gauss–Legendre rule (its derivatives use h″), elsewhere the direct quotient is used (for g ahead of both fronts, the linear part τ is divided exactly and the remainder is evaluated with `erfcx`).

**Sampling design.** Simple Monte Carlo: M = 1000 independent draws, the inverse-CDF transform of `numpy.random.default_rng(1985).random((1000, 8))` (one uniform column per input, in the order of `NAMES`). All draws lie inside the stated supports and the physical domain (C is an untruncated normal; its smallest draw is 769.4 J/(kg K)). Field values below the double-precision range underflow to exactly 0: ΔT 3993 and p 2941 of 100,000 values each, in 6 grid cells each (large r, early t, where the heat front has not arrived); u_r never. These are the model's values and are kept; the notebook's grid sweep excludes cells whose median |f| is below 1e-6 of its grid maximum (29 cells for ΔT, 23 for p, 0 for u_r).

**Published results.** The 1985 paper has no uncertainty or sensitivity study and no active-subspace results. Its Fig. 1 plots f and g (compared in the notebook with digitized values: differences ≤ 0.012 and ≤ 0.013), and Figs. 2–6 treat a cylindrical source by integrating the point solution (not reproduced). Chaudhry et al. (2021) computed local and first-order Sobol' sensitivities over the grid; the UQTestFuns documentation (`docs/test-functions/saturated-point-heat-source.md` in the same unreleased tree) transcribes from their Fig. 16 a ranking by the maximum first-order index over the grid, which the notebook compares with its own Monte Carlo estimate (agreement for ΔT and the top two inputs of p; α_s ranks 7th, not 3rd, for p, and 1st, not 3rd, for u_r, where K is not 1st). For u_r, ranks 3–5 (K, φ, C) lie within 0.02 of each other: at N = 50,000 their order is not resolved; an independent N = 200,000 estimate puts K third (K − φ = 0.016 ± 0.007, K − C = 0.018 ± 0.003, bias bound of the K maximum 0.0056), probably but narrowly, and leaves φ vs C unresolved (0.002 ± 0.005). No published active-subspace analysis of this model was found.

## Regenerating

```bash
python generate_data.py
```

Run time 0.5 s (this container, numpy 2.5.4, scipy 1.18.1, Python 3.13). The draws are fixed by the seed, so a run reproduces the files bit for bit on the same numpy/scipy versions; the inverse CDFs come from `scipy.stats`, so a different scipy version could change the last digits of the inputs. The notebook runs in about 30–65 s wall clock in this container (two runs: 32 s and 64 s, the spread due to other load), including a 50,000-sample Monte Carlo Sobol' estimate over the whole grid and a 200,000-sample estimate for u_r.

## Accuracy

Checked with scratch scripts (not committed), on the 1000 shipped samples unless noted:

- **Against UQTestFuns** (GitHub `dev`, commit `82d5c8b`, unreleased; `evaluate` of `SaturatedPointHeatSource`, an independent implementation that uses a Taylor expansion near c = κ): largest relative difference over all 300 000 nonzero field values 2.8e-13 (ΔT), 1.5e-12 (p), 1.2e-11 (u_r); the same values underflow to 0 in both.
- **Against the 1985 Eq. (23)** typed directly from the scan (its Y, Z form, no special treatment of c = κ), for the 842 samples with |1 − c/κ| > 0.2: 4.4e-16 (ΔT), 1.1e-14 (p), 2.2e-13 (u_r).
- **Field equations.** u_r agrees with an independent radial integration of M ∇·u = p + b′ΔT (Gauss–Legendre, 400 nodes) to 2.7e-14 (45 cases: 5 samples × r ∈ {2, 6, 25} m × t ∈ {7 d, 1 y, 5 y}). The residuals of ∂T/∂t = κ∇²T and ∂p/∂t = c∇²p + X ∂T/∂t by central differences (45 cases) fall with the step h (relative): 9.7e-3, 9.8e-5, 3.5e-6 (heat) and 6.5e-3, 6.5e-5, 6.5e-7 (pressure) for h = 1e-2, 1e-3, 1e-4, i.e. the truncation error of the differences.
- **Gradients against central differences** (step h·sd_j per input; error measured as ‖FD − G‖/‖G‖ with both scaled by the input standard deviations): for the three QoIs at (6 m, 1 y), worst sample 6.4e-10, 3.0e-7, 5.0e-8 at h = 1e-4 (truncation), 1.0e-9, 3.1e-9, 6.9e-10 at h = 1e-5, and 8.6e-9, 3.4e-9, 5.6e-9 at h = 1e-6 (round-off). Over all 1000 samples and every analysed grid cell at h = 1e-5: worst 1.9e-9 (ΔT), 2.3e-8 (p), 1.7e-8 (u_r); median 1.7e-10, 1.1e-10, 1.5e-10. The notebook's check cell repeats the headline check and prints 1.0e-09, 3.1e-09, 6.9e-10.
- **Linear trend vs. mean gradient** (a coordinate check): the least-squares slope of each QoI on the normalized inputs (x − mean)/sd and the mean normalized gradient point the same way (cosines 0.99991 ΔT, 0.9953 p, 0.9772 u_r) and differ in norm by 1.6 %, 23 % and 28 % relative to the mean gradient; the larger gaps for p and u_r come from their strong curvature, not from a scaling error.
- **Divided differences.** Compared with composite Gauss–Legendre quadrature of h′ and h″ (400 panels × 30 nodes; 1500 random cases per function over the diffusivities and grid of this study, both branches): the value and both partial derivatives agree to ≤ 3.2e-13 (relative) for f, and the value and κ-derivative to ≤ 5.2e-14 for g. The c-derivative of g agrees to ≤ 4.1e-13 wherever c ∂D/∂c is at least 1e-3 of κ ∂D/∂κ; far ahead of the front, where it is exponentially small, its relative error grows (up to 100 % where it underflows), but its absolute error stays below 2e-15 of κ ∂D/∂κ, so the gradient is unaffected.

## How these files relate to the paper

- **Not a computation from the paper.** Booker and Savvidou give the solution but no parameter values, ranges or samples. The inputs, distributions, held parameters and grid come from Chaudhry et al. (2021) through UQTestFuns, and the samples, gradients and analysis are new. Nothing here is a value printed in the 1985 paper except the functions f and g.
- **The model.** `heat_source_model.py` is an independent numpy/scipy implementation of the equations above; no UQTestFuns code is copied. The input distributions, held parameters and grid are taken from the UQTestFuns parameter file, and its MIT notice is reproduced at the end of the module.
- **Open questions** (only the authors can settle them): (1) the ranking of α_s for p (3rd in the UQTestFuns transcription of Chaudhry et al. 2021, Fig. 16; 7th here, first-order index 0.019 ± 0.002) and of K and α_s for u_r (1st and 3rd there; here α_s is 1st, 0.642 ± 0.005, and K is not 1st: 0.245 ± 0.003 at N = 50,000, probably 3rd at N = 200,000 but within 0.02 of φ and C) — which estimator and which grid quantity underlie Fig. 16, the held parameters, or the transcription; and whether Fig. 16 supports α_s 3rd for p, given that α_s contributes only about 17 % of X at the nominal point; (2) whether Chaudhry et al. (2021) used α_w = 4e-4 1/K and μ_w = 1e-3 Pa s, as UQTestFuns does; (3) the 1985 paper's illustrative parameters for the cylindrical source are partly illegible in the scan (we read ν = 0.4 and c/κ = 2) and are not used.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model: J. R. Booker and C. Savvidou (1985), confirmed for these fields by A. A. Chaudhry, J. Buchwald, O. Kolditz and T. Nagel (2019). Input distributions: A. A. Chaudhry, J. Buchwald and T. Nagel (2021), via UQTestFuns (D. Wicaksono and M. Hecht, *J. Open Source Softw.* 8(90):5671, 2023, doi:10.21105/joss.05671; MIT licence, © 2022 Damar Wicaksono; values from the unreleased GitHub source https://github.com/damar-wicaksono/uqtestfuns, commit `82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f`, not from the 0.7.0 PyPI release). The MIT notice is reproduced at the end of `heat_source_model.py`. Cite J. R. Booker and C. Savvidou, "Consolidation around a point heat source", Int. J. Numer. Anal. Methods Geomech. 9(2):173–184, 1985, doi:10.1002/nag.1610090206, and Chaudhry et al. (2021), doi:10.1016/j.ijrmms.2021.104662.
