# Simply_Supported_Tube_Deflection: data

Kind: **Python model**. The model is `tube_model.py`, three closed-form beam formulas. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun reproduces them bit for bit.

## Files

| file | contents | role |
|---|---|---|
| `Simply_Supported_Tube_Deflection.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2011; outputs committed |
| `tube_model.py` | `outputs`, `gradients`, `second_moment`, `sample`, `virtual_work`; `NAMES`, `DESCRIPTIONS`, `DISTRIBUTIONS`, `QOI_NAMES`, `MEAN`, `SD` | the model (numpy; scipy only in the `virtual_work` check) |
| `generate_data.py` | draws the samples and writes the three CSVs; prints the gradient and quadrature checks; `--check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 1000 × 6 physical inputs `F, L, a, De, di, E` | model input |
| `outputs.csv` | 1000 × 3: `deflection` v(a), `theta_left` v′(0), `theta_right` v′(L) | model output |
| `gradients.csv` | 1000 × 18: columns 0–5 ∂deflection/∂(F, L, a, De, di, E), 6–11 ∂theta_left/∂(…), 12–17 ∂theta_right/∂(…); with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names) and are read by `np.loadtxt(path, delimiter=',')`; `gradients.csv` reshapes to (1000, 3, 6). Values are written with 17 significant digits, so they round-trip exactly. Units are not stated by the benchmark: any consistent set (force, length, force/length²); the deflection has the length unit and the slopes are in radians.

## Model

**Source.** There is no journal article. The benchmark is the OpenTURNS use case "Vertical deflection of a tube" (class `DeflectionTube`; OpenTURNS 1.22 documentation, https://openturns.github.io/openturns/1.22/usecases/use_case_deflection_tube.html, © 2005–2023 Airbus-EDF-IMACS-ONERA-Phimeca, no licence stated on the page). The formulas are the classical simply-supported-beam results with an intermediate point load, which the page attributes to textbook beam tables: J. M. Gere and B. J. Goodno, *Mechanics of Materials*, 7th ed., Cengage, 2009; R. G. Budynas and J. K. Nisbett, *Shigley's Mechanical Engineering Design*, 9th ed., McGraw-Hill, 2011 (and web pages: Wikipedia, MechaniCalc, clag.org.uk). We did not open the textbooks; the formulas were checked independently (see Accuracy). The OpenTURNS source (`python/src/usecases/deflection_tube.py`, tag v1.22, LGPL) was read to confirm the formulas; none of it is copied.

**Equations.** Euler–Bernoulli beam, length L, simple supports at x = 0 and x = L, downward force F at x = a, b = L − a, v positive upward:

- v(a) = −F a² b² / (3 E L I) (deflection under the load)
- v′(0) = −F b (L² − b²) / (6 E L I) (slope at the left support)
- v′(L) = F a (L² − a²) / (6 E L I) (slope at the right support)
- I = π (De⁴ − di⁴) / 64 (second moment of area of a circular tube about a diameter).

**The factor of two.** OpenTURNS writes I = π (De⁴ − di⁴)/32 (on the page and in the code), which is the polar moment of the section, not the bending second moment of area. Its outputs are therefore exactly half of ours (`tube_model.outputs(X, openturns_convention=True)`; the notebook prints the ratio 0.5). We ship the textbook (Euler–Bernoulli) values. A constant factor multiplies every eigenvalue by 4 and changes no eigenvector, active variable or R². The /32 looks like a genuine slip (J used for I = J/2); it is still in the OpenTURNS source on the `master` branch (`python/src/usecases/deflection_tube.py`, checked 2026-10-31), and is worth reporting upstream. The page also uses b in the right-slope formula without defining it; the source defines it (`var b:=L - a`), as we do.

**Range of validity (disclosure).** Both conventions are Euler–Bernoulli values, and this tube is far too stubby for that theory to be accurate: L/De = 1.875 at the means, whereas Euler–Bernoulli theory needs a span of roughly ten depths or more. Timoshenko beam theory (shear strain V/(κGA), G = E/(2(1 + ν)), A = π(De² − di²)/4) leaves the bending moment and the cross-section rotations unchanged (the beam is statically determinate) and adds F a b/(κ G A L) to the deflection under the load, a fraction 3EI/(κGA a b) of the bending deflection. With ν = 0.3 (assumed; the benchmark gives none) and Cowper's shear coefficient for a hollow circular section (κ = 0.852 at the means; Cowper 1966), the fraction is 0.712 at the input means (0.684 with the solid-section κ = 0.886), so a shear-deformable tube deflects 1.712 times the Euler–Bernoulli value, which is short by 0.416; over the 1000 samples the fraction runs from 0.590 to 1.142. The centreline end slopes gain 0.382 (left) and 0.552 (right) of their bending values at the means. The notebook prints these numbers (section *The model*). **The data are the Euler–Bernoulli benchmark as posed, not a physically accurate prediction for this tube**; we keep the model Euler–Bernoulli on purpose, as a test function with the product structure of beam theory. The factor of two between conventions is a departure of the same order.

**Quantities of interest.** `deflection`, `theta_left`, `theta_right` as above (the OpenTURNS outputs "Deflection", "Left angle", "Right angle", times 2). Each is F/(E I) times a function of a and L: |v(a)| ∝ a²b²/L peaks at a = L/2; |v′(0)| ∝ ab(2L − a)/L peaks at a = L(1 − 1/√3) ≈ 0.634 (outside the sampled range of a); v′(L) ∝ ab(L + a)/L peaks at a = L/√3 ≈ 0.866 (inside it).

**Inputs** (m = 6, independent, all six random). Distributions from the input table on the OpenTURNS page (the spreadsheet's note "not stated (uniform assumed)" is superseded: the page states them). The page's API section, and the class's `inputDistribution`, instead fix L, a, De, di at Dirac values (1.5, 1.0, 0.8, 0.1) with only F and E random; we follow the input table, as the use case's stated study, and use the Dirac point only as a reference point.

| name | symbol | description | distribution | mean | sd | source |
|---|---|---|---|---|---|---|
| `F` | F | point force (downward) | Normal(1, 0.1) | 1 | 0.1 | OpenTURNS page, input table |
| `L` | L | tube length (span) | Normal(1.5, 0.01) | 1.5 | 0.01 | OpenTURNS page, input table |
| `a` | a | distance of the force from the left support | Uniform(0.7, 1.2) | 0.95 | 0.14434 | OpenTURNS page, input table |
| `De` | D | outer diameter | Triangular(0.75, 0.8, 0.85) | 0.8 | 0.020412 | OpenTURNS page, input table |
| `di` | d | inner diameter | Triangular(0.09, 0.1, 0.11) | 0.1 | 0.0040825 | OpenTURNS page, input table |
| `E` | E | Young's modulus | Normal(200000, 2000) | 200000 | 2000 | OpenTURNS page, input table |

Standard deviations are exact: uniform half-width/√3, symmetric triangular half-width/√6. At the means: v(a) = −1.509059e−05, v′(0) = −2.960355e−05, v′(L) = 3.537986e−05. di⁴ is only 2.4e−4 of De⁴, so di is nearly inert.

**Samples and physical meaning.** None outside the physical domain: F ∈ [0.683, 1.277] (P(F ≤ 0) ≈ 1e−23 under the normal), the load is always inside the span (a ∈ [0.7005, 1.1996], L ∈ [1.4693, 1.5322], smallest L − a = 0.2885), di < De and E > 0 everywhere. None dropped; no duplicate rows. 98 samples have a < L/2 and 338 have a < L/√3 (the rising branches of |v(a)| and v′(L)).

**Scaling.** The inputs are normal, uniform and triangular; active_subspaces v1 has only uniform and Gaussian spaces. Every input is scaled the same way, by its own standard deviation, z_j = (x_j − mean_j)/sd_j, by passing `ac.Gaussian(MEAN, SD**2)` as the space; the library uses it only to standardize the physical X and gradients. The samples come from the true distributions, so the gradient average estimates C under the true joint density.

**Gradients.** Exact: each QoI is a product of powers, g = ±c F a^p b^q (2L − a or L + a) / (E L I), so ∂g/∂x_j = g ∂ln|g|/∂x_j, with the logarithmic derivatives written out in `tube_model.gradients` (e.g. ∂ln|v(a)|/∂a = 2/a − 2/b, ∂ln|v(a)|/∂L = 2/b − 1/L, ∂ln I/∂De = 4De³/(De⁴ − di⁴)).

**Sampling design.** M = 1000 independent Monte Carlo draws (not Latin hypercube) from `numpy.random.default_rng(2011)`, columns drawn in the order F, L, a, De, di, E (`tube_model.sample`). M ≥ 10m and ≥ 300.

**Published results.** None: the OpenTURNS page gives the model and distributions but no moments, sensitivity indices or subspace. The notebook computes reference moments from 10⁶ fresh samples (deflection mean −1.42113e−05 ± 7.6e−09, sd 3.8821e−06 ± 4.5e−09, 95 %; slopes in the notebook).

**Active-subspace results (notebook).** n = 2 for all three outputs (set by hand; the library's absolute-gap rule picks 1): λ1/λ2 = 8.3, 14.0, 3.1 and λ2/λ3 = 161.3, 70.9, 157.7 (deflection, left slope, right slope). Each two-dimensional subspace contains the load position e_a (projection norms 0.9987, 0.9983, 0.9987) and the amplitude direction of F/(EI) (0.9913, 0.9968, 0.9955); the slope subspaces are within 0.0534 (left) and 0.0380 (right) of the deflection's (2-norm of the difference of projectors). Quadratics in (y1, y2) give LOO R² 0.9975, 0.9986, 0.9966.

## Regenerating

```bash
python generate_data.py                            # writes the three CSVs here
python generate_data.py --out /tmp/tube --check    # writes elsewhere and compares
```

The run takes about 0.05 s (0.85 s with Python start-up) and prints the two checks. `--check` on 2026-10-31 (numpy 2.5.4, scipy 1.18.1, Python 3.13.18) reported the three files identical to the shipped ones. The notebook runs in 3–5 s by its own timer and 7–11 s wall time under `nbconvert`, depending on machine load (3.4 s and 7.1 s in the last run, 2026-10-31).

## Accuracy

- **Values.** Closed form in double precision; `tube_model` reproduces `outputs.csv` exactly (printed in the notebook).
- **Formulas.** Checked against the unit-load (virtual-work) method, numerical quadrature of the bending moment with `scipy.integrate.quad` (`tube_model.virtual_work`), which does not use the tabulated formulas: largest relative difference 5.6e−16 at the first 20 samples (`generate_data.py`) and 6.7e−16 at the first 50 (notebook).
- **Gradients.** Exact. Against central finite differences with relative step 10⁻⁶ per input at all 1000 samples, the largest relative error (row 2-norm) is 5.1e−10 (deflection), 4.9e−10 (left slope), 5.0e−10 (right slope), the size of the differencing error.

## How these files relate to the source

- **New samples of a documented benchmark.** OpenTURNS publishes the model and the distributions, not data. The inputs are new Monte Carlo draws from the page's distributions; the outputs and gradients are computed here.
- **Outputs are twice the OpenTURNS outputs** because of the I convention above; the factor is exact and has no effect on the active subspace.
- **Distributions follow the page's input table** (all six inputs random), not the class's Dirac defaults (see Inputs).
- **Euler–Bernoulli benchmark as posed.** The outputs are the closed-form Euler–Bernoulli values; at L/De ≈ 1.9 they are not an accurate prediction for a real tube (Timoshenko shear adds about 0.7 times the bending deflection; see Range of validity).

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the textbook equations; no data were copied, and no outside code was used (OpenTURNS, LGPL, was read to confirm formulas, not copied). The input distributions are parameter values from the OpenTURNS use-case page (© 2005–2023 Airbus-EDF-IMACS-ONERA-Phimeca; no licence stated on the page). Cite:

- OpenTURNS, "Use case: vertical deflection of a tube", OpenTURNS 1.22 documentation, https://openturns.github.io/openturns/1.22/usecases/use_case_deflection_tube.html (benchmark and distributions).
- M. Baudin, A. Dutfoy, B. Iooss and A.-L. Popelin, "OpenTURNS: An industrial software for uncertainty quantification in simulation", in *Handbook of Uncertainty Quantification*, pp. 2001–2038, Springer, Cham, 2017, doi:10.1007/978-3-319-12385-1_64 (checked on Crossref).
- J. M. Gere and B. J. Goodno, *Mechanics of Materials*, 7th ed., Cengage Learning, 2009; R. G. Budynas and J. K. Nisbett, *Shigley's Mechanical Engineering Design*, 9th ed., McGraw-Hill, 2011 (formulas, as cited by OpenTURNS; no DOI; not opened).
- G. R. Cowper, "The shear coefficient in Timoshenko's beam theory", *Journal of Applied Mechanics* 33(2), 335–340, 1966, doi:10.1115/1.3625046 (shear coefficient for the validity estimate; checked on Crossref).
