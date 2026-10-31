# Golinski_Speed_Reducer: data

Kind: **Python model**. The model is `golinski_model.py`, a closed-form numpy implementation of Golinski's speed reducer (gearbox) volume and two of Ray's constraint functions, with exact gradients; `generate_data.py` writes the three CSVs from it with a saved seed.

## Files

| file | contents | role |
|---|---|---|
| `Golinski_Speed_Reducer.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 1970`; outputs committed; about 3 s |
| `golinski_model.py` | `volume`, `volume_grad`, `volume_geometric`, `volume_rounded`, `g3`, `g3_grad`, `g5`, `g5_grad`, `qoi`, `grad`, `sample`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `NOMINAL`, `Z = 17`, `RATIO = 3` | the model |
| `generate_data.py` | writes the three CSVs; `--out DIR --check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 1000 × 6 physical inputs (cm), columns in the order of `NAMES`: b, m_t, l1, l2, d1, d2 | model input |
| `outputs.csv` | 1000 × 3: volume V (cm³), g3, g5 (dimensionless, feasible if ≤ 0) | model output |
| `gradients.csv` | 1000 × 18: column 6 i + j is ∂(output i)/∂(input j), with respect to the **physical** inputs | model output |

The CSVs are comma-separated numbers (17 significant digits) after two `#` lines (a description, then the column names with units), so `np.loadtxt(path, delimiter=',')` reads them.

## Model

**Citation.** J. Golinski, "Optimal synthesis problems solved by means of nonlinear programming and random methods", *Journal of Mechanisms* 5(3):287–309, 1970, doi:[10.1016/0022-2569(70)90064-9](https://doi.org/10.1016/0022-2569(70)90064-9). Section 3.1, "Speed reducer", pp. 297–300, Figs. 10–12, Table 1. The PDF read for this build is the publisher's scan (file tagged `_oa`; the printed objective is partly garbled in the text layer and was read from a 300 dpi rendering of p. 299). The constraint functions and the input box come from T. Ray, "Golinski's speed reducer problem revisited", *AIAA Journal* 41(3):556–558, 2003, doi:[10.2514/2.1984](https://doi.org/10.2514/2.1984), as printed in psdr (J. M. Hokanson, `psdr/demos/golinski.py`, github.com/jeffrey-hokanson/PSDR). Ray's paper itself was not read.

**Fixed assumptions** (p. 297): transmitted power N = 100 (unit printed "km", presumably the Polish KM, metric horsepower), pinion speed n = 1500 min⁻¹, transmission ratio i = 3, allowable stresses k_g = 900, p_d = 5800, k_g1 = 1100, k_g2 = 850 kG cm⁻², tooth form factor q = 2.54. These enter Golinski's constraints, not his objective.

**Volume** (p. 299, Fig. 11). For gear k = 1, 2 with z_1 = z, z_2 = i z teeth: d_st = m_t z_k − 2.4 m_t, d_w = d_st − 4 m_t, d_p = d_k + 2 e_k with e_k = 0.7 d_k (so d_p = 2.4 d_k), and

V = Σ_k [ (d_st² − d_w²) π b/4 + (d_w² − d_p²) π b/12 + (d_p² − d_k²)(π/4) 2 d_k ] + (π/4) Σ_k d_k² l_k

(rim of width b, web of width b/3, hub of length 2 d_k, shafts). Two readings: the scan prints the hub factor as "2b" and the shaft term as π Σ (d_k/4) l_k; we use the hub length 2 d_k shown in Fig. 11, with the factor π/4 of a cylinder, and d_k² for the shaft. Expanded with i = 3,

V = (π/4) b m_t² (10/3 z² + 224/15 z − 646.4/15) − 0.48 π b (d_1² + d_2²) + 2.38 π (d_1³ + d_2³) + (π/4)(l_1 d_1² + l_2 d_2²),

and the rounded coefficients (0.7854, 3.3333, 14.9334, 43.0934, π·5.76/12 = 1.50796, π·4.76/2 = 7.47699) are those of the standard formulation (Ray 2003; NASA Langley MDO test suite; psdr uses 1.5079 and 7.477). This agreement in every coefficient is the main evidence for the two readings. The formula does not check that the geometry is realizable: at 57.8 % of the samples the pinion's hub diameter 2.4 d_1 exceeds its inner rim diameter 10.6 m_t, so its web term is negative; this is inherited by the standard benchmark and left as is. We use the exact constants; on the 1000 samples the exact and the psdr-rounded volumes differ by at most a relative 1.5e-6, and their first eigenvectors agree to 1.0000000 (notebook).

**Constraints** (Ray 2003 as printed in psdr; each feasible when ≤ 0). Two of Ray's eleven, both for shaft 1:

- g3 = 1.93 l_1³ / (m_t z d_1⁴) − 1, transverse deflection of shaft 1 (Golinski's condition 7);
- g5 = sqrt((745 l_1/(m_t z))² + 16.9×10⁶) / (110 d_1³) − 1, substitute stress in shaft 1 (condition 9).

Both are smooth on the box (no max, no kinks) and depend on three inputs. The others were left out: g1, g2, g7, g8, g9 are monomials or ratios in (b, m_t), g24 and g25 are the linear-fractional design conditions, and g4, g6 are the shaft-2 analogues of g3 and g5 (same structure). Golinski's own constraint formulas need constants he does not print (B, E, the moments), so Ray's constants were not checked against the 1970 paper.

**Inputs** (m = 6). The paper gives no box, only constraint-type limits (p. 298–299). The box is psdr's `build_golinski_design_domain` (after Ray 2003):

| name | symbol | units | lower | upper | source |
|---|---|---|---|---|---|
| `b` | face width b (x1) | cm | 2.6 | 3.6 | psdr / Ray 2003; variable defined p. 298 |
| `m_t` | teeth module m (x2) | cm | 0.7 | 0.8 | psdr / Ray 2003; p. 298 |
| `l1` | shaft 1 length between bearings (x4) | cm | 7.3 | 8.3 | psdr / Ray 2003; p. 298 |
| `l2` | shaft 2 length between bearings (x5) | cm | 7.3 | 8.3 | psdr / Ray 2003 (other sources: [7.8, 8.4]); p. 298 |
| `d1` | shaft 1 diameter (x6) | cm | 2.9 | 3.9 | psdr / Ray 2003; p. 298 |
| `d2` | shaft 2 diameter (x7) | cm | 5.0 | 5.5 | psdr / Ray 2003; p. 298 |
| (held) | number of pinion teeth z (x3) | – | 17 | 17 | integer; z ≥ 17 is condition 3, p. 298; fixed at 17 in psdr |
| (held) | transmission ratio i | – | 3 | 3 | p. 297 |

The spreadsheet note (from arXiv 1203.6586, citing Golinski 1973) gives l2 ∈ [7.8, 8.4] and z ∈ [17, 28]. The notebook reruns the analysis with l2 ∈ [7.8, 8.4]: the first eigenvectors agree to 0.99993 (volume) and 1.00000 (g3, g5). z is held at 17, as in psdr and in Ray's six-variable form, because it is an integer.

**Quantities of interest.** V (cm³; Golinski calls it the weight, as a volume), g3, g5 (dimensionless).

**Gradients.** Exact, by differentiating the closed forms by hand (`volume_grad`, `g3_grad`, `g5_grad`), with respect to the physical inputs. The notebook passes physical X and physical gradients with `ac.Uniform(LOWER, UPPER)`, which maps them to [−1, 1]⁶.

**Sampling design.** N = 1000 independent uniform samples on the box (`numpy.random.default_rng(1970)`, `LOWER + (UPPER − LOWER) * rng.random((1000, 6))`). This is ≥ 10 m and ≥ 300, as needed for the gradient estimator. The samples are not restricted to the feasible set: g3 ≤ 0 at 94.1 % of them and g5 ≤ 0 at 53.0 %.

**Published results.** Golinski reports no parameter study, sensitivity or dimension reduction. Table 1 (p. 300) gives two optimized designs: crude Monte Carlo, f = 2236.35 cm³ at b = 4.4, m = 0.6, l1 = 7.3, l2 = 8.1, d1 = 3.4, d2 = 5.0, z = 17 (734,631 samples, 36,937 feasible, 5 h); stray process, f = 2247.79 cm³ at b = 3.6, m = 0.7, l1 = 6.6, l2 = 8.2, d1 = 2.8, d2 = 5.2, z = 18 (166,796 samples, 45,998 feasible, 2 h). Hokanson and Constantine (SIAM J. Sci. Comput., doi:10.1137/20M1364850, per the screening spreadsheet) analyse the six-input standard volume with psdr; their numbers were not available for this build.

## Regenerating

```bash
python generate_data.py                                   # writes the three CSVs here
python generate_data.py --out /tmp/gs --check             # writes elsewhere and compares
```

The 2026-10-30 run (numpy 2.5.4, Python 3.13.18) took 0.03 s for the 1000 samples (0.17 s including start-up). A rerun reproduces the files exactly: `--check` prints 0.0e+00 for inputs, outputs and gradients.

## Accuracy

- **Gradients.** Against central differences with step 1e-6 times the input range, on 50 samples (in the notebook), the largest relative errors of the gradient vectors are 1.6e-9 (V), 1.6e-9 (g3) and 1.5e-9 (g5), the size of the rounding error in the differences.
- **Volume.** The expanded closed form and the term-by-term geometric sum of Fig. 11 (`volume_geometric`) agree to a relative 6.1e-16. g3 and g5 agree with the expressions printed in psdr to 4.4e-16 (scratch check; psdr not installed or imported).

## How these files relate to the paper

- **Same objective, different numbers.** The volume is Golinski's formula with the two readings above. It does **not** reproduce his Table 1: at his two designs it gives 2671.92 cm³ (crude Monte Carlo, +19.5 %) and 3049.87 cm³ (stray process, +35.7 %), against 2236.35 and 2247.79. The two printed values differ by 0.5 % and ours by 14 %, so no constant factor explains it. Other readings tried (scratch, cm³ for the two designs): hub term as printed, (d_p² − d_k²) 2b, with d_k² l_k shafts: 2974.8 and 3029.8; the same with d_k l_k shafts: 2800.8 and 2863.0; hub (π/4)(d_p² − d_k²) b: 2044.8 and 2303.9; hub (π/4)(d_p² − d_k²)(2d_k − b/3): 2471.5 and 2893.4. Solving for free hub and shaft coefficients from the two Table 1 values gives a negative hub coefficient. So Table 1 is inconsistent with the printed formula under every reading we tried; we keep the reading whose expansion is the standard formulation.
- **Not the paper's design space.** Both Table 1 designs lie partly outside the standard box (b = 4.4, m = 0.6; l1 = 6.6, d1 = 2.8), and the stray design has z = 18. Under Ray's constants the stray design violates g5 (+0.710), which suggests that Ray's constraint constants differ from Golinski's.
- **New computation.** The paper has no parameter study; the uniform sampling, the box, the held z and the active subspace analysis are ours (box and z from psdr).

## License and credit

Code: MIT. Data (the CSVs): CC BY 4.0. Model: J. Golinski (1970), in the six-variable form of T. Ray (2003); input box and the fixed z = 17 follow psdr (J. M. Hokanson). No code was copied from psdr (its licence was not checked); `golinski_model.py` was written from the equations. The paper gives no acknowledgments or funding. Cite Golinski, J. Mechanisms 5(3):287–309, 1970, doi:10.1016/0022-2569(70)90064-9, and Ray, AIAA J. 41(3):556–558, 2003, doi:10.2514/2.1984.
