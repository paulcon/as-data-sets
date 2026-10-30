# Runoff_Permeability_Field: data

Kind: **stored data**. ParFlow produced the runoff files, and they are the only record of these runs. The notebook reads them as shipped and changes no value. Rows with zero Hortonian runoff are excluded in the notebook by a stated rule (f > 0), not deleted from the file.

## Files

| file | contents | role |
|---|---|---|
| `Runoff_Permeability_Field.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom; outputs committed; no random numbers |
| `inputs-ShortCorr.GW02.txt` | 10,000 × 100 KL coefficients | stored |
| `RunOff02_Saturated_10k.txt` | 10,000 Dunnian runoff volumes | stored |
| `RunOff02_Unsaturated_10k.txt` | 10,000 Hortonian runoff volumes | stored |

Row i of each runoff file is the output of the run whose inputs are row i of `inputs-ShortCorr.GW02.txt`. The two scenarios use the same 10,000 inputs.

| file | bytes | md5 |
|---|---|---|
| `inputs-ShortCorr.GW02.txt` | 25500463 | 97713bb987811b4cd81c42b1586c74f5 |
| `RunOff02_Saturated_10k.txt` | 240000 | 321214425a749f01e43fc744b9171b44 |
| `RunOff02_Unsaturated_10k.txt` | 239999 | 684567841f06f230a944b28d69afe49b |

## Inputs (`inputs-ShortCorr.GW02.txt`)

Space-separated text, no header, 10,000 rows × 100 columns, `%.18e`. Column i is the coefficient x_i of the i-th spatial mode in the truncated Karhunen–Loève (KL) series of the log permeability, log κ = μ + B x (eq. (12) of the paper). The coefficients are dimensionless.

- They are independent standard Gaussian draws, used unnormalized. The notebook prints the checks: column means within ±0.028, standard deviations 0.987–1.016, largest off-diagonal correlation 0.041, pooled Kolmogorov–Smirnov test against N(0, 1) p = 0.45.
- Low column indices are large-scale modes and high indices small-scale ones. In the paper, B holds the first 100 principal components of 2,000 turning-bands log permeability fields (§2.4, Table 2: geometric mean permeability 0.04465 m h⁻¹, correlation lengths λx = λy = 50 m and λz = 5 m).
- "ShortCorr" refers to the short correlation length of the permeability correlation model (author, 2026-10-30). What "GW02" means is **unknown**.
- The basis B, the mean μ and the 30 × 30 × 16 grid are **not available** (author, 2026-10-30). So the weights cannot be mapped to the spatial sensitivity fields B ŵ of the paper's Figs. 8 and 9, and the notebook does not plot them.

The file loads in well under a second with `np.loadtxt`. At 25.5 MB it is under GitHub's limits, and it stays as text: a `.npy` copy would add a blob without removing the old one from history.

## Outputs (`RunOff02_*_10k.txt`)

One value per line with a trailing space, 10,000 lines. `RunOff02_Unsaturated_10k.txt` has no final newline (harmless for `np.loadtxt`).

| file | scenario | quantity | range | notes |
|---|---|---|---|---|
| `RunOff02_Saturated_10k.txt` | Dunnian (initially saturated): near-surface water table, no precipitation | cumulative runoff volume at the outlet over the 48-h simulation, m³ | 45,219 – 46,168 | mean 45,483 |
| `RunOff02_Unsaturated_10k.txt` | Hortonian (initially unsaturated): deep water table, 3-h, 5 cm/h storm in hours 1–3 | same | 0 – 1,345 | mean 66.1; **3,137 runs exactly 0**; smallest non-zero 8.8e-5 |

The quantity and units are those of the paper (§2.6, p. 35; the axis of Figs. 6b and 7b reads "Cumulative Runoff Volume [cubic meters]"). The paper drops the zero Hortonian runs from Fig. 7 (p. 37), and so does the notebook, for every Hortonian fit: the least-squares direction, its summary plot and the quadratic-model extension, which is not in the paper (author, 2026-10-30). That extension fits 5,151 coefficients from 6,863 runs (1.33 runs per coefficient), so its Hessian is noisy. The Dunnian fits use all 10,000 runs.

## Provenance

- Generating code: ParFlow, on the idealized tilted-v domain of the paper (300 m × 300 m × 16 m, 30 × 30 × 16 cells; §2.6, Table 1). The ParFlow version, the run date and who ran the simulations are **not recorded** (author, 2026-10-30). The script that generated the KL coefficients and permeability fields is not in the repository.
- The data appear to be the runs behind the paper's figures. The weights computed from them match Figs. 6a and 7a, and the active-variable ranges match Figs. 6b and 7b, once the inputs are rescaled column by column to [−1, 1] by their sample minimum and maximum (Hortonian w₁ = −0.646 and w₅ = −0.501 against the figure's ≈ −0.65 and −0.50). That rescaling is inferred from these matches; the paper states a standard Gaussian density, and the notebook uses the Gaussian scale (author, 2026-10-30). Both scales give nearly the same direction (cosine 0.9992 Hortonian, 0.9999 Dunnian). The notebook prints all of these numbers.

## Regenerating

The files cannot be regenerated from this repository. The notebook reads them and runs in one to two minutes in one Python process; the two quadratic-model fits take most of that time. It uses no random numbers, and two runs give identical printed output. It reproduces the 2016 notebook's least-squares eigenvalues ‖â‖² (11372.3058 Dunnian, 6568.8455 Hortonian, all runs) and its Dunnian in-sample R² table. Its Hortonian R² table is computed on the non-zero runs, so it is not comparable with the 2016 one, which used all runs.

## License and credit

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). The original license of the runs is not recorded (author, 2026-10-30). Credit the ParFlow study of Gilbert, Jefferson, Constantine and Maxwell (Colorado School of Mines). The 2016 notebook was written by Ryan Howard. Cite J. M. Gilbert, J. L. Jefferson, P. G. Constantine and R. M. Maxwell, "Global spatial sensitivity of runoff to subsurface permeability using the active subspace method", Adv. Water Resour. 92:30–42, 2016, doi:10.1016/j.advwatres.2016.03.020.
