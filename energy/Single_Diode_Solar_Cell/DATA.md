# Single_Diode_Solar_Cell: data

Kind: **stored data**. Mark Campanelli's MATLAB implementation of the single-diode model produced `SingleDiodePV-Pmax.txt`. That code is not in this repository, so the file is the only record of those runs. `single_diode.py` is a Python re-implementation of the model that is used to *check* the file. It does not replace the file, and any disagreement is reported, not corrected.

## Files

| file | contents | role |
|---|---|---|
| `Single_Diode_Solar_Cell.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed |
| `SingleDiodePV-Pmax.txt` | 10,000 samples of the inputs, P_max and its gradient | stored data |
| `single_diode.py` | `pmax`, `current`, `thermal_voltage`, `to_physical`, `to_normalized`, bounds `XL`, `XU` | Python model, used only to check the stored data |

## `SingleDiodePV-Pmax.txt`

CSV with one header row, 10,000 rows × 12 columns, values written as `%.16e`. Size 2,780,045 bytes, md5 `71b17000d76d7ca0ccb73be3aa1d3bfd`. There are no missing values. Load it without pandas with `np.genfromtxt('SingleDiodePV-Pmax.txt', delimiter=',', names=True)`.

| column | quantity | units | coordinates |
|---|---|---|---|
| `Run` | row label 1, …, 10000 | | not used |
| `ISC` | short-circuit current I_SC | A | physical |
| `IS` | diode reverse saturation current I_S | A | physical (not log) |
| `n` | ideality factor | none | physical |
| `RS` | series resistance R_S | Ω | physical |
| `RP` | parallel (shunt) resistance R_P | Ω | physical |
| `Pmax` | maximum power P_max, 0.0143 to 0.2032 | W | |
| `dISC`, `dIS`, `dn`, `dRS`, `dRP` | ∂P_max/∂x_i | W | **normalized** x, with I_S through log I_S |

**Gradient coordinates.** The gradients are taken with respect to x ∈ [−1, 1]⁵, where x_i = 2(q_i − L_i)/(U_i − L_i) − 1, q = (I_SC, log I_S, n, R_S, R_P), and L, U are the bounds below (the paper's eq. (13)). To get gradients with respect to q, divide by (U − L)/2. active_subspaces v1 expects gradients with respect to the inputs of its space and applies the factor (U − L)/2 itself. So with `ac.Uniform(lower=L, upper=U)` on q, pass `G / ((U - L) / 2)`. Passing the stored `G` unchanged scales it twice, and the first eigenvector then comes out dominated by R_P (docs/port-2026/SingleDiodePV.md §10).

**Inputs.** The paper's Table 1 gives the ranges (typical 2 cm² crystalline silicon cells). I_S is log-uniform, and the other inputs are uniform:

| input | L | U | units |
|---|---|---|---|
| I_SC | 0.05989 | 0.23958 | A |
| log I_S | ln 2.2e-11 = −24.540 | ln 2.2e-7 = −15.330 | log A |
| n | 1 | 2 | none |
| R_S | 0.16625 | 0.665 | Ω |
| R_P | 93.75 | 375 | Ω |

The stored inputs fill the box: in normalized coordinates they span [−0.99998, 0.99995] (printed by the notebook). The paper draws its samples uniformly at random in x (§4.2).

**Model.** The single-diode equation (paper eqs. (9)–(10)) with N_S = 1 and the thermal voltage V_th = k_B T/q at T = 25 °C. P_max = max IV (eq. (11)). The paper gives no value for V_th. V_th = k_B · 298.15 K / q = 0.025693 V reproduces the file. **Gradients**: first-order finite differences with a step of 10⁻⁶ in the normalized coordinates (paper §4.2).

## Check against `single_diode.py`

`single_diode.py` solves eq. (9) for I explicitly with the Lambert W function, and maximizes VI over [0, V_oc] with a bounded scalar search. The notebook runs the check (seed 2016):

- P_max at all 10,000 rows: relative difference from −1.6e-6 to −9.6e-7.
- Gradient by central differences (step 10⁻⁶ in x) at 20 random rows: `dIS`, `dn`, `dRS` and `dRP` agree within a relative 6.0e-6. `dISC` is off by a nearly constant relative **4.7e-5** at every row. The cause is unknown (author, 2026-10-30; `docs/port-2026/UNKNOWNS.md`). It is consistent with the original code using slightly different I_SC bounds than Table 1. The bias is far smaller than the bootstrap variability, and the notebook uses the stored gradients as they are.

These checks confirm the column meanings and the gradient coordinates above.

## Provenance and unknowns

- **Generating code:** Campanelli's MATLAB single-diode code (paper §3.1: "Requests for the MATLAB code may be sent to the third author Campanelli"). The paper's MATLAB scripts for its figures were at `bitbucket.org/paulcon/active-subspaces-in-a-single-diode-model` (§4). Code version: unknown.
- **Who ran it, sampling seed:** unknown. Paul Constantine committed the file in this repository's initial commit (7c26e3a, 2014-10-23).
- **Relation to the paper:** the paper's figures use M = 1000 samples, while the file has 10,000. Whether the paper's samples are a subset of these is unknown. The notebook reproduces the paper's settings on a random subset of 1000 rows.
- **Unknown:** the cause of the `dISC` bias (above).

## License and credit

Code (notebook, `single_diode.py`): MIT (`LICENSE`). Data (`SingleDiodePV-Pmax.txt`): CC BY 4.0 (`LICENSE-DATA`). Credit Mark Campanelli (National Renewable Energy Laboratory) for the model code. Cite P. G. Constantine, B. Zaharatos and M. Campanelli, "Discovering an active subspace in a single-diode solar cell model", Stat. Anal. Data Min. 8(5–6):264–273, 2015, doi:10.1002/sam.11281 (arXiv:1406.7607). Ryan Howard updated the notebook in 2016.
