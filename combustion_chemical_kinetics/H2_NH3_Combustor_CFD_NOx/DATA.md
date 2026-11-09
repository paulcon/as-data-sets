# H2_NH3_Combustor_CFD_NOx: data

Kind: **stored data**. The values come from 175 RANS simulations that Piscopo et al. (2026) ran with Ansys
Fluent v22.1 and deposited as cell-centre fields at Zenodo,
[doi:10.5281/zenodo.19348021](https://doi.org/10.5281/zenodo.19348021) (v2.0, CC BY 4.0). They were not
regenerated (one case cost about 3360 core-hours) and are not altered: `extract_data.py` copies the deposited
values of the outlet cell row (all 15 fields, 37 cells, 175 cases) and the operating conditions unchanged to
`h2nh3_outlet_cells.npz`, and reduces those cell values to outlet quantities by the formula stated below
(`h2nh3_outlet.csv`). The CSV is therefore a derived reduction, not a format conversion; the `.npz` is the
lossless copy it can be traced to without the download. The 1.26 GB field file is not part of the repository.

## Files

| file | contents | role |
|---|---|---|
| `H2_NH3_Combustor_CFD_NOx.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 491; outputs committed |
| `h2nh3_outlet_cells.npz` | the deposited values the reduction uses, unchanged (float64): `P` (175 × 3, from `P.npy`), `cell` (the 37 outlet cell indices into `xy.npy`), `z`, `r` (37, m, from `xy.npy`), `features` (the 15 field names, in the order of `X.npy`), `fields` (15 × 37 × 175, the `X.npy` values of those cells; species as mole fractions, rho kg/m³, p gauge Pa, velocities m/s, T K) | lossless copy written by `extract_data.py` from the Zenodo files (784 kB) |
| `h2nh3_outlet.csv` | 175 × 15: case number, the 3 operating conditions (`phi1` [1], `xH2` [mol/mol], `p0` [Pa]), duplicate flag `dup_of`, and per case the outlet NOx, NO, NO2, NH3 (ppmv dry at 16 % O2), O2 (% dry), H2O (mol/mol), T (K), area-weighted NOx and T, outlet mass flow (kg/s) | reduction of the `.npz` cell values by `extract_data.py`; the file the notebook analyses |
| `extract_data.py` | checks the SHA-256 of `P.npy`, `xy.npy`, `X.npy`, memory-maps `X.npy`, reads the outlet rows only, writes the `.npz` and the CSV computed from it; `--check` compares both with the shipped files; `--cells` rebuilds the CSV from the shipped `.npz` alone | the copy and the reduction (numpy only) |
| `check_accuracy.py` | the sensitivity of the reduction to its choices and the checks of the fields quoted under Accuracy | check (needs the Zenodo files) |

The CSV is plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads it. Rows are in the order of the columns of `X.npy` (case
k is column k − 1). The operating conditions are written as the shortest decimal string that reads back as
the same float64 (Python `repr`, e.g. `1.3`, `101325.0`), so they are the deposited values; the outlet
quantities with 12 significant digits. The `.npz` is read by `np.load(path)` (no pickled objects).

## Model

**Citation.** A. Piscopo, L. Giuntini, C. Novelli, W. De Paepe, A. Parente, "A CFD dataset for supporting
machine-learning-based models of hydrogen–ammonia combustion", *Machine Learning: Engineering* 2(1):010601
(2026), doi:10.1088/3049-4761/ae61b5 (checked on Crossref; CC BY 4.0). The PDF read is the IOP Accepted
Manuscript; page numbers below are its printed page numbers, which may differ from the Version of Record.

**Combustor and CFD** (pp. 2–4, Fig. 1, Table 1). ULB modified stagnation-point reverse-flow combustor, C-type
design: a rich first stage at $\phi_1$, four holes in the sleeve wall that inject air until the mixture reaches
$\phi = 1$ at the fourth hole, then dilution air. 2-D axisymmetric RANS in Ansys Fluent v22.1 (pressure-based,
realizable $k$–$\epsilon$ with enhanced wall treatment, PaSR with $C_{mix} = 0.1$, Shrestha et al. 2018
NH$_3$/H$_2$ mechanism, discrete ordinates with WSGG radiation and a UDF for the WSGG coefficients, SIMPLE,
second-order schemes), about 63,000 cells; about 7 days on 20 CPUs per case (3360 core-hours; 588,000 for the
data set). Validated on the A-type geometry only (outlet NO 9 %, outlet T 0.57 % average relative error, p. 4;
the abstract, p. 1, rounds these to 10 % and "less than 1 %").

**Operating conditions** (p. 5, eqs. 1–5). Mass flows scale with $p_o/p_o^0$ ($p_o^0$ = 1 bar), fuel from
$P_{th}^0$ = 20 kW and the blend's LHV; first-stage air from $\phi_1$ (eq. 3, printed as
$\dot m_{air,1} = \phi_1 (F/A)_{st}\,\dot m_{fuel}$. Both sides are mass flows, since $(F/A)_{st}$ is a kg/kg
ratio, but the printed form inverts the definition of the equivalence ratio,
$\phi_1 = (\dot m_{fuel}/\dot m_{air,1})/(F/A)_{st}$: it gives more air for a richer stage and is smaller than
the defined air flow by the factor $[\phi_1 (F/A)_{st}]^2$. The definition gives
$\dot m_{air,1} = \dot m_{fuel}/[\phi_1 (F/A)_{st}]$); hole air to reach
$\phi = 1$ at the fourth hole; total air by an energy balance for $T_{out}$ = 1200 K; dilution air the rest
(eq. 4). Fuel at 338 K, air preheated to 900 K (p. 4).

| name | symbol | units | range | source |
|---|---|---|---|---|
| `phi1` | $\phi_1$ | 1 | [1, 1.6] | eq. (5), p. 5; abstract, p. 1 |
| `xH2` | $x_{H_2}$ | mol/mol of fuel | [0, 0.75] | eq. (5), p. 5; abstract, p. 1 |
| `p0` | $p_o$ | Pa | [101325, 303975] | eq. (5), p. 5 ("[1, 3]", bar in the abstract); stored in Pa |

**Pressure units.** `P.npy` stores the pressure in Pa from 101325 to 303975, i.e. 1 to 3 atm (1.013 to
3.040 bar); the grid levels the paper calls 1 and 2 bar (Table 2) are stored as 101325 and 202650 Pa. The
notebook uses the stored end points as the box. The deposit's README states "Pressure (Pa): 0 ÷ 303975",
"Equivalent ratio … 0 ÷ 1.6" and "H2 fuel content (%) : 0 ÷ 70" (the stored H2 fraction is a mole fraction,
maximum 0.7443); these lower bounds and the percent are not what the files hold.

**Quantities of interest.** The paper reports outlet NO$_x$ and N$_2$O in ppmv, dry basis, at 16 % O$_2$
(Table 2, p. 5) and says the outlet quantities "can be computed from a surface integral at the outlet from the
field files" (p. 6), without stating the integral or its weighting. The deposit's 15 fields (README.ipynb
order: N2, NH2, NO2, H2O, OH, O2, H2, NH3, NO mole fractions; rho kg/m³; p, gauge, Pa; vmag, vr, vz m/s; T K)
do not include N$_2$O, so only NO$_x$ = NO + NO$_2$ is computable. The paper (p. 6) and the deposit's README
call the species values "molar concentrations"; they are read here as mole fractions, because the README's
plot labels give them in mol/mol and the nine stored species sum to 0.99972–0.99997 in the outlet cells
(`check_accuracy.py`). **The reduction is this data set's
choice** (`extract_data.py`):

1. Outlet cells: the last row of cells below the outlet plane, i.e. every cell centre within 0.5 mm of the
   highest one (37 cells, $z_c$ = 0.64541–0.64542 m, from the axis to the outlet wall). Cell-centre values
   stand in for the face values.
2. Areas: radial face edges midway between neighbouring centres, the axis as inner edge, and the wall placed
   as far beyond the last centre as that centre lies beyond its inner edge ($r_{wall}$ = 32.58 mm);
   $A_k = \pi (e_{k+1}^2 - e_k^2)$.
3. Weights: mass flux $w_k = \rho_k v_{z,k} A_k$ ($v_z > 0$ in all outlet cells of all cases).
4. $\bar X = \sum w_k X_k / \sum w_k$ for NO, NO$_2$, NH$_3$, O$_2$, H$_2$O and T; dry basis
   $\bar X/(1 - \bar X_{H_2O})$; correction to 16 % O$_2$ by $(21 - 16)/(21 - \%O_{2,dry})$. The 21 % is the O$_2$
   mole fraction of the simulated inlet air (0.21/0.79 in the fields, `check_accuracy.py`).

$T_{out}$ is the mass-flux-weighted outlet temperature. Area-weighted NO$_x$ and $T_{out}$ are stored as well.

**Gradients.** None: the CFD code computes none, and a run costs 3360 core-hours. The notebook uses the
global quadratic estimator (QPHD), $M$ = 173 distinct cases against the 10 coefficients of a quadratic in
$m$ = 3 inputs, in normalized coordinates on the box of the table above.

**Sampling design** (pp. 5–6, Fig. 3). Three stages, not one sample: 70 Latin hypercube points "under uniform
input distributions"; "previously computed C-type simulations and additional corner points"; adaptive
refinement inserting points "midway between eligible pairs" with large scaled NO$_x$ differences (eq. 6). No
seed is given. The deposit carries no block label; by case number and checks (notebook): cases 1–70 are a
Latin hypercube on the box (each of 70 equal strata per input holds one case); 71–98 the $7\times2\times2$
grid of Table 2 ($\phi_1$ = 1.00, 1.15, 1.20, 1.25, 1.30, 1.45, 1.60; $x_{H_2}$ = 0, 0.3; 1 and 2 atm); 99–104
six corners with $x_{H_2}$ = 0.7 (not 0.75); 105–175 the refinement. Two refinement cases repeat earlier
inputs: case 130 equals case 126 in all 903,465 field values; case 133 has the inputs of case 127 and fields
that differ by at most 2.8e-05 of each field's maximum (a second converged run; outlet NO$_x$ differs by
1e-04 ppmvd; `check_accuracy.py`). The notebook drops cases 130 and 133 by the stored flag `dup_of`.

**Published results.** No active subspace or sensitivity index. Table 2 (p. 5): outlet NO$_x$ and N$_2$O of
the B- and C-type designs on the grid; statements that higher pressure generally reduced emissions (p. 4) and
that the data identified low-NO$_x$ operating regions (p. 6). Surrogates built on the data elsewhere (PCE,
ref. 30; POD–GPR, ref. 31) are not reproduced here.

## Regenerating

The CFD runs cannot be regenerated here (Ansys Fluent is commercial, and one case took about 3360
core-hours). Both files are regenerated from the deposit, and the CSV also from the shipped `.npz` alone:

```
python extract_data.py <folder with P.npy, xy.npy, X.npy> --out DIR
python extract_data.py <same folder> --out DIR --check   # DIR outside this directory
python extract_data.py --cells h2nh3_outlet_cells.npz --out DIR --check   # no download needed
```

`--check` compares the copied cell values at a relative tolerance of 1e-15, the CSV's five input columns as
text and its outlet columns at a relative tolerance of 1e-9. From the deposit it printed "agree" for both
files, with largest relative differences printed as 0.0e+00; from the `.npz` alone it printed "agree" for the
CSV, likewise. Run time about 6 s from the deposit (2026-11-08, two shared CPUs, `X.npy` already in the page cache;
longer from a cold disk), most of it the SHA-256 of `X.npy`, memory about 100 MB; under 1 s from the `.npz`.
`--out` is required, so nothing is written by default, and `--check` refuses a `DIR` inside this directory.

**Source.** Zenodo record 19348021, v2.0, CC BY 4.0, DOI
[10.5281/zenodo.19348021](https://doi.org/10.5281/zenodo.19348021) (the DOI of ref. 29 of the paper). The
record's description names the files `P_175.npy` and `X_175.npy`; the files are `P.npy` and `X.npy`.

| file | bytes | MD5 (as listed by the record) | SHA-256 |
|---|---|---|---|
| `X.npy` | 1,264,851,128 | `e71f2904f367b7cd0ef7b100d021f298` | `1ec080ded6c9fe879b1cef334f2d438679c046dbf51b9b1f2ec4baec3e9934ef` |
| `P.npy` | 4,328 | `a4eb4abf347badb7797d88c2814ea5e7` | `e7ba58b83cb6937fed3604ebbfc11e3133581684803cfdaa7ae0975731861c9b` |
| `xy.npy` | 963,824 | `51ca0d4d61b6283acf83a4aa3436929d` | `b5857e4bb34abbc8cadadec669733d39cd8ad330998a5095ba80a8bd37780f77` |
| `README.ipynb` | 1,055,187 | `37e301d7a77045d3b2222892fc6d8ab3` | `a5e0139cccdf66ab7c567aaaf9ba0250f6b391194b99b79bf84c273428668398` |

`X.npy` is float64, shape (903465, 175) = (60231 cells × 15 fields, 175 cases), C order; `xy.npy` (60231, 2)
holds axial $z$ and radius $r$ in m; `P.npy` (175, 3) holds $\phi_1$, $x_{H_2}$, $p_o$ [Pa].

The notebook printed a kernel run time of 7 s (20 s for `jupyter nbconvert --execute` including start-up,
2026-11-08, two shared CPUs).

## Accuracy

`check_accuracy.py <folder>` prints the numbers marked with its name (it skips the SHA-256); without the
deposit it stops with a message naming the Zenodo record to download. Its duplicate check needs two full
columns of `X.npy`, which is stored row by row, so it reads the whole 1.26 GB file once, in blocks; it took
3 s with the file in the page cache (longer from a cold disk), peak memory about 320 MB.

- Reduction choices (`check_accuracy.py`; largest relative change over the 175 cases): area instead of
  mass-flux weighting, NO$_x$ 1.8e-03 and $T_{out}$ 7.6e-03 (the notebook prints these too); molar-flux
  weighting, NO$_x$ 3.8e-05, $T_{out}$ 1.9e-04; the cell row 1 mm upstream, NO$_x$ 1.0e-05, $T_{out}$ 3.0e-05;
  face edges with each centre midway between its edges (wall at 32.575 against 32.581 mm), NO$_x$ 7.1e-06,
  mass flow 1.4e-03; a 20.95 % instead of 21 % O$_2$ reference multiplies NO$_x$ by 1.0095–1.0134. The outlet
  is radially well mixed, so the choices matter little; the O$_2$ reference matters most, and 21 % is the
  simulated air.
- Field checks (`check_accuracy.py`): the nine stored mole fractions sum to 0.99972–0.99997 in the outlet cells;
  the inlet air (cells below $z$ = 0 at 900 ± 1 K) has O$_2$ 0.21000–0.21001 and N$_2$ 0.78999–0.79000; the outlet
  mass flow per atm of $p_o$ is 0.05054–0.05116 kg/s, consistent with the pressure scaling of eq. (1).
- The reduction from the shipped cells (notebook): recomputing NO$_x$ and $T_{out}$ from
  `h2nh3_outlet_cells.npz` reproduces the CSV to a largest relative difference of 4.2e-12 and 4.1e-12 (the
  CSV keeps 12 significant digits).
- Against the paper (notebook): at the 14 grid cases with 0 % H$_2$ the reduced NO$_x$ is within 2.4 % of
  Table 2 (10 within 1 %); at 30 % H$_2$, 10 within 5.0 % and four differ by 12.5–39.7 % (cases 80, 82, 92,
  93). At $\phi_1$ = 1.00 Table 2's B and C values coincide in all four blocks, as expected (a $\phi_1 = 1$
  stage is already at the $\phi = 1$ the C-type holes aim for, so they add no air, pp. 2–5); three of these
  agree with the deposit (1034.3, 872.9, 1435.7 against 1033, 873, 1428). The fourth (case 92, 30 % H$_2$,
  2 bar) does not (959.6 against 828), and 828 is printed again in that block as the B-type N$_2$O, where every
  other N$_2$O entry at $\phi_1$ = 1.00, including the C-type one B should match, is 0. That points to a
  misplaced entry in that block of Table 2; for cases 80, 82 and 93 the tabulated and deposited runs may differ.
- No gradients, so no gradient check. Fit quality (notebook): the quadratic behind QPHD has LOO $R^2$ 0.685
  for NO$_x$ and 0.948 for $T_{out}$; a cubic 0.896 and 0.965. The cubic and the 70 Latin hypercube cases
  alone give $w_1$ within 6.6° and 1.8° (NO$_x$), 2.5° and 4.2° ($T_{out}$) of the QPHD direction; QPHD on
  $\log$ NO$_x$ gives $w_1$ within 4.2° of the NO$_x$ direction (its quadratic LOO $R^2$ 0.657).
- Solver convergence of the runs: assessed by the authors through residuals, heat balance and outlet NO$_x$
  stability (p. 4); external run, not reproducible from the repository.

## How these files relate to the paper

The `.npz` and the CSV hold exactly the 175 deposited cases, the runs behind the paper's Fig. 3 design and its Table 2
C-type columns (cases 71–98). The outlet values are this data set's reduction of the deposited fields, not
values the authors published; they reproduce Table 2 at 0 % H$_2$ within 2.4 % and disagree at four 30 % H$_2$
grid points (above). The paper analyses no active subspace; the notebook estimates one for outlet NO$_x$
($n$ = 2) and outlet temperature ($n$ = 1) on the uniform box of operating conditions and compares the
directions with the paper's qualitative statements.

## License and credit

Data: © Piscopo, Giuntini, Novelli, De Paepe, Parente, Zenodo 10.5281/zenodo.19348021, CC BY 4.0. The `.npz`
is an unchanged extract of those data and the CSV a reduction of it; both are CC BY 4.0 with the same credit.
Article: CC BY 4.0 (Crossref licence of the Version of Record and the Accepted Manuscript cover page; see
Unknowns for the accepted manuscript's acknowledgments). Cite the paper and the Zenodo record when using
these files. `extract_data.py`, `check_accuracy.py` and the notebook: MIT (this repository).

## Unknowns

- The paper does not state how its outlet quantities were integrated (weighting, plane, O$_2$ reference);
  the reduction here is a reconstruction from cell-centre values without the mesh's face geometry. Its
  agreement with Table 2 at 0 % H$_2$ suggests it is close.
- Why four 30 % H$_2$ grid cases disagree with Table 2 by 12.5–39.7 %. For case 92 the 828 printed for both
  designs, and again as the B-type N$_2$O in the same block, suggests a misplaced entry; for cases 80, 82 and
  93 the tabulated runs may differ from the deposited ones.
- The pressure unit: the paper says bar, the files hold 1–3 atm.
- Which earlier case each refinement point was inserted between, and the thresholds $\lambda_r$, $\lambda_g$
  of eq. (6); the block assignment by case number is inferred (and checked) here.
- Why case 133 repeats the inputs of case 127 (and case 130 those of case 126).
- The corner points use $x_{H_2}$ = 0.7 while the range is stated as [0, 0.75].
- The acknowledgments (p. 9) state that a CC BY-NC-ND 4.0 licence applies to the accepted manuscript under
  the grant's conditions, while its cover page and Crossref give CC BY 4.0 for the article; the data are
  CC BY 4.0 either way.
