# Luo_Rudy_Cardiac_Action_Potential: data

Kind: **stored data**. The values come from MATLAB runs of the Luo–Rudy 1991 (LR1991) ventricular myocyte model by Chang, Strong and Clayton (2015). They published the runs as the supporting files S1 Data and S2 Data of their PLOS ONE article. The two files are in `original/`, unaltered. `convert_data.py` converts them losslessly to the repository's CSV format. Nothing here was rerun or refitted.

## Files

| file | contents | role |
|---|---|---|
| `Luo_Rudy_Cardiac_Action_Potential.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2015; outputs committed |
| `original/pone.0130252.s002.csv` | S1 Data: 200 design runs, as published (header row, CRLF line ends, 4 decimals) | stored data, unaltered |
| `original/pone.0130252.s003.csv` | S2 Data: 20 test runs, as published | stored data, unaltered |
| `convert_data.py` | writes `design.csv` and `test.csv` from `original/` and checks the conversion; `NAMES`, `NOMINAL`, `LOWER`, `UPPER`, `OUTPUTS`, `UNITS` | lossless conversion |
| `design.csv` | 200 × 14: the six conductances in mS/cm², then the eight outputs | the design data (S1) |
| `test.csv` | 20 × 14, same columns | the test data (S2) |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Columns:

| column | PLOS header | quantity | units |
|---|---|---|---|
| `GNa`, `Gsi`, `GK`, `GK1`, `GKp`, `Gb` | `G Na` … `G b` | maximum conductances (physical, see below) | mS/cm² |
| `dVdt_max` | `dVdt max (mV/ms)` | maximum upstroke velocity dV_m/dt | mV/ms |
| `Vm_max` | `Vm max (mV)` | peak V_m | mV |
| `V_dome` | `Dome V (mV)` | dome (plateau) V_m | mV |
| `APD90` | `APD90 (ms)` | action potential duration to 90 % repolarization | ms |
| `V_rest` | `Rest v (mV)` | resting V_m | mV |
| `APD50` | `APD50 (ms)` | APD to 50 % repolarization | ms |
| `APDr_slope` | `APDr slope` | maximum slope of the APD restitution curve | 1 |
| `DI_min` | `Min DI (ms)` | minimum diastolic interval of the S2 restitution sequence | ms |

SHA-256 of the originals: s002 `af735fd071cf5b1ca5c40ce5feeabe02909870e2da565977ee2db4dd7ac07f8d`, s003 `98781f56a34af2c360e1524bf1a68706d01bb932a25bad109c9bdfd112bd350b`. They were downloaded through a browser on 2026-10-30 from the article's supplementary-file links (`…article/file?type=supplementary&id=10.1371/journal.pone.0130252.s002` and `.s003`). In PLOS numbering, s001 is S1 Text (a PDF) and is not used here.

## Model

From the paper card. All page and table references are to Chang, Strong and Clayton, PLoS ONE 10(6):e0130252 (2015).

- **Simulator**: LR1991 (Luo and Rudy, Circ. Res. 68:1501–1526, 1991): eight ODEs, for V_m, six gating variables and the intracellular calcium concentration. The paper ran MATLAB code generated automatically from the CellML repository (www.cellml.org) and solved it with `ode15s` (p. 4). It does not state solver tolerances, the MATLAB version or the CellML model revision.
- **Pacing protocol** (p. 5): nine S1 stimuli of −25.5 µA/cm² for 2 ms at a cycle length of 1000 ms. The six action-potential outputs come from the final (ninth) beat.
- **Restitution** (p. 5): the final S1 beat is the initial condition for runs with an S2 stimulus (same strength and duration) at progressively shorter cycle lengths, until no S2 action potential is elicited. The step is not stated. APD90 and DI of each S2 beat give the restitution curve. APD = a − b e^(−DI/c) (Eq. (1)) is fitted with `fminsearch`, and the maximum slope is taken as (a/b) e^(−DI_min/b) (Eq. (2), as printed). The minimum DI is "the shortest DI in the S2 sequence" (p. 5), and "the shortest DI that could elicit an action potential with APD90 longer than 100 ms" (p. 17).
- **Inputs** (m = 6): the maximum conductances, independent uniform on G_x ± G_x/4, where G_x is the LR1991 value (p. 4, Table 1 p. 5). The paper's normalized input u ∈ [0, 1] has G_x at u = 0.5. So G = G_x (3/4 + u/2). This is the map `convert_data.py` applies, in exact decimal arithmetic.

  | input | units | G_x (u = 0.5) | exact range G_x(1 ∓ 1/4) | Table 1, p. 5 (as printed) | sensitivity-analysis distribution, Table 1 |
  |---|---|---|---|---|---|
  | GNa | mS/cm² | 23.0 | [17.25, 28.75] | U(17.250, 28.750) | N(23.0000, 0.84640000) |
  | Gsi | mS/cm² | 0.09 | [0.0675, 0.1125] | U(0.0675, 0.1125) | N(0.09000, 0.00001296) |
  | GK | mS/cm² | 0.282 | [0.2115, 0.3525] | U(0.2115, 0.3525) | N(0.28200, 0.00012544) |
  | GK1 | mS/cm² | 0.6047 | [0.453525, 0.755875] | U(0.4535, 0.7559) | N(0.60470, 0.00058564) |
  | GKp | mS/cm² | 0.0183 | [0.013725, 0.022875] | U(0.0137, 0.0229) | N(0.01830, 0.00000064) |
  | Gb | mS/cm² | 0.03921 | [0.0294075, 0.0490125] | U(0.0294, 0.0490) | N(0.03921, 0.00000256) |

  Table 1 rounds the range ends to four decimals. The text (p. 4: "Gx ± Gx/4", "0.0 corresponding to 17.25 … 1.0 corresponding to 28.75") fixes the exact ends used here. The normal distributions, N(0.5, variance 0.04) in u, are the paper's sensitivity-analysis distribution. The notebook uses them only to compare with the paper's indices; the active subspace is computed on the uniform design distribution.
- **Quantities of interest**: the eight outputs above (p. 4, Table 2). APD90 is the paper's worked example.
- **Gradients**: none. Only values are published, so the notebook uses the global quadratic (QPHD) estimator, with M = 200 ≥ 2 × 28, and OLS as a cross-check.
- **Sampling design** (p. 4): 200 design points by Latin hypercube sampling of the uniform ranges, and 20 separate test runs. No seed is given. Both files are valid Latin hypercubes in u: with the values sorted, the i-th value of every column lies in [i/M, (i+1)/M] (checked for M = 200 and M = 20). There are no duplicate rows, and no test point is a design point.
- **Cost** (p. 10): 2000 simulator runs (one Monte Carlo set for Fig. 4(b)) took 9 h 13 min to 9 h 34 min, about 17 s per run.
- **Published results**: GP emulator summaries (Table 2: design mean, emulator mean and CoV under the normal inputs, Mahalanobis distance on the test runs); mean effects (Figs. 5, 7); main-effect indices (Fig. 6 for APD90, Fig. 8 for all outputs, with the values printed in the cells); PLS regression coefficients (Fig. 9). The APD90 main effects sum to 98.4 % (p. 10). The sums are over 98 % for all outputs except the APDr slope (85 %) and the minimum DI (26 %, 0.256; pp. 13, 15, 17). There is no active subspace in the paper.

## Regenerating

```bash
python convert_data.py            # writes design.csv and test.csv from original/ (0.1 s)
python convert_data.py --check    # rebuilds them in memory and compares with the shipped files
```

The conversion copies each output field character for character. It computes each conductance as an exact decimal G_x (3/4 + u/2) and writes it with all its digits (at most 10 decimals). It checks that the inverse map returns the published 4-decimal u exactly, in decimal and in floating point, and that `np.loadtxt` reads back the published outputs bit for bit. The notebook's load cell repeats the `--check` comparison.

The simulator runs themselves cannot be regenerated here. They were MATLAB runs of CellML-generated code, and no LR1991 implementation is in this repository. Rerunning LR1991 (e.g. from the CellML model repository) would give a new data set, not a reproduction of these files.

## Accuracy

- **Conversion**: lossless, as checked above.
- **Agreement with the paper, six action-potential outputs**: the design means equal Table 2 (column 2) to 0.01 (APD50: 325.78 here, 325.77 in the table). From a least-squares quadratic fitted to the design data, with the paper's normal inputs, the mean and CoV are within 0.4 of the emulator's values in Table 2 (APD90: 362.86 ms vs 363.02, CoV 9.66 % vs 9.48 %; largest, dome V_m: CoV 27.23 % vs 26.87 %). The main-effect indices are within 0.02 of Fig. 8. So these columns are the data behind the paper's results for those outputs.
- **Disagreement, restitution outputs**: the design means of `APDr_slope` and `DI_min` are 1.36 and 6.00 ms. Table 2 gives 2.02 and 10.02 ms. Their quadratic-based main-effect indices also differ from Fig. 8. For `DI_min` they sum to 0.78, against the paper's 0.26, and the top index is GK 0.49 against Fig. 8's 0.16. The file values match the ranges plotted in the paper's Fig. 2 (APDr slope about 0.9–2.5, min DI about 4–9.4 ms). The paper's text says increasing Gsi increases the APDr slope (p. 13, Fig. 7(f)), but in these data the slope falls with Gsi (correlation −0.55), as Fig. 2 also shows. The files are kept as published, and the notebook reports these outputs without interpreting them.
- **Smoothness**: a full quadratic in the six inputs has leave-one-out R² ≥ 0.9992 for the six action-potential outputs, but only 0.894 (APDr slope) and 0.608 (min DI) for the restitution outputs. The paper suspects a discontinuity in the minimum DI (p. 17).
- **Precision**: all values are given to 4 decimals. That is ≤ 5e-5 in u, and 5e-5 in outputs whose ranges are 3–240 units, so rounding is negligible for the analysis.

## How these files relate to the paper

- These are the paper's own simulator runs: S1 Data (design, 200) and S2 Data (test, 20), the inputs and outputs from which its GP emulators were fitted and tested (p. 4, Supporting Information p. 17). After testing, the paper refitted its emulators on all 220 runs (p. 8). The notebook estimates subspaces from the 200 design runs only, and uses the 20 test runs for validation.
- What changed in the files: the inputs are stored in physical units (mS/cm²) instead of u. The PLOS header row became two `#` lines, the column names are Python identifiers with units, and the line ends are LF. No value was changed, and the originals are kept.
- The analysis is new. The paper uses GP emulators, main-effect indices and PLS, and does not compute active subspaces.

## Open questions (only the authors can settle)

1. Why the design means of the APDr slope and minimum DI in Table 2 (2.02, 10.02 ms) differ from S1 Data (1.36, 6.00 ms), which matches Fig. 2. Were the restitution outputs recomputed after the emulators were fitted?
2. Whether the APDr slope rises with Gsi (text p. 13, Fig. 7(f)) or falls with it (S1 Data, Fig. 2).
3. Eq. (2), (a/b) e^(−DI_min/b), is not the derivative of Eq. (1) at DI_min, which is (b/c) e^(−DI_min/c). Is this a typesetting error, or the formula that was used?
4. The S2 cycle-length decrement, the ode15s tolerances, the MATLAB version, the CellML model revision, and the exact definitions of dome V_m and resting V_m (Fig. 1(a) shows them only graphically).

## License and credit

Code: MIT (`LICENSE`). Data: the original files are © 2015 Chang et al., published under the Creative Commons Attribution License (CC BY 4.0) with the article. The converted CSVs are CC BY 4.0 (`LICENSE-DATA`). When using the data, cite E. T. Y. Chang, M. Strong and R. H. Clayton, "Bayesian sensitivity analysis of a cardiac cell model using a Gaussian process emulator", PLoS ONE 10(6):e0130252, 2015, doi:10.1371/journal.pone.0130252, and, for the model, C. H. Luo and Y. Rudy, Circ. Res. 68(6):1501–1526, 1991, doi:10.1161/01.RES.68.6.1501. The paper states that the runs were produced on the iceberg compute resource at the University of Sheffield.
