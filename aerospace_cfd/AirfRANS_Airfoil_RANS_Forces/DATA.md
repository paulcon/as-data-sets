# AirfRANS_Airfoil_RANS_Forces: data

Kind: **stored data**. The flow fields were computed by Bonnet et al. (2022) with OpenFOAM v2112 (simpleFoam,
k–ω SST) and published as `Dataset.zip` (ODbL v1.0; source, size and SHA-256 under "Regenerating"). The
archive (about 10 GB) is not part of this repository. `convert_data.py` reduces it to one row per simulation:
the inputs copied from each case's name, the split flags copied from the archive's `manifest.json`, and six
force coefficients computed from the stored fields by the authors' own `airfrans` library. The coefficients
are not altered after that step; the notebook drops rows only by the stated rules below.

## Files

| file | contents | role |
|---|---|---|
| `AirfRANS_Airfoil_RANS_Forces.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1705; outputs committed |
| `airfrans_forces.csv` | 1000 × 17: family (4 or 5), `U_inf` (m/s), `alpha` (deg), `M_or_L`, `P`, `Q` (NaN for 4-digit rows), `XX`, `CD`, `CD_p`, `CD_v`, `CL`, `CL_p`, `CL_v` (dimensionless), and the flags `full_train`, `scarce_train`, `reynolds_train`, `aoa_train` (1 = training set of that split) | computed from `Dataset.zip` by `convert_data.py` |
| `airfrans_case_names.txt` | the 1000 case names (`airFoil2D_SST_<U_inf>_<alpha>_<digits>`), one per line, row i ↔ row i of the CSV | from `Dataset.zip` |
| `convert_data.py` | reads the archive without extracting it, checks its size and SHA-256, calls `airfrans` for each case, writes the two files; `--check` compares with the shipped table | the conversion (needs `airfrans`, `pyvista`, `vtk`) |
| `check_accuracy.py` | recomputes the six coefficients a second way (numpy and the standard library only: its own VTK XML reader, face pressures, a one-sided wall gradient) and compares them with the shipped table | check (needs the archive) |

The two data files are offered under ODbL v1.0 ("License and credit"). If they are not in this folder, make
them from the archive as described under "Running the notebook from the archive" below.

The CSV has two `#` lines (a description, then the column names with units) and is read by
`np.loadtxt('airfrans_forces.csv', delimiter=',')`. Rows are in sorted case-name order. The coefficients are
written with Python's `repr` (shortest round-trip float64), the inputs character for character as they
appear in the case names (three decimals).

## Model

**Citation.** F. Bonnet, J. A. Mazari, P. Cinnella, P. Gallinari, "AirfRANS: High Fidelity Computational
Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions", *Advances in Neural
Information Processing Systems 35* (NeurIPS 2022), Datasets and Benchmarks Track, pp. 23463–23478,
doi:10.52202/068431-1705. The version read is the arXiv preprint v3 (arXiv:2212.07564v3, 1 June 2023); the
page, table and equation numbers below are its numbers. The `airfrans` README states that the arXiv
version corrects an inconsistency in the machine-learning experiments of the NeurIPS version; the dataset
description is the same.

**Simulator.** Steady incompressible RANS equations (Eqs. 1–2, p. 3) with the k–ω SST turbulence model
(p. 4; Appendix F). OpenFOAM v2112 (p. 21): blockMesh C-grids with the boundaries 200 chords away, first cells
2 µm high (y⁺ around 1 in the worst case), 250,000–300,000 cells (p. 4); simpleFoam with SIMPLEC, run until
the drag and lift coefficients converge (p. 5); boundary conditions in Appendix J (Tables 7–9, pp. 28–31).
Air at 298.15 K and sea level, ν = 1.56×10⁻⁵ m²/s (Table 6, p. 21); chord 1 m (p. 4). About 25 min per
simulation on 16 cores of an AMD Threadripper 3960X and about 20 days for the 1000 (Table 4, p. 9).
Validation against NACA 0012 and NACA 4412 experiments (Appendix K, pp. 31–35; figures only; drag slightly
over-estimated, p. 32). Generator code: https://github.com/Extrality/NACA_simulation (ODbL; not copied).

**Inputs.** Each simulation is a NACA airfoil, a Reynolds number and an angle of attack (p. 4).

| input | symbol | units | range or set | source |
|---|---|---|---|---|
| inlet velocity | `U_inf` | m/s | Re ∈ [2×10⁶, 6×10⁶] → [31.2, 93.6] with ν = 1.56×10⁻⁵ m²/s, chord 1 m | p. 4; Table 6, p. 21; 93.6 m/s at Re = 6×10⁶, p. 31 |
| angle of attack | `alpha` | deg | [−5, 15] | p. 5 |
| 4-digit max camber | `M_or_L` (M) | % chord | [0, 7] | Table 1, p. 4 |
| 4-digit camber position | `P` | tenths of chord | uniform on [0, 7], values below 1.5 set to 0: {0} ∪ [1.5, 7] | Table 1 and caption, p. 4; p. 26 |
| 5-digit camber digit | `M_or_L` (L) | 1 (design lift coefficient 0.15 L) | [0, 4] | Table 1; p. 25 |
| 5-digit camber position | `P` | twentieths of chord | [3, 8] | Table 1; p. 25 |
| 5-digit reflex switch | `Q` | 1 | {0, 1} (0 standard, 1 reflex camber) | Table 1; p. 4 |
| thickness (both families) | `XX` | % chord | [5, 20] | Table 1 |

The paper's text says the inputs are uniform over each interval or set (Table 1 caption) and gives the ranges
of Re and α without their distribution (Fig. 4, p. 27, shows histograms). The digits need not be integers
(p. 26). The 4-digit and 5-digit counts are not reported; the data have 489 and 511, and 120 of the 4-digit
airfoils have P = 0 (0.245 against 0.214 under the stated rule, binomial sd 0.019; notebook).

**Quantities of interest.** $C_D = D/q_\infty$ and $C_L = L/q_\infty$, $q_\infty = \rho U_\infty^2 A/2$,
$A = 1$ m² (p. 5), and their pressure and viscous parts. The pitching moment is not used (p. 5).

**How the coefficients are computed.** For each case, `convert_data.py` calls
`airfrans.Simulation(root, name).force_coefficient(reference=True)` (airfrans 0.1.5.1). The library takes the
pressure of the internal field at the wall nodes, computes the wall shear stress from the vtk point gradient
(Jacobian) of the stored velocity with ν = 1.5498×10⁻⁵ m²/s (its polynomial for air at 298.15 K), averages
both to the faces of the aerofoil patch, integrates with the patch normals and lengths, and projects on the
free-stream directions. ρ cancels because the stored pressure is kinematic. The paper computes the wall
shear stress with ParaView's gradient filter (p. 6). These are therefore post-processed coefficients, not
OpenFOAM's own `forceCoeffs`, which the archive does not contain.

**Gradients.** None. The simulations give values only; a finite-difference gradient would cost six RANS runs
of about 25 min each per point.

**Sampling design.** Stated as independent uniform draws (a Monte Carlo design, not a Latin hypercube):
Table 1's caption states uniform sampling for the airfoil inputs, and the public generator code
(`dataset_generator.py` of Extrality/NACA_simulation) draws every input independently with
`np.random.uniform` (and `randint` for Q), rounds them to three decimals, and sets U_inf = Re·ν. No seed is
reported. The stored angles of attack do not fit uniform draws on [−5°, 15°]: 85 cases lie above 12.5°
against 125 expected (z = −3.8), the angle-of-attack training set holds 804 cases against 750 (binomial
p = 5.8e-05), and the Kolmogorov–Smirnov and Cramér–von Mises tests give p = 0.007 and 0.003 (0.008 and 0.014
in the cambered 4-digit subset; notebook). The shortfall does not depend on camber, thickness or inlet velocity
(Mann–Whitney p 0.17 to 0.94; notebook). The inlet velocity, camber, camber position and thickness are
consistent with uniform draws (KS p 0.184 to 0.890; the 5-digit L gives KS p 0.020 and Cramér–von Mises
p 0.172; notebook). The notebook nevertheless uses the uniform density on every range, α included (its own
choice, matching the stated design); the high-incidence corner, where drag curves most, holds fewer samples
than that density gives it.

**Published results.** No active subspace and no table of force coefficients. Table 3 (p. 8) gives errors and
Spearman correlations of machine-learning surrogates for $C_D$ and $C_L$; Fig. 9 (p. 33) shows NACA 0012
polars against experiment. The splits (p. 6): full 800/200; scarce 200 training cases; Reynolds extrapolation
(training Re 3–5 million); angle-of-attack extrapolation (training −2.5° to 12.5°).

## Regenerating

The flow fields cannot be regenerated here (OpenFOAM, about 25 min on 16 cores per case). The table is
regenerated from the archive.

**Source.** `Dataset.zip`, https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip, the
"pre-processed" version of the data in the README and download function of the `airfrans` library (the raw
OpenFOAM cases are a separate archive, `OF_dataset.zip`, not used here; no DOI), 10,029,067,577 bytes, SHA-256
`6b301d75dee77fc6c7de6e551c44332be4acc84e30b253e9c60cfa756f6c96db`. It holds `Dataset/manifest.json` and
1000 case folders, each with `<name>_internal.vtu`, `<name>_aerofoil.vtp` and `<name>_freestream.vtp`
(4002 entries, 15,033,455,903 bytes uncompressed). `convert_data.py` reads only `manifest.json` and the
`internal.vtu` and `aerofoil.vtp` of each case.

**Recipe** (external run; the shipped files were written this way on macOS on 2026-11-08 with airfrans 0.1.5.1,
pyvista 0.49.2, vtk 9.7.2):

```
python -m venv venv && . venv/bin/activate
pip install airfrans==0.1.5.1 pyvista==0.49.2 vtk==9.7.2
python convert_data.py --zip Dataset.zip --out DIR --max-seconds 600   # repeat until it reports 1000 cases
python convert_data.py --zip Dataset.zip --out DIR --check             # compare with the shipped table
```

`--out` must be a folder outside this directory (not this directory or a folder in it). The run is resumable (rows are appended to
`DIR/airfrans_forces.partial.csv`); `--rows N` converts N evenly spaced cases only. The archive's size and
SHA-256 are checked first (about a minute). `--check` compares the inputs and flags as text and the
coefficients with the tolerance |a − b| ≤ 1e-9 + 1e-6 |b|; the tolerance allows for other vtk/pyvista
versions and platforms. Run time of the shipped run: about 0.8 s per case (its log records 212 cases in
166 s), so about 13 min for the 1000 cases. Without `airfrans` the script exits with the install command
above. `--check` compares with the `airfrans_forces.csv` in this folder and stops with a message if that file
is not there. The notebook needs numpy, scipy, matplotlib and active_subspaces and runs in about 5 s (4 s printed in the committed `.venv` run).

### Running the notebook from the archive

The notebook reads `airfrans_forces.csv` and `airfrans_case_names.txt` from its own folder. If they are not
there (the cell that loads them then stops with a pointer to this section), make them:

1. Download `Dataset.zip` from https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip (about 10 GB;
   size and SHA-256 above).
2. Install the libraries and run `convert_data.py --zip Dataset.zip --out DIR` for all 1000 cases (no
   `--rows`), as in the recipe above; DIR is any folder outside this one. Repeat the command until it reports
   1000 cases. It writes `DIR/airfrans_forces.csv` and `DIR/airfrans_case_names.txt`.
3. Copy those two files from DIR into this folder (the folder of the notebook).
4. Run the notebook. `check_accuracy.py` also reads the two files from this folder.

## Accuracy

Two checks were run outside this repository on `Dataset.zip` (external runs, macOS, 2026-11-08, airfrans
0.1.5.1, pyvista 0.49.2, vtk 9.7.2; they need the archive). The numbers from `check_accuracy.py` are those
its summary prints; the scale numbers are printed by the notebook.

- **Reproducibility of the table.** `convert_data.py --zip Dataset.zip --out DIR --rows 25 --check`: PASS for 25
  evenly spaced cases: inputs and flags identical as text, coefficients within the tolerance with no
  difference at printed precision. This ran on the same machine and library versions as the shipped table, so
  it is a reproducibility check, not a cross-platform one.
- **Totals: independent integration.** `check_accuracy.py --zip Dataset.zip --out DIR --rows 50` recomputes the
  six coefficients of 50 cases, evenly spaced in name order (18 of them in the notebook's cambered 4-digit
  subset), with its own VTK reader, face pressures and a one-sided wall gradient, and compares them with the
  shipped table:
  - $C_D$: median relative difference 2.06e-3, 95th percentile 5.75e-3, largest 7.88e-3; largest absolute
    difference 9.47e-5.
  - $C_L$: median relative difference 2.64e-5, 95th percentile 3.43e-4, largest 8.32e-4; largest absolute
    difference 5.38e-5.

  The totals of the two methods agree to better than 0.8 % ($C_D$) and 0.09 % ($C_L$). For scale, in the
  cambered subset the standard deviation of $C_D$ is 5.39e-03 and the leave-one-out RMS error of the cubic in
  all five inputs is 6.75e-04 for $C_D$ and 7.74e-03 for $C_L$ (notebook): the largest $C_D$ difference is
  1.8 % of that spread and 14 % of that error, and the largest $C_L$ difference 0.7 % of its error (ratios of the numbers above). The
  analysis uses only $C_D$ and $C_L$, so its conclusions do not depend on the integration method.
- **Pressure and viscous parts: the split depends on the method.** Same run:
  - $C_{D,p}$: median relative difference 4.11e-2, 95th percentile 1.07e-1, largest 1.33e-1; median difference
    +1.12e-4, largest absolute 1.66e-4.
  - $C_{D,v}$: median relative difference 1.87e-2, 95th percentile 2.08e-2, largest 2.39e-2; median difference
    −1.31e-4, largest absolute 1.52e-4.
  - $C_{L,p}$ agrees like $C_L$ (largest relative difference 1.07e-3). $C_{L,v}$: median relative difference
    2.46e-2, 95th percentile 3.18e-1, largest 1.40; largest absolute 1.86e-5.

  The two $C_D$ shifts have opposite signs and mostly cancel in the total. The large relative values for
  $C_{D,p}$ come from its small size (median 3.34e-03 over the 1000 cases, below 2e-3 in 0.25 of them;
  notebook), not from a larger absolute difference. So the shipped pressure/viscous split is `airfrans`'
  decomposition: how $C_D$ divides between the two parts depends on the discretization of the wall pressure and
  the wall velocity gradient by about 1.5e-4, about 2 % of $C_{D,v}$ (median 7.07e-03; notebook). The viscous
  part of $C_L$ (median absolute value 1.12e-04; notebook) is not resolved below about 2e-5 and should not be
  used.
- **Which arrays the check read.** `check_accuracy.py` uses the cell values OpenFOAM wrote (face pressure of
  `aerofoil.vtp`, wall-cell velocity of `internal.vtu`) and falls back to point-value means where a file has
  none; it prints which it used (line "arrays used") and its own SHA-256. A rerun of the shipped script
  (SHA-256 876ad351…2fad, external run, macOS, 2026-11-08) used the cell values in 50 of 50 cases for both
  arrays and printed the same numbers as above to every digit shown.
- **Viscosity.** The inlet velocities were set with ν = 1.56×10⁻⁵ m²/s: with that value the Reynolds numbers
  span 2.005×10⁶ to 5.999×10⁶ and the `reynolds_train` flags match the paper's definition in all 1000 cases;
  with the library's 1.5498×10⁻⁵ six cases would exceed 6×10⁶ and nine flags would disagree (notebook). If the
  solver also used 1.56×10⁻⁵, the viscous parts computed by `airfrans` are low by the ratio 1.0066 (notebook),
  a 0.66 % change, smaller than the method dependence of the split above (about 2 % of $C_{D,v}$). In the
  `airfrans` split the viscous part of $C_D$ is 0.071 to 0.936 of the total (median 0.686), and the viscous part
  of $C_L$ is at most 6.35e-04 in absolute value (notebook).
- **Consistency.** Pressure plus viscous part equals the total for every row to rounding (differences of order
  1e-16 and below; notebook).
- **Solver.** Run until the force coefficients converged (p. 5); no tolerance or residual level is reported.
  Mesh: y⁺ around 1 in the worst case (p. 4). Validation: Appendix K, figures only.
- **Precision of the inputs.** Three decimals, as written in the case names (the solver ran with these values).

## How these files relate to the paper

The 1000 rows are the paper's 1000 simulations (p. 4), with its four splits as flags. The paper uses the
coefficients to score machine-learning surrogates of the fields; it reports no sensitivity or active-subspace
analysis. The notebook analyses the 369 cambered NACA 4-digit cases (rule: family 4 and P > 0) with QPHD for
$C_L$ and $C_D$, checks the directions with OLS, with the gradients of a cubic fit and, for lift, with
thin-airfoil theory, and repeats the estimate on the 120 4-digit cases with P = 0 and on the 5-digit cases for
each Q. Two 5-digit cases with $C_D$ above 0.034 at about −4.4° (thin, cambered, mostly pressure drag) are
kept; the notebook reports the fit with and without them. The comparison table of the notebook lists the
checks of the paper's statements (ranges, splits, "roughly 0" lift at −5°).

## License and credit

The AirfRANS dataset is © Bonnet, Mazari, Cinnella and Gallinari (Sorbonne Université, ISIR; Extrality),
released under the Open Data Commons Open Database License v1.0 (Appendix A, p. 20). `airfrans_forces.csv` and
`airfrans_case_names.txt` are derived from that database and are therefore offered under **ODbL v1.0**, not
CC BY 4.0, and are listed as exceptions in `LICENSE-DATA`; cite the paper when using them. The notebook's
figures and numbers are produced works of the database (credit: AirfRANS, Bonnet et al. 2022, ODbL v1.0).
`convert_data.py` calls the `airfrans` library (the package metadata names MIT, its README ODbL v1.0) without
copying any of its code; `check_accuracy.py` uses numpy and the standard library only. Both scripts and the
notebook: MIT (this repository). No code of Extrality/NACA_simulation or Extrality/AirfRANS is included.

## Unknowns

- Which viscosity the solver used (1.56×10⁻⁵ from Table 6, with which the inlet velocities were set, or the
  1.5498×10⁻⁵ of the current generator code and of `airfrans`). It scales the viscous parts by 1.0066, less
  than the method dependence of the pressure/viscous split (about 2 % of $C_{D,v}$; "Accuracy").
- Why the angles of attack fall short of uniform above about 11° (85 cases above 12.5° against 125 expected;
  "Sampling design"). The paper's text does not state the distributions of Re and α; uniform is taken from the
  public generator code, whose version for the 2022 runs is not recorded. A different sampler, or
  high-incidence runs that were dropped or redrawn (for example unconverged ones; see the convergence item below),
  would both explain it. The other inputs are consistent with uniform draws.
- Whether the paper's $C_D$ and $C_L$ (the targets of Table 3) were computed with `airfrans` or with ParaView's
  gradient filter (p. 6). The totals agree between methods; the pressure/viscous split differs by about 1e-4.
- How P = 0 airfoils are built: the paper calls them symmetric (p. 26); the public generator code sets the
  camber line to zero but keeps a camber slope of −2(M/100)x in the thickness direction, so they may be
  slightly asymmetric for M > 0. The stored geometries were not checked; in the data M has almost no weight
  for these cases (0.003 in the leading lift direction; notebook).
- Convergence level of each run (not reported) and whether any run ended unconverged or was replaced (the
  paper says that runs of its validation study that could not be made to converge are missing from Fig. 9,
  p. 32); the two 5-digit cases with high drag at negative incidence are consistent with separated flow but
  were not inspected. This may be linked to the α shortfall above.
