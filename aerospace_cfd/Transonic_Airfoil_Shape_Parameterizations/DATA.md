# Transonic_Airfoil_Shape_Parameterizations: data

Kind: **stored data** for the outputs; the design inputs are **regenerated** (below). The lift and drag values come from SU2 runs by Z. J. Grey and P. G. Constantine for "Active Subspaces of Airfoil Shape Parameterizations" (AIAA Journal 56(5):2003–2017, 2018, doi:10.2514/1.J056054). The authors posted the SU2 output files in [zgrey/ASAP](https://github.com/zgrey/ASAP). `convert_data.py` reads them from the authors' zip files and writes four CSV files; no value was rerun, refitted or corrected. **The design inputs were not posted.** `convert_data.py` regenerates them from the random seeds recorded in the authors' MATLAB script. They are therefore a reconstruction. The generator is checked bit for bit against design files of the authors' later study, and the rows against the outputs (see "Inputs" below). The zips themselves (20.8 MB) are not shipped. Their URLs and SHA-256 digests are below, and `convert_data.py --check` rebuilds the CSVs from them.

## Files

| file | contents | role |
|---|---|---|
| `Transonic_Airfoil_Shape_Parameterizations.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2018; outputs committed |
| `convert_data.py` | reads the three ASAP zips, regenerates the designs, writes and checks the CSVs; `CST_NAMES`, `PARSEC_NAMES`, `PARSEC_DESCRIPTIONS`, Table 2/3 bounds as printed, `LOWER`/`UPPER` (normalized), `DROPPED`, `SOURCE_SHA256`, `physical_approx` (approximate physical values) | conversion |
| `parsec_inputs.csv` | 6472 × 11: normalized PARSEC design variables `x1` … `x11` in [−1, 1], paper Table 1 numbering | regenerated inputs |
| `parsec_outputs.csv` | 6472 × 5: `sample_set` (500, 1000 or 5000), `run` (i of `forces/airfoil<i>.su2.dat`), `design_row` (1-based row of the regenerated design matrix), `CL`, `CD` (dimensionless) | stored outputs |
| `cst_inputs.csv` | 6500 × 10: normalized CST design variables `x1` … `x10` in [−1, 1], paper numbering | regenerated inputs |
| `cst_outputs.csv` | 6500 × 5: the same five columns | stored outputs |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are written as the shortest decimal that reads back as the same double. `CL` and `CD` are written exactly as SU2 printed them (six significant digits). Rows follow the authors' `mixsamples.m`: the 500-design set, then 1000, then 5000, each in run order.

**Source.** [zgrey/ASAP](https://github.com/zgrey/ASAP), commit `79ae7ed62f282c7ee0bebf508af55530f69579e7` (2019-03-13, branch master), fetched 2026-10-30:

| file | bytes | SHA-256 | contents |
|---|---|---|---|
| `500_Samples.zip` | 1,598,859 | `ea90e7eed0c1761a4ecf070beb68911f3eae02f2f628bba397080cb1f3c99aa0` | `CST_forces.zip` (500 runs, dated 2016-10-12), `PARSEC_forces.zip` (497 runs, 2016-10-05) |
| `1000_Samples.zip` | 3,204,684 | `55271fbec2c171f8f6895e81e0577be0ee7af525d7c868f002a179bd5444469d` | CST 1000 runs, PARSEC 998 runs (2016-10-21) |
| `5000_Samples.zip` | 16,017,270 | `639f37a3394b5be8cfaa2cc9f927767eda1a82af212c3d5e94ce94807b926239` | CST 5000 runs, PARSEC 4977 runs (2016-10-27) |

Each inner zip holds one SU2 `forces_breakdown.dat` per run, renamed `forces/airfoil<i>.su2.dat`, numbered 1 to the number of runs without gaps. The repository also holds MATLAB scripts (`mixsamples.m`, `AUTO/*.m`), an SU2 configuration (`AUTO/inv_airfoil.cfg`) and, under `AUTO/` and `NEW_DATA/`, designs and forces of a later study (fine meshes; m = 9 PARSEC and m = 10, 100 CST; ±20 % and ±60 %). The later study is not used here and was not compared with the paper. The scripts are the 2019 versions. They differ from those that made the 2016 runs (see below).

## Model

Content of the paper card. Page, table and figure numbers refer to the published AIAA Journal article.

- **Citation.** Z. J. Grey and P. G. Constantine, "Active Subspaces of Airfoil Shape Parameterizations", *AIAA Journal* 56(5):2003–2017, 2018, doi:10.2514/1.J056054 (presented as AIAA Paper 2017-0507; preprint arXiv:1702.02909, not used).
- **Simulator.** SU2 release 4.2.0 "Cardinal" (stated in every output file) solves the 2-D compressible Euler equations on an unstructured mesh. The free stream is at Mach 0.8, 273.15 K and 101,325 Pa ("NIST conditions", p. 2008), with a free-stream density of 1.29225 kg m⁻³ and a speed of 265.057 m/s (from the output files). The authors' meshing chain (2019 scripts) is: the airfoil coordinates at 300 points, `dat2gmsh.py`, gmsh 2.10.1 with one refinement, then conversion to SU2 format. The runs were made on the Colorado School of Mines Mio cluster. The cost per run is not reported.
- **Angle of attack.** The paper says the runs cover 0–2.5° (p. 2008). Every one of the 12,972 posted files states a free-stream velocity of (265.057, 0) m/s, so **all posted runs are at 0°**. In every file the lift equals the y-force and the drag the x-force coefficient (CL = CFy, CD = CFx), as at 0°. The flow agrees: CST sections that are mirror-symmetric (x1 = x6 in the normalized inputs, see "Physical values") have C_l ≈ 0 (mean |C_l| 0.0035 over the 125 runs with |x1 − x6| < 0.02; quadratic value −5e-5 at x1 = x6 = 0), which holds only at zero incidence. `convert_data.py` checks the velocity, the Mach number, the free-stream state, the SU2 release and zero friction drag in every file.
- **Inputs, PARSEC** (m = 11; Table 1, p. 2005; Table 2, p. 2008; uniform on the box; bounds are ±20 % of a NACA 0012 fit):

  | name | description | units | Table 2 "Lower" | Table 2 "Upper" | source |
  |---|---|---|---|---|---|
  | `x1` | upper-surface domain coordinate (crest position) | chord | 0.242 | 0.363 | T1, T2 |
  | `x2` | lower-surface domain coordinate | chord | 0.242 | 0.363 | T1, T2 |
  | `x3` | upper-surface shape coordinate (crest height) | chord | 0.048 | 0.072 | T1, T2 |
  | `x4` | lower-surface shape coordinate | chord | −0.072 | −0.048 | T1, T2 |
  | `x5` | trailing-edge shape coordinate (offset) | chord | −0.004 | 0.004 | T1, T2 |
  | `x6` | trailing-edge half-thickness | chord | 0.008 | 0.012 | T1, T2 |
  | `x7` | wedge angle | deg (assumed; Appendix A uses tan) | −2.223 | −3.335 | T1, T2 |
  | `x8` | wedge half-angle | deg (assumed) | 7.40 | 11.10 | T1, T2 |
  | `x9` | upper-surface inflection (crest curvature) | 1/chord | −0.400 | −0.600 | T1, T2 |
  | `x10` | lower-surface inflection | 1/chord | 0.400 | 0.600 | T1, T2 |
  | `x11` | leading-edge radius | chord | 0.012 | 0.018 | T1, T2 |

  The paper prints no units. Those above are our reading. The stored values are normalized; "Physical values" below gives an approximate conversion. Appendix A gives the linear system: rows of M are s(l_int), s(1), s′(l_int), s′(1), s″(l_int) and the leading coefficient, with p_U = (x3, x5 + x6/2, 0, tan(x7 − x8/2), x9, √(2 x11)) at l_int = x1, and p_L = (x4, x5 − x6/2, 0, tan(x7 − x8/2), x10, √(2 x11)) at l_int = x2. The text says the lower-surface leading coefficient takes the negative sign.
- **Inputs, CST** (m = 10; Table 3, p. 2010). Each surface is s(l) = l^½ (1 − l) Σ_{j=0}^{4} x_j l^j (Eqs. 11–13). The paper numbers the upper coefficients x1–x5 and the lower ones x6–x10 (p. 2011); the data suggest the 2016 code had them the other way round ("Physical values", item 5). Table 3 bounds: x1 ∈ [0.12, 0.18]; x2–x5 ∈ [0.8, 1.2]; x6 ∈ [−0.18, −0.12]; x7–x10 ∈ [−1.2, −0.8].
- **Quantities of interest.** The lift and drag coefficients C_l and C_d, dimensionless, at unit reference length and area ("Total CL", "Total CD" in SU2's forces breakdown). The flow is inviscid, so C_d is pressure (wave) drag, and its friction component is 0 in every file. No transform is applied. Ranges in the data: PARSEC C_l −0.477 to 0.218 and C_d 0.00096 to 0.0426; CST C_l −0.252 to 0.250 and C_d 0.00090 to 0.0180. No value is non-finite or has C_d ≤ 0.
- **Gradients.** None (no adjoint runs were posted). The paper estimates the active subspace from a global least-squares quadratic (Sec. III.B, Eqs. 19–22), and so does the notebook (`estimator="qphd"`; the eigenvalues agree with the paper's, so its C has the same 1/3 factor, gg^T + H H/3).
- **Sampling design.** Independent uniform (Monte Carlo) samples on the normalized cube, drawn by MATLAB `rand` in three sets of 500, 1000 and 5000 with seeds 43, 44 and 45. PARSEC designs whose surfaces cross were dropped before meshing (3, 2 and 23 designs). CST and PARSEC used the same seeds, so within a set they share their random numbers: CST column j equals column j of the PARSEC design matrix in the authors' code order (j ≤ 10), row for row up to the first dropped PARSEC design. For example, CST run 1 of the 5000 set has x1 = 0.978023 and x6 = −0.795466, and PARSEC run 1 has x11 = 0.978023 and x4 = −0.795466. The two parameterizations are therefore not independent samples of each other. The paper reports N = 6500 for each parameterization and 500 bootstrap replicates (pp. 2008, 2010).
- **Published results.** Fig. 3 (PARSEC eigenvalues), Fig. 4 (subspace errors), Fig. 5 (first two eigenvectors), Figs. 6–7 (shadow plots), and the same for CST in Figs. 8–11; Fig. 12 (subspace error against N); Fig. 13 (lift over the drag active variables). There are no printed numbers. The notebook's "Comparison with the paper" lists values read from the figures. Figs. 14–19 (Pareto front, flows at angles of attack up to 2.5°, a 50-parameter CST study) use runs not in these zips.

## Inputs: how they were reconstructed and checked

`mixsamples.m` loads the inputs from `CST_designs*.mat` and `PARSEC_designs*.mat`, which are not in the repository. The 2019 `AUTO/parsec_samples.m` records the seeds: `rng(43); % Used for 500 Sample Designs`, `rng(44)` for 1000 and `rng(45)` for 5000. It then draws `X = 2*rand(N, m) - 1`. MATLAB's default generator is the Mersenne Twister with `init_genrand(seed)` and 53-bit doubles. numpy's `RandomState(seed).random_sample` is the same generator, so the MATLAB matrix is `RandomState(seed).random_sample(N*m).reshape(m, N).T` (MATLAB fills column by column; MATLAB maps `rng(0)` to `init_genrand(5489)`, which does not matter for the nonzero seeds here). The checks:

- **The generator, against the authors' own files.** The repository tracks three design files of the authors' later study (`AUTO/designs/FINE/PARSEC_designs_m9_N1000_pm20pct.mat`, `AUTO/designs/FINE/designs_m10_N1000_pm20pct.mat`, `AUTO/designs/COARSE/designs_m10_N500_pm20pct.mat`). The expression above reproduces their `X` **bit for bit** with seed 44 (the PARSEC file) and 47 (the two CST files), and their physical designs obey `X0 = lb0 + (ub0 − lb0)(X + 1)/2` exactly. So the MATLAB–numpy equivalence and the column order are verified, not assumed. What remains assumed is that the 2016 scripts drew `2*rand(N, m) - 1` right after `rng(seed)`; the CST fit below confirms it. `convert_data.py --verify --mat DIR` repeats the check.

- **CST** (all designs were run): with these seeds, a linear fit of C_l in the regenerated inputs has R² = 0.9987, 0.9988 and 0.9988 in the three sets. For every other seed from 0 to 1999, the best R² was 0.063 (500 set) and 0.11 (1000 and 5000 sets, first 300 runs), the level expected for unrelated rows. Filling the matrix row by row instead gives 0.03–0.04 at the right seeds. So the CST seeds were also 43, 44 and 45 (the 2019 `cst_samples.m` holds `rng(47)` and a different setup).
- **PARSEC** (feasible designs only). The runs are numbered 1 to N − NF, so the dropped rows must be found. `convert_data.py --verify` aligns the runs to the regenerated rows by dynamic programming: it chooses where the NF skips fall to minimize the misfit of C_l and C_d under a least-squares quadratic, refitting three times. The dropped rows (0-based) are 106, 258 and 312 (500 set); 9 and 257 (1000 set); and 23 rows of the 5000 set, listed in `convert_data.py`. Every one of the 28 lies in the corner x5 > 0.85, x6 < −0.85 of the normalized inputs (high trailing-edge offset, thin trailing edge), which holds 52 of the 6500 drawn rows (0.8 %; for 28 random rows the chance is about 1e-63). **How sure the rows are.** Which block of runs follows which offset is certain: a quadratic fitted only to the first 370 runs of the 5000 set (before any drop) gives each run, without any ordering constraint, the same offset as the dropped rows imply for 85 % (5000 set; chance level 4 %), 97 % (1000) and 73 % (500) of the runs, and a majority in every block. The exact row of a drop is fixed by the outputs alone for 18 of the 28. For the other 10 (500 set: 106, 312; 1000 set: 9; 5000 set: 376, 778, 1107, 1972, 2906, 3723, 4833), a row one or two places away fits almost as well (the sum of squared standardized residuals rises by 0.03 to 1.8), and the choice rests on the corner pattern: the chosen rows are in the corner, the alternatives are not. If those 10 were wrong, 10 runs would carry a neighbouring design; moving all 10 changes |w1·w1| and |w2·w2| by less than 5e-5 for lift and drag (checked by moving them). `--verify` lists these near-ties. With these rows dropped, the quadratic R² of C_l is 0.9795 (5000 set) and 0.9787 (1000 set); for C_d it is 0.9927 and 0.9924. The columns are put in the paper's order with the permutation in `mixsamples.m`. The first eigenvectors then agree with the paper's Fig. 5 to within 0.005 in every component (notebook).
- **The PARSEC 500-design set is anomalous.** Its runs fit the regenerated inputs (quadratic R² of C_l 0.914; unrelated rows would give about 78/497 = 0.16), but less well than the other sets. The quadratic fitted to the 5000 set misses it by a median 0.48 standard deviations of C_l, against 0.08–0.09 for the 1000 and 5000 sets. The linear sensitivities differ too: for example, the leading-edge radius coefficient is −0.006 there, against −0.001 in the other sets. A per-input affine change of the 500-set inputs reduces the median misfit only from 0.36 to 0.23 (scratch check, not shipped), so a simple change of bounds does not explain it; neither does a swap of two columns, a sign flip or a rescaling of one column (best median misfit 0.35–0.43). That set was run first (2016-10-05), and its parameterization or mesh may have differed. The CST 500-design set shows no such difference (median misfit 0.023 against 0.020). The rows are kept in the files. The notebook pools all sets as the paper does, then shows that dropping this set changes the directions by less than 2e-4 in |w·w|.
- **Not reproduced: the feasibility check.** We tried to recompute which PARSEC designs have crossing surfaces from the paper's Appendix A, as printed and in four sign variants, with the bounds of Table 2. None selects the 28 dropped designs (scratch check, not shipped). The 2016 PARSEC code is not in the repository, and the posted `parsec.m` and `parsec_samples.m` are later versions (m = 9, sharp trailing edge, a nominal labelled "WRONG").

## Physical values

**The files hold the normalized inputs.** The authors stored and analyzed the normalized X, the paper's eigenvectors refer to it, and the seeds regenerate it exactly, so this is what is shipped. The physical values can be recovered only **approximately**, as follows.

**The map.** The authors' sampling scripts set `lb0 = (1 − pct)·dv`, `ub0 = (1 + pct)·dv` with pct = 0.2 and `X0 = lb0 + (ub0 − lb0)(X + 1)/2`, that is, physical = nominal × (1 + 0.2 X): X = −1 gives 0.8 × nominal, also for a negative nominal. The posted m = 9 PARSEC file obeys this exactly (above). Table 2 is printed in this order for x7 (−2.223, −3.335) and x9 (−0.400, −0.600), and the 2019 nominal reproduces Table 2's x1, x3, x7, x8 and x11 to the printed digits (0.8 × 0.3025 = 0.242; 0.8 × −2.7791 = −2.223; 1.2 × 9.2496 = 11.10; 0.8 × 0.0146 = 0.0117, printed 0.012). The data fix the two intervals that are printed the other way round:

- **PARSEC x4** (printed −0.072, −0.048). Drag rises along its first vector, which has +0.41 on x3 and +0.80 on x4 (notebook), and the paper reads this vector as thickness (p. 2009). That holds only if X4 = +1 gives −0.072, the deeper lower crest. The lift vector (+0.47 on x3, −0.50 on x4) is then camber, as the paper also says.
- **CST x6** (printed −0.18, −0.12). The runs are mirror-symmetric under exchanging x1 and x6: a quadratic in (x1, x6) has C_l coefficients −0.12787 and +0.12796, C_d coefficients 0.004282 and 0.004289, and C_l ≈ 0 on the line x1 = x6 (above). A section is symmetric only when its upper and lower leading coefficients have equal magnitude, so both magnitudes grow with X: X6 = +1 gives −0.18.

`convert_data.physical_approx(kind, X)` applies this map with the midpoints of Tables 2 and 3 as nominals:

| PARSEC | nominal | units | | CST | nominal |
|---|---|---|---|---|---|
| x1, x2 | 0.3025 | chord | | x1 | 0.15 |
| x3, x4 | 0.06, −0.06 | chord | | x2–x5 | 1.0 (meaning unknown) |
| x5 | 0 (mapped to 0.004 X) | chord | | x6 | −0.15 |
| x6 | 0.01 | chord | | x7–x10 | −1.0 (meaning unknown) |
| x7, x8 | −2.779, 9.25 | deg | | | |
| x9, x10 | −0.5, 0.5 | 1/chord | | | |
| x11 | 0.015 | chord | | | |

**What remains uncertain, and why the files stay normalized.**

1. Tables 2 and 3 are rounded to 2–3 digits, and the 2016 nominals are not posted. For example x11 is 0.015 from the table but 0.0146 in the 2019 script (2.7 %), and x9 is −0.5 against −0.4928.
2. PARSEC x5 has nominal 0, so the ±20 % rule cannot give its printed range ±0.004; how the 2016 code set it is unknown. The 2016 lower-surface sign convention also differs from the 2019 script (x4 and x10 change sign).
3. The PARSEC 500-design set follows a different input–output map (above), so its physical meaning may differ.
4. **CST.** Table 3's nominal values of ±1 for x2–x5 and x7–x10 cannot be Eq. (13) coefficients of a NACA 0012 fit. A least-squares fit of Eq. (13) to the NACA 0012 thickness gives (0.170, −0.059, 0.063, −0.085, 0.057), and coefficients near 1 would give a 44 %-thick section. In the data, C_l and C_d do not depend on these eight inputs at all: the quadratic R² in x1 and x6 alone is 0.998751 against 0.998765 with all ten for C_l, and 0.999298 against 0.999304 for C_d (a partial F-test of the 60 extra terms gives F = 1.21 and 0.95, noise level). So the CST finding that "x2, …, x5, x7, …, x10 have no impact" (p. 2011) is a property of the 2016 code that made the runs; in the authors' later CST study (`AUTO/subspaces/FINE/AS_m10_N1000_pm20pct_QOI2.mat`, Bernstein coefficients, ±20 %) all ten coefficients carry weight in the lift vector (0.19–0.51).
5. **Which surface is CST x1.** With the map above, moving along the lift vector w1 = (e1 − e6)/√2 enlarges the leading coefficient in column 1 and shrinks the one in column 6, and C_l *falls* (correlation −0.9994, also visible in the paper's Fig. 11a). If column 1 is the upper surface, as the paper says (p. 2011), lift falls as positive camber grows. The authors' later CST study at the same conditions shows conventional behaviour (its lift vector has all ten components positive and C_l rises along it, columns 1–5 being the lower surface as the 2019 `cst_samples.m` states). The more likely reading is therefore that the 2016 code also had columns 1–5 on the lower surface, the reverse of the paper's labels. The posted files cannot prove this; it is listed for the authors.

The Table 2 and 3 bounds as printed are in `convert_data.py` (`PARSEC_TABLE_LOWER`, …) for reference.

## Regenerating

```bash
python convert_data.py --zips /path/to/ASAP              # writes the four CSVs (about 4 s)
python convert_data.py --zips /path/to/ASAP --check      # rebuilds them in memory and compares
python convert_data.py --zips /path/to/ASAP --verify     # also repeats the PARSEC alignment search
python convert_data.py --zips /path/to/ASAP --verify --mat /path/to/ASAP/AUTO/designs   # and checks the generator
```

`/path/to/ASAP` is a directory holding the three zips (for example a clone of zgrey/ASAP). The script refuses a zip whose SHA-256 differs from the table above. It checks every forces file for the flow conditions and the CL/CD lines, checks that the runs are numbered without gaps, and checks that each CSV reads back with `np.loadtxt` equal to the values written, bit for bit. The shipped files were written on 2026-10-30 with Python 3.13.18 and numpy 2.5.4. `--check` reports them identical, and `--verify` finds the same 28 dropped rows and lists the 10 near-ties. The design `.mat` files for `--mat` are tracked in zgrey/ASAP under `AUTO/designs/` (in a sparse checkout of the zips, add them with `git -C ASAP checkout HEAD -- AUTO/designs`); `--verify --mat` reports all three identical bit for bit. There is no randomness beyond the authors' fixed seeds. Running SU2 again is a different matter: see the EMA notes kept outside the repository.

## Accuracy

- **Conversion**: exact (bit for bit, as above). C_l and C_d carry SU2's six printed significant digits. The smallest |C_l| values, near 1e-5, therefore have few correct digits.
- **Outputs**: the configuration posted in `AUTO/inv_airfoil.cfg` (2019) asks for a residual reduction of 8 orders or 10,000 iterations, with a Cauchy criterion on drag (ε = 1e-8, 100 iterations). Whether a run converged is not recorded in the forces files. The paper reports no mesh-convergence study. That file also sets `GAS_CONSTANT= 287.87`, while the 2016 output files report 287.058 (`STANDARD_AIR`), so the 2016 configuration was not identical to the posted one.
- **Inputs**: the generator is verified bit for bit against the authors' later design files, and the seeds and rows against the outputs (R² values above). The inputs are exact unless one of the 10 near-tie PARSEC drops is off by one or two rows (effect on the directions below 5e-5). Physical values are approximate ("Physical values").

## How these files relate to the paper

- **The same runs.** The SU2 values are the authors' own output, and we pool the three sets as `mixsamples.m` does. With the paper's estimator, the notebook reproduces the paper's figures. Eigenvalues agree with Figs. 3 and 8 to within the reading error (about 10–20 %). Eigenvector components agree with Figs. 5 and 10 to ≤ 0.007. The bootstrap subspace errors agree with Figs. 4 and 9. The subspace error against N agrees with Fig. 12 at N = 100 and 6400 (notebook table).
- **N = 6500.** CST has 6500 runs. PARSEC has 6472, because 28 designs failed the shape check and were never run. The paper's "6500" for PARSEC counts the designs drawn.
- **Angle of attack.** All posted runs are at 0°. The runs at other angles that the paper mentions are not in these zips.
- **What is not here.** The design `.mat` files (inputs regenerated instead), the meshes and flow fields, the runs behind Figs. 14–19, and the 2016 shape and sampling scripts.

## License and credit

The source repository, zgrey/ASAP, states **no licence**. It has no LICENSE file, and its README does not mention one; checked on 2026-10-30. The only licence file in it, `AUTO/paretopoints/license.txt`, is a BSD-style notice by Eduard Polityko for the bundled `prtp.m` Pareto routine and does not cover the data. The repository owner (P. Constantine, who is also an author of the paper) decided on 2026-10-30 that data without a licence may be hosted here. The C_l and C_d values are therefore redistributed unchanged, with credit to Z. J. Grey and P. G. Constantine and the zgrey/ASAP repository (commit `79ae7ed`, SHA-256 digests above). This repository's CC BY 4.0 data licence (`LICENSE-DATA`) does not apply to those values until their authors license them. The regenerated inputs are produced by `convert_data.py` from the authors' seeds. Code here (`convert_data.py`, the notebook): MIT (`LICENSE`). No ASAP script was copied; the scripts were read only to document the setup.

Cite: Z. J. Grey and P. G. Constantine, "Active Subspaces of Airfoil Shape Parameterizations", *AIAA Journal* 56(5):2003–2017, 2018, doi:10.2514/1.J056054.
