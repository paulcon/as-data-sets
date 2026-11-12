# OpenAeroStruct_Wing_Aerostructural: data

Kind: **stored data, generated for this repository with an outside code, OpenAeroStruct 2.12.0 on OpenMDAO 3.45.2; regeneration script included**. `oas_wing_driver.py` sets up the analysis and `generate_data.py` draws the inputs and writes the three CSVs; they record exactly how the files were made and reproduce them where OpenAeroStruct is installed (see Regenerating). The shipped files stay the record. **Regenerating the data, or rerunning the accuracy checks (`check_accuracy.py`), needs OpenAeroStruct 2.12.0** (`pip install openaerostruct==2.12.0`, which pulls OpenMDAO); both scripts stop with an error that says what to install when it is missing. The repo's `.venv` has no OpenAeroStruct. The notebook reads the files and needs only numpy, matplotlib and active_subspaces.

## Files

| file | contents | role |
|---|---|---|
| `OpenAeroStruct_Wing_Aerostructural.ipynb` | the case study (Python 3, active_subspaces v1); analyses the 805 rows with failure ≤ 0 and, for comparison, all 2000 | executed top to bottom in `.venv` with seed 2018; outputs committed; run time about 6 s |
| `oas_wing_driver.py` | `OASWing` (`evaluate`, `evaluate_with_gradient`), `NAMES`, `QOIS`, `LOWER`, `UPPER`, `HELD` | the psdr test function rebuilt on OpenAeroStruct 2.x (imports `openaerostruct` only when a model is built); the CSV headers name it by its earlier file name, `oas_wing.py` |
| `generate_data.py` | draws the inputs (seed 2018) and writes the three CSVs; `--rows a:b --out DIR [--check]` for a chunk or a checked subset, `--merge DIR ...` to join chunks | regenerates the data (needs OpenAeroStruct 2.12.0) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" and "How these files relate" below (`--psdr DIR` for the comparison with psdr's test points, `--slow` for the two long checks) | accuracy checks (needs OpenAeroStruct 2.12.0) |
| `inputs.csv` | 2000 × 12 physical inputs, columns in the order of `NAMES` | Monte Carlo design |
| `outputs.csv` | 2000 × 5: fuelburn (kg), structural_mass (kg), lift (N), total_mass (kg), failure (-) | model output |
| `gradients.csv` | 2000 × 60: column 12 i + j is ∂QoI_i/∂x_j with respect to the **physical** inputs | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are written with 16 significant digits, outputs with 13 and gradients with 11. The stored inputs are therefore the float64 inputs the model was run at, rounded to 16 significant digits: they differ from the design drawn from the seed by at most 1.8e-15 of the half-range, and 62 % of the entries are exact (`check_accuracy.py` check 0, which runs without OpenAeroStruct).

## Model

**Citation.** J. P. Jasa, J. T. Hwang, J. R. R. A. Martins, "Open-source coupled aerostructural optimization using Python", *Structural and Multidisciplinary Optimization* 57(4):1815–1827, 2018, https://doi.org/10.1007/s00158-018-1912-8 (publisher PDF read). The paper describes OpenAeroStruct (https://github.com/mdolab/OpenAeroStruct). It couples a vortex-lattice method (Sec. 2.2) with 6-DOF spatial beam elements (Sec. 2.3), solves the coupled system with OpenMDAO, and gets total derivatives by the coupled adjoint (Sec. 3). Fuel burn comes from the Breguet range equation (Sec. 3.4, p. 1823). Table 1 (p. 1826) lists the design variables (spar thickness, twist, α, root chord, taper) and the KS failure constraint. **The paper contains no parameter study**, since it is an educational article.

**The parameter study** is the `OpenAeroStruct` test function of J. Hokanson's psdr package: `psdr/demos/oas.py` (inputs and ranges) and `psdr/demos/oas/oas_driver.py` (problem setup and outputs), from https://github.com/jeffrey-hokanson/PSDR (AGPL-3.0; read 2026-10-31 and, for `check_accuracy.py`, 2026-11-02 at commit a3103a2, the head of `master`). psdr runs it in the Docker image `jeffreyhokanson/oas:v1`, built from OpenAeroStruct v1 (Hokanson's fork) on OpenMDAO 1 under Python 2. psdr's `Dockerfile` names OpenMDAO 1 commit `d71f7477` and fork commit `8ee82d3e6edf87b9d8435773c4b4531e1c66ca2b` in `ARG OM_VER` / `ARG OAS_VER`, but it declares both *before* `FROM`, so they are out of scope in the build steps: the fork is cloned without any `git checkout`, and `git checkout ${OM_VER}` expands to a bare `git checkout`. The image was therefore built from the branch heads at build time, not from pinned commits. The fork's head is still `8ee82d3` (2018-06-20), and swapping v1's load transfer into the rebuild reproduces psdr's stored outputs (below), so the image very likely used that fork commit; the OpenMDAO 1 commit is inferred, not pinned. This data set does not use the image: `oas_wing_driver.py` rebuilds the same analysis on the maintained OpenAeroStruct 2.12. The settings come from `oas_driver.py` and from the v1 defaults in that fork's `run_classes.py`. No psdr or v1 code is copied.

**Inputs** (independent uniform; source: psdr `oas.py`). psdr states units only for E, G and density; it passes twist and thickness directly to v1's `twist_cp` (deg) and `thickness_cp` (m), and with these units the rebuild with v1's load transfer reproduces psdr's lift to 1.9e-8 (`check_accuracy.py` check 7), so they are confirmed:

| name | units | range | OAS variable | source |
|---|---|---|---|---|
| twist_1, twist_2, twist_3 | deg | U[−1, 1] each | `twist_cp` (tip → root) | psdr `oas.py` |
| thickness_1, thickness_2, thickness_3 | m | U[0.005, 0.05] each | spar tube wall `thickness_cp` (tip → root), v1 convention (below) | psdr `oas.py` |
| root_chord | – (× 1 m) | U[0.7, 1.3] | `chord_cp = [taper_ratio, 1] × root_chord` | psdr `oas.py`, `oas_driver.py` |
| taper_ratio | – | U[0.75, 1.25] | as above | psdr `oas.py` |
| alpha | deg | U[2, 5] | angle of attack | psdr `oas.py` |
| E | Pa | U[0.8, 1.2] × 70e9 | spar Young's modulus | psdr `oas.py` |
| G | Pa | U[0.8, 1.2] × 30e9 | spar shear modulus | psdr `oas.py` |
| mrho | kg/m³ | U[0.8, 1.2] × 3e3 | spar material density | psdr `oas.py` |

**Held settings** (psdr `oas_driver.py` and the OAS v1 defaults): a rectangular wing with span 10 m and base chord 1 m, `num_y = 7` (4 nodes and 3 elements on the symmetric half), `num_x = 2`, uniform spanwise spacing, and symmetry. The spar is a tube at 35 % chord. Its outer radius is fixed at 0.06 m, because v1 computes it once as t/c · chord / 2 = 0.12 · 1 m / 2 from the undeformed mesh and does not update it when the chord changes. This is faithful to v1 but not physical at small chords: at c_r = 0.7 and taper 0.75 the tip airfoil is 0.12 × 0.525 m = 0.063 m deep, about half the 0.12 m spar diameter. The aerodynamics are inviscid (no viscous or wave drag, CL0 = CD0 = 0) with S_ref of the wetted type. The flight condition is M = 0.84 at a = 295.4 m/s (V = 248.1 m/s), ρ∞ = 0.38 kg/m³, CT = 9.80665 × 17e-6 s⁻¹, R = 11.165e6 m and W0 = 1000 kg. The failure output uses OpenAeroStruct's yield/2.5 convention (yield stress 500 MPa): the paper's constraint is KS σ_2.5g ≤ σ_yield (Table 1, p. 1826), with 2.5 the 2.5 g sizing load, and OpenAeroStruct evaluates it as the stress at the analysed lift against yield/2.5. With the lift untrimmed (lift/weight 0.89–7.63) the two are not equivalent. The KS ρ is 100. There is no structural-weight relief, and the wing-weight ratio is 1. The lift is not trimmed to the weight; psdr only analyzes.

**Reconstruction choices**, each checked against psdr's 10 stored test outputs (`tests/oas.input`, `tests/oas.output` in the psdr repository) unless noted:

1. *B-spline convention.* OAS v1 evaluates a 3-control-point, order-3 B-spline at equally spaced parameters from tip to root: at the 4 nodes for twist, and at 3 points for the 3 element thicknesses. The element thicknesses are therefore [t1, (t1 + 2 t2 + t3)/4, t3]. OAS 2.x evaluates thickness at element midpoints instead, so `oas_wing_driver.py` maps the v1 control points exactly to OAS 2.x control points with a fixed 3 × 3 matrix. Twist and chord use the same evaluation points in both versions. Reversing the order of the thickness control points or dropping the map raises the largest fuel-burn difference from psdr's 10 test outputs from 0.35 % to 5.2 % and 2.2 % (`check_accuracy.py` check 7).
2. *Fixed spar radius* (above), set with `radius_cp`.
3. *Outputs.* psdr returns, in order, `fuelburn` (kg), `wing_perf.structural_weight` (N), `wing_perf.L` (N), `total_weight` (N) and `wing_perf.failure`. Here the two weights are masses in kg: structural_mass, and total_mass = W0 + structural_mass + fuelburn. v1 computed the structural weight with g = 9.81, so psdr's value is 9.81 × structural_mass, which agrees with these files to rounding (4e-16 relative; check 7).
4. *Material inputs.* OAS 2.x takes E, G and density as surface options, not OpenMDAO variables. `oas_wing_driver.py` builds the problem once and writes new values into the surface dictionaries and into the von Mises component's cached E and G. At rows 0–2 this gives the same outputs and gradients as a fresh build, to 7e-14 and 2e-14 relative (rounding; `check_accuracy.py` check 5).
5. *Coupled solver.* OAS's NonlinearBlockGS with Aitken acceleration, with atol 1e-10, rtol 1e-13 and at most 200 iterations; non-convergence raises an error. psdr used v1's default solver. A Newton solver with a direct linear solve failed when the data were made (singular Jacobian), so it is not used.

**Quantities of interest:** fuel burn W_f = (W0 + W_s)[exp(R·CT/V · CD/CL) − 1] (paper Sec. 3.4) in kg. This is the main QoI; psdr's documentation does not say which output it intends. The other outputs are the spar mass, lift, total mass, and failure (KS of von Mises stress / (500 MPa / 2.5) − 1 at the analysed lift; > 0 means the spar fails).

**Gradients**, all with respect to the physical inputs:
- the 9 geometric and flight inputs: OpenMDAO `compute_totals` (reverse mode, direct linear solve of the coupled system; the coupled adjoint the paper describes), chained through the thickness map and through chord_cp = [taper_ratio · root_chord, root_chord];
- E and G: central differences with relative step 1e-4;
- mrho: exact, because density enters only the spar mass (no weight relief): ∂m_s/∂ρ_m = m_s/ρ_m and ∂W_f/∂ρ_m = W_f/(W0 + m_s) · m_s/ρ_m. Lift and failure do not depend on it.

**Sampling design:** M = 2000 independent uniform Monte Carlo samples, `numpy.random.default_rng(2018).random((2000, 12))` scaled to the ranges (167 per input). No solve failed, and fuel burn, spar mass and lift are positive at every sample.

**Plausible wings: the rows the notebook analyses.** psdr's box contains many wings that would not be built. 1195 of the 2000 samples have failure > 0 (the spar exceeds the allowable stress at the analysed lift; maximum 6.74), and lift/weight ranges from 0.89 to 7.63 (the lift is not trimmed; 1 sample below 1). The samples with the thinnest root spars are close to static divergence: the spar at 35 % chord sits behind the aerodynamic center at 25 %, so lift twists the wing nose-up. Inside the box but outside the sample, the corner with every thickness 5 mm, c_r = 1.3, taper 1.25, α = 5°, twist +1°, E and G at their minimum and nominal density gives fuel burn 843.3 kg, lift 329.7 kN (17.7 times the weight) and failure 25.80 (`check_accuracy.py` check 6). These are converged analyses, not model errors. **The notebook drops the rows with failure > 0 and analyses the 805 with failure ≤ 0; the files keep every row** (stored data). The rule is the paper's structural constraint (Table 1, p. 1826), it uses only a stored output, and it removes the near-divergent wings: on the kept rows the root thickness is at least 8.2 mm, lift/weight at most 3.47 and fuel burn at most 111.1 kg (all rows: 5.0 mm, 7.63 and 261.8 kg), and the dropped rows hold 84.2 % of trace(C) of fuel burn (notebook). It is close to a cut on the root thickness: it keeps 1 % of the rows with a root spar under 10 mm and 58 % of those over 30 mm. A lift/weight criterion is not used: a cruise condition L ≈ W would keep only a few dozen rows (37 have lift/weight ≤ 1.5). The rule applies OpenAeroStruct 2.12's failure value; with v1's load transfer (psdr's) it would keep 645 rows, since the two transfers give failure of opposite sign at 160 rows, all of them among the 805 (`check_accuracy.py` check 8).

**What the rule changes** (notebook). On the 805 kept rows fuel burn has a one-dimensional active subspace: λ1/λ2 = 49.1 (bootstrap 95 % interval [45.2, 53.5]), 46.1 to 56.7 over the kept rows of the four disjoint blocks of 500 file rows, and 45.7 on the 154 rows with failure ≤ 0 of an independent 400-sample design (`default_rng(99)`, `check_accuracy.py` check 9). w1 is led by α, the root chord, the twist and the taper; the root spar thickness has a component of 0.03. The gradients are not heavy-tailed (the largest sample holds 0.3 % of trace(C), the largest 5 % hold 8.7 %). On all 2000 rows (the notebook's last section) the fuel-burn gradients are heavy-tailed (the largest sample holds 6.3 % of trace(C), the largest 1 % hold 31.9 %, and the 90 samples with root thickness < 7.5 mm hold 39.7 %), the eigenvalues change between draws (over four disjoint blocks of 500 rows, λ1/λ2 from 2.08 to 4.18 and λ2/λ3 from 13.5 to 42.4; the independent 400-sample design gives 6.4 and 8.6), and only a two-dimensional subspace led by the root thickness, α and the root chord is stable (at M = 2000: λ1/λ2 = 2.37, λ2/λ3 = 28.4). The plausible w1 lies inside that plane (projection length 0.996).

**Published results:** none. The paper reports no sensitivity analysis or active subspace, and no published active subspace analysis of this setup was found.

## Regenerating

```bash
python3 -m venv oasvenv && oasvenv/bin/pip install openaerostruct==2.12.0    # pulls openmdao (3.45.2 when the files were made)
oasvenv/bin/python generate_data.py                                          # all 2000 samples, one process, writes here
oasvenv/bin/python generate_data.py --rows 0:10 --out DIR --check            # a subset, compared with the shipped files
oasvenv/bin/python check_accuracy.py --psdr PSDR/tests --slow                # the accuracy numbers below
```

`generate_data.py` stops if OpenAeroStruct is missing (in `.venv` it exits with status 3 and the install message) or if another version is installed (override with `--any-oas`); `--check` and `--rows` need `--out` outside this directory, so that the shipped files are never overwritten.

**Provenance of the shipped files.** Written on 2026-10-31 in a Linux container (2 CPUs shared with other jobs; Python 3.13, OpenAeroStruct 2.12.0, OpenMDAO 3.45.2, numpy 2.5.4, scipy 1.18.1), one process at a time, with `nice -n 10`, in four chunks of 500 rows joined with `generate_data.py --merge c0 c1 c2 c3` (which checks that the chunk inputs are the design rows in order). Run time of that original run: 1573 s wall in all (420.8, 487.8, 362.0 and 302.3 s for the four chunks; about 0.29 s of CPU per sample in the last three). One evaluation costs about 0.05–0.3 s; the 4 extra solves for the E and G differences and the total-derivative solve make up the rest of a sample. There, `--rows 0:10 --check` and `--rows 1995:2000 --check` reproduced outputs and gradients bit for bit at the digits written.

## Accuracy

`check_accuracy.py` recomputes every number in this section and in the next one that the notebook does not print. It needs OpenAeroStruct 2.12.0; the values below are from a run in a separate environment on macOS x86-64 (Python 3.11.16, OpenAeroStruct 2.12.0, OpenMDAO 3.45.2, numpy 2.4.6, scipy 1.17.1; 2026-11-02; 604 s with `--psdr` and `--slow`, 71 s with `--psdr` alone, on a shared machine). The checks of `.venv` itself (the notebook) do not need it.

- **Reproduction on another platform** (check 1). At rows 0, 1, 2, 100, 250 and 499, the outputs recomputed with `oas_wing_driver.py` differ from the files by at most 4.1e-13 relative and the gradients by 2.1e-11 (relative 2-norm), against 13 and 11 significant digits written: the files reproduce to rounding and solver tolerance.
- **Gradients vs. independent central differences** of the whole model (check 1), with step 1e-5 × half-range in every input, at the same six rows (root thicknesses from 5.4 mm to 42 mm). The relative error of the normalized gradient vector is at most 5.3e-9 (fuel burn), 7.3e-11 (spar mass), 5.2e-9 (lift), 7.2e-9 (total mass) and 4.4e-9 (failure).
- **E and G step** (check 2): relative steps 1e-4 and 1e-3 give E and G columns that agree to 4.2e-6 relative to the largest entry of each column, at the same rows. These columns are small: at the nominal point fuel burn changes by 0.33 kg over the whole E range, against 85.4 kg over the α range (check 6).
- **Fourth-order central differences** (check 3; step 1e-3 × half-range) at rows 23, 241 and 249 agree with the shipped gradients to a relative 6.7e-10 or better; at row 56, the largest-gradient row (root thickness 5.3 mm, near divergence), to 4.7e-9 with step 1e-4 × half-range (1.2e-7 with step 1e-3 × half-range, whose truncation error is larger at this near-divergent row).
- **Solver tolerance** (check 4): re-solving rows 0–2 with atol 1e-11 changes the outputs by at most 3.5e-13 relative.

## How these files relate to the paper

- **Not the paper's computation.** The paper has no parameter study. These data are a sampling study of psdr's test function, which uses the paper's design variables (Table 1) plus three material constants, run with today's OpenAeroStruct.
- **Agreement with psdr's own outputs.** On psdr's 10 stored test points (normalized inputs in `tests/oas.input`, mapped to the ranges above; `tests/oas.input` sha256 bd3f4e73…, `tests/oas.output` 584b1f54…, read from the psdr repository and not shipped), the files' model gives (`check_accuracy.py` check 7):

  | output | stock OAS 2.12 (these files) | OAS 2.12 with v1's load transfer (check only) |
  |---|---|---|
  | fuel burn, max relative difference | 3.5e-3 | 8.2e-5 |
  | structural weight (9.81 × mass), max relative difference | 4e-16 | 4e-16 |
  | lift, max relative difference | 3.9e-3 | 1.9e-8 |
  | total weight (9.81 × spar mass + 9.80665 × (W0 + fuel burn)), max relative difference | 2.6e-4 | 6.6e-6 |
  | failure, difference (OAS 2.12 − psdr) | −0.139 to −0.068 (same sign at 8 of 10) | ≤ 4.2e-7 in magnitude |

  The remaining difference comes from the load transfer. OAS 2.x (`transfer/load_transfer.py`) gives each panel's aerodynamic moment to its two beam nodes with arms from the panel's aerodynamic center to each node. OAS v1 (`transfer.py`, `TransferLoads`) uses one arm to the element midpoint, split equally. The total moment is the same, but the bending-moment distribution differs, which lowers the KS stress ratio by 0.07–0.14 at these points. Replacing the compute method of OAS 2.12's `LoadTransfer` with the v1 formula (re-typed in `check_accuracy.py` from v1's `transfer.py` at the fork commit 8ee82d3; used for this check only) reproduces psdr's lift and failure to rounding. The rest of the fuel-burn difference (8e-5) is v1's g = 9.81 in the structural weight. psdr's total weight fits 9.81 × spar mass + 9.80665 × (W0 + fuel burn) (v1's g in the structural weight only) to 6.6e-6 with v1's transfer, so the total-weight row is compared in that form; with 9.80665 × total mass the two columns would read 3.0e-4 and 8.2e-5.

  **Across the design the difference is larger** than at psdr's 10 points, because it grows on the thin-root-spar samples. Running all 2000 shipped inputs with the v1 transfer (`check_accuracy.py` check 8, 217 s), the stock 2.12 values differ from the v1-transfer values as follows: fuel burn by a median of 0.14 %, a 90th percentile of 0.52 % and a maximum of 13.0 % (row 56, root thickness 5.3 mm, the sample with the largest gradient); lift by up to 12.3 %; and failure is lower at every sample, by 0.023 to 1.86 (median 0.10). The v1 transfer gives failure > 0 at 1355 of the 2000 samples against 1195 here, and the sign of failure differs at 160 samples (8.0 %). **On the 805 rows the notebook analyses the difference is small**: fuel burn by a median of 0.08 %, a 90th percentile of 0.16 % and at most 0.30 %, lift by at most 0.3 %, and failure by 0.023 to 0.112 (median 0.072); all 160 sign changes are among these rows, where v1's failure is slightly positive.

  The shipped data use the **unmodified** OAS 2.12, so the model is the maintained code that anyone can install. Its inputs, ranges and spar mass are psdr's. Its fuel burn, lift and total mass agree with psdr's to within 0.4 % at psdr's 10 test points; against the same model with v1's load transfer (psdr's), they agree to within 0.3 % on the plausible wings the notebook analyses but only to within 13 % across the whole design box. Its failure output is not psdr's failure output.
- **Which psdr output is "the" QoI** is not stated in psdr's documentation. Fuel burn is the main QoI here because it is the paper's objective and psdr's first column.

## Unknowns

- The exact build of psdr's Docker image: its `Dockerfile` pins no commit in effect. The OAS v1 fork commit `8ee82d3` is very likely (the rebuild with v1's load transfer reproduces psdr's stored outputs); the OpenMDAO 1 commit is inferred, not pinned.
- Which psdr output is "the" QoI: psdr's documentation does not say.

## License and credit

Code (`oas_wing_driver.py`, `generate_data.py`, `check_accuracy.py`, the notebook): MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were computed with OpenAeroStruct (Apache-2.0; J. P. Jasa, J. T. Hwang, J. R. R. A. Martins and contributors) and OpenMDAO (Apache-2.0). The parameter study is J. Hokanson's psdr `OpenAeroStruct` test function (AGPL-3.0); its settings are described and reimplemented here, and none of its code is copied; `check_accuracy.py` reads psdr's two test files from a psdr clone, which is not shipped. `check_accuracy.py` re-types the few lines of OAS v1's load-transfer formula, for a check only. Cite Jasa, Hwang and Martins (2018), https://doi.org/10.1007/s00158-018-1912-8, and credit psdr (https://github.com/jeffrey-hokanson/PSDR) for the study design.
