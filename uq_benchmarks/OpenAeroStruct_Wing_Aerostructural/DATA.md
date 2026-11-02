# OpenAeroStruct_Wing_Aerostructural: data

Kind: **stored data**, generated with an outside code: OpenAeroStruct 2.12.0 on OpenMDAO 3.45.2. The notebook needs only numpy, matplotlib and active_subspaces. `generate_data.py` and `oas_wing.py` record exactly how the files were made, and they reproduce the files when OpenAeroStruct is installed (see Regenerating).

## Files

| file | contents | role |
|---|---|---|
| `OpenAeroStruct_Wing_Aerostructural.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2018; outputs committed |
| `oas_wing.py` | `OASWing` (`evaluate`, `evaluate_with_gradient`), `NAMES`, `QOIS`, `LOWER`, `UPPER`, `HELD` | the psdr test function rebuilt on OpenAeroStruct 2.x (needs `openaerostruct`) |
| `generate_data.py` | draws the inputs (seed 2018) and writes the three CSVs; `--rows a:b --out DIR [--check]` for a chunk or a checked subset, `--merge DIR ...` to join chunks | regenerates the data |
| `inputs.csv` | 2000 × 12 physical inputs, columns in the order of `NAMES` | Monte Carlo design |
| `outputs.csv` | 2000 × 5: fuelburn (kg), structural_mass (kg), lift (N), total_mass (kg), failure (-) | model output |
| `gradients.csv` | 2000 × 60: column 12 i + j is ∂QoI_i/∂x_j with respect to the **physical** inputs | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are written with 16 significant digits, outputs with 13 and gradients with 11.

## Model

**Citation.** J. P. Jasa, J. T. Hwang, J. R. R. A. Martins, "Open-source coupled aerostructural optimization using Python", *Structural and Multidisciplinary Optimization* 57(4):1815–1827, 2018, https://doi.org/10.1007/s00158-018-1912-8 (publisher PDF read). The paper describes OpenAeroStruct (https://github.com/mdolab/OpenAeroStruct). It couples a vortex-lattice method (Sec. 2.2) with 6-DOF spatial beam elements (Sec. 2.3), solves the coupled system with OpenMDAO, and gets total derivatives by the coupled adjoint (Sec. 3). Fuel burn comes from the Breguet range equation (Sec. 3.4, p. 1823). Table 1 (p. 1826) lists the design variables (spar thickness, twist, α, root chord, taper) and the KS failure constraint. **The paper contains no parameter study**, since it is an educational article.

**The parameter study** is the `OpenAeroStruct` test function of J. Hokanson's psdr package: `psdr/demos/oas.py` (inputs and ranges) and `psdr/demos/oas/oas_driver.py` (problem setup and outputs), from https://github.com/jeffrey-hokanson/PSDR (AGPL-3.0; cloned 2026-10-31). psdr runs it in the Docker image `jeffreyhokanson/oas:v1`, built from OpenAeroStruct v1 (Hokanson's fork) on OpenMDAO 1 under Python 2. psdr's `Dockerfile` names OpenMDAO 1 commit `d71f7477` and fork commit `8ee82d3e6edf87b9d8435773c4b4531e1c66ca2b` in `ARG OM_VER` / `ARG OAS_VER`, but it declares both *before* `FROM`, so they are out of scope in the build steps: the fork is cloned without any `git checkout`, and `git checkout ${OM_VER}` expands to a bare `git checkout`. The image was therefore built from the branch heads at build time, not from pinned commits. The fork's head is still `8ee82d3` (2018-06-20), and swapping v1's load transfer into the rebuild reproduces psdr's stored outputs (below), so the image very likely used that fork commit; the OpenMDAO 1 commit is inferred, not pinned. Docker Hub cannot be reached here, so `oas_wing.py` rebuilds the same analysis on OpenAeroStruct 2.12. The settings come from `oas_driver.py` and from the v1 defaults in that fork's `run_classes.py`. No psdr or v1 code is copied.

**Inputs** (independent uniform; source: psdr `oas.py`). psdr states units only for E, G and density; it passes twist and thickness directly to v1's `twist_cp` (deg) and `thickness_cp` (m), and the v1-transfer diagnostic below reproduces psdr's lift to 1.9e-8 with these units, so they are confirmed:

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

**Held settings** (psdr `oas_driver.py` and the OAS v1 defaults): a rectangular wing with span 10 m and base chord 1 m, `num_y = 7` (4 nodes and 3 elements on the symmetric half), `num_x = 2`, uniform spanwise spacing, and symmetry. The spar is a tube at 35 % chord. Its outer radius is fixed at 0.06 m, because v1 computes it once as t/c · chord / 2 = 0.12 · 1 m / 2 from the undeformed mesh and does not update it when the chord changes. This is faithful to v1 but not physical at small chords: at c_r = 0.7 and taper 0.75 the tip airfoil is 0.12 × 0.525 m = 0.063 m deep, about half the 0.12 m spar diameter. The aerodynamics are inviscid (no viscous or wave drag, CL0 = CD0 = 0) with S_ref of the wetted type. The flight condition is M = 0.84 at a = 295.4 m/s (V = 248.1 m/s), ρ∞ = 0.38 kg/m³, CT = 9.80665 × 17e-6 s⁻¹, R = 11.165e6 m and W0 = 1000 kg. The yield stress is 500 MPa with safety factor 2.5, and the KS ρ is 100. There is no structural-weight relief, and the wing-weight ratio is 1. The lift is not trimmed to the weight; psdr only analyzes.

**Reconstruction choices**, each checked against psdr's 10 stored test outputs (`tests/oas.input`, `tests/oas.output` in the psdr repository) unless noted:

1. *B-spline convention.* OAS v1 evaluates a 3-control-point, order-3 B-spline at equally spaced parameters from tip to root: at the 4 nodes for twist, and at 3 points for the 3 element thicknesses. The element thicknesses are therefore [t1, (t1 + 2 t2 + t3)/4, t3]. OAS 2.x evaluates thickness at element midpoints instead, so `oas_wing.py` maps the v1 control points exactly to OAS 2.x control points with a fixed 3 × 3 matrix. Twist and chord use the same evaluation points in both versions. Reversing the order or dropping the map raises the fuel-burn error against psdr from 0.35 % to 2–12 %.
2. *Fixed spar radius* (above), set with `radius_cp`.
3. *Outputs.* psdr returns, in order, `fuelburn` (kg), `wing_perf.structural_weight` (N), `wing_perf.L` (N), `total_weight` (N) and `wing_perf.failure`. Here the two weights are masses in kg: structural_mass, and total_mass = W0 + structural_mass + fuelburn. v1 computed the structural weight with g = 9.81, so psdr's value is 9.81 × structural_mass, which agrees with these files to 2e-16.
4. *Material inputs.* OAS 2.x takes E, G and density as surface options, not OpenMDAO variables. `oas_wing.py` builds the problem once and writes new values into the surface dictionaries and into the von Mises component's cached E and G. During the build, three random samples gave the same outputs and gradients as a fresh build, to within 2e-15 (rounding).
5. *Coupled solver.* OAS's NonlinearBlockGS with Aitken acceleration, with atol 1e-10, rtol 1e-13 and at most 200 iterations; non-convergence raises an error. psdr used v1's default solver. A Newton solver with a direct linear solve failed here (singular Jacobian), so it is not used.

**Quantities of interest:** fuel burn W_f = (W0 + W_s)[exp(R·CT/V · CD/CL) − 1] (paper Sec. 3.4) in kg. This is the main QoI; psdr's documentation does not say which output it intends. The other outputs are the spar mass, lift, total mass, and failure (KS of von Mises stress / (500 MPa / 2.5) − 1; > 0 means the spar fails).

**Gradients**, all with respect to the physical inputs:
- the 9 geometric and flight inputs: OpenMDAO `compute_totals` (reverse mode, direct linear solve of the coupled system; the coupled adjoint the paper describes), chained through the thickness map and through chord_cp = [taper_ratio · root_chord, root_chord];
- E and G: central differences with relative step 1e-4;
- mrho: exact, because density enters only the spar mass (no weight relief): ∂m_s/∂ρ_m = m_s/ρ_m and ∂W_f/∂ρ_m = W_f/(W0 + m_s) · m_s/ρ_m. Lift and failure do not depend on it.

**Sampling design:** M = 2000 independent uniform Monte Carlo samples, `numpy.random.default_rng(2018).random((2000, 12))` scaled to the ranges (167 per input; the gradient estimator needs 10·m = 120). The first release had M = 500; its rows are the first 500 rows here, unchanged, because the generator draws the stream row by row. The design was enlarged because the fuel-burn gradients are heavy-tailed: a few samples with a thin root spar (near static divergence: the spar at 35 % chord sits behind the aerodynamic center at 25 %, so lift twists the wing nose-up) hold much of trace(C). In the 2000 samples the largest one holds 6.3 %, the largest 1 % hold 31.9 %, and the 90 samples with root thickness < 7.5 mm hold 39.7 %. The eigenvalues therefore change between draws: over four disjoint blocks of 500 rows, λ1/λ2 ranges from 2.08 to 4.18 and λ2/λ3 from 13.5 to 42.4 (an independent 400-sample draw by the reviewer gave 6.4 and 8.6), while the two-dimensional subspace stays within 0.012–0.037 of the 2000-sample one. The notebook reports all of these. At M = 2000: λ1/λ2 = 2.37, λ2/λ3 = 28.4 (bootstrap 95 % interval [18.4, 36.4], which understates the draw-to-draw spread).

**Samples outside the design's comfort zone** are kept, and the notebook keeps every row: 1195 of the 2000 samples have failure > 0 (the spar exceeds the allowable stress; maximum 6.74), and lift / weight ranges from 0.89 to 7.63, with 1 sample below 1. Fuel burn ranges from 28.2 to 261.8 kg. Outside the sampled points but inside the box, the near-divergence corner (all thicknesses 5 mm, c_r = 1.3, taper 1.25, α = 5°, twist +1°, E and G at their minimum) gives fuel burn 843.3 kg, lift 329.7 kN and failure 25.80 (scratch check with `oas_wing.py`). Fuel burn is psdr's untrimmed output, a function of CD/CL at the given α, not a mission fuel burn. Both are converged analyses of wings that an optimizer would reject, not model errors. No solve failed, and fuel burn, spar mass and lift are positive at every sample.

**Published results:** none. The paper reports no sensitivity analysis or active subspace, and no published active subspace analysis of this setup was found when the item was screened.

## Regenerating

```bash
pip install openaerostruct==2.12.0          # pulls openmdao; the files used openmdao 3.45.2, numpy 2.5.4, scipy 1.18.1, Python 3.13
python generate_data.py                      # all 2000 samples, one process, writes here
python generate_data.py --rows 0:10 --out /tmp/oas --check     # quick check of a subset against the shipped files
```

The shipped files were written in a Linux container (2 CPUs shared with other jobs), one process at a time, with `nice -n 10`. Rows 0–499 are the first release's run (2026-10-31, 420.8 s wall, 0.84 s per sample). Rows 500–1999 were run on 2026-10-31 in three chunks of 500 (`--rows 500:1000 --out c1`, and so on: 487.8, 362.0 and 302.3 s wall; 154.1, 142.2 and 135.4 s CPU, about 0.29 s CPU per sample) and joined with `python generate_data.py --merge c0 c1 c2 c3`, which checks that the chunk inputs are the design rows in order (max normalized difference 8.9e-16). One evaluation costs about 0.05–0.3 s; the 4 extra solves for the E and G differences and the total-derivative solve make up the rest of a sample. `--rows 0:10 --check` and `--rows 1995:2000 --check`, run with the final `oas_wing.py` and generator on the merged files, reproduced both outputs and gradients to 0.0 (bit for bit, at the digits written), and the merged first 500 rows are byte-identical to the first release.

## Accuracy

Scratch checks, not committed:

- **Gradients vs. independent central differences** of the whole model, with step 1e-5 × half-range in every input, at six shipped rows (0, 1, 2, 100, 250, 499; their root thicknesses range from 5.4 mm to 42 mm). The relative error of the normalized gradient vector is at most 5.3e-9 (fuel burn), 6.9e-11 (spar mass), 5.2e-9 (lift), 7.1e-9 (total mass) and 4.4e-9 (failure).
- **E and G step:** relative steps 1e-4 and 1e-3 give E and G columns that agree to 1.5e-6 relative to their largest entry. These columns are small: fuel burn changes by about 0.3 kg over the whole E range, against about 85 kg over the α range at the nominal point.
- **Independent review check:** 4th-order central differences of fresh solves at rows 23, 241, 249 and at the largest-gradient row 56 agree with the shipped gradients to a relative 6.7e-10 (4.7e-9 at row 56).
- **Solver tolerance:** re-solving three rows with atol 1e-11 changes the outputs by at most 3.5e-13 relative.

## How these files relate to the paper and to psdr

- **Not the paper's computation.** The paper has no parameter study. These data are a sampling study of psdr's test function, which uses the paper's design variables (Table 1) plus three material constants, run with today's OpenAeroStruct.
- **Agreement with psdr's own outputs.** On psdr's 10 stored test points (normalized inputs in `tests/oas.input`, mapped to the ranges above), the files' model gives:

  | output | stock OAS 2.12 (these files) | OAS 2.12 with v1's load transfer (diagnostic only) |
  |---|---|---|
  | fuel burn, max relative difference | 3.5e-3 | 8.2e-5 |
  | structural weight (9.81 × mass), max relative difference | 2.2e-16 | 2.2e-16 |
  | lift, max relative difference | 3.9e-3 | 1.9e-8 |
  | total weight, max relative difference | 3.0e-4 | 8.2e-5 |
  | failure, difference (OAS 2.12 − psdr) | −0.139 to −0.068 (same sign at 8 of 10) | ≤ 4.2e-7 in magnitude |

  The remaining difference comes from the load transfer. OAS 2.x (`transfer/load_transfer.py`) gives each panel's aerodynamic moment to its two beam nodes with arms from the panel's aerodynamic center to each node. OAS v1 (`transfer.py`, `TransferLoads`) uses one arm to the element midpoint, split equally. The total moment is the same, but the bending-moment distribution differs, which lowers the KS stress ratio by 0.07–0.14 at these points. Replacing the OAS 2.12 load transfer with the v1 formula, in a scratch diagnostic only, reproduces psdr's lift and failure to rounding. The rest of the fuel-burn difference (8e-5) is v1's g = 9.81 in the structural weight. The independent review re-implemented v1's `TransferLoads` and confirmed the same agreement (lift 1.9e-8, failure 4.2e-7, fuel burn 8.2e-5); this revision re-ran the swap with its own re-typing of the v1 formula and got the same three numbers.

  **Across the design the difference is larger** than at psdr's 10 points, because it grows on the thin-root-spar samples. Running all 2000 shipped inputs with the v1 transfer (scratch diagnostic, 416 s), the stock 2.12 values differ from the v1-transfer values as follows: fuel burn by a median of 0.14 %, a 90th percentile of 0.52 % and a maximum of 13.0 % (row 56, root thickness 5.3 mm, the sample with the largest gradient); lift by up to 12.3 %; and failure is lower at every sample, by 0.023 to 1.86 (median 0.10). The v1 transfer gives failure > 0 at 1355 of the 2000 samples against 1195 here, and the sign of failure differs at 160 samples (8.0 %). (The review's first estimate, on 100 of the first 500 rows, was a maximum of 4.2 % in fuel burn and 0.64 in failure.)

  The shipped data use the **unmodified** OAS 2.12, so the model is the maintained code that anyone can install. Its inputs, ranges and spar mass are psdr's. Its fuel burn, lift and total mass agree with psdr's to within 0.4 % at psdr's 10 test points, but only to within 13 % across the design. Its failure output is not psdr's failure output.
- **Which psdr output is "the" QoI** is not stated in psdr's documentation. Fuel burn is the main QoI here because it is the paper's objective and psdr's first column.

## License and credit

Code (`oas_wing.py`, `generate_data.py`, the notebook): MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were computed with OpenAeroStruct (Apache-2.0; J. P. Jasa, J. T. Hwang, J. R. R. A. Martins and contributors) and OpenMDAO (Apache-2.0). The parameter study is J. Hokanson's psdr `OpenAeroStruct` test function (AGPL-3.0); its settings are described and reimplemented here, and none of its code is copied. Cite Jasa, Hwang and Martins (2018), https://doi.org/10.1007/s00158-018-1912-8, and credit psdr (https://github.com/jeffrey-hokanson/PSDR) for the study design.
