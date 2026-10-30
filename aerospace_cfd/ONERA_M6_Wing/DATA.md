# ONERA_M6_Wing: data

Kind: **stored data**. SU2 computed the values and gradients in `ONERA-M6.txt`, and the file is the only record of those runs. It must not be regenerated or corrected.

## Files

| file | contents | role |
|---|---|---|
| `ONERA_M6_Wing.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2014; outputs committed |
| `ONERA-M6.txt` | CSV with a header line, 297 rows × 153 columns, `%.16e` | SU2 output (stored); md5 689d8463d4824537dcf953de7ddc5969 |
| `baseline_ffdbox_original.png`, `baseline_ffdbox_deformed.png` | the FFD box around the baseline wing, before and after a deformation (1200 × 1067) | illustrations in the notebook's first cell |
| `baseline_surface_flow.png` | surface pressure contours on the baseline wing from SU2 (1200 × 1067) | illustration in the notebook's first cell |

## Columns

| columns | meaning | units and coordinates |
|---|---|---|
| `Run` | run number, 1–297 | — |
| `x01`–`x50` | free-form deformation (FFD) design variables; 0 is the baseline ONERA-M6 wing | physical, in [−0.05, 0.05] (Lukaczyk et al. 2014, §IV). The unit of the control-point displacements is not recorded |
| `Lift`, `Drag` | lift and drag coefficients C_L, C_D | dimensionless |
| `dlift01`–`dlift50`, `ddrag01`–`ddrag50` | SU2 adjoint gradients ∂C_L/∂x_i, ∂C_D/∂x_i | with respect to the **physical** inputs |

- **Gradient coordinates.** The gradients are physical: the slope of a least-squares linear fit of each output on the physical inputs is close to the mean gradient (ratio 1.10 for lift, 0.91 for drag; printed by the notebook). Normalized gradients would give a ratio near 20. With normalized inputs z ∈ [−1, 1]⁵⁰, x = 0.05 z, so ∂f/∂z_i = 0.05 ∂f/∂x_i. `ac.active_subspace(space=ac.Uniform(np.full(50, -0.05), np.full(50, 0.05)), grads=G)` applies that factor, so its eigenvalues are 0.05² = 2.5 × 10⁻³ times those of the 2016 notebook, which used the raw gradients. The eigenvectors are the same, because all 50 inputs have the same range.
- **Design.** The paper samples the inputs with a 300-point Latin hypercube design. The file agrees: in every column, the 297 values fall in 297 distinct strata of the 300 (checked in the notebook), and their standard deviation is 0.02886, against 0.05/√3 = 0.02887 for the uniform distribution.

## Provenance and unknowns

- **Source.** These are the runs of T. W. Lukaczyk, F. Palacios, J. J. Alonso and P. G. Constantine, "Active subspaces for shape optimization", AIAA 2014-1171, §IV: the ONERA-M6 wing in inviscid flow (the SU2 Euler solver) at free-stream Mach 0.8395 and angle of attack 3.06°, with lift and drag gradients from the SU2 adjoint solver. The paper's Fig. 1 gives C_L = 0.2864 and C_D = 0.0118 for "a typical direct solution", and its lift constraint is C_L ≥ 0.2864 (eq. 1). The file has no run at the baseline design (all inputs 0).
- **Code.** SU2, the Stanford Aerospace Design Laboratory's open-source suite (flow, adjoint, and FFD mesh deformation). The design variables are control points of a Bézier box (FFD) enclosing the wing (the paper's §II).
- **Who ran it.** Not recorded. The paper's authors are credited; the runs were presumably made at Stanford by Trent Lukaczyk, the first author. Paul Constantine committed the file, with a first notebook, in this repository's initial commit on 2014-10-23 (7c26e3a), and the file has not changed since. Ryan Howard rewrote the notebook and added the three PNG files in 2016 (79789d9); where the PNGs were rendered is not recorded.
- **Unknown** (author, 2026-10-30), and not stated in the paper:
  - which 3 of the 300 Latin hypercube runs are missing, and why. The paper says the bounds were shrunk "until a majority of the flow evaluations would successfully converge", so they may be failed runs; that is a guess;
  - which FFD control point, and which displacement direction, each of `x01`–`x50` is;
  - the SU2 version, the mesh, and the solver and adjoint settings.
- **A different ONERA-M6 data set.** Constantine (2015), *Active Subspaces*, §5.3.2, studies the ONERA-M6 wing with 60 FFD parameters, 248 runs and no gradients. Those data are not in this repository and are not available (author, 2026-10-30).

## Regenerating

The data cannot be regenerated from this repository. The notebook reads `ONERA-M6.txt` and computes everything from it in about 15–20 s. The only randomness is the bootstrap (500 replicates), and it all comes from one `np.random.default_rng(2014)`. Two runs give identical printed output.

## License and credit

Code: MIT (`LICENSE`). Data (`ONERA-M6.txt` and the PNG files): CC BY 4.0 (`LICENSE-DATA`). Credit the authors of, and cite, T. W. Lukaczyk, F. Palacios, J. J. Alonso and P. G. Constantine, "Active subspaces for shape optimization", 10th AIAA Multidisciplinary Design Optimization Conference, National Harbor, MD, 2014, AIAA 2014-1171, doi:10.2514/6.2014-1171.
