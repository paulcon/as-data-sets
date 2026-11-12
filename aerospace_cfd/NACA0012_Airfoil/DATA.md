# NACA0012_Airfoil: data

Kind: **stored data**. SU2 computed the values and gradients in `NACA0012.txt`, and the file is the only record of those runs. It must not be regenerated or corrected. The notebook drops rows by a stated rule and leaves the file unchanged.

## Files

| file | contents | role |
|---|---|---|
| `NACA0012_Airfoil.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2015; outputs committed |
| `NACA0012.txt` | CSV with a header line, 1756 rows × 57 columns, `%.16e` | SU2 output (stored); md5 d4f2187e2da2a935b25b012896b67050 |

| columns | meaning | units and coordinates |
|---|---|---|
| `Run` | run number, 1–1756 | — |
| `x01`–`x18` | heights of 18 Hicks–Henne bump functions; 0 is the baseline NACA0012 | physical, in [−0.01, 0.01] (the book, §5.3.1: each parameter in [−0.01, 0.01]). The bump heights' unit (e.g. fraction of chord) is not recorded |
| `Lift`, `Drag` | lift and drag coefficients C_L, C_D | dimensionless |
| `dlift01`–`dlift18`, `ddrag01`–`ddrag18` | SU2 adjoint gradients ∂C_L/∂x_i, ∂C_D/∂x_i | with respect to the **physical** inputs |

- **Duplicate.** Run 945 repeats run 1, the nominal design (all inputs 0), with identical outputs and gradients. The file seems to join two batches, runs 1–944 and 945–1756, each starting at the nominal design. The notebook drops every row whose inputs repeat an earlier row's, which removes run 945 only, and analyzes the other 1755.

## Model

- **Code.** SU2, the Stanford Aerospace Design Laboratory's compressible flow solver, with its adjoint solver for the gradients (the book, p. 82). The book describes the flow only as "transonic, compressible".
- **Who ran it.** Trent Lukaczyk (Stanford University) ran SU2. Paul Constantine committed the file and the first notebook to this repository on 2014-10-27 (commit 51a93cd).
- **Inputs** (m = 18). The heights of 18 Hicks–Henne bump functions that deform the airfoil, each in [−0.01, 0.01], 0 being the baseline NACA0012 (columns `x01`–`x18` under "Files").
- **Quantities of interest.** The lift and drag coefficients C_L and C_D. Each output has its own active subspace.
- **Gradient coordinates.** The gradients are with respect to the physical inputs: the slope of a least-squares linear fit of each output on the physical inputs equals the mean gradient (ratio 1.01, correlation 0.999; printed by the notebook). With normalized inputs z ∈ [−1, 1]¹⁸, x = 0.01 z, so ∂f/∂z_i = 0.01 ∂f/∂x_i. `ac.active_subspace(space=ac.Uniform(np.full(18, -0.01), np.full(18, 0.01)), grads=G)` applies that factor, so its eigenvalues are 10⁻⁴ times those of the book's Fig. 5.11, which used the raw gradients. The eigenvectors are the same, because all 18 inputs have the same range.
- **Distribution.** The inputs look uniform on the box (standard deviation 0.00576, against 0.01/√3 = 0.00577). How they were drawn (independent uniform, Latin hypercube, …) is not recorded.

## Regenerating

The data cannot be regenerated from this repository. The notebook reads `NACA0012.txt` and computes everything from it in a few seconds. The only randomness is the bootstrap (200 replicates for the full analysis, 1000 for the book's figure), and it all comes from one `np.random.default_rng(2015)`. Two runs give identical printed output.

## Accuracy

No accuracy numbers are recorded for these runs: the SU2 version, the mesh and the adjoint settings are not known (Unknowns). The notebook prints one check on the gradients, the comparison of a linear fit's slope with the mean gradient under "Model".

## How these files relate to the paper

- **The book's samples.** Constantine (2015), §5.3.1, uses M = 122 gradient samples (Algorithm 3.3 with k = 7, α = 6) and does not say which. The eigenvalues of rows 1–122 match its Fig. 5.11, and the notebook reproduces that figure and Fig. 5.12 from them. That these are the book's rows is inferred from the match, not recorded.
- **Eigenvalue scale.** The book's eigenvalues were computed from the raw (physical) gradients, so they are 10⁴ times the notebook's (see "Gradient coordinates" under "Model").
- **Drag dimension.** With the first 122 samples the book (p. 83) calls drag "one- or four-dimensional"; on all the kept samples the notebook uses n = 2 for drag.

## Unknowns

Not stated in the book or the AIAA 2014-1171 paper:

- The flow conditions: Mach number, angle of attack, Euler or RANS.
- The SU2 version, the mesh, and the adjoint settings.
- Where the second batch (runs 945–1756) came from.
- Which bump each of `x01`–`x18` is: upper or lower surface, and chord station.
- The unit of the bump heights (e.g. fraction of chord).
- How the inputs were drawn (independent uniform, Latin hypercube, …).
- Which samples the book used; rows 1–122 are inferred from the match with Fig. 5.11.

## License and credit

Code: MIT (`LICENSE`). Data (`NACA0012.txt`): CC BY 4.0 (`LICENSE-DATA`). Credit Trent Lukaczyk (SU2 runs) and cite P. G. Constantine, *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies*, SIAM, 2015, §5.3.1, doi:10.1137/1.9781611973860. Background on the method applied to shape optimization: T. W. Lukaczyk, F. Palacios, J. J. Alonso and P. G. Constantine, "Active subspaces for shape optimization", AIAA 2014-1171, doi:10.2514/6.2014-1171.
