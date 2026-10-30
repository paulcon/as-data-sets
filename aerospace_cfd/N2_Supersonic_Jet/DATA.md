# N2_Supersonic_Jet: data

Kind: **stored data**. SU2 computed the values and adjoint gradients in `LM104X.txt`, and the file is the only record of those runs. It must not be regenerated or corrected. The notebook uses every row and leaves the file unchanged.

## Files

| file | contents | role |
|---|---|---|
| `N2_Supersonic_Jet.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2015; outputs committed |
| `LM104X.txt` | CSV with a header line, 310 rows × 424 columns, `%.16e` | SU2 output (stored); md5 d0bcc428d046e8d228227d1c43af3aa5 |

The file keeps the name of the 2016 notebook ("Lockheed-Martin 104X"). What "104X" stands for is not recorded; the dissertation calls the aircraft the Lockheed N+2.

## Columns

| columns | meaning | units and coordinates |
|---|---|---|
| `Run` | run number, 1–310 | — |
| `x01`–`x105` | z-displacements of 105 free-form deformation (FFD) control points; 0 is the baseline design | physical, in [−0.3, 0.3]. The units are not recorded. Dissertation Fig. 2.20 (p. 42) puts 50 control points on the main wing, 24 on the tail, 16 on the aft deck and 15 on the fuselage; which columns belong to which component is not recorded |
| `Lift`, `Drag` | lift and drag coefficients C_L, C_D | dimensionless |
| `Equiv Area` | equivalent-area functional ΔA_e (dissertation Eq. (2.2), p. 17): the mean squared difference between the near-field equivalent-area distribution and a target distribution | scale not recorded: the stored values run from 5.5 to 1666, while Eq. (2.21) constrains ΔA_e ≤ 10⁻⁴ |
| `d-lift01`–`d-lift105`, `d-drag01`–`d-drag105`, `d-ea01`–`d-ea105` | SU2 continuous-adjoint gradients ∂f/∂x_i | with respect to the **physical** inputs |

- **Gradient coordinates.** With normalized inputs z ∈ [−1, 1]¹⁰⁵, x = 0.3 z, so ∂f/∂z_i = 0.3 ∂f/∂x_i. `ac.active_subspace(space=ac.Uniform(np.full(105, -0.3), np.full(105, 0.3)), grads=G)` applies that factor, so its eigenvalues are 0.09 times those of the 2016 notebook, which passed the raw gradients. The eigenvectors are the same, because all 105 inputs have the same range. With each output also min–max scaled to [−1, 1], the eigenvalues reproduce dissertation Fig. 4.14 (p. 108): λ₁ = 0.19 (lift), 0.084 (drag), 1.42e-3 (equivalent area). So the dissertation treated the gradients as physical too.
- **Gradient accuracy** (printed by the notebook's "Gradient quality" cell). The values and the gradients disagree:
  - Lift is linear in the inputs (least-squares R² = 1.000), so its exact gradient is the regression slope s at every run. The adjoint w₁ is 24.8° from s, each gradient's component along s is 0.62–0.65 of ‖s‖, and the gradients differ from s by a median of 0.47 ‖s‖.
  - Drag: along w₁, the adjoint derivatives are 0.70 of the slope of a quadratic fit of the values.
  - Equivalent area: along w₁, the adjoint derivatives follow the slope of a quadratic fit (correlation 0.94) but are 15.1 times smaller. Either the stored values and gradients are on different scales, or the gradients are wrong by a nearly constant factor; the cause is not recorded.
  - The dissertation measures SU2's continuous-adjoint error only for NACA0012 (Table 3.4, p. 72: 9.9 % lift, 2.0 % drag) and explains the lift error by artificial dissipation calibrated for drag (§3.3.2.1, p. 73). It has no gradient check for the N+2.
- **Distribution.** The inputs look uniform on the box (standard deviation 0.1729, against 0.3/√3 = 0.1732). How they were drawn (independent uniform, Latin hypercube, …) is not recorded.

## Rows

- 310 rows, all with distinct inputs. Run 1 is the baseline design (all inputs 0).
- The dissertation (p. 107) reports "315 sampled designs, 305 of which were retained after filtering for results that converged more than 1 × 10⁻⁶". Why the file has 310 rows, and whether it includes runs that failed that filter, is unexplained (author, 2026-10-30). The notebook uses all 310 rows. Its eigenvalues match Fig. 4.14 within 14 %, so they cannot tell 305 rows from 310.

## Provenance and unknowns

- **Code.** SU2, the Stanford Aerospace Design Laboratory's flow solver, with its continuous adjoint for the gradients. Dissertation §2.5.3.2 (pp. 41–42): the "boom mesh" with active engines and a near-field region, simulated as a half-body; freestream Mach 1.7, angle of attack 2.1°, baseline cruise C_L 0.14 (§2.5.3, p. 38). Each run took 120 minutes on 96 cores (p. 107).
- **Who ran it.** Trent Lukaczyk (Stanford University, Aerospace Design Laboratory; adviser Juan J. Alonso), for his dissertation. Paul Constantine committed the file and the first notebook to this repository on 2014-10-23 (commit 7c26e3a). They were removed from master on 2016-09-09 (9c1b8a7), kept on `dev`, and restored to `main` in the 2026 branch merge (#32).
- **Funding and release.** The work was funded by the NASA N+2 Supersonics Program and the Lockheed Martin Corporation (dissertation p. vii). The author cleared the release of this file (2026-10-30). The file holds only FFD displacements, integrated coefficients and their gradients, no geometry.
- **Unknown** (author, 2026-10-30, unless marked new):
  - which columns of `x01`–`x105` belong to the wing, tail, aft deck and fuselage;
  - the units of the displacements, and why the data lie in [−0.3, 0.3] while dissertation Eq. (2.21) gives −3.0 ≤ x_i ≤ 3.0;
  - which 310 runs these are (dissertation: 315 sampled, 305 retained);
  - the SU2 version, mesh size and adjoint settings;
  - (new) the scale of the stored `Equiv Area` values relative to Eq. (2.21), and why its gradients are about 15 times smaller than the values imply;
  - (new) what "104X" in the file name stands for.
- **Dissertation errata noted.** Fig. 4.15 (p. 109) computes the active variable from the raw ±0.3 inputs (its axis, −0.48 to 0.57, is reproduced exactly that way), contradicting §4.2 (pp. 78–79) and Fig. 4.14, which normalize the inputs. "Passengjer" in the captions of Figs. 2.16 and 2.19.

## Regenerating

The data cannot be regenerated from this repository. The notebook reads `LM104X.txt` and computes everything from it in about 30 seconds (one Python process). The only randomness is the bootstrap (200 replicates per output), and it all comes from one `np.random.default_rng(2015)`. Two runs give identical printed output.

## License and credit

Code: MIT (`LICENSE`). Data (`LM104X.txt`): CC BY 4.0 (`LICENSE-DATA`). Credit Trent Lukaczyk (SU2 runs) and cite T. W. Lukaczyk, *Surrogate Modeling and Active Subspaces for Efficient Optimization of Supersonic Aircraft*, PhD dissertation, Stanford University, June 2015, http://purl.stanford.edu/xx611nd3190, §2.5.3 and §4.3.3. The dissertation is licensed CC BY-NC 3.0 US; that license applies to the dissertation's text and figures, not to this data file. Background on the method applied to shape optimization: T. W. Lukaczyk, F. Palacios, J. J. Alonso and P. G. Constantine, "Active subspaces for shape optimization", AIAA 2014-1171, doi:10.2514/6.2014-1171.
