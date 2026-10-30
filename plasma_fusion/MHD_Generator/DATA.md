# MHD_Generator: data

Kind: **both**. The Hartmann problem is a **Python model**: `hartmann_model.py`, closed form, with no data files. The 3-D MHD generator is **stored data**: two files of runs of Sandia National Laboratories' Drekar code, which are the only record of those runs and cannot be regenerated here.

## Files

| file | contents | role |
|---|---|---|
| `MHD_Generator.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2017; outputs committed |
| `hartmann_model.py` | `uavg`, `Bind`, `uavg_grad`, `Bind_grad`, `NAMES`, `LABELS`, `LOWER`, `UPPER` | the Hartmann model (renamed from the 2016 `gradients.py`) |
| `MHD_Generator_Data_Uavg.txt` | 483 rows × 12 columns, 62 KB | generator runs, u_avg (stored) |
| `MHD_Generator_Data_Bind.txt` | 483 rows × 12 columns, 67 KB | generator runs, B_ind (stored) |

## Hartmann problem (Python model)

The average velocity u_avg and the induced magnetic field B_ind of laminar flow between two plates a distance 2ℓ apart, in a uniform magnetic field perpendicular to the flow: Eqs. (36) and (37) of Glaws et al. (2017).

- **Inputs**: m = 5 log-inputs, in the order log μ, log ρ, log ∂p0/∂x, log η, log B0 (`NAMES`), uniform on the logs of the Table 2 ranges (`LOWER`, `UPPER`). Neither quantity depends on ρ.
- **Fixed parameters**: `l` and `mu0` are keyword arguments, both 1 by default. The paper sets ℓ = 1 m (p. 318). It does not state μ0 for the Hartmann problem; μ0 = 1 is the 2016 notebook's value, confirmed by the author (2026-10-30). μ0 only scales B_ind (its eigenvalues are proportional to μ0², and its eigenvectors do not depend on it).
- **Gradients**: `uavg_grad` and `Bind_grad` return ∂f/∂log x_i = x_i ∂f/∂x_i, shape (..., 5). `ac.active_subspace` scales them to the normalized inputs. The 2016 `gradients.py` returned gradients already scaled to the normalized inputs with module-level Hartmann bounds, and built the ρ-derivative with `0.0*np.empty(...)`, which can give NaN; the port returns log-input gradients, uses zeros, and has no module globals. On the 11⁵ quadrature nodes the new functions agree with the 2016 ones (after removing that scaling) to a relative 1e-13 (scratch check). The notebook checks the gradients against central finite differences at 100 random points (maximum error 1.1e-9 for u_avg and 1.2e-9 for B_ind, relative to the largest component).
- **Quadrature**: C is computed with an 11-point Gauss–Legendre rule per dimension (11⁵ = 161,051 points), as in the paper. The weights are the scipy `roots_legendre(11)` weights halved, so their tensor product sums to 1 and already includes the density 2⁻⁵ on the normalized inputs. The 2016 notebook multiplied the integrand by 2⁻⁵ a second time, so its Hartmann eigenvalues were 32 times too small. The notebook prints both values.

### Regenerating

There are no shipped Hartmann outputs and no `generate_data.py`: the notebook evaluates the model directly. The whole notebook takes about 11–15 s in one Python process (about 30 s wall time with kernel start-up). All draws come from one `np.random.default_rng(2017)`: the gradient-check points, the 5000 quadrature nodes in the summary plots, and the generator bootstrap. The Hartmann eigenvalues and eigenvectors do not depend on the seed. Two runs give identical printed output apart from the runtime line.

### Relation to the paper

- u_avg: λ1 = 36.57, λ2 = 1.2275, λ1/λ2 = 29.8. The eigenvectors match the paper's Fig. 2B, with the same signs. The paper's Fig. 2A shows λ1 ≈ 1.4 and λ2 ≈ 0.05: its scale matches the 2016 values (1.14, 0.038), so the figure appears to carry the same extra 2⁻⁵. Its text (p. 319) calls the gap between λ1 and λ2 "three orders of magnitude"; its own figure shows a ratio of about 28. The notebook does not repeat that sentence.
- B_ind: λ1 = 0.0863, λ1/λ2 = 13.6, w1 = [0.381, 0, −0.663, 0.493, −0.414]. This does **not** match the paper's Fig. 3: Fig. 3B shows w1 ≈ [0.38, 0, −0.56, 0.49, −0.56], and Fig. 3A shows λ1 ≈ 3 and λ1/λ2 ≈ 45. μ0 cannot explain it. A smaller ℓ (about 0.4) reproduces the published w1 but not the eigenvalues (specs/MHD.md §10). How Fig. 3 was computed is not known (author, 2026-10-30); the notebook keeps ℓ = μ0 = 1, states it, and says that Fig. 3 differs.

## MHD generator (stored data)

### Provenance

- **Generating code**: Sandia National Laboratories' Drekar multiphysics code (Pawlowski et al., SAND2012-2697), with its adjoint capability for the gradients (Shadid et al., J. Comput. Phys. 321, 2016). The Drekar version, the mesh and the solver settings are not recorded.
- **Who ran it**: not recorded. The files were added to this repository by Ryan Howard on 2016-07-19 (commit 2788d24).
- **Setup** (paper §4.2, pp. 320–321): steady resistive MHD duct flow in a 15 m × 1 m × 1 m duct, inlet velocity [1, 0, 0] m/s, applied vertical field ½B0[tanh((x − x_on)/δ) − tanh((x − x_off)/δ)] with x_on = 4, x_off = 6, δ = 0.1, and μ0 = 1. Each run took 5.3 CPU-hours (10 minutes on 32 cores).
- **Inputs**: μ, ρ, ∂p0/∂x, η, B0, sampled independently and uniformly on the logs of the paper's Table 3 ranges, [.001, .01], [.1, 10], [.1, .5], [.1, 10], [.1, 1]. All 483 rows of both files lie inside these ranges. The sampling seed is not recorded.

### Format

Both files are pandas `to_csv` output: a header row `,0,1,2,3,4,5,6,7,8,9,10` and then one row per run. The column names are not in the files. The meaning below comes from the 2016 notebook.

| column | contents |
|---|---|
| 0 | pandas index label (an integer 0–499; see below) |
| 1–5 | μ, ρ, ∂p0/∂x, η, B0: physical values, not their logs |
| 6 | the quantity of interest: u_avg (`…_Uavg.txt`) or B_ind (`…_Bind.txt`) |
| 7–11 | the gradient of the quantity of interest with respect to the five **physical** inputs, in the same order |

The gradient coordinates are checked in the notebook. For u_avg the median of (∂p0/∂x)·∂u/∂(∂p0/∂x)/u is 1.0002, and for B_ind 1.0060, as expected for quantities roughly proportional to the pressure gradient (exactly proportional in the Hartmann problem). The notebook uses ∂f/∂log x_i = x_i ∂f/∂x_i.

**Units**: not recorded. The paper gives the setup in SI units (with μ0 = 1), but it does not give the units of the inputs or outputs, or say over what region u_avg and B_ind are averaged. u_avg ranges over 0.38–14.4 and B_ind over 1.1e-4–0.123.

Load without pandas:

```python
D = np.loadtxt('MHD_Generator_Data_Uavg.txt', delimiter=',', skiprows=1)
label, X, f, df = D[:, 0].astype(int), D[:, 1:6], D[:, 6], D[:, 7:]
```

### Row issues and the run-ID rule

The files are kept exactly as shipped. The notebook prints each fact below and asserts the rule.

- **Index gaps.** The labels run from 0 to 499 with 17 missing, the same in both files: 5, 40, 72, 107, 115, 153, 253, 258, 295, 305, 309, 347, 398, 411, 453, 467, 470. We take them to be failed runs (see run 72 below), but this is not recorded.
- **The label shift.** Matching rows with identical inputs across the two files gives 482 pairs. In 434 pairs the labels agree. In the other 48, the u_avg label is the B_ind label + 1, and these are exactly the u_avg labels 50–99 (all of them except 72, a gap, and 73, see below). So in the u_avg file the row labelled L holds the run that the B_ind file labels L − 1, for L = 50…99, and for no other label.
- **The duplicate is explained by the shift.** Rows 47 and 48 (0-based) of `…_Uavg.txt`, labels 49 and 50, are identical in all 11 data columns: both are run 49. The shifted block starts with a second copy of the run just before it.
- **The unmatched points fit the shift.**
  - u_avg label 73 holds run 72, which is one of the 17 gaps (the B_ind file has no run 72). Its gradient is a placeholder, [0.0027695, 0.0027695, 0.0027695, 0.0027695, 0], so (∂p0/∂x)·∂u/∂(∂p0/∂x)/u = 1e-4 where every other row gives about 1 (the other 482 rows range over 0.96–1.26), and its u_avg value lies 9 times the median residual off a quadratic fit of log u_avg in the log-inputs (the 95th percentile is 5; scratch check). It is a failed run.
  - Run 71 would sit at u_avg label 72, which is a gap, and run 99 at u_avg label 100, which holds run 100. So both are missing from the u_avg file. They are the two B_ind rows (labels 71 and 99) with no u_avg match.
- **One relabeling rule makes the files consistent.** Take the B_ind labels as run IDs, and relabel the u_avg rows labelled 50–99 as L − 1. Then every run ID present in both files has identical inputs in both, and every input point present in both files has the same run ID. What is left: run 49 twice in u_avg, run 72 (failed) only in u_avg, and runs 71 and 99 only in B_ind. The reverse rule (B_ind labels 49–98 relabeled as L + 1) fails: two B_ind rows would get run ID 99, and the u_avg duplicate stays unexplained.
- **What the notebook drops.** After relabeling, it drops (i) rows that repeat an earlier row exactly in all 11 data columns, keeping the first, and (ii) runs whose ID is one of the 17 gaps. That drops u_avg labels 50 (run 49 again) and 73 (run 72), and no B_ind rows. u_avg uses 481 rows and B_ind 483.
- **What stays unknown.** Why the u_avg rows in the label block 50–99 (one block of 50) were shifted, and why the block starts with a repeat of run 49; whether runs 71 and 99 have u_avg outputs anywhere; and whether the 17 gaps are all failed runs (run 72 is: its u_avg row is a placeholder). The original Drekar output cannot be recovered (author, 2026-10-30).

**Joint or separate.** Each quantity of interest is analyzed from all its own valid runs (u_avg 481, B_ind 483). The active subspace of one quantity does not need the other file. Use the run IDs (not the raw labels) whenever the two files must be paired, for example to look at u_avg and B_ind at the same inputs: the 481 u_avg runs are all in the B_ind file.

### Results

From all 483 rows of each file, as in the 2016 notebook: u_avg λ = [28.98, 0.928, 0.202, 0.0306, 7.33e-5] and B_ind λ = [4.156e-3, 4.94e-4, 1.10e-4, 1.58e-5, 4.92e-11] (the spec's §9 values). With the rule above, u_avg (481 rows) λ = [29.08, 0.931, 0.203, 0.0308, 7.32e-5], at most 0.42 % from the 483-row values, and B_ind is unchanged. Restricting B_ind to the 481 runs in both files would change its eigenvalues by at most 0.42 %. λ5 is 0.000242 % (u_avg) and 0.000001 % (B_ind) of the sum, as in the paper (0.00024 % and 0.000001 %, pp. 322–323), which used all 483 rows. The eigenvectors match the paper's Figs. 5B and 6B, with the same signs.

### Checksums

| file | md5 |
|---|---|
| `MHD_Generator_Data_Uavg.txt` | fbd9bd7bd77c9738acb5ab5cd853e96a |
| `MHD_Generator_Data_Bind.txt` | 47508035ca2bd86c122667e45a744e05 |

## License and credit

Code (`hartmann_model.py`, the notebook): MIT (`LICENSE`). Data (the two generator files): CC BY 4.0 (`LICENSE-DATA`). Redistribution is approved (author, 2026-10-30). The generator data were computed with Sandia National Laboratories' Drekar code: credit **Sandia National Laboratories**. Cite A. Glaws, P. G. Constantine, J. N. Shadid and T. M. Wildey, "Dimension reduction in magnetohydrodynamics power generation models: Dimensional analysis and active subspaces", Stat. Anal. Data Min. 10(5):312–325, 2017, doi:10.1002/sam.11355.
