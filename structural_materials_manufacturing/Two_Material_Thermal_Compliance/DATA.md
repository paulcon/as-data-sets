# Two_Material_Thermal_Compliance: data

Kind: **Python model**. The model is `thermal_model.py` (numpy/scipy), and `generate_data.py` writes `thermal_compliance.npz` from it with a saved seed. A rerun reproduces the file.

## Files

| file | contents | role |
|---|---|---|
| `Two_Material_Thermal_Compliance.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with seed 2025; outputs committed; run time in `.venv` about 3 min (158 s on a quiet machine, 195–198 s in two other runs, 290–440 s on a busy one), most of it the 200-replicate bootstrap |
| `thermal_model.py` | `ThermalCompliance(nx=16, k1=2, k2=1, boundary='paper')` with `energy`, `grad`, `energy_and_grad`, `volume`; `source`, `mesh`, `boundary_nodes`; `K1`, `K2`, `E_MAX`, `M_INPUTS` | the model |
| `generate_data.py` | writes `thermal_compliance.npz`; `--check` (optionally with `--rows a:b`), `--fd-check`, `--casm DIR` | regenerates and checks the data |
| `check_accuracy.py` | recomputes the numbers below that neither the notebook nor `generate_data.py` prints (numpy and scipy only, about 1 s) | accuracy checks |
| `thermal_compliance.npz` | `theta` (1500 × 512, float32), `E` (1500, float64), `dE_dtheta` (1500 × 512, float64), `E_max` (1.35), `description` | samples, outputs and exact gradients (8.5 MB) |

The data are one compressed `.npz` file (8.5 MB). Read it with `np.load('thermal_compliance.npz')`; no pickle is needed. Column `e` of `theta` and `dE_dtheta` is element `e` of the mesh: square row `e // 32`, square column `(e % 32) // 2`, lower-left triangle for even `e` and upper-right for odd `e` (squares are numbered from the corner (−1, −1), row by row). This is the element numbering of the author's code, so the eigenvector maps onto the mesh in the same way as the paper's Fig. 7.

**Why float32 inputs.** The design is drawn as float64 and rounded to float32 *before* the model is run, and the model is run at the rounded values (cast back to float64). The stored `theta` are therefore exactly the inputs of the stored `E` and `dE_dtheta`; nothing is lost. This halves the input storage. The gradients are stored in float64.

## Model

**Citation.** P.-A. Luneau, "Conservative surrogate models for optimization with the active subspace method", *Engineering Optimization* 57(12):3757–3777, 2025, https://doi.org/10.1080/0305215X.2025.2453542. The PDF read is the arXiv preprint (arXiv:2403.15678v1, 23 March 2024), not the accepted manuscript or the version of record. The published article is the one cited; its bibliographic data (volume 57, issue 12, pages 3757–3777, online 25 March 2025) are from Crossref, but its figures and numbers were not compared with the preprint's. Section, equation, page and figure numbers refer to the preprint. Code: https://github.com/paluneau/CASM (MIT licence; release v1.0, commit 20a6bd1, archived on Zenodo, https://doi.org/10.5281/zenodo.10850850, file `paluneau/CASM-v1.0.zip`, sha256 6083dfddad82ea8b4b93cf36dd25e02ab690774e9a11ec99f707132721ef4941), with its finite element submodule https://github.com/paluneau/pyTOP (MIT, commit 90c61d4 pinned by CASM).

**Equations.** Sec. 4.3, Eq. (8): find u ∈ H¹₀(Ω), Ω = [−1, 1]², with ∫ (k₁θ + k₂(1 − θ)) ∇u·∇v dx = ∫ f v dx for all v ∈ H¹₀(Ω), where θ is the fraction of material 1. Quantity of interest: the thermal energy (heat compliance) E(θ) = ½ ∫ (k₁θ + k₂(1 − θ)) |∇u|² dx = ½ ∫ f u dx (p. 16), nondimensional (no units in the paper). The paper's design problem, Eq. (9), is min V(θ) = ∫ θ dx subject to E(θ) ≤ E_max.

| constant | value | source |
|---|---|---|
| k₁ (material 1, θ = 1) | 2 | Sec. 4.3, p. 16 |
| k₂ (material 2, θ = 0) | 1 | Sec. 4.3, p. 16 |
| heat source f(x, y) | 56 (1 − \|x\| − \|y\|)⁶, used as written (no clipping at \|x\| + \|y\| = 1); half of its integral lies outside the diamond (below) | Sec. 4.3, p. 16; same in the CASM notebook |
| E_max | 1.35 | Sec. 4.3, p. 16 |
| mesh | structured, 16 × 16 squares, D = 512 triangles, second-order Lagrange elements | Sec. 4.3, p. 16 ("D = 512 triangular second-order Lagrange elements"); 16 × 16 from the CASM notebook |
| boundary condition | u = 0 on ∂Ω | Sec. 4.3, p. 15 |

**Discretization.** As in the author's code: the 33 × 33 grid of P2 nodes, each square cut along its lower-right-to-upper-left diagonal, θ constant per element, and the element integrals evaluated with the 4-point degree-3 Hammer rule (exact for the P2 stiffness; approximate for the load ∫ f φ). `thermal_model.py` is written from these facts in vectorized numpy/scipy (sparse direct solve); it is not a copy of pyTOP.

**Boundary nodes (a difference from the author's code).** The CASM notebook fixes u = 0 at a hard-coded list of 124 nodes, which leaves four of the 128 boundary nodes free: 32 and 1088 (the lower-right and upper-right corners) and 1023 and 1055 (the nodes just below the top corners on the left and right edges), apparently an off-by-one in the `range` calls. The paper states u = 0 on the whole boundary, so this data set fixes all 128 boundary nodes (`boundary='paper'`). The author's list is available as `boundary='casm'` and is used only to check this implementation against the author's files (below). At the author's 50 test designs (`Xtest.npy`) the full boundary changes E by 0.58–0.80 % from the value with the author's list, and E − E_max by up to 11 % (`generate_data.py --casm`); it changes λ₁ by −0.92 % (notebook).

**The source outside the diamond.** Used as written, 56 (1 − |x| − |y|)⁶ is not zero outside the diamond |x| + |y| ≤ 1: it equals 56 at the four corners, as much as at the centre. **Half of the heat lies in the four corner triangles outside the diamond**: f integrates to 8 over Ω, 4 inside the diamond and 4 in the corners (exactly 1 on each quarter of the diamond and on each corner triangle, because 56 = 1/∫ (1 − x − y)⁶ over the unit simplex; notebook, source cell, and `check_accuracy.py` check 2). That normalization suggests that the source was meant to vanish outside the diamond (Unknowns, question 4), but the paper writes the formula without clipping, the author's code uses it so, and the paper's Fig. 8 range (E − E_max about 0.10–0.16) matches the unclipped source (the clipped E(0.5) = 1.3631 would put E − E_max near 0.01). This data set keeps the source as written. The corner heat enters next to the cold boundary, so it adds much less to E than the central heat: clipping lowers E(0.5) by 5.6 % (1.443779 → 1.363099).

The four free nodes matter only because of that corner heat: two of them (32 and 1088) are corners. At θ ≡ 0.5, the author's node list gives E = 1.453713 and the full boundary 1.443779 (−0.68 %) with the source as written; with the source set to zero outside |x| + |y| ≤ 1 they give 1.363110 and 1.363099 (−0.0008 %). `check_accuracy.py` (check 2) computes these values by building the model with the clipped formula; the shipped model keeps the formula as written.

**Rebuilding the author's 124-node variant.** It is not shipped as data, but `ThermalCompliance(boundary='casm')` is the model with the author's node list, and

```python
import thermal_model as tm, generate_data as gd
E_casm, G_casm = gd.compute(gd.design(), model=tm.ThermalCompliance(boundary='casm'))  # 1500 rows, 10-20 s
```

gives its energies and gradients at this data set's inputs. `python generate_data.py --casm DIR` (DIR a clone of github.com/paluneau/CASM, which holds `dfData.npy`, `fData.npy` and `Xtest.npy`) checks the variant against the author's files and prints λ₁, λ₂ for both boundary lists; it writes no file. The notebook's comparison section runs the variant at the author's 512 inputs.

**Inputs (m = 512).** θ_e ∈ [0, 1], one per element (Sec. 4.3, p. 16, Eq. (9)). Sec. 4.3 states no sampling density for θ (the method samples uniformly over the input domain, Sec. 3.4, p. 11, and the two-dimensional example of Sec. 4.1 uses a uniform density); the author's code draws x ~ U[−1, 1]⁵¹² and sets θ = (x + 1)/2, and this data set samples θ_e independently and uniformly on [0, 1]. There is no nominal design in the paper; the notebook uses θ_e = 0.5 as a reference design.

| name | symbol | units | nominal | range | distribution | source |
|---|---|---|---|---|---|---|
| `theta_000` … `theta_511` | θ_e, e = 0 … 511 | – (volume fraction of material 1 in element e) | not reported (0.5 used as a reference) | [0, 1] | independent uniform | range: Sec. 4.3, Eq. (9), p. 16; density: the author's code |

**Gradients.** E = ½ FᵀU with K(θ)U = F, so E is self-adjoint and ∂E/∂θ_e = −½ (k₁ − k₂) u_eᵀ K_e u_e, with K_e the unit-conductivity element stiffness and u_e the element's nodal values. This is the exact derivative of the discrete E; one solve gives E and all 512 components. The author's code uses the same formula for ∂E/∂x = ½ ∂E/∂θ (the paper does not state the gradient method). Every component is negative. `dE_dtheta` is with respect to the physical θ; the library maps it to the normalized x = 2θ − 1 (a factor ½).

**Sampling design.** Monte Carlo: `numpy.random.default_rng(20250129).random((1500, 512))`, rounded to float32. No duplicate rows, no values at exactly 0 or 1 (range 1.98e-7 to 0.9999985), per-column kurtosis 1.71–1.88 (mean 1.802, Pearson, biased; 1.8 for a uniform) (`check_accuracy.py` check 1). Every sample has E > E_max (E from 1.4219 to 1.5327; notebook): random designs with half of each material on average do not meet the paper's threshold. This is a property of the design box, not an error, and all samples are kept.

**Sample size.** M = 1500 (2.9 m) samples; the size was chosen for this data set, and the file is 8.5 MB. It exceeds the heuristic M = α k log m of Constantine (2015, Ch. 3), with oversampling α ≤ 10, for k = 6 eigenvalues (10 × 6 × ln 512 ≈ 374), and with M > m the estimated C has full rank. The bootstrap distance for n = 1 is 0.0054 (notebook), so λ₁, w₁ and n = 1 are well determined. The tail eigenvalues are not converged at this M: in 50 subsamples per size, the mean λ₂ falls from 2.2216e-6 at M = 100 to 1.8023e-6 at 256, 1.6697e-6 at 512 and 1.6084e-6 at 1000, and is 1.5839e-6 with all 1500 (notebook, eigenvalue cell). So λ₂ … λ₆ are still biased upward, and λ₁/λ₂ = 551.6 is a lower estimate; its 95 % bootstrap interval is [504.3, 568.6] (λ₂/λ₃: [1.033, 1.221], no second gap). The paper's complete spectrum (Fig. 6b), whose axis reaches 1e-15, is a sampling effect of M = m, not rounding: the author's 512 × 512 `dfData.npy` has full rank, and its smallest eigenvalue, 2.72e-15, is about 1.4e4 times the rounding level ε λ₁ = 2.0e-19 (`generate_data.py --casm` and notebook). With M = m the smallest sample eigenvalues are pushed toward zero (mean λ₅₁₂ = 8.13e-16 over 50 subsamples of 512 of this data set's gradients); at M = 1500 λ₅₁₂ = 4.66e-12. The paper does not report its number of gradient samples; the author's code used M = m = 512.

**Published results.** Fig. 6: eigenvalues (10 largest, and the full spectrum), with a drop after the first; d = 1 chosen. Fig. 7: the first eigenvector, an annular pattern around the centre. Fig. 8: E − E_max along the 1-D active subspace, 50 samples, with a linear fit. Table 2: optima of the full, AS and CASM problems (not reproduced here; they need the paper's GPR/bias machinery and the MMA optimizer). The author's notebook prints λ₁ = 0.000882448329127676 and λ₂ = 1.5953761821851023e-06.

## Regenerating

```bash
python generate_data.py                       # all 1500 samples, writes thermal_compliance.npz here
python generate_data.py --check               # recompute all rows and compare with the file (writes nothing)
python generate_data.py --rows 0:50 --check   # the same for a subset
python generate_data.py --fd-check            # exact gradients vs central differences, 3 samples
python generate_data.py --casm /path/to/CASM  # reproduce the author's dfData.npy and fData.npy (124-node list)
python check_accuracy.py                      # the numbers of DATA.md that the notebook does not print
```

The shipped file was written on 2026-11-01 in a Linux container (Python 3.13). In `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS x86-64, 2026-11-02) the full run takes 10–20 s (1500 samples; 9.8 s and 17.7 s in two runs on a shared machine, about 7–12 ms per sample, one process). `generate_data.py --check` reproduces the shipped file with largest relative differences 0 for `theta`, 7.7e-15 for `E` and 4.6e-14 for `dE_dtheta`; a full run written to another directory gives the same differences. Differences at the level of 1e-14 are rounding and change between platforms (the notebook's re-solve of five samples gives 5e-15 and 2e-14). `--check` prints these differences (relative to the largest shipped value) and the largest of them, 4.6e-14 for `dE_dtheta` (in `.venv` on macOS, Python 3.11, 2026-11-12; 5.8 s with start-up); it exits with code 1 only on a structural mismatch: a missing array, a different shape or NaN pattern, or a `theta` that differs from the shipped design by more than 10⁻⁸ (relative).

## Accuracy

- **Gradients.** `generate_data.py --fd-check` (`.venv`): at the first three samples, central differences with step 1e-5 in θ (one-sided near a bound) agree with the adjoint gradient to a relative 3.9e-8, 4.2e-8 and 4.5e-8 (2-norm; largest single-entry relative difference 3.8e-6, at the smallest entries). The notebook's check cell compares a directional derivative with a central difference (relative difference 7e-8) and re-solves five samples (differences from the file of order 1e-14).
- **Against the author's code.** `generate_data.py --casm DIR`, run in `.venv` on 2026-11-02 with DIR the unpacked Zenodo archive of CASM v1.0 (above; `dfData.npy` sha256 c64652b0…, `fData.npy` 67d9c10a…, `Xtest.npy` 8d3e14d1…): with `boundary='casm'` and the author's inputs (NumPy legacy `np.random.seed(42)`, the notebook's draw order), the gradients match the author's `dfData.npy` (512 × 512, ∂E/∂x) to a relative 2.8e-14 (Frobenius; worst row 3.7e-14), and E − E_max matches `fData.npy` (50 values at `Xtest.npy`) to 3.0e-13. The resulting λ₁, λ₂ equal the author's printed values to all digits shown (8.82448329e-4, 1.59537618e-6). `dfData.npy` itself has rank 512 and smallest eigenvalue 2.72e-15 (printed by `--casm`). With `boundary='paper'` the gradients differ from `dfData.npy` by 4.8 % (Frobenius) and E − E_max from `fData.npy` by up to 11 %.
- **Discretization (for information; the model is the paper's 16 × 16 discretization).** At θ ≡ 0.5, E = 1.443779 on the 16 × 16 mesh, 1.447183 on 32 × 32 and 1.447535 on 64 × 64, so the 16 × 16 value is 0.26 % below the finest (`check_accuracy.py` check 3). Integrating the load exactly (collapsed 6 × 6 Gauss rule) instead of with the 4-point rule changes E by 3.8e-4 (θ ≡ 0.5) and 3.1e-4 (sample 0), relative (check 4).
- **Storage.** float32 inputs are exact (see Files). Gradients and E are float64, as computed.

## How these files relate to the paper

- **An independent computation of the paper's model, not the author's files.** The author's repository stores 512 gradients (`dfData.npy`), 50 test inputs (`Xtest.npy`) and their E − E_max (`fData.npy`), computed with the four free boundary nodes. They are not shipped here; this data set is new: 1500 designs, the boundary condition of the paper, and E itself.
- **What is new relative to `uq_benchmarks/Elliptic_PDE_Random_Coefficients`.** That data set is also a 2-D elliptic PDE with a high-dimensional coefficient, but its inputs are Gaussian KL coefficients of a log-conductivity random field and its QoI is a boundary average of the solution. Here the inputs are bounded design variables (an element-wise mixture of two materials, so the conductivity is affine in each input and bounded between 1 and 2), the QoI is the energy that serves as the constraint of a design optimization, and its gradient is one-signed, which makes w₁ the normalized mean gradient.
- The paper's text has the rescaling backwards: "First, a rescaling is applied to map [−1, 1]^D to [0, 1]^D, because the AS theory is applicable only to the former" (p. 16). The code maps the design variables θ ∈ [0, 1]^D to x ∈ [−1, 1]^D for the analysis (x = 2θ − 1), which is what is done here (through the library's normalization).

## Unknowns

- The figures and numbers of the published article were not compared with the preprint's.
- Sec. 4.3 states no sampling density for θ, the paper does not state the gradient method, and it does not report its number of gradient samples (the author's code used M = m = 512).
- **Questions for the author** (open):
  1. Were the four free boundary nodes in the CASM notebook (32, 1023, 1055, 1088) intended, or should u = 0 hold on the whole boundary as Eq. (8) states?
  2. How many gradient samples were used for Fig. 6 in the published version (the code uses 512)?
  3. What do the values 0.5 to 0.56 in the caption of Fig. 7 measure?
  4. Is the source meant to be 56 (1 − |x| − |y|)⁶ everywhere (as coded), or only inside the diamond |x| + |y| ≤ 1 (zero outside)? As written, half of its integral (4 of 8) lies in the four corner triangles outside the diamond, and the constant 56 normalizes each quarter of the diamond to 1. This also decides whether question 1 matters: at θ ≡ 0.5 the free nodes change E by 0.69 % with the unclipped source and by 0.0008 % with the clipped one.

## License and credit

Code (`thermal_model.py`, `generate_data.py`, `check_accuracy.py`, the notebook): MIT (`LICENSE`). Data (`thermal_compliance.npz`): CC BY 4.0 (`LICENSE-DATA`). Model and discretization: P.-A. Luneau. The author's code, CASM (github.com/paluneau/CASM) and its finite element submodule pyTOP (github.com/paluneau/pyTOP), is MIT-licensed (LICENSE files in both repositories: "Copyright (c) 2023 Philippe-André Luneau"). No file of that code is copied here; `thermal_model.py` re-implements the discretization from it, and the `--casm` check and the notebook's comparison reproduce the author's stored files with credit to CASM. Cite P.-A. Luneau, Engineering Optimization 57(12):3757–3777, 2025, doi:10.1080/0305215X.2025.2453542, and the code archive doi:10.5281/zenodo.10850850.

