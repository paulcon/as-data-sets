# Aquifer_Tracer_Advection_Diffusion: data

Kind: **Python model**. The model is `tracer_model.py` (numpy and scipy only), and `generate_data.py` writes every CSV except `kle_basis.csv` from it, with a saved seed. `kle_basis.csv` is an input computed by PyApprox 1.0.3 and is treated as stored data (see "KLE basis").

## Files

| file | contents | role |
|---|---|---|
| `Aquifer_Tracer_Advection_Diffusion.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom, bootstrap seed 20231825; outputs committed |
| `tracer_model.py` | `TracerModel(order=21, nsteps=64, m=10)` with `qoi`, `qoi_and_grad`, `solve`; `tracer_qoi`, `tracer_qoi_and_grad`; the model constants | the model |
| `generate_data.py` | draws the inputs (seed 20231825) and writes the outputs and gradients | regenerates the data |
| `pyapprox_reference.py` | writes `kle_basis.csv` from PyApprox 1.0.3 and compares `tracer_model.py` with PyApprox | provenance only; needs PyApprox 1.0.3 (not in `.venv`; exits with status 1 and an install message without it) |
| `check_accuracy.py` | recomputes the numbers of this file that the notebook and `generate_data.py` do not print (numpy, scipy) | check |
| `kle_basis.csv` | 144 × 12: node coordinates s1, s2 and the 10 KLE modes φ_d = √λ_d v_d on the 12 × 12 Chebyshev mesh | stored input |
| `inputs.csv` | 500 × 10: z_1 … z_10, Monte Carlo draws from N(0, I) | inputs |
| `outputs.csv` | 500 × 1: q on the converged order-21 mesh (22 × 22 nodes, 64 steps) | model output (main) |
| `gradients.csv` | 500 × 10: ∂q/∂z_d on order 21, exact discrete derivatives | model output (main) |
| `outputs_f0.csv` | 500 × 1: q on PyApprox's highest-fidelity mesh f0 (order 11, 64 steps) | benchmark variant |
| `gradients_f0.csv` | 500 × 10: ∂q/∂z_d on f0 (median 15 %, max 67 % relative error vs order 21) | benchmark variant |
| `outputs_order27.csv` | 50 × 1: q on order 27, rows 0–49 of `inputs.csv` | convergence check |
| `gradients_order27.csv` | 50 × 10: ∂q/∂z_d on order 27 | convergence check |

The CSVs are comma-separated numbers after two `#` lines (a description; column names with units), so `np.loadtxt(path, delimiter=',')` reads them. All quantities are nondimensional; the inputs are their own physical units (standard normal KLE coefficients), so physical and normalized coordinates coincide.

## Model

**Citation.** J. D. Jakeman, "PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling", Environmental Modelling & Software 170:105825, 2023, doi:10.1016/j.envsoft.2023.105825 (volume and article number checked with Crossref). Read from the accepted manuscript (SAND2023-13522J); page numbers below are the manuscript's.

**Equations** (paper Eq. (1), p. 4; Sec. 3, pp. 7–8; footnote 1, p. 8). On Ω = [0, 1]², t ∈ (0, T]:

- ∂c/∂t = ∇·(K ∇c) − v·∇c + G, with v = (1, 0) (footnote 1); the paper writes the advection term as −∇·(vc), which is the same for this constant v.
- Robin condition on all four sides: K ∇c·n + 0.1 c = 0 (PyApprox 1.0.3 code; see "Paper versus code").
- K(s, z) = exp(Σ_d φ_d(s) z_d), with φ_d = √λ_d v_d from the squared-exponential kernel exp(−‖s − s′‖²/(2ℓ²)), ℓ = 0.5, σ = 1, K0 = 1 (p. 7).
- Initial condition: the steady state with the source G(s) = 100 exp(−‖s − (0.25, 0.75)‖²/0.1²) on; G = 0 for t > 0 (p. 8). T = 0.2 (Sec. 3.5.6, p. 24).

**Inputs** (m = 10).

| name | symbol | units | nominal | distribution | source |
|---|---|---|---|---|---|
| z_1 … z_10 | z_d | – | 0 | i.i.d. N(0, 1) | prior, Sec. 3, p. 8; m = 10 is our choice (paper example m = 3, p. 8; PyApprox default 2) |

The discrete KLE eigenvalues λ_d (= ‖φ_d‖²) are 66.578, 27.178, 27.178, 11.094, 3.813, 3.813, 1.557, 1.557, 0.324, 0.324; the 11th is 0.218, so m = 10 does not split an equal pair, and the 10 modes carry 99.59 % of the field variance on the 144 nodes (Σλ_d/144).

**Quantity of interest.** q(z) = ∫ over S = [0.75, 1] × [0, 0.25] of c(s, T, z) ds, the tracer mass in the subdomain at the final time, as returned by PyApprox's benchmark (`final_time_functional` of `subdomain_integral_functional`) and its documentation. The paper's text (Sec. 3.5.6, p. 24) instead names "the left QoI in (2)", the integral over t ∈ [0, T] as well; we follow the code.

**Discretization** (`tracer_model.py`). Chebyshev-Gauss-Lobatto collocation of order p per direction ((p+1)² nodes, PyApprox `CartesianProductCollocationMesh`): p = 21 (22 × 22 nodes, the default) for the main files, p = 11 (12 × 12 nodes, PyApprox's highest-fidelity member f0, `tm.ORDER_F0`) for the `_f0` variant, p = 27 for the convergence check; with the boundary rows replaced by the Robin condition (left and right rows include the corners); PyApprox's two-stage Crank-Nicolson scheme (`im_crank2`, "Wildey" form: each stage's boundary rows satisfy the Robin condition) with 64 steps of Δt = 0.2/64 = 0.003125; the source enters the first step at t = 0, as in PyApprox; q by a (p+2) × (p+2) Gauss-Legendre rule on S applied to the collocation interpolant. The steady state is one linear solve. No PyApprox code is copied; the scheme was rebuilt from reading the PyApprox 1.0.3 source and checked against it (Accuracy). Meshes other than order 11 (including order 21) interpolate the log field from the 12 × 12 KLE mesh, as PyApprox's `InterpolatedMeshKLE` does.

**Gradients.** Exact derivatives of the discrete model, by forward sensitivities: each linear system (steady state, both stages of every step) is differentiated with respect to z_d through K = exp(Φz), and the 10 sensitivity vectors are carried with the same LU factors. Physical gradients are normalized gradients here (σ = 1).

**Sampling design.** Monte Carlo: `np.random.default_rng(20231825).standard_normal((500, 10))`. The paper's own design was a pilot of 20 evaluations of each of 8 fidelities at posterior samples (Sec. 3.5.6, p. 24), with no seed; it is not reproduced.

**Published results.** None for an active subspace (the notebook finds n = 3 on order 21: λ1/λ2 = 2.65, λ2/λ3 = 8.75, λ3/λ4 = 11.56; cubic LOO R² 0.665 / 0.953 / 0.983 for n = 1 / 2 / 3). The paper reports that the best ACV estimator uses f0 and f4 (finest mesh, coarsest time step) and shows the variance reduction (Fig. 6); no statistic of q is printed.

**Physical meaning.** No sample has a negative nodal concentration at t = 0 or t = T on any of the three meshes (counted by `generate_data.py`); q lies in [0.0948, 0.6686] on order 21 and [0.0909, 0.6626] on f0. q > 0 everywhere.

## Paper versus code

The PyApprox 1.0.3 code (the version current when the paper was accepted) differs from what the paper prints in five places. We follow the code, because it produced the paper's results:

1. **Source.** The paper prints G = s/(2πh²) exp(−|s − s_src|²/(2h²)) (p. 7); the code uses s exp(−|s − s_src|²/h²) (`gauss_forc_fun`). The PDE is linear in c, so the amplitude (100 vs 100/(2π·0.01) = 1591.5) only scales q and every eigenvalue by a constant factor and leaves the subspace unchanged; only the width differs (e-folding radius 0.1 in the code, 0.1·√2 in the paper).
2. **Boundary sign.** The paper prints K∇c·n = c/10 (Eq. (1), p. 4; the benchmark's docstring writes K∇u·n − 0.1u = 0, the same sign). That is an inward diffusive flux proportional to c, i.e. a boundary that injects tracer where tracer is present: unphysical. The code enforces K∇c·n + 0.1c = 0, the physical outflow condition; following it is clearly right.
3. **QoI.** Space integral at t = T (code, the benchmark's docstring and API documentation) versus space-time integral (one sentence of Sec. 3.5.6, p. 24). The code's definition is the documented one.
4. **KLE without Nyström weights.** The paper says the KLE is computed "using the Nyström method" (p. 7), but PyApprox's `MeshKLE` is called with `quad_weights=None`: `eigh(K)` of the unweighted 144 × 144 nodal kernel matrix on the boundary-clustered Chebyshev nodes, so the modes are weighted towards the nodes near the boundary rather than being a quadrature-weighted discretization of the kernel's eigenfunctions. `kle_basis.csv` keeps the code's basis.
5. **Source acting after t = 0.** The paper sets G = 0 for t > 0 (p. 8), but the first Crank-Nicolson step evaluates the source at t_0 = 0 (stage 0 uses f(t_0)), so the source acts for half a step, Δt/2 = 0.0016, after t = 0. The effect vanishes as Δt → 0; we keep it to match PyApprox.

**Effect of items 1 and 3** (`check_accuracy.py`, check 6: order 21, rows 0–299, exact gradients; the space-time QoI by the trapezoid rule over the Crank-Nicolson levels with tangent-linear gradients, checked against central differences at row 0 to 1.1e-7): the code's subspace has ratios λ1/λ2, λ2/λ3, λ3/λ4 = 2.91, 8.87, 9.56 on these rows; with the paper's printed source 2.67, 8.97, 9.58 and subspace distances to the code's of 0.033, 0.035, 0.018 for k = 1, 2, 3 (1.9°, 2.0°, 1.0°); with the space-time QoI 3.45, 8.07, 9.91 and distances 0.014, 0.051, 0.030 (0.8°, 2.9°, 1.7°). These are comparable to the order-21 bootstrap 95th percentiles at 500 rows (0.066, 0.039, 0.042). The check takes about 8 min in `.venv`. So the paper's printed versions change the subspace by at most 0.051 (2.9°), less than the f0 mesh does (9.6° for k = 2), and the qualitative result (dominant directions in z_1–z_4) does not depend on the choice. Item 2 is not tested: the printed sign is unphysical. Items 4 and 5 are part of the benchmark's definition.

## KLE basis (stored)

`kle_basis.csv` holds PyApprox 1.0.3's `MeshKLE` basis on the f0 mesh: the 10 leading eigenvectors v_d of the 144 × 144 kernel matrix (no quadrature weights) scaled by √λ_d, written by `pyapprox_reference.py --kle` (PyApprox 1.0.3, torch 2.2.2, numpy 1.26.4, scipy 1.11.4, Python 3.11, Linux x86_64, 2026-11-01; md5 1514976238ae5fe36f23cb566d526c56). Checked with numpy (`check_accuracy.py`, check 1): the v_d are orthonormal to 7e-16 and eigenvectors of the kernel matrix to a residual of 2e-14, and the λ_d match its 10 largest eigenvalues to 1e-14; λ11 = 0.2184, and the 10 modes carry 99.59 % of the variance. Four eigenvalues come in exactly equal pairs (by the x–y symmetry of the mesh), so within each pair the basis is an arbitrary rotation chosen by the eigensolver; the stored file fixes it, so z_d have the same meaning as in PyApprox 1.0.3. Do not recompute it.

## Regenerating

```bash
python generate_data.py --fd                                   # all files, written here, one process
python generate_data.py --out DIR --check --fd                 # all files, written to DIR and compared with the files here
python generate_data.py --out DIR --nmain 20 --nconv 5 --check # a subset
python check_accuracy.py [--skip-slow]                         # the numbers of "Accuracy" that no other script prints
```

`--check`, `--nmain` and `--nconv` refuse to run without `--out DIR` outside this directory, so that the shipped files are never overwritten. The inputs come from the seed, so a run reproduces every file. The shipped files were written on 2026-11-01 in a Linux container (Python 3.13). In `.venv` (Python 3.11, numpy 2.4.6, scipy 1.17.1, 2026-11-02) the full `--out DIR --check --fd` run took 517 s for the three meshes (order 21: 359 s for the 500 values and gradients, 0.72 s each; f0: 61 s, 0.12 s each; order 27: 97 s for 50, 1.93 s each; about 9.5 min including the finite-difference check), on one process of a Mac shared with other jobs; timings vary several-fold with load. It reproduced the inputs exactly and the outputs and gradients to a largest relative difference of 4.2e-10 and 1.6e-9 (order 21), 4.2e-11 and 1.8e-10 (f0) and 1.6e-9 and 4.9e-9 (order 27). The stated tolerance is 1e-8 relative: the differences come from rounding in the dense LU solves and change between platforms.

## Accuracy

- **Against PyApprox 1.0.3**: an external run (PyApprox 1.0.3 in a separate Python 3.11 venv in a Linux container, 2026-11-01). `pyapprox_reference.py --compare` repeats its comparison of q where PyApprox 1.0.3 is installed (it exits with an install message otherwise): q agrees to a relative 2e-13 – 9e-13 at three random points, and q(0) = 0.537336364601 (`tracer_model.py`, recomputed by `check_accuracy.py`, check 4) against PyApprox's 0.537336364600886. The comparisons of the internals in the same run (the mesh, derivative matrices to 1.4e-14 and boundary index sets identical; steady and final states to 1e-12; the lower-fidelity members, order 7 and 16 steps, to 1e-12) used code that is not in the repository and are not reproducible from it. One PyApprox evaluation took about 4.5 s there; one `tracer_model.py` value on f0 takes 0.08 s in `.venv` (`check_accuracy.py`, check 4, on a shared machine).
- **Gradients against central differences** (h = 1e-3, relative 2-norm error per row; `generate_data.py --fd`, `.venv`): 1.2e-7, 4.2e-7, 2.6e-7, 3.6e-7, 7.3e-7 at rows 0–4 on order 21, and 1.1e-7, 6.0e-8, 2.0e-7, 1.3e-7, 1.1e-7 on f0 (the notebook's check cell repeats row 0 on order 21: 1.2e-7). h = 1e-4 and 1e-5 give larger errors (6e-8 to 8e-5, growing as h shrinks: rounding; `check_accuracy.py`, check 5), so the agreement is at the finite-difference floor.
- **Discretization.** Paired comparison on the shipped rows (printed by the notebook). f0 against order 21, 500 rows: q differs by a median of 1.25 % and at most 9.31 %; the gradients by a relative 2-norm error of median 0.148, 90th percentile 0.285, max 0.668; the 1-, 2- and 3-D subspaces by 0.074 (4.2°), 0.166 (9.6°) and 0.077 (4.4°), against the order-21 bootstrap 95th percentiles 0.066, 0.039, 0.042. Order 21 against order 27, rows 0–49: q at most 0.19 %, gradients at most 0.72 %, subspaces 0.001, 0.002, 0.003 (f0 against order 27: 0.076, 0.178, 0.088). Order 21 is therefore converged for this purpose and is the main data; f0 is accurate in q to a median 1 % but not in the gradients, which is why it is shipped only as the labelled benchmark variant. On rows 0–3 (`check_accuracy.py`, checks 2 and 3): q with 64 steps differs from q with 1024 steps by 0.04–0.06 % (f0 and order 21 alike), and q with 16 steps, the coarsest time step of PyApprox's fidelities, from q with 64 steps by 0.12–0.19 %; against order 35, order 21 differs by at most 0.04 %, order 11 (f0) by 0.2–2.3 %, and order 7 (PyApprox's coarsest mesh) by 25–32 % (row 0: q = 0.304 on order 7 against 0.438 on f0).

## How these files relate to the paper

- The model is the paper's benchmark (PyApprox `setup_multi_index_advection_diffusion_benchmark(kle_nvars, kle_length_scale=0.5, time_scenario=True)`), re-implemented in numpy/scipy and matched to PyApprox 1.0.3's highest-fidelity member f0 to rounding. The main files use the same model on a finer mesh (order 21) than any PyApprox fidelity; `*_f0.csv` are the benchmark's own discretization. Why a re-implementation: PyApprox 2.0.0 (current on PyPI) no longer has the `benchmarks` module; 1.0.3 builds only from a Python 3.11 wheel with torch, numpy < 2 and scipy < 1.12, has no gradient for the transient model ("return_grad=True is not supported"), and takes 4.5 s per value.
- The number of KLE terms (10), the mesh of the main files (order 21), the prior distribution, the sample size and the seed are ours. The paper's use of the model (ACV with posterior samples) is not reproduced.
- Overlap with `uq_benchmarks/Elliptic_PDE_Random_Coefficients`: both are unit-square PDEs with a log-normal KLE coefficient, standard normal inputs and an integral QoI. This one is transient, with advection, Robin conditions on all sides, a squared-exponential kernel with 10 modes (vs. an exponential kernel with 100), spectral collocation, and a subdomain QoI.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The model is J. D. Jakeman's PyApprox benchmark; PyApprox is MIT-licensed (Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS)); no PyApprox code is included here, and `kle_basis.csv` is output of PyApprox 1.0.3. Cite J. D. Jakeman, Environ. Model. Softw. 170:105825, 2023, doi:10.1016/j.envsoft.2023.105825.
