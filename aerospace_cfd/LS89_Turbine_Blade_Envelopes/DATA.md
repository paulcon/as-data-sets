# LS89_Turbine_Blade_Envelopes: data

Kind: **stored data**. The values come from SU2 runs by Wong, Seshadri, Scillitoe, Duncan and Parks for "Blade Envelopes Part I". The authors published them as `blade_envelopes/blade_envelopes.npz` in the [equadratures/data-sets](https://github.com/equadratures/data-sets) repository. `convert_data.py` copied the four arrays the notebook uses into CSV files, losslessly. Nothing was rerun, refitted or corrected. The original archive (11.9 MB, six arrays) is not shipped. Its URL and SHA-256 are below, and `convert_data.py --check` rebuilds the CSVs from it.

## Files

| file | contents | role |
|---|---|---|
| `LS89_Turbine_Blade_Envelopes.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2022; outputs committed |
| `convert_data.py` | writes the four CSVs from `blade_envelopes.npz` and checks them bit for bit; `NAMES`, `LOWER`, `UPPER`, `SOURCE_URL`, `SOURCE_SHA256` | lossless conversion |
| `inputs.csv` | 1000 × 20: the FFD design variables `x1` … `x20`, scaled to [−1, 1] (npz key `X`) | stored inputs |
| `outputs.csv` | 1000 × 1: stagnation pressure loss coefficient `Yp`, dimensionless (npz key `loss`). Rows 74, 224 and 474 (0-based) are `inf` | stored output |
| `blade_x.csv` | 240 × 1: x coordinates of the blade-surface points, m. They are the same for every design (npz key `x_coords`) | stored geometry |
| `blade_y.csv` | 1000 × 240: y coordinates of the same points for each design, in the row order of `inputs.csv`, m (npz key `y_coords_random`) | stored geometry |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. `inf` reads as `numpy.inf`. Every value is written as the shortest decimal that reads back as the same double.

**Source archive.** `https://raw.githubusercontent.com/equadratures/data-sets/main/blade_envelopes/blade_envelopes.npz`, downloaded 2026-10-30. Size 11,939,157 bytes. SHA-256 `519dc43c8125a98ee5a2fadf6329eeef2edc4d68382aad2b8994929bae870395`. The equadratures function `eq.datasets.load_eq_dataset('blade_envelopes')` loads the same file. Its arrays are all float64:

| key | shape | shipped as | what it is (our reading) |
|---|---|---|---|
| `X` | (1000, 20) | `inputs.csv` | the 1000 random designs, scaled FFD variables |
| `loss` | (1000,) | `outputs.csv` | Y_p of the 1000 designs |
| `x_coords` | (240,) | `blade_x.csv` | surface x coordinates |
| `y_coords_random` | (1000, 240) | `blade_y.csv` | surface y coordinates of the 1000 designs. Their row order matches `X`: a linear map of `X` reproduces them to 6.2e-9 m, while the same fit with the rows of `X` reversed leaves a relative residual of 0.99 |
| `train_y_coords` | (5000, 240) | not shipped | probably the H = 5000 inactive-subspace designs of Sec. 4.2 (unused here) |
| `y_coords_loss` | (500, 240) | not shipped | probably the 500 loss-invariant designs run through CFD for Fig. 10. Their loss values are not in the archive |

## Model

Content of the paper card. Page, table and figure numbers refer to the arXiv preprint (arXiv:2011.11636v5, the accepted manuscript TURBO-20-1443). The published version is C. Y. Wong, P. Seshadri, A. Scillitoe, A. B. Duncan and G. Parks, "Blade Envelopes Part I: Concept and Methodology", J. Turbomach. 144(6), 2022, doi:10.1115/1.4053239. Check numbers against the ASME version before quoting them.

- **Simulator.** SU2 deforms the mesh and solves the steady 2-D RANS equations with the Spalart–Allmaras model. The flow is fully turbulent, with no transition model (Sec. 2.2, p. 4). The case is the VKI LS89 linear turbine cascade (Arts and Lambert de Rouvroit 1992): a highly loaded transonic nozzle guide vane with chord ≈ 67 mm and stagger 55° (p. 4). The inlet is stagnation-pressure driven, the outlet is at static pressure, the pitchwise boundaries are periodic, and the freestream turbulent viscosity ratio is 100. The code, meshes and configuration files are at https://github.com/ncywong/VKI_CFD (cited on p. 4; see "How these files relate to the paper").
- **Flow conditions** (Table 1, p. 5):

  | property | symbol | value |
  |---|---|---|
  | inlet stagnation pressure | p01 | 1.1 × 10⁶ N m⁻² |
  | inlet stagnation temperature | T01 | 592.295 K |
  | inlet density | ρ | 1.2866 kg m⁻³ |
  | freestream Reynolds number | Re | 6.0 × 10⁵ |
  | exit static pressure | p2 | 5.23 × 10⁵ N m⁻² |
  | heat capacity ratio | γ | 1.4 |

- **Inputs** (m = 20; Sec. 2.1, p. 4). There are d = 20 independent free-form-deformation design variables. They are "scaled to lie within the range −1 to 1", and the FFD nodes move only perpendicular to the inflow (Fig. 3).

  | name | symbol | units | nominal | range / distribution | source |
  |---|---|---|---|---|---|
  | `x1` … `x20` | x(1) … x(20) | scaled (−) | 0 (the LS89) | U[−1, 1], independent | Sec. 2.1, p. 4; density τ = 2⁻ᵈ, Eq. 7, p. 6 |

  The paper does not state the physical amplitude behind the scaling. In VKI_CFD `makedirs.py`, variable j (1-based) is the y-displacement of FFD control point (i, k) = ((j−1) // 10, (j−1) mod 10) of a box of degree (1, 9). The displacement is `0.01 · x_j` in mesh units (`bound = 0.01`). We have not verified that the published runs used this default. From the stored surfaces, the largest surface displacement per unit x_j is 0.63–4.42 mm for 18 of the variables. For **x10 and x20 it is zero** (below 6e-10 m), so these two variables move no stored surface point. The paper says the deformations "grossly exaggerate" real manufacturing deviations.
- **Quantity of interest.** The stagnation pressure loss coefficient Y_p, dimensionless (Eq. 2, p. 5), with p02 the circumferentially mass-averaged exit stagnation pressure. The preprint prints Eq. 2 as (p02 − p01)/(p02 − p2), which is negative. The authors' post-processing (`get_loss_from_file`, VKI_CFD `util_funcs.py`) computes **Y_p = (p01 − p02)/(p02 − p2)**, mass-averaged on planes at x = −0.054999 and x = 0.07999. All 997 finite values are positive (0.0297–0.1055). No transform is applied. The paper also records the exit mass flow function (Eq. 3) and the isentropic Mach number at 240 surface points (Eqs. 4–5). Neither is in the archive.
- **Failed runs.** In `get_loss_from_file`, `inf` is the value returned when no `flow.vtk` was written, or when the final density residual `Res_Flow[0]` (log10) in `history.csv` is above −8 ("diverged"). The archive has three `inf` losses. They are kept in `outputs.csv`, and the notebook drops them by the rule "loss not finite".
- **Gradients.** None from CFD. The paper estimates gradients from a sparse total-order polynomial of order 3 (O = 1771 terms), fitted by compressed sensing (ε = 0.001) with equadratures on 800 training runs. Its test R² is 0.957 on 200 runs (Sec. 4.1, pp. 12–13, Fig. 8). The notebook uses the library's global quadratic estimator (QPHD, 231 terms, M = 997 ≥ 2 × 231) and local linear gradients as a cross-check.
- **Sampling design.** K = 1000 uniform Monte Carlo samples in [−1, 1]²⁰ (Sec. 2.3, p. 5; Sec. 4.1, p. 12), split into 800 training and 200 test runs. No seed or split is given. VKI_CFD's `gendesign_doe.py` makes a Latin hypercube instead (2000 points). We have not checked which design the archive holds; the notebook needs only that the inputs are spread over the cube. A second set of 1000 designs with d = 30 (Sec. 4.4) is not in the archive.
- **Cost per run.** Not reported. VKI_CFD's `auto_run.py` runs each design with `mpirun -n 4 SU2_CFD`.
- **Published results.** Fig. 9(a) shows the eigenvalues of C_Yp: one dominant eigenvalue, read as ≈ 2.5e-4, then ≈ 1.0e-5 and ≈ 2.6e-6. The paper takes a 1-D active subspace (p. 13). Fig. 9(b) is the sufficient summary plot of the CFD loss: Y_p ≈ 0.03–0.105 over an active coordinate ≈ [−1.7, 1.7]. Fig. 8(b) is the polynomial validation (R² = 0.957). On p. 13, the "two-standard-deviation interval" of Y_p is 0.0185 for random designs and 0.0011 for 500 inactive-subspace designs. There is no eigenvector plot or table.

## Regenerating

```bash
python convert_data.py --npz /path/to/blade_envelopes.npz           # writes the four CSVs (0.7 s)
python convert_data.py --npz /path/to/blade_envelopes.npz --check   # rebuilds them in memory and compares
```

The script refuses an archive whose SHA-256 differs from the one above. It checks that each array has the expected shape and dtype, and that each CSV reads back with `np.loadtxt` equal to the source array bit for bit (as uint64), `inf` included. In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-10-31) `--check` reports all four identical and every array read back bit for bit, in 0.7 s. There is no randomness. Running SU2 again is a different matter: see the EMA notes kept outside the repository.

## Accuracy

- **Conversion:** exact (bit for bit, as above).
- **Outputs:** the authors' converged-residual criterion (log10 density residual ≤ −8) is the only stated accuracy control. The paper reports no mesh-convergence study and refers to earlier SU2 validation on this blade (ref. [18] of the paper). There are no gradients to check.
- **Geometry:** `blade_y.csv` is an affine function of `inputs.csv` to 6.2e-9 m, while the displacements reach 8.0 mm, so the stored surfaces are a linear FFD map of the design variables. The notebook uses this map to draw the active direction as a blade shape.

## How these files relate to the paper

- **The same runs.** These are the paper's CFD values as the authors published them, not a regeneration. With QPHD on the 997 finite runs, the notebook gets λ1 = 3.17e-4, λ2 = 1.24e-5 and n = 1. Local linear gives 2.52e-4 and 5.68e-6. The first eigenvectors of the two methods agree to |w1·w1| = 0.9957. The active-variable range and the positions of the highest losses match Fig. 9(b).
- **Three runs without a loss.** The paper says Fig. 9(b) plots "all 1000 geometries", but rows 74, 224 and 474 hold `inf`. The paper does not say how it treated them in the polynomial fit.
- **x10 and x20.** They move no stored surface point. Dropping them raises the leave-one-out R² of a quadratic (0.9208 → 0.9251), and their weights in w1 are within bootstrap noise of 0. The data are consistent with Y_p not depending on them. We do not know whether the SU2 meshes moved with them, or why they do not move the stored surfaces: with the Bézier blending of the VKI_CFD box (`gen_ffd_box.cfg`, degree (1, 9)), every control point would be expected to move the blade.
- **VKI_CFD vs Table 1.** The repository's `run_cfd.cfg` (commit `cfe6e19`, 2019-12-09) sets `FREESTREAM_PRESSURE= 1100000.0`, `FREESTREAM_TEMPERATURE= 592.295` and `REYNOLDS_NUMBER= 6.0E5`, as in Table 1. But its Riemann boundaries are a total pressure of 1.4e5 Pa and total temperature of 420 K at the inlet, and a static pressure of 0.90405e5 Pa at the outlet, a pressure ratio of 1.55. Table 1's p01/p2 = 1.1e6/5.23e5 is 2.10. `FREESTREAM_DENSITY` is 1.2886, while Table 1 gives 1.2866. Which settings produced the archive is a question for the authors.

## License and credit

The source repository, equadratures/data-sets, states **no licence**. It has no LICENSE file, and its README does not mention one; checked on 2026-10-30. The repository owner (P. Constantine) decided on 2026-10-30 that data without a licence may be hosted here. The four CSV files are therefore redistributed unchanged in value, with credit to C. Y. Wong, P. Seshadri, A. Scillitoe, A. B. Duncan and G. Parks, and to the equadratures/data-sets repository (https://github.com/equadratures/data-sets, `blade_envelopes/blade_envelopes.npz`, SHA-256 `519dc43c8125a98ee5a2fadf6329eeef2edc4d68382aad2b8994929bae870395`). This repository's CC BY 4.0 data licence (`LICENSE-DATA`) does not apply to them, since their authors have not licensed them. If the authors object or attach a licence, follow it. Code here (`convert_data.py`, the notebook): MIT (`LICENSE`). VKI_CFD (the run scripts and configuration files cited above) also has no licence file, so nothing from it is copied here. It was read only to document the setup.

Cite: C. Y. Wong, P. Seshadri, A. Scillitoe, A. B. Duncan and G. Parks, "Blade Envelopes Part I: Concept and Methodology", J. Turbomach. 144(6), 2022, doi:10.1115/1.4053239.
