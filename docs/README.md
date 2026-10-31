# Repository documentation

Notes about the repository, not data sets.

- [TODO.md](TODO.md): the open work and the record of decisions, with links to the PRs. This is the one file here that is still updated.
- This file: the two kinds of data sets (the rules every data set follows) and the table of data sets by kind (below).
- [port-2026/](port-2026/): the October 2026 review for the port to Python 3 and `active_subspaces` v1, without pandas or numpad, kept as a record. It holds one spec per 2016 notebook (inventory, text and math versus the literature, code, data management, the v1 call map, an update plan, acceptance criteria, prototype results and open questions), the cross-cutting documents listed at the end, and the acceptance scripts for the `active_subspaces` changes. The DATA.md files cite these specs by section. The port is finished, so these files are not updated; they were written in `specs/` (moved here on 2026-10-31) and still refer to that directory, and to the old data set paths mapped below. The working files of the review (the other prototype scripts and their outputs, `BRANCH_MERGE.md`, `ACTIVE_SUBSPACES_V1_CHANGES.md`, `ACTIVE_SUBSPACES_V1_RESPONSE.md` and `Reentry_freestream_conditions.md`) were removed on 2026-10-31; they are kept at the tag [`port-review-2026`](https://github.com/paulcon/as-data-sets/tree/port-review-2026/docs/port-2026), which is where the specs' mentions of `prototypes/g1/` … `prototypes/numpad/` and of those documents point.

## Paths

On 2026-10-30 the data sets moved into category directories and were renamed, and each notebook was renamed after its directory. The specs in `port-2026/` were written before the move and use the old paths. Map:

| Old directory (old notebook) | New directory and notebook |
|---|---|
| `HyShotII/` (`HyShotII.ipynb`) | `aerospace_cfd/HyShotII_Scramjet/HyShotII_Scramjet.ipynb` |
| `NACA0012/` (`active-subspace-naca0012.ipynb`) | `aerospace_cfd/NACA0012_Airfoil/NACA0012_Airfoil.ipynb` |
| `ONERA-M6/` (`active-subspace-oneraM6.ipynb`) | `aerospace_cfd/ONERA_M6_Wing/ONERA_M6_Wing.ipynb` |
| `LM104X/` (`active-subspace-LM104X.ipynb`) | `aerospace_cfd/N2_Supersonic_Jet/N2_Supersonic_Jet.ipynb` |
| `Hydrology/` (`PF-CLM_Active_Subspaces.ipynb`) | `hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/Land_Surface_Energy_Fluxes.ipynb` |
| `Stomatal_Resistance/` (`Stomatal_Resistance.ipynb`) | `hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/Photosynthesis_Stomatal_Resistance.ipynb` |
| `Subsurface_Permeability/` (`Subsurface_permeability.ipynb`) | `hydrology_subsurface_geoscience/Runoff_Permeability_Field/Runoff_Permeability_Field.ipynb` |
| `Lithium_Battery/` (`Li_Battery.ipynb`) | `energy/Lithium_Ion_Battery/Lithium_Ion_Battery.ipynb` |
| `MHD/` (`MHD.ipynb`) | `plasma_fusion/MHD_Generator/MHD_Generator.ipynb` |
| `NREL_Wind/` (`NREL_Wind.ipynb`) | `energy/Offshore_Wind_Turbine_Fatigue/Offshore_Wind_Turbine_Fatigue.ipynb` |
| `SingleDiodePV/` (`active-subspace-pv.ipynb`) | `energy/Single_Diode_Solar_Cell/Single_Diode_Solar_Cell.ipynb` |
| `Ebola/` (`Ebola.ipynb`) | `epidemiology_systems_biology/Ebola_SEIR/Ebola_SEIR.ipynb` |
| `HIV/` (`HIV.ipynb`) | `epidemiology_systems_biology/HIV_Long_Term_Dynamics/HIV_Long_Term_Dynamics.ipynb` |
| `Elliptic_PDE/` (`PDE_notebook.ipynb`) | `uq_benchmarks/Elliptic_PDE_Random_Coefficients/Elliptic_PDE_Random_Coefficients.ipynb` |
| `MCMC/` (`MCMC.ipynb`) | `uq_benchmarks/PDE_Inverse_Problem_MCMC/PDE_Inverse_Problem_MCMC.ipynb` |

The prototype scripts in `port-2026/prototypes/` were updated to the new paths. Data file names did not change.

## Two kinds of data sets

Every data set is one of two kinds, and the kind decides what the port must deliver, what DATA.md records, and what "reproducible" means.

**Stored data.** An outside model generated the data: a CFD, hydrology, battery or wind code, or MATLAB code that is not in the repo. The shipped files are the only record of those runs and cannot be regenerated here.
- Never regenerate, refit or "correct" stored values. Allowed changes: lossless format conversion (text → CSV with a `#` header, or `.npz`) by a committed script that checks the values are unchanged, and full originals moved to Zenodo (decision §0). Rows the analysis drops (duplicates, overshoots, validation runs) are dropped in the notebook by a stated rule or a stored flag, never deleted from the file.
- DATA.md records the generating code and version, who ran it, the input ranges and distributions, units, the gradient coordinates (physical or normalized), license and credit, and every unknown (see [UNKNOWNS.md](port-2026/UNKNOWNS.md)).
- Acceptance: the notebook's numbers reproduce from the shipped files.
- A Python re-implementation may be added to *check* stored data (SingleDiodePV's `single_diode.py`), but the shipped files stay the record, and any disagreement is reported, not fixed.

**Python model.** The data-generating model is Python code in the data set's directory, so its outputs can be regenerated and the notebook can call the model as needed.
- The model lives in a module named `*_model.py` (e.g. `HIV_model.py`, `pde_model.py`). It imports only numpy and scipy and loads its own files relative to `__file__`, so it works from any working directory. A closed-form model may stay in the notebook if it is a few lines, but a module is preferred so it can be called outside the notebook.
- Each shipped output file (samples, gradients, chains) is written by `generate_data.py` in the directory, with fixed seeds and all settings saved, alongside the data or in the script. The script must reproduce the shipped files to a stated tolerance. Its runtime is stated in DATA.md and the notebook.
- Inputs the model needs but that cannot be regenerated (Elliptic_PDE's KL bases, whose generating script is lost) are treated as stored data.
- The notebook calls the model directly when that is cheap, and loads the shipped files when regeneration is expensive.
- DATA.md records how to regenerate, the runtime, the tolerance, and how the shipped files relate to the paper's original runs (e.g. HIV's CSVs are a Python regeneration, not the paper's MATLAB runs).
- Acceptance: `generate_data.py`, or the notebook's own model calls, reproduces the shipped files and the notebook's numbers.

| Data set (spec) | Kind | Model in the repo | Outside generating code (stored data) |
|---|---|---|---|
| Ebola | Python model | `ebola_model.py` (closed-form R0 and its gradient); no data files | — |
| HIV | Python model | `HIV_model.py` (ODE); `generate_data.py` writes the CSVs | — |
| MHD | both | Hartmann problem: closed-form `hartmann_model.py` (was `gradients.py`) | 3-D MHD generator: Drekar (Sandia), 483 runs with adjoint gradients |
| Elliptic_PDE | Python model | `pde_model.py` (shared solver); KL bases `kl-*.npz` are stored | — |
| MCMC | Python model | `pde_model.py` (shared solver); `generate_data.py` writes the chains | — |
| HyShotII | stored data | — | HyShot II scramjet simulations (JCP 2015; book §5.1) |
| NACA0012 | stored data | — | SU2 with adjoint gradients |
| ONERA-M6 | stored data | — | SU2 with adjoint gradients |
| LM104X | stored data | — | SU2 with adjoint gradients (Lukaczyk dissertation) |
| Hydrology | stored data | — | ParFlow-CLM |
| Stomatal_Resistance | stored data | — | ParFlow-CLM |
| Subsurface_Permeability | stored data | — | ParFlow |
| Lithium_Battery | stored data | — | Newman-model runs by Hadigol et al. |
| NREL_Wind | stored data | — | FAST runs by Graf (NREL) |
| SingleDiodePV | stored data | `single_diode.py`, which checks the data | Campanelli's MATLAB code |
| Atacamac (excluded from main; Annular_Combustor_Thermoacoustics replaces it) | stored data | — | ATACAMAC annular-combustor model runs (Bauerheim et al., JFM 2016) |
| Reentry_freestream_conditions (excluded from main) | stored data | — | EXPERT reentry simulations |

The 15 data sets added in October 2026 (14 in [#56](https://github.com/paulcon/as-data-sets/pull/56), Transonic_Fan_Blade_Performance_Maps in [#60](https://github.com/paulcon/as-data-sets/pull/60)) have no spec in this directory; their DATA.md files record the kind. They are listed by directory name.

| Data set (directory) | Kind | Model in the repo | Outside generating code (stored data) |
|---|---|---|---|
| Annular_Combustor_Thermoacoustics | Python model | `annular_combustor_model.py` (a new implementation of the ATACAMAC dispersion relation); `generate_data.py` writes the CSVs | — |
| Fire_Satellite_Coupled_System | Python model | `fire_satellite_model.py`; `generate_data.py` writes the CSVs | — |
| Geological_CO2_Storage_Leakage | Python model | `co2_leakage_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Golinski_Speed_Reducer | Python model | `golinski_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Hodgkin_Huxley_Action_Potential | Python model | `HH_model.py`; `generate_data.py` writes the CSVs | — |
| Lillgrund_FLORIS_Wake_Model | Python model | `lillgrund_model.py` (numpy port of FLORIS 2.0.1); `generate_data.py` writes the CSVs | — |
| LS89_Turbine_Blade_Envelopes | stored data | — | SU2 runs by Wong, Seshadri, Scillitoe, Duncan and Parks (equadratures/data-sets); `convert_data.py` converts them losslessly |
| Luo_Rudy_Cardiac_Action_Potential | stored data | — | MATLAB LR1991 runs by Chang, Strong and Clayton (PLOS ONE 2015, S1 and S2 Data); `convert_data.py` converts them losslessly |
| River_Flood_Dyke_Overflow | Python model | `flood_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| STAT5_Dimerization_Signaling | Python model | `STAT5_model.py`; `generate_data.py` writes the CSVs | — |
| Steel_Arch_Inplane_Buckling | Python model | `arch_buckling_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Sulfur_Aerosol_Radiative_Forcing | Python model | `sulfur_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Transonic_Fan_Blade_Performance_Maps | stored data | — | Rolls-Royce Hydra RANS runs by Seshadri, Shahpar, Constantine, Parks and Adams (equadratures/data-sets); `convert_data.py` converts them losslessly |
| Vlasov_Poisson_Instability_Growth_Rate | Python model | `plasma_dispersion_model.py`; `generate_data.py` writes the CSVs | — |
| Zika_SEIR_SEI_Brazil | Python model | `zika_model.py`; `generate_data.py` writes the CSVs | the weekly case counts the model is fitted to (`data_cases.csv`, Brazilian Ministry of Health) are stored data |

## Specs

| Directory | Kind | m (verified) | Most serious finding | Spec |
|---|---|---|---|---|
| Ebola | Python model | 8 | Eigenvalues 256× too small (extra quadrature factor) | [Ebola.md](port-2026/Ebola.md) |
| HIV | Python model | 27 | Legacy solver no longer reproduces the (correct) shipped data in Python 3 | [HIV.md](port-2026/HIV.md) |
| MHD | both | 5 | Hartmann eigenvalues 32× too small (extra quadrature factor) | [MHD.md](port-2026/MHD.md) |
| Elliptic_PDE | Python model | 100 | SENS index off by one; surrogate length scale edited after training | [Elliptic_PDE.md](port-2026/Elliptic_PDE.md) |
| MCMC | Python model | 100 | Chains unreproducible (seeds and synthetic data never saved); 80 MB .mat | [MCMC.md](port-2026/MCMC.md) |
| HyShotII | stored data | 7 | Validation runs inside the bootstrap; safe ranges differ from paper Table 4 | [HyShotII.md](port-2026/HyShotII.md) |
| NACA0012 | stored data | 18 | Gradients are physical, comment says normalized | [NACA0012.md](port-2026/NACA0012.md) |
| ONERA-M6 | stored data | 50 | Quoted R² are in-sample; leave-one-out is much lower for drag | [ONERA-M6.md](port-2026/ONERA-M6.md) |
| Reentry_freestream_conditions (excluded from main) | stored data | 7 | Wrong variance for uniform inputs; cited paper not found | [Reentry_freestream_conditions.md](https://github.com/paulcon/as-data-sets/blob/port-review-2026/docs/port-2026/Reentry_freestream_conditions.md) |
| Hydrology | stored data | 19 | Quadratic prediction bands use a linear formula | [Hydrology.md](port-2026/Hydrology.md) |
| Stomatal_Resistance | stored data | 20 | Hard-coded sign flip breaks under v1; hidden nominal row | [Stomatal_Resistance.md](port-2026/Stomatal_Resistance.md) |
| Subsurface_Permeability | stored data | 100 | 31% zero runoffs silently dropped from log plot; meaningless 3-D OLS R² | [Subsurface_Permeability.md](port-2026/Subsurface_Permeability.md) |
| Lithium_Battery | stored data | 19 | One run silently dropped; 67 MB text data | [Lithium_Battery.md](port-2026/Lithium_Battery.md) |
| NREL_Wind | stored data | 5 | Text says ~500,000 samples, file has 50,000; inputs not on [−1, 1] | [NREL_Wind.md](port-2026/NREL_Wind.md) |
| SingleDiodePV | stored data | 5 | Stored gradients are normalized; passing them to v1 as physical scales them twice | [SingleDiodePV.md](port-2026/SingleDiodePV.md) |
| Atacamac (excluded from main) | stored data | 38 | Chain-rule sign error changes eigenvalue ratios and R² | [Atacamac.md](port-2026/Atacamac.md) |
| LM104X | stored data | 105 | Data release unconfirmed (cleared 2026-10-30); notebook says 50 inputs | [LM104X.md](port-2026/LM104X.md) |

Papers: the published versions used for these checks are in `papers/` at the repository root (not for redistribution).

Cross-cutting documents:

- [UNKNOWNS.md](port-2026/UNKNOWNS.md): every question the author answered as unknown, which the notebooks and DATA.md must state as unknown. Add new unknowns here.
- [numpad_replacement.md](port-2026/numpad_replacement.md): scipy.sparse finite-volume solver with adjoint gradients, validated against the legacy residual and finite differences.
- [prototypes/v1report/](port-2026/prototypes/v1report/): the three acceptance scripts for the `active_subspaces` v1 changes (`accept_spaces.py`, `accept_subspaces.py`, `accept_surrogates.py`), with `common.py` (data loaders) and `fvsolver.py` (the review's finite-volume solver, moved here from `prototypes/g2/`). Re-run them after an upstream change (TODO §5). They hard-code absolute paths to this checkout.

At the tag only: [BRANCH_MERGE.md](https://github.com/paulcon/as-data-sets/blob/port-review-2026/docs/port-2026/BRANCH_MERGE.md) (master + dev → main), [ACTIVE_SUBSPACES_V1_CHANGES.md](https://github.com/paulcon/as-data-sets/blob/port-review-2026/docs/port-2026/ACTIVE_SUBSPACES_V1_CHANGES.md) (the changes the port needed in `active_subspaces` v1, with repro scripts) and [ACTIVE_SUBSPACES_V1_RESPONSE.md](https://github.com/paulcon/as-data-sets/blob/port-review-2026/docs/port-2026/ACTIVE_SUBSPACES_V1_RESPONSE.md) (what landed, at `ca41b3d`).

The general guide for building a new notebook from a paper is the Claude Doc "Plan: New Active Subspaces Notebook from a Published Article" (https://claude.ai/artifact/TrkTwESVSbEwLbFzj2Ue26).
