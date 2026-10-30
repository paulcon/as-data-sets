# Update specs (2026-10-29)

Review of every notebook in `paulcon/as-data-sets` for the port to Python 3 and `active_subspaces` v1, without pandas or numpad. Each spec covers inventory, text and math versus the literature, code, data management, the v1 call map, an update plan, acceptance criteria, prototype results and open questions. Start with [TODO.md](TODO.md).

## Paths

On 2026-10-30 the data sets moved into category directories and were renamed, and each notebook was renamed after its directory. The specs below were written before the move and use the old paths. Map:

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

The prototype scripts in `prototypes/` were updated to the new paths. Data file names did not change.

| Directory | m (verified) | Most serious finding | Spec |
|---|---|---|---|
| Ebola | 8 | Eigenvalues 256× too small (extra quadrature factor) | [Ebola.md](Ebola.md) |
| HIV | 27 | Legacy solver no longer reproduces the (correct) shipped data in Python 3 | [HIV.md](HIV.md) |
| MHD | 5 | Hartmann eigenvalues 32× too small (extra quadrature factor) | [MHD.md](MHD.md) |
| Elliptic_PDE | 100 | SENS index off by one; surrogate length scale edited after training | [Elliptic_PDE.md](Elliptic_PDE.md) |
| MCMC | 100 | Chains unreproducible (seeds and synthetic data never saved); 80 MB .mat | [MCMC.md](MCMC.md) |
| HyShotII | 7 | Validation runs inside the bootstrap; safe ranges differ from paper Table 4 | [HyShotII.md](HyShotII.md) |
| NACA0012 | 18 | Gradients are physical, comment says normalized | [NACA0012.md](NACA0012.md) |
| ONERA-M6 | 50 | Quoted R² are in-sample; leave-one-out is much lower for drag | [ONERA-M6.md](ONERA-M6.md) |
| Reentry_freestream_conditions (excluded from main) | 7 | Wrong variance for uniform inputs; cited paper not found | [Reentry_freestream_conditions.md](Reentry_freestream_conditions.md) |
| Hydrology | 19 | Quadratic prediction bands use a linear formula | [Hydrology.md](Hydrology.md) |
| Stomatal_Resistance | 20 | Hard-coded sign flip breaks under v1; hidden nominal row | [Stomatal_Resistance.md](Stomatal_Resistance.md) |
| Subsurface_Permeability | 100 | 31% zero runoffs silently dropped from log plot; meaningless 3-D OLS R² | [Subsurface_Permeability.md](Subsurface_Permeability.md) |
| Lithium_Battery | 19 | One run silently dropped; 67 MB text data | [Lithium_Battery.md](Lithium_Battery.md) |
| NREL_Wind | 5 | Text says ~500,000 samples, file has 50,000; inputs not on [−1, 1] | [NREL_Wind.md](NREL_Wind.md) |
| SingleDiodePV | 5 | Stored gradients are normalized; passing them to v1 as physical scales them twice | [SingleDiodePV.md](SingleDiodePV.md) |
| Atacamac (dev) | 38 | Chain-rule sign error changes eigenvalue ratios and R² | [Atacamac.md](Atacamac.md) |
| LM104X (dev) | 105 | Data release unconfirmed; notebook says 50 inputs | [LM104X.md](LM104X.md) |

Papers: the published versions used for these checks are in `../papers/` (not for redistribution).

Cross-cutting documents:

- [BRANCH_MERGE.md](BRANCH_MERGE.md): master + dev → main, with commands, GitHub steps and hygiene.
- [ACTIVE_SUBSPACES_V1_CHANGES.md](ACTIVE_SUBSPACES_V1_CHANGES.md): verified changes needed in the `active_subspaces` v1 library, for its maintainer agent (repro scripts in `prototypes/v1report/`).
- [UNKNOWNS.md](UNKNOWNS.md): every question the author answered as unknown, which the notebooks and DATA.md must state as unknown. Add new unknowns here.
- [numpad_replacement.md](numpad_replacement.md): scipy.sparse finite-volume solver with adjoint gradients, validated against the legacy residual and finite differences.
- [prototypes/](prototypes/): the scripts the reviewers ran (Python 3.11 `.venv`, v1). They hard-code absolute paths to this checkout.

The general guide for building a new notebook from a paper is the Claude Doc "Plan: New Active Subspaces Notebook from a Published Article" (https://claude.ai/artifact/TrkTwESVSbEwLbFzj2Ue26).
