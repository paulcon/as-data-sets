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
- Data generated for this repository with an outside code that is not a numpy/scipy model (SMRT, Cantera, ngspice, SIPNET; decision 1 of 2026-10-30, confirmed 2026-11-02) are stored data too. Their `generate_data.py` records how the files were made and can regenerate them where that code is installed, but the shipped files stay the record, and DATA.md names the tool, its version, and where the files were written.
  - When the outside code is not installed, `generate_data.py` exits with an error that names what to install and says to rerun it. The checks of a data set do not need the outside code: a data set passes them without it.
  - The outside code is rerun only when a problem is found in the shipped data (author, 2026-11-02).
  - A Python module that drives the outside code is named `*_driver.py` (e.g. `snow_smrt_driver.py`, `h2o2_ignition_driver.py`); `*_model.py` is reserved for numpy/scipy models.
- Numbers in a DATA.md Accuracy section that the notebook and the generating or converting script do not print come from `check_accuracy.py` in the data set directory. Like `generate_data.py`, it runs where the outside code it needs is installed and otherwise exits with an error naming what to install. A number that no script here can produce (a commercial code, an unrecoverable run) is labelled "external run, not reproducible from the repository".
- Data with no stated licence are hosted with the source credited (decision of 2026-10-30) and are listed, with their source, in the exceptions list of `LICENSE-DATA`.

**Python model.** The data-generating model is Python code in the data set's directory, so its outputs can be regenerated and the notebook can call the model as needed.
- The model lives in a module named `*_model.py` (e.g. `HIV_model.py`, `pde_model.py`). It imports only numpy and scipy and loads its own files relative to `__file__`, so it works from any working directory. A closed-form model may stay in the notebook if it is a few lines, but a module is preferred so it can be called outside the notebook.
- Each shipped output file (samples, gradients, chains) is written by `generate_data.py` in the directory, with fixed seeds and all settings saved, alongside the data or in the script. The script must reproduce the shipped files to a stated tolerance. Its runtime is stated in DATA.md (author, 2026-11-02).
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

The 28 data sets added in October 2026 (14 in [#56](https://github.com/paulcon/as-data-sets/pull/56), Transonic_Fan_Blade_Performance_Maps in [#60](https://github.com/paulcon/as-data-sets/pull/60), 13 in [#64](https://github.com/paulcon/as-data-sets/pull/64)) and the 65 added in November 2026 (25 in [#68](https://github.com/paulcon/as-data-sets/pull/68), 13 in [#76](https://github.com/paulcon/as-data-sets/pull/76), 27 in [#78](https://github.com/paulcon/as-data-sets/pull/78)) have no spec in this directory; their DATA.md files record the kind. They are listed by directory name.

| Data set (directory) | Kind | Model in the repo | Outside generating code (stored data) |
|---|---|---|---|
| AirfRANS_Airfoil_RANS_Forces | stored data | — | OpenFOAM v2112 RANS runs by Bonnet et al. (AirfRANS `Dataset.zip`, ODbL, not hosted); `convert_data.py` computes the force coefficients from the archive with the `airfrans` library |
| Ammonia_Syngas_Optimized_Kinetics | stored data | `ammonia_idt_driver.py` (a Cantera driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when Cantera and the mechanism file are available; `convert_mechanism.py` makes the mechanism file from the article's ESI zip (not hosted) | Cantera 3.2.0 with the optimized mechanism of Li, Zou and Yao (2023, ESI), run for this repository |
| Annular_Combustor_Thermoacoustics | Python model | `annular_combustor_model.py` (a new implementation of the ATACAMAC dispersion relation); `generate_data.py` writes the CSVs | — |
| Aquifer_Tracer_Advection_Diffusion | Python model | `tracer_model.py` (numpy/scipy re-implementation of the PyApprox 1.0.3 advection-diffusion benchmark); `generate_data.py` writes the CSVs | the KLE basis `kle_basis.csv` (computed by PyApprox 1.0.3) is stored data |
| Arctic_Snow_Depth_SMRT | stored data | `snow_smrt_driver.py` (sets up SMRT; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when SMRT is installed | SMRT 1.5.1 (LGPL-3.0, not shipped), run for this repository |
| Atmospheric_Selenium_Lifetime | stored data | — (`convert_data.py` copies the deposited inputs and integrates the deposited annual-mean fields over the globe) | SOCOL-AER with Se chemistry (ECHAM5 licence, code on request), 400 training and 50 validation runs by Feinberg et al. (ETH Research Collection 10.3929/ethz-b-000357105) |
| Atrial_Cell_Models_Courtemanche_Maleckar | stored data | — | MATLAB Courtemanche and Maleckar runs by Coveney and Clayton (figshare 11309756); `convert_data.py` converts them losslessly |
| BISON_Fuel_Performance_Surrogates | stored data | — (`convert_data.py` copies the pyMAISE 0.0.2 files losslessly) | BISON (INL, licensed) runs of the PCMI benchmark Case 1 by Radaideh and Kozlowski (2020), distributed in pyMAISE 0.0.2 (Apache-2.0) and Zenodo record 20140559 |
| Cavity_Shielding_Effectiveness | Python model | `cavity_se_model.py` (closed-form SE bound, numpy, complex-step gradients); `generate_data.py` writes the CSVs | — |
| CiPA_Ventricular_Proarrhythmia_Metrics | stored data | `cipaord_driver.py` (drives Myokit; not a numpy/scipy model); `generate_data.py` regenerates the CSV when Myokit is installed | Myokit 1.39.2 with `cipaord_endo.mmt` (a Myokit re-expression of CiPAORdv1.0), run for this repository |
| CityCOVID_Chicago_Calibration | stored data | — (`convert_data.py` converts the authors' CSVs losslessly) | CityCOVID agent-based model (Argonne; not public), 700 parameter sets × 50 seeds by Robertson et al. (github.com/sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest, MIT, commit e4c1e0e) |
| CMF_Soil_Moisture_SPOTPY | stored data | — (`convert_data.py` converts the PLOS ONE S4 zip) | CMF (GPL-3.0) driven by SPOTPY, runs by Houska et al. 2015 |
| CO2_Injection_Benchmark_Saturation | stored data | — (`generate_data.py` runs the benchmark's executable and converts its output losslessly) | the benchmark's finite-volume executable `prog` (Linux x86-64, no source; Zenodo 10.5281/zenodo.933827), run for this repository on the benchmark's 10,000 input samples |
| Cole_Cole_Cell_Monolayer_Impedance | Python model | `cole_model.py` (closed form, exact gradients); `generate_data.py` writes the CSVs | — |
| COVID19_Exit_Strategies_virsim | stored data | `R/strategies.R`, `R/run_chunk.R` (an R driver of virsim, not a numpy/scipy model); `generate_data.py` regenerates the runs when R and virsim 1.0.5 are installed; `convert_data.py` rebuilds the CSVs from the per-chunk run files | virsim 1.0.5 (L. E. Coffeng; R package, CC BY-NC-ND 4.0, not shipped), run for this repository on AWS EC2 (2288 runs, 63.9 process-hours) |
| COVID19_SEIR_Epidemic_Peak | Python model | `seir_model.py` (SEIR ODE with forward sensitivities); `generate_data.py` writes the CSVs | — |
| CovidSim_Cumulative_Deaths | stored data | — (`convert_data.py` converts the authors' HDF5 and `vary.py` losslessly) | CovidSim (MRC GIDA, Imperial College; GPL-3.0, version not stated), 3000 runs by W. Edeling on Altair (PSNC), published in github.com/wedeling/deep_active_subspace_data (MIT) |
| DTMB_5415_Hull_Resistance | stored data | `l2sea_driver.py` (drives L2-Sea; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when L2-Sea is built | L2-Sea (Fortran, GPL-3.0; A. Serani and M. Diez, CNR-INM) at commit 100da62, run for this repository |
| Earthquake_Rupture_Seismic_Moment | stored data (hybrid: outputs as printed, inputs regenerated) | `fdfault_driver.py` (drives fdfault; not a numpy/scipy model); `generate_data.py` regenerates the design inputs and, where fdfault is installed, the five reruns | fdfault runs by Daub et al. (Tables 2–3); five reruns with fdfault v1.0.1 for this repository |
| EMT_Switch_ODE | Python model | `emt_model.py` (the two-equation model, Eqs. 1-2 of Gasior 2026, with implicit-function gradients); `generate_data.py` writes the CSVs | — |
| Epoxy_MD_Force_Field_Sensitivity | stored data | — | LAMMPS runs by Edeling et al. (github.com/wedeling/MD-active-subspace tag 1.1; Zenodo 10.5281/zenodo.10817481); `convert_data.py` converts them losslessly |
| EU_DEMO_PROCESS_Major_Radius | stored data | `process_driver.py` and the deck `demo_template_IN.DAT` (run PROCESS; not a numpy/scipy model); `generate_data.py` regenerates the CSV when PROCESS is installed | PROCESS v3.4.2 (UKAEA, MIT), run for this repository on a public deck at the paper's input ranges, not the paper's runs |
| European_LV_Feeder_Probabilistic_Power_Flow | stored data | `opendss_driver.py` (drives OpenDSS; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when OpenDSSDirect.py is installed | OpenDSS (DSS C-API 0.14.5, OpenDSS SVN 3723) through OpenDSSDirect.py 0.9.4 (BSD), on the IEEE European LV Test Feeder files of dss-extensions/electricdss-tst at commit 3b20839, run for this repository |
| FaIR_Climate_Response_Calibration | stored data | `fair_driver.py` (sets up and runs FaIR; not a numpy/scipy model); `generate_data.py` regenerates the run files and the SSP2-4.5 emissions when FaIR is installed; `convert_data.py` converts the Zenodo posterior losslessly | FaIR 2.1.3 (Apache-2.0, not shipped), run for this repository on the fair-calibrate v1.4.1 posterior (Zenodo 10.5281/zenodo.10566813) |
| Fire_Satellite_Coupled_System | Python model | `fire_satellite_model.py`; `generate_data.py` writes the CSVs | — |
| Four_Chamber_Heart_Electromechanics | stored data | — | CARP + CircAdapt four-chamber runs and ToR-ORd runs by Strocchi et al. (Zenodo 10.5281/zenodo.7405335); `convert_data.py` converts them losslessly |
| Freyberg_MODFLOW_Forecast | stored data | `freyberg_driver.py` (writes the MODFLOW input, kriging and head interpolation; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when MODFLOW-2005 is installed | MODFLOW-2005 1.12.00 (USGS, public domain; double-precision build), run for this repository on the model files of the USGS data release 10.5066/P96A7ZC9 |
| Geological_CO2_Storage_Leakage | Python model | `co2_leakage_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Golinski_Speed_Reducer | Python model | `golinski_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| H2_Flame_Kinetics_Bayesian_Calibration | stored data | `h2_flame_driver.py` (a Cantera driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when Cantera is installed (`--procs`, `--qoi` for parallel or per-QoI runs) | Cantera 3.2.0 with the Burke et al. (2012) mechanism plus X1 and X6 (You et al. 2012), run for this repository (Su20 gradients on a 16-core AWS instance) |
| H2_NH3_Combustor_CFD_NOx | stored data | — (`extract_data.py` copies the outlet cell row of the deposited fields losslessly to `h2nh3_outlet_cells.npz`, reduces it to the outlet quantities of `h2nh3_outlet.csv`, and checks both with `--check`) | Ansys Fluent v22.1 RANS runs by Piscopo et al. (Zenodo 10.5281/zenodo.19348021, CC BY 4.0) |
| H2_O2_Ignition_Delay_Kinetics | stored data | `h2o2_ignition_driver.py` (a Cantera driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when Cantera is installed | Cantera 3.2.0 with the Li et al. (2004) mechanism, run for this repository |
| Hall_Thruster_Multi_Component | stored data | `spt100_components.py` (numpy cathode and plume components); `run_hallthruster.jl` drives HallThruster.jl; `generate_data.py` regenerates the CSV when Julia and HallThruster.jl are installed | HallThruster.jl v0.18.7 (MIT, Julia 1.13.3), run for this repository |
| Hat_Stiffened_Panel_Shear_Buckling | Python model | `panel_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| HBV_SASK_Oldman_Streamflow | Python model | `hbv_sask_model.py` (numpy reimplementation of VARS-TOOL's HBV-SASK, checked against it as an outside tool); `generate_data.py` writes the CSVs | the Oldman forcing (`oldman_daily.csv`, `oldman_monthly.csv`, `oldman_basin.csv`, from VARS-TOOL's `tutorials/Oldman Basin`, `convert_forcing.py`) is stored data |
| Hector_Matilda_Warming | stored data | `hector_driver.py` and `hector_worker.R` (an R driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when R, hector and matilda are installed | Hector 3.1.1 (GPL-3, not shipped) through R, with Matilda v1.0.0 (MIT), run for this repository |
| HIV_Three_Stage_Reservoir | Python model | `hiv3_model.py` (five-ODE three-stage HIV model with forward sensitivities); `generate_data.py` writes the CSVs | — |
| Hodgkin_Huxley_Action_Potential | Python model | `HH_model.py`; `generate_data.py` writes the CSVs | — |
| HTGR_Microreactor_Control_Drums | stored data | — | Serpent runs by Price, Radaideh and Kochunas (deanrp2/MicroControl); `convert_data.py` converts them losslessly |
| HyMod_Leaf_River_Rainfall_Runoff | Python model | `hymod_model.py` (clean-room numpy re-implementation of SAFE's `hymod_sim`); `generate_data.py` writes the CSVs | the Leaf River forcing (`forcing.csv`, `forcing_other_years.csv`, from SAFE's `LeafCatch.txt`) is stored data |
| Ice_Sheet_Basal_Melt_Ensemble | stored data | — (`convert_data.py` converts the authors' CSVs) | CISM v2.1, Antarctic 4 km runs by Berdahl et al. (github.com/mberdahl-uw/EmulatorPaper) |
| JAG_ICF_Implosion_Scalars | stored data | — (`convert_data.py` converts the released .npy members losslessly) | JAG semi-analytic ICF simulator (LLNL, not released), 10,000 runs released by Anirudh et al. in github.com/llnl/macc (MIT) |
| Kaesbach_Catchment_Subsurface_Flow | stored data | — | HydroGeoSphere runs by Erdal and Cirpka (residence times by Tecplot 360 particle tracking), from the HESS supplement; `extract_data.py` converts them losslessly |
| Karst_Aquifer_LuKARS_Calibration | Python model | `lukars_model.py` (numpy port of the authors' LuKARS R script); `generate_data.py` writes the data files | the measured forcing and discharge (`forcing.csv`, `monthly_mean_temp.csv`, from HydroShare) are stored data |
| Kurchatov_Sr90_Groundwater_Migration | stored data | — | MARTHE (BRGM) runs by Volkova, Iooss and Van Dorpe, as hosted by the VLSE (`marthedata.txt`); `convert_data.py` converts them losslessly |
| Lillgrund_FLORIS_Wake_Model | Python model | `lillgrund_model.py` (numpy port of FLORIS 2.0.1); `generate_data.py` writes the CSVs | — |
| Lithium_Ion_DFN_Drive_Cycle | stored data | `dfn_us06_driver.py` (sets up PyBaMM; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when PyBaMM is installed | PyBaMM 24.9.0 (BSD-3-Clause, not shipped), run for this repository |
| Lithium_Ion_Impedance_Calibration | stored data | `spme_eis_driver.py` (sets up PyBaMM/PyBOP and the adjoint gradient; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when they are installed | PyBaMM 26.10.1.1 and PyBOP 26.3 (BSD-3-Clause, not shipped), run for this repository |
| Lorenz96_Initial_State_Inference | Python model | `lorenz96_model.py` (RK4 with exact tangent-linear gradients); `generate_data.py` writes the CSVs | — |
| LS89_Turbine_Blade_Envelopes | stored data | — | SU2 runs by Wong, Seshadri, Scillitoe, Duncan and Parks (equadratures/data-sets); `convert_data.py` converts them losslessly |
| Luo_Rudy_Cardiac_Action_Potential | stored data | — | MATLAB LR1991 runs by Chang, Strong and Clayton (PLOS ONE 2015, S1 and S2 Data); `convert_data.py` converts them losslessly |
| Malaria_Bed_Net_Reproduction_Number | Python model | `malaria_model.py` (closed-form R_eff with exact gradient); `generate_data.py` writes the CSVs | — |
| MAPK_Cascade_Bistable_Oscillatory | Python model | `MAPK_model.py` (ODE with forward sensitivities); `generate_data.py` writes the CSVs | — |
| Marine_Carbonate_System_PyCO2SYS | stored data | `carbonate_driver.py` (drives PyCO2SYS; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when PyCO2SYS is installed | PyCO2SYS 1.8.3 (GPLv3, not shipped), run for this repository |
| Minnesota_COVID19_Deaths | stored data | `mn_covid_driver.py` and `mn_covid_eval.R` (an R driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when R 4.4.3 + plyr and the article's supplement are available | Minnesota COVID-19 model (Univ. of Minnesota, GPL-3.0, not shipped), `covid_allinone.R` from the supplement of Wycoff et al. (figshare 10.6084/m9.figshare.13611284), run for this repository with R 4.4.3 (900 runs) |
| NACA2412_Flapped_Airfoil | stored data | `xfoil_driver.py` (drives XFOIL; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when XFOIL is built | XFOIL 6.99 (GPL; M. Drela, H. Youngren), built from Ubuntu's source package, run for this repository; `davis_training_98.csv` holds the authors' own XFOIL runs (csdlpsu/uqchallenge2) |
| NACA4412_Shape_Lift_OpenFOAM | stored data | — (`convert_data.py` converts the authors' `.npy` files losslessly) | OpenFOAM v6 unsteady RANS runs with ITHACA-FV mesh morphing by Tezzele et al. (github.com/mtezzele/MORtech-2019-Special-Issue, commit ec05216; no licence stated), 170 runs |
| NH3_H2_Kinetics_Uncertainty | stored data | `nh3h2_kinetics_driver.py` (a Cantera driver, not a numpy/scipy model); `generate_data.py` regenerates the CSVs when Cantera and the supplement's mechanism file are present | Cantera 3.2.0 with the authors' NH3/H2 mechanism (paper supplement, linked, not redistributed), run for this repository |
| OpenAeroStruct_Wing_Aerostructural | stored data | `oas_wing_driver.py` (sets up OpenAeroStruct; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when OpenAeroStruct is installed | OpenAeroStruct 2.12.0 on OpenMDAO 3.45.2 (Apache-2.0, not shipped), run for this repository |
| Parcel_Droplet_Activation | stored data | `pyrcel_parcel_driver.py` (drives pyrcel; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when pyrcel is installed | pyrcel 2.0.0 (JAX/diffrax, BSD-3-Clause, not shipped), run for this repository |
| Pipe_Flow_Dimensional_Analysis | Python model | `pipe_model.py` (Colebrook friction factor by Newton, exact implicit gradients); `generate_data.py` writes the CSVs | — |
| Point_Heat_Source_Consolidation | Python model | `heat_source_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Powder_Bed_Fusion_Residual_Stress | stored data | — | Abaqus runs by Guo and Kramer (yulin-g/BPOF-optimization-AM); `convert_data.py` converts them losslessly |
| Pulmonary_Pulse_Wave_Network | stored data | `pulmonary_wk_driver.py` (drives the solver; constants only); `generate_data.py` regenerates the CSVs where the solver can be fetched and compiled | the authors' C++ solver `sor06` (Colebank_Identifiability_PDE at commit 9b2741f, CC BY-NC 4.0; cloned and compiled with g++ at run time, not shipped), run for this repository |
| PyGEM_Glacier_Mass_Balance | Python model | `pygem_mb_model.py` (numpy re-implementation of PyGEM v0.1.0 `runmassbalance`, checked against PyGEM v0.1); `generate_data.py` writes the CSVs | the glacier bins and ERA5 forcing (`glacier_bins.csv`, `climate_monthly.csv`, `site.csv`, reduced from OGGM sample data by `prepare_inputs.py`) are stored data |
| Pyloric_Circuit_Energy_Consumption | stored data | — (`generate_data.py` drives pyloric; no `*_driver.py`) | pyloric (mackelab, commit 89dd27d, MIT), run for this repository |
| PyPSA_Eur_Investment_Ranges | stored data | — (`convert_data.py` converts the authors' stored runs and PCE files losslessly) | PyPSA-Eur capacity-expansion LPs solved with Gurobi by Neumann and Brown (github.com/fneum/broad-ranges, results CC BY 4.0): 50,500 low-fidelity and 1,515 high-fidelity runs |
| Rabbit_Ventricular_Action_Potential | stored data | `shannon_driver.py` (drives Myokit; not a numpy/scipy model); `generate_data.py` regenerates the CSV when Myokit is installed | Myokit 1.39.2 with Yang et al.'s Shannon 2004 model file, run for this repository |
| RAE2822_SHAP_Design_Exploration | stored data | — | SU2 RANS runs by Palar et al. (optimuspram/SHAPMODE); `convert_data.py` converts them losslessly |
| Raven_Salmon_River_Blended_Model | stored data | `raven_salmon_driver.py` (fills the authors' xSSA deck templates, fetched at run time, runs Raven and reads its output; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when Raven is built | Raven (C++, commit cb745e1), run for this repository on the authors' xSSA deck and forcing; regenerated byte for byte on 2026-11-02 |
| Residential_Building_Stock_Energy_Use | stored data | — | EnergyPlus driven by jEPlus with DesignBuilder templates, run by Ali et al. (versions not stated; Mendeley Data 10.17632/m6vv9k9gcd.1, CC BY 4.0); `convert_data.py` extracts one building-type x weather x renewables cell losslessly |
| River_Flood_Dyke_Overflow | Python model | `flood_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Shared_Subspace_H2_Ignition_Flame | stored data | `h2_konnov_driver.py` (a Cantera driver, not a numpy/scipy model); `build_mechanism.py` rebuilds the mechanism file, which is not shipped; `generate_data.py` regenerates the CSVs when Cantera and the mechanism are available | Cantera 3.2.0 with Konnov's (2008) H₂ mechanism, run for this repository |
| Simply_Supported_Tube_Deflection | Python model | `tube_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| SIPNET_Ecosystem_Carbon_Flux | stored data | — (`generate_data.py` regenerates the CSVs with a compiled SIPNET binary) | SIPNET v2.2.0 (C), run for this repository in a substitute set-up, not the paper's runs |
| Speed_Reducer_Shaft_Limit_State | Python model | `shaft_model.py` (closed form, exact gradients); `generate_data.py` writes the CSVs | — |
| Stagnation_Temperature_Probe_Design | stored data | — | ANSYS Fluent RANS runs by Scillitoe, Ubald, Seshadri and Shahpar (equadratures/data-sets `probes/probes.npz`); `convert_data.py` converts them losslessly |
| STAT5_Dimerization_Signaling | Python model | `STAT5_model.py`; `generate_data.py` writes the CSVs | — |
| Steel_Arch_Inplane_Buckling | Python model | `arch_buckling_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Steel_Column_Reliability | Python model | `steel_column_model.py` (closed-form limit state, exact gradient, inverse-CDF maps); `generate_data.py` writes the CSVs | — |
| Subglacial_Drainage_GlaDS | stored data | — | GlaDS (ISSM v4.24) runs by Hill et al. (Zenodo 10.5281/zenodo.15053542, MIT); `convert_data.py` converts them losslessly |
| Sulfur_Aerosol_Radiative_Forcing | Python model | `sulfur_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Transonic_Airfoil_Shape_Parameterizations | stored data | — | SU2 runs by Grey and Constantine (zgrey/ASAP); `convert_data.py` converts the outputs losslessly and regenerates the design inputs |
| Transonic_Fan_Blade_Performance_Maps | stored data | — | Rolls-Royce Hydra RANS runs by Seshadri, Shahpar, Constantine, Parks and Adams (equadratures/data-sets); `convert_data.py` converts them losslessly |
| Truss_23_Bar_Midspan_Deflection | Python model | `truss_model.py` (bar finite elements with adjoint gradients); `generate_data.py` writes the CSVs | — |
| Tumour_Competition_ODE | Python model | `tumour_model.py` (Lotka-Volterra ODE, LSODA with forward sensitivities); `generate_data.py` writes `lv_samples.npz` | — |
| Two_DOF_Equipment_Structure_Oscillator | Python model | `oscillator_model.py` (closed form); `generate_data.py` writes the CSVs | — |
| Two_Material_Thermal_Compliance | Python model | `thermal_model.py` (P2 finite elements, exact adjoint gradient); `generate_data.py` writes `thermal_compliance.npz` | — |
| Two_Stage_CMOS_Op_Amp | stored data | `opamp_spice.py` and the circuit deck (run ngspice; not a numpy/scipy model); `generate_data.py` regenerates the CSVs when ngspice is installed | ngspice 42 with a BSIM4 45 nm model card, run for this repository |
| Undamped_Oscillator_Limit_State | Python model | `oscillator_model.py` (yielding oscillator solved event by event with forward-mode gradients; closed-form linear benchmark); `generate_data.py` writes the CSVs | — |
| Vlasov_Poisson_Instability_Growth_Rate | Python model | `plasma_dispersion_model.py`; `generate_data.py` writes the CSVs | — |
| Wrench_Elasticity_Displacement | Python model | `wrench_model.py` (P1 plane-stress finite elements, adjoint gradient) and `wrench_mesh.py`; `generate_data.py` writes the mesh, `kl_basis.npz` and the CSVs | — |
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
