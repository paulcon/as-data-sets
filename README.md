# Active subspaces

Many studies in computational science and engineering involve varying model inputs and examining effects on model outputs: averaging, approximating, calibrating, and optimizing. Unfortunately, these problems suffer from the _curse of dimensionality_, where the costs grow exponentially with the number of input parameters. *Active subspaces* are tools for subspace-based dimension reduction in the input parameter space of complex simulation models. This repository contains data sets and associated [Jupyter](http://jupyter.org) notebooks that apply active subspace methods to a range of science and engineering models to discover low-dimensional structures that are advantageous for parameter variation studies. The Python implementations of the methods are available on [Github](http://github.com/paulcon/active_subspaces/). More information on active subspaces can be found at [activesubspaces.org](http://activesubspaces.org).

# Data sets

Each data set's directory contains a Jupyter notebook (`.ipynb`), the data or model it analyzes, and a `DATA.md` that describes the files, their provenance and license, and what is unknown about them. Data files are mostly text (`.txt`, `.csv`) or NumPy `.npz`. Github renders the notebooks when you click an `.ipynb` file, but its rendering of mathematical expressions is not as robust as [nbviewer.org](https://nbviewer.org/). To view a notebook with nbviewer, paste its Github link into the box at [nbviewer.org](https://nbviewer.org/). Each notebook also has links to associated publications for more details.

[docs/using-the-notebooks.md](docs/using-the-notebooks.md) explains how to run the notebooks and read their results. [docs/notebook-index.md](docs/notebook-index.md) lists the notebooks in tables, one per category, that give each notebook's number of inputs, the dimension of the active subspace it finds, how it estimates the subspace, how it models the inputs, and what else it does with the subspace. The data sets are grouped by application area, one directory per category. Each data set is one of two kinds, marked at the end of its entry:

* **Stored data:** an outside simulation code generated the data. The files in the directory are the record of those runs. Most cannot be regenerated from this repository. Where the data were made for this repository with an outside code, `generate_data.py` can regenerate them where that code is installed, and otherwise exits with a message saying what to install.
* **Python model:** the model that generates the data is Python code in the directory (a `*_model.py` module, or a closed-form formula in the notebook). The notebook can call it as needed, and a `generate_data.py` script regenerates any shipped data files.

## Aerospace and CFD (`aerospace_cfd/`)

* **[AirfRANS_Airfoil_RANS_Forces](aerospace_cfd/AirfRANS_Airfoil_RANS_Forces/):** lift and drag of NACA 4- and 5-digit airfoils from the 1000 AirfRANS OpenFOAM RANS simulations. [[paper]](https://doi.org/10.52202/068431-1705) _Stored data._
* **[DTMB_5415_Hull_Resistance](aerospace_cfd/DTMB_5415_Hull_Resistance/):** calm-water resistance of the DTMB 5415 destroyer hull. [[paper]](https://doi.org/10.1016/j.jcp.2024.113542) _Stored data._
* **[HyShotII_Scramjet](aerospace_cfd/HyShotII_Scramjet/):** exit pressure from a multiphysics hypersonic scramjet. [[paper]](https://doi.org/10.1016/j.jcp.2015.09.001) _Stored data._
* **[LS89_Turbine_Blade_Envelopes](aerospace_cfd/LS89_Turbine_Blade_Envelopes/):** shape-design model of the profile loss of the VKI LS89 turbine cascade. [[paper]](https://doi.org/10.1115/1.4053239) _Stored data._
* **[N2_Supersonic_Jet](aerospace_cfd/N2_Supersonic_Jet/):** shape-design model of a Lockheed Martin N+2 supersonic passenger jet. [[dissertation]](https://purl.stanford.edu/xx611nd3190) _Stored data._
* **[NACA0012_Airfoil](aerospace_cfd/NACA0012_Airfoil/):** shape-design model for a NACA 0012 2D airfoil. [[chapter 5]](https://doi.org/10.1137/1.9781611973860) _Stored data._
* **[NACA2412_Flapped_Airfoil](aerospace_cfd/NACA2412_Flapped_Airfoil/):** lift, drag and moment of the NACA 2412 airfoil. [[paper]](https://doi.org/10.48550/arXiv.2606.28871) _Stored data._
* **[NACA4412_Shape_Lift_OpenFOAM](aerospace_cfd/NACA4412_Shape_Lift_OpenFOAM/):** late-time mean lift of a NACA 4412 airfoil. [[paper]](https://doi.org/10.1186/s40323-020-00177-y) _Stored data._
* **[ONERA_M6_Wing](aerospace_cfd/ONERA_M6_Wing/):** shape-design model of an ONERA M6 transonic wing. [[paper]](https://doi.org/10.2514/6.2014-1171) _Stored data._
* **[Pipe_Flow_Dimensional_Analysis](aerospace_cfd/Pipe_Flow_Dimensional_Analysis/):** pressure loss of viscous pipe flow. [[paper]](https://doi.org/10.48550/arXiv.1708.04303) _Python model._
* **[RAE2822_SHAP_Design_Exploration](aerospace_cfd/RAE2822_SHAP_Design_Exploration/):** drag and absolute pitching moment of the RAE 2822 airfoil. [[paper]](https://doi.org/10.1007/s00158-024-03769-z) _Stored data._
* **[Stagnation_Temperature_Probe_Design](aerospace_cfd/Stagnation_Temperature_Probe_Design/):** shape-design model of the pressure loss and temperature recovery of a Kiel stagnation-temperature probe. [[paper]](https://doi.org/10.1115/GT2020-16277) _Stored data._
* **[Transonic_Airfoil_Shape_Parameterizations](aerospace_cfd/Transonic_Airfoil_Shape_Parameterizations/):** transonic lift and drag of perturbed NACA 0012 airfoils under two global shape parameterizations. [[paper]](https://doi.org/10.2514/1.J056054) _Stored data._
* **[Transonic_Fan_Blade_Performance_Maps](aerospace_cfd/Transonic_Fan_Blade_Performance_Maps/):** a shape-design model of the efficiency and pressure ratio of Rolls-Royce transonic research fan blades. [[paper]](https://doi.org/10.1115/1.4038839) _Stored data._

## Hydrology, subsurface and geoscience (`hydrology_subsurface_geoscience/`)

* **[Aquifer_Tracer_Advection_Diffusion](hydrology_subsurface_geoscience/Aquifer_Tracer_Advection_Diffusion/):** tracer mass reaching a downstream subdomain in a transient advection–diffusion model of an aquifer. [[paper]](https://doi.org/10.1016/j.envsoft.2023.105825) _Python model._
* **[CMF_Soil_Moisture_SPOTPY](hydrology_subsurface_geoscience/CMF_Soil_Moisture_SPOTPY/):** the RMSE of a CMF Richards-equation soil column against topsoil soil moisture at the Linden FACE site. [[paper]](https://doi.org/10.1371/journal.pone.0145180) _Stored data._
* **[CO2_Injection_Benchmark_Saturation](hydrology_subsurface_geoscience/CO2_Injection_Benchmark_Saturation/):** the CO₂ saturation after 100 days of injection into a radially symmetric saline aquifer. [[paper]](https://doi.org/10.1007/s10596-018-9785-x) _Stored data._
* **[Earthquake_Rupture_Seismic_Moment](hydrology_subsurface_geoscience/Earthquake_Rupture_Seismic_Moment/):** integrated final slip (seismic moment) of a 2D dynamic rupture simulation of a rough fault. [[paper]](https://doi.org/10.1098/rsta.2020.0076) _Stored data._
* **[Freyberg_MODFLOW_Forecast](hydrology_subsurface_geoscience/Freyberg_MODFLOW_Forecast/):** the head forecast at a pumping well of the Freyberg (1988) water-table aquifer after a river is lined. [[paper]](https://doi.org/10.1111/gwat.12907) _Stored data._
* **[Geological_CO2_Storage_Leakage](hydrology_subsurface_geoscience/Geological_CO2_Storage_Leakage/):** a one-dimensional Darcy model of long-term CO₂ leakage from two geological storage sites. [[preprint]](https://doi.org/10.20944/preprints202609.2139.v1) _Python model._
* **[HBV_SASK_Oldman_Streamflow](hydrology_subsurface_geoscience/HBV_SASK_Oldman_Streamflow/):** Nash–Sutcliffe efficiency, mean flow and mean annual peak flow of the HBV-SASK rainfall-runoff model of the Oldman River (Alberta) over 1982–1996. [[paper]](https://doi.org/10.1016/j.envsoft.2018.10.005) _Python model._
* **[HyMod_Leaf_River_Rainfall_Runoff](hydrology_subsurface_geoscience/HyMod_Leaf_River_Rainfall_Runoff/):** Nash–Sutcliffe efficiency, mean flow and peak flow of the HyMod rainfall-runoff model of the Leaf River catchment (Mississippi, USA). [[paper]](https://doi.org/10.1016/j.envsoft.2015.04.009) _Python model._
* **[Kaesbach_Catchment_Subsurface_Flow](hydrology_subsurface_geoscience/Kaesbach_Catchment_Subsurface_Flow/):** stream discharge, boundary outflow and residence times from a HydroGeoSphere model of the Käsbach catchment (Germany). [[paper]](https://doi.org/10.5194/hess-23-3787-2019) _Stored data._
* **[Karst_Aquifer_LuKARS_Calibration](hydrology_subsurface_geoscience/Karst_Aquifer_LuKARS_Calibration/):** the Bayesian data misfit of the 21-parameter LuKARS lumped karst model calibrated to the discharge of the Kerschbaum spring (Austria). [[paper]](https://doi.org/10.1029/2019WR024739) _Python model._
* **[Kurchatov_Sr90_Groundwater_Migration](hydrology_subsurface_geoscience/Kurchatov_Sr90_Groundwater_Migration/):** Sr-90 concentrations predicted at 10 observation wells of the RRC Kurchatov Institute radwaste site. [[paper]](https://doi.org/10.1007/s00477-006-0093-y) _Stored data._
* **[Land_Surface_Energy_Fluxes](hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/):** an integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1016/j.cageo.2015.07.001) _Stored data._
* **[Photosynthesis_Stomatal_Resistance](hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/):** a model of stomatal resistance in an integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1175/JHM-D-16-0053.1) [[thesis]](http://hdl.handle.net/11124/170080) _Stored data._
* **[PyGEM_Glacier_Mass_Balance](hydrology_subsurface_geoscience/PyGEM_Glacier_Mass_Balance/):** glacier-wide specific mass balance 2000–2017 of Hintereisferner (Ötztal Alps) from the PyGEM v0.1.0 degree-day and mass-redistribution model. [[paper]](https://doi.org/10.1017/jog.2019.91) _Python model._
* **[Raven_Salmon_River_Blended_Model](hydrology_subsurface_geoscience/Raven_Salmon_River_Blended_Model/):** the 20-year mean flow, Nash–Sutcliffe efficiency and monthly mean flows of a Raven hydrologic model of the Salmon River (British Columbia). [[paper]](https://doi.org/10.5194/hess-24-5835-2020) _Stored data._
* **[Runoff_Permeability_Field](hydrology_subsurface_geoscience/Runoff_Permeability_Field/):** a subsurface permeability model for groundwater flow. [[paper]](https://doi.org/10.1016/j.advwatres.2016.03.020) _Stored data._
* **[Subglacial_Drainage_GlaDS](hydrology_subsurface_geoscience/Subglacial_Drainage_GlaDS/):** channel discharge fraction, sheet transit time and channel network length of the GlaDS subglacial drainage model on a synthetic ice-sheet margin. [[paper]](https://doi.org/10.5194/gmd-18-4045-2025) _Stored data._

## Climate, atmosphere and ocean (`climate_atmosphere_ocean/`)

* **[Arctic_Snow_Depth_SMRT](climate_atmosphere_ocean/Arctic_Snow_Depth_SMRT/):** a SMRT microwave radiative transfer model of the brightness temperatures of dry snow on Arctic sea ice in the 12 AMSR2 channels. [[paper]](https://doi.org/10.5194/tc-20-6157-2026) _Stored data._
* **[Atmospheric_Selenium_Lifetime](climate_atmosphere_ocean/Atmospheric_Selenium_Lifetime/):** SOCOL-AER aerosol-chemistry-climate model of the global atmospheric selenium cycle. [[paper]](https://doi.org/10.5194/acp-20-1363-2020) _Stored data._
* **[FaIR_Climate_Response_Calibration](climate_atmosphere_ocean/FaIR_Climate_Response_Calibration/):** FaIR 2.1.3 simple climate model on the 841-member fair-calibrate v1.4.1 posterior. [[paper]](https://doi.org/10.5194/gmd-17-8569-2024) _Stored data._
* **[Hector_Matilda_Warming](climate_atmosphere_ocean/Hector_Matilda_Warming/):** 2081–2100 warming under SSP2-4.5 in the Hector simple climate model. [[paper]](https://doi.org/10.1371/journal.pclm.0000295) _Stored data._
* **[Ice_Sheet_Basal_Melt_Ensemble](climate_atmosphere_ocean/Ice_Sheet_Basal_Melt_Ensemble/):** the Antarctic sea-level contribution at year 200 from CISM v2.1. [[paper]](https://doi.org/10.5194/tc-15-2683-2021) _Stored data._
* **[Lorenz96_Initial_State_Inference](climate_atmosphere_ocean/Lorenz96_Initial_State_Inference/):** a Bayesian inverse problem: the misfit of noisy observations of a chaotic 36-variable Lorenz-96 trajectory. [[paper]](https://doi.org/10.48550/arXiv.1712.02749) _Python model._
* **[Marine_Carbonate_System_PyCO2SYS](climate_atmosphere_ocean/Marine_Carbonate_System_PyCO2SYS/):** a PyCO2SYS model of seawater pCO₂, pH and carbonate saturation states in a measurement-uncertainty study and a surface-ocean study. [[paper]](https://doi.org/10.5194/gmd-15-15-2022) _Stored data._
* **[Parcel_Droplet_Activation](climate_atmosphere_ocean/Parcel_Droplet_Activation/):** the maximum supersaturation of an adiabatic cloud parcel. [[paper]](https://doi.org/10.1175/JAS-D-15-0223.1) _Stored data._

## Ecology and agriculture (`ecology_agriculture/`)

* **[SIPNET_Ecosystem_Carbon_Flux](ecology_agriculture/SIPNET_Ecosystem_Carbon_Flux/):** two-year totals of net ecosystem exchange, evapotranspiration and gross primary production from the SIPNET ecosystem model. [[paper]](https://doi.org/10.5194/bg-15-5801-2018) _Stored data._

## Energy: batteries, PV, wind, nuclear, power grids, buildings (`energy/`)

* **[BISON_Fuel_Performance_Surrogates](energy/BISON_Fuel_Performance_Surrogates/):** four fuel-performance outputs of 400 BISON runs of the PCMI benchmark Case 1 power ramp. [[paper]](https://doi.org/10.1016/j.ress.2019.106731) _Stored data._
* **[European_LV_Feeder_Probabilistic_Power_Flow](energy/European_LV_Feeder_Probabilistic_Power_Flow/):** phase-A voltage at bus 899, a line current and the feeder loss of the IEEE European LV Test Feeder. [[paper]](https://doi.org/10.1109/TPWRS.2016.2558622) _Stored data._
* **[HTGR_Microreactor_Control_Drums](energy/HTGR_Microreactor_Control_Drums/):** k<sub>eff</sub> and the quadrant powers of the HOLOS-Quad gas-cooled microreactor. [[paper]](https://doi.org/10.1016/j.nucengdes.2022.111776) _Stored data._
* **[Lillgrund_FLORIS_Wake_Model](energy/Lillgrund_FLORIS_Wake_Model/):** FLORIS wake model of the 48-turbine Lillgrund offshore wind farm. [[paper]](https://doi.org/10.3390/en14051293) _Python model._
* **[Lithium_Ion_Battery](energy/Lithium_Ion_Battery/):** a model of a lithium-ion battery. [[paper]](https://doi.org/10.1002/sam.11347) _Stored data._
* **[Lithium_Ion_DFN_Drive_Cycle](energy/Lithium_Ion_DFN_Drive_Cycle/):** the terminal voltage of the Doyle–Fuller–Newman lithium-ion cell model (PyBaMM) under a scaled US06 drive cycle. [[paper]](https://doi.org/10.1016/j.est.2026.121751) _Stored data._
* **[Lithium_Ion_Impedance_Calibration](energy/Lithium_Ion_Impedance_Calibration/):** the least-squares misfit between the impedance spectra of a grouped single particle model with electrolyte (SPMe) and synthetic data at nine states of charge. [[paper]](https://doi.org/10.1149/1945-7111/add41b) _Stored data._
* **[Offshore_Wind_Turbine_Fatigue](energy/Offshore_Wind_Turbine_Fatigue/):** a model of offshore wind turbine fatigue. [[model]](https://doi.org/10.1002/we.1870) _Stored data._
* **[PyPSA_Eur_Investment_Ranges](energy/PyPSA_Eur_Investment_Ranges/):** the total annual cost of a fully renewable European power system from PyPSA-Eur capacity-expansion runs. [[paper]](https://doi.org/10.1016/j.isci.2023.106702) _Stored data._
* **[Residential_Building_Stock_Energy_Use](energy/Residential_Building_Stock_Energy_Use/):** the energy use intensity of an Irish semi-detached house. [[paper]](https://doi.org/10.1016/j.dib.2024.110241) _Stored data._
* **[Single_Diode_Solar_Cell](energy/Single_Diode_Solar_Cell/):** a lumped-parameter single-diode photovoltaic solar cell model. [[paper]](https://doi.org/10.1002/sam.11281) _Stored data._

## Plasma and fusion (`plasma_fusion/`)

* **[EU_DEMO_PROCESS_Major_Radius](plasma_fusion/EU_DEMO_PROCESS_Major_Radius/):** the optimized major radius of an EU DEMO-like tokamak in the fusion systems code PROCESS. [[paper]](https://doi.org/10.1109/TPS.2022.3180233) _Stored data._
* **[Hall_Thruster_Multi_Component](plasma_fusion/Hall_Thruster_Multi_Component/):** corrected thrust and discharge current of a coupled cathode–discharge–plume model of the SPT-100 Hall thruster. [[paper]](https://doi.org/10.1063/5.0283796) _Stored data._
* **[JAG_ICF_Implosion_Scalars](plasma_fusion/JAG_ICF_Implosion_Scalars/):** LLNL's JAG semi-analytic inertial-confinement-fusion implosion simulator. [[paper]](https://doi.org/10.1073/pnas.1916634117) _Stored data._
* **[MHD_Generator](plasma_fusion/MHD_Generator/):** a magnetohydrodynamics power generation model. [[paper]](https://doi.org/10.1002/sam.11355) _Stored data and Python model._
* **[Vlasov_Poisson_Instability_Growth_Rate](plasma_fusion/Vlasov_Poisson_Instability_Growth_Rate/):** models of the growth rate of electrostatic plasma instabilities from the linearized Vlasov–Poisson dispersion relation. [[paper]](https://doi.org/10.1016/j.cnsns.2024.107994) _Python model._

## Epidemiology and systems biology (`epidemiology_systems_biology/`)

* **[CityCOVID_Chicago_Calibration](epidemiology_systems_biology/CityCOVID_Chicago_Calibration/):** cumulative COVID-19 deaths in Chicago on 2020-06-04 from CityCOVID, an agent-based model of 2.7 million agents. [[paper]](https://doi.org/10.1002/sim.70029) _Stored data._
* **[COVID19_Exit_Strategies_virsim](epidemiology_systems_biology/COVID19_Exit_Strategies_virsim/):** the peak intensive-care load and the IC overload of four COVID-19 exit strategies (flattening the curve, contact tracing, intermittent lockdown, phased opening) in virsim, a stochastic individual-based model of one million people. [[paper]](https://doi.org/10.1371/journal.pcbi.1009355) _Stored data._
* **[COVID19_SEIR_Epidemic_Peak](epidemiology_systems_biology/COVID19_SEIR_Epidemic_Peak/):** the time and size of the epidemic peak of an SEIR model of COVID-19 in Denmark. [[paper]](https://doi.org/10.1051/mmnp/2022014) _Python model._
* **[CovidSim_Cumulative_Deaths](epidemiology_systems_biology/CovidSim_Cumulative_Deaths/):** cumulative deaths after 801 days from Imperial College's CovidSim individual-based COVID-19 code. [[paper]](https://doi.org/10.1137/21M1463240) _Stored data._
* **[Ebola_SEIR](epidemiology_systems_biology/Ebola_SEIR/):** a model for the geographic spread of Ebola. [[paper]](https://doi.org/10.1016/j.amc.2017.11.039) [[arXiv]](https://arxiv.org/abs/1603.04955) _Python model._
* **[EMT_Switch_ODE](epidemiology_systems_biology/EMT_Switch_ODE/):** a two-equation switch model of the epithelial–mesenchymal transition. [[paper]](https://doi.org/10.1038/s41540-026-00757-5) _Python model._
* **[HIV_Long_Term_Dynamics](epidemiology_systems_biology/HIV_Long_Term_Dynamics/):** a dynamical system model for in-host HIV. [[paper]](https://doi.org/10.3934/mbe.2017040) _Python model._
* **[HIV_Three_Stage_Reservoir](epidemiology_systems_biology/HIV_Three_Stage_Reservoir/):** the CD4⁺ T-cell count at 5 and 10 years and the time to AIDS in a five-equation within-host HIV model. [[paper]](https://doi.org/10.1007/s00285-024-02056-1) _Python model._
* **[Malaria_Bed_Net_Reproduction_Number](epidemiology_systems_biology/Malaria_Bed_Net_Reproduction_Number/):** the closed-form effective reproduction number of a malaria model. [[paper]](https://doi.org/10.3390/mca24010032) _Python model._
* **[MAPK_Cascade_Bistable_Oscillatory](epidemiology_systems_biology/MAPK_Cascade_Bistable_Oscillatory/):** a three-state MAPK (RAF/MEK/ERK) cascade in its bistable and oscillatory regimes. [[paper]](https://doi.org/10.1371/journal.pcbi.1010651) [[model]](https://doi.org/10.1038/srep12569) _Python model._
* **[Minnesota_COVID19_Deaths](epidemiology_systems_biology/Minnesota_COVID19_Deaths/):** cumulative COVID-19 deaths over one year from the University of Minnesota age-structured compartmental model. [[paper]](https://doi.org/10.1080/10618600.2021.1874962) _Stored data._
* **[STAT5_Dimerization_Signaling](epidemiology_systems_biology/STAT5_Dimerization_Signaling/):** a benchmark ODE model of STAT5A/STAT5B phosphorylation and dimerization in intracellular signaling. [[paper]](https://doi.org/10.1093/bioinformatics/btz020) _Python model._
* **[Zika_SEIR_SEI_Brazil](epidemiology_systems_biology/Zika_SEIR_SEI_Brazil/):** a SEIR–SEI model of the 2016 Zika outbreak in Brazil. [[paper]](https://doi.org/10.1016/j.amc.2018.06.024) _Python model._

## Biomedical and cardiovascular (`biomedical_cardiovascular/`)

* **[Atrial_Cell_Models_Courtemanche_Maleckar](biomedical_cardiovascular/Atrial_Cell_Models_Courtemanche_Maleckar/):** 11 action-potential biomarkers of the Courtemanche and Maleckar human atrial cell models. [[paper]](https://doi.org/10.3389/fphys.2020.00364) _Stored data._
* **[CiPA_Ventricular_Proarrhythmia_Metrics](biomedical_cardiovascular/CiPA_Ventricular_Proarrhythmia_Metrics/):** eight proarrhythmia metrics of the CiPAORdv1.0 human ventricular myocyte. [[paper]](https://doi.org/10.3389/fphar.2019.01054) _Stored data._
* **[Cole_Cole_Cell_Monolayer_Impedance](biomedical_cardiovascular/Cole_Cole_Cell_Monolayer_Impedance/):** a 4-parameter Cole–Cole impedance of a cell monolayer. [[paper]](https://doi.org/10.3390/molecules27185861) _Python model._
* **[Four_Chamber_Heart_Electromechanics](biomedical_cardiovascular/Four_Chamber_Heart_Electromechanics/):** 20 pressure–volume features of a patient-specific four-chamber heart electromechanics simulator and 4 calcium-transient features of the ToR-ORd ventricular cell model. [[paper]](https://doi.org/10.1371/journal.pcbi.1011257) _Stored data._
* **[Hodgkin_Huxley_Action_Potential](biomedical_cardiovascular/Hodgkin_Huxley_Action_Potential/):** a Hodgkin–Huxley model of the nerve action potential. [[paper]](https://doi.org/10.1186/2190-8567-5-3) _Python model._
* **[Luo_Rudy_Cardiac_Action_Potential](biomedical_cardiovascular/Luo_Rudy_Cardiac_Action_Potential/):** a Luo–Rudy 1991 model of the ventricular cardiac action potential. [[paper]](https://doi.org/10.1371/journal.pone.0130252) _Stored data._
* **[Pulmonary_Pulse_Wave_Network](biomedical_cardiovascular/Pulmonary_Pulse_Wave_Network/):** pressures in a 1-D pulse-wave model of the main, left and right pulmonary arteries with Windkessel outlets. [[paper]](https://doi.org/10.1371/journal.pcbi.1013553) _Stored data._
* **[Pyloric_Circuit_Energy_Consumption](biomedical_cardiovascular/Pyloric_Circuit_Energy_Consumption/):** the energy consumption of a three-neuron conductance-based model of the crab pyloric network. [[paper]](https://doi.org/10.1073/pnas.2207632119) _Stored data._
* **[Rabbit_Ventricular_Action_Potential](biomedical_cardiovascular/Rabbit_Ventricular_Action_Potential/):** six action-potential and two Ca biomarkers of the Shannon et al. (2004) rabbit ventricular myocyte. [[paper]](https://doi.org/10.1371/journal.pone.0328261) _Stored data._
* **[Tumour_Competition_ODE](biomedical_cardiovascular/Tumour_Competition_ODE/):** the total tumour burden over 56 days in a two-population competitive Lotka–Volterra tumour-spheroid ODE. [[paper]](https://doi.org/10.3934/mbe.2026079) _Python model._

## Combustion and chemical kinetics (`combustion_chemical_kinetics/`)

* **[Ammonia_Syngas_Optimized_Kinetics](combustion_chemical_kinetics/Ammonia_Syngas_Optimized_Kinetics/):** ignition delays of ammonia at three shock-tube/RCM conditions. [[paper]](https://doi.org/10.1039/D3RE00160A) _Stored data._
* **[Annular_Combustor_Thermoacoustics](combustion_chemical_kinetics/Annular_Combustor_Thermoacoustics/):** a model of the azimuthal thermoacoustic modes of an annular combustor with 19 burners. [[paper]](https://doi.org/10.1017/jfm.2015.730) _Python model._
* **[H2_Flame_Kinetics_Bayesian_Calibration](combustion_chemical_kinetics/H2_Flame_Kinetics_Bayesian_Calibration/):** an ignition delay and two laminar flame speeds of hydrogen. [[paper]](https://doi.org/10.1016/j.combustflame.2019.04.023) _Stored data._
* **[H2_NH3_Combustor_CFD_NOx](combustion_chemical_kinetics/H2_NH3_Combustor_CFD_NOx/):** outlet NOₓ and temperature of a staged ammonia–hydrogen combustor. [[paper]](https://doi.org/10.1088/3049-4761/ae61b5) _Stored data._
* **[H2_O2_Ignition_Delay_Kinetics](combustion_chemical_kinetics/H2_O2_Ignition_Delay_Kinetics/):** the ignition delay of a fuel-rich hydrogen/oxygen mixture. [[paper]](https://doi.org/10.1016/j.combustflame.2019.03.006) _Stored data._
* **[NH3_H2_Kinetics_Uncertainty](combustion_chemical_kinetics/NH3_H2_Kinetics_Uncertainty/):** the NH₃/O₂/Ar shock-tube ignition delay and the NH₃/H₂-air laminar flame speed. [[paper]](https://doi.org/10.1016/j.fuel.2025.135810) _Stored data._
* **[Shared_Subspace_H2_Ignition_Flame](combustion_chemical_kinetics/Shared_Subspace_H2_Ignition_Flame/):** 16 ignition delays and 5 laminar flame speeds of hydrogen/air. [[paper]](https://doi.org/10.1016/j.combustflame.2017.11.021) _Stored data._

## Structural, materials and additive manufacturing (`structural_materials_manufacturing/`)

* **[Epoxy_MD_Force_Field_Sensitivity](structural_materials_manufacturing/Epoxy_MD_Force_Field_Sensitivity/):** the stiffness and Poisson ratio of a cross-linked epoxy resin from LAMMPS molecular dynamics. [[paper]](https://doi.org/10.1038/s41524-024-01272-z) _Stored data._
* **[Powder_Bed_Fusion_Residual_Stress](structural_materials_manufacturing/Powder_Bed_Fusion_Residual_Stress/):** the maximum residual stress after one electron-beam scan of a Ti–6Al–4V powder layer. [[paper]](https://doi.org/10.1007/s00158-025-04120-w) _Stored data._
* **[Steel_Arch_Inplane_Buckling](structural_materials_manufacturing/Steel_Arch_Inplane_Buckling/):** a closed-form model of the in-plane elastic buckling load of shallow pinned and fixed steel arches. [[paper]](https://doi.org/10.48084/etasr.3833) _Python model._
* **[Two_Material_Thermal_Compliance](structural_materials_manufacturing/Two_Material_Thermal_Compliance/):** the thermal energy of a square plate made of two conducting materials. [[paper]](https://doi.org/10.1080/0305215X.2025.2453542) _Python model._
* **[Wrench_Elasticity_Displacement](structural_materials_manufacturing/Wrench_Elasticity_Displacement/):** the vertical displacement of a point on a wrench modeled by plane-stress linear elasticity. [[paper]](https://doi.org/10.1137/18M1214123) _Python model._

## Electronics and circuits (`electronics_circuits/`)

* **[Cavity_Shielding_Effectiveness](electronics_circuits/Cavity_Shielding_Effectiveness/):** the matched power-balance bound on the shielding effectiveness of a slotted cylindrical cavity. [[paper]](https://doi.org/10.1115/1.4051906) _Python model._
* **[Two_Stage_CMOS_Op_Amp](electronics_circuits/Two_Stage_CMOS_Op_Amp/):** the gain, unity-gain bandwidth, phase margin and bias current of a 45 nm two-stage CMOS operational amplifier. [[paper]](https://doi.org/10.23919/DATE48585.2020.9116200) _Stored data._

## UQ benchmark collections (`uq_benchmarks/`)

* **[Elliptic_PDE_Random_Coefficients](uq_benchmarks/Elliptic_PDE_Random_Coefficients/):** a 100-parameter Poisson equation whose log-coefficient is a Karhunen–Loève expansion. [[paper]](https://doi.org/10.1137/130916138) _Python model._
* **[Fire_Satellite_Coupled_System](uq_benchmarks/Fire_Satellite_Coupled_System/):** a coupled three-discipline model of a fire-detection satellite (orbit, attitude control, power). [[paper]](https://doi.org/10.1115/1.4005619) _Python model._
* **[Golinski_Speed_Reducer](uq_benchmarks/Golinski_Speed_Reducer/):** the volume and two shaft constraints of Golinski's 6-parameter speed reducer. [[paper]](https://doi.org/10.1016/0022-2569(70)90064-9) _Python model._
* **[Hat_Stiffened_Panel_Shear_Buckling](uq_benchmarks/Hat_Stiffened_Panel_Shear_Buckling/):** a closed-form model of the shear buckling load of a hat-stiffened panel. [[report]](https://ntrs.nasa.gov/citations/19950011075) _Python model._
* **[OpenAeroStruct_Wing_Aerostructural](uq_benchmarks/OpenAeroStruct_Wing_Aerostructural/):** fuel burn, spar mass, lift and spar failure of a coupled aerostructural wing. [[paper]](https://doi.org/10.1007/s00158-018-1912-8) _Stored data._
* **[PDE_Inverse_Problem_MCMC](uq_benchmarks/PDE_Inverse_Problem_MCMC/):** data misfit of a parameterized PDE model. [[paper]](https://doi.org/10.1137/15M1042127) _Python model._
* **[Point_Heat_Source_Consolidation](uq_benchmarks/Point_Heat_Source_Consolidation/):** a closed-form model of the temperature, excess pore pressure and displacement around a point heat source buried in saturated soil or rock. [[paper]](https://doi.org/10.1002/nag.1610090206) _Python model._
* **[River_Flood_Dyke_Overflow](uq_benchmarks/River_Flood_Dyke_Overflow/):** a closed-form river flood model of the overflow above a dyke. [[paper]](https://doi.org/10.1016/j.matcom.2013.02.002) _Python model._
* **[Simply_Supported_Tube_Deflection](uq_benchmarks/Simply_Supported_Tube_Deflection/):** a closed-form model of the deflection and support slopes of a simply supported tube under an off-center point load. [[benchmark]](https://openturns.github.io/openturns/1.22/usecases/use_case_deflection_tube.html) _Python model._
* **[Speed_Reducer_Shaft_Limit_State](uq_benchmarks/Speed_Reducer_Shaft_Limit_State/):** a closed-form reliability benchmark, strength minus bending–torsion stress of a speed-reducer shaft. [[paper]](https://doi.org/10.2514/1.3877) _Python model._
* **[Steel_Column_Reliability](uq_benchmarks/Steel_Column_Reliability/):** the yield/buckling limit state of a steel column. [[paper]](https://doi.org/10.1007/BF01194859) _Python model._
* **[Sulfur_Aerosol_Radiative_Forcing](uq_benchmarks/Sulfur_Aerosol_Radiative_Forcing/):** a model of the direct climate forcing by anthropogenic sulfate aerosols. [[paper]](https://doi.org/10.1126/science.255.5043.423) _Python model._
* **[Truss_23_Bar_Midspan_Deflection](uq_benchmarks/Truss_23_Bar_Midspan_Deflection/):** a finite-element model of the midspan deflection of a 23-bar plane truss. [[paper]](https://doi.org/10.1016/j.jcp.2010.12.021) _Python model._
* **[Two_DOF_Equipment_Structure_Oscillator](uq_benchmarks/Two_DOF_Equipment_Structure_Oscillator/):** a closed-form model of the response of light equipment mounted on a structure. [[paper]](https://doi.org/10.1061/(ASCE)0733-9399(1985)111:1(1)) _Python model._
* **[Undamped_Oscillator_Limit_State](uq_benchmarks/Undamped_Oscillator_Limit_State/):** the limit state of an undamped single-degree-of-freedom oscillator with a yielding spring. [[paper]](https://doi.org/10.1016/0167-4730(93)90003-J) _Python model._

# Contributing

If you have a data set from a model, feel free to contribute a notebook to our repository. [CONTRIBUTING.md](CONTRIBUTING.md) says what a data set directory contains and how to submit it.

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run each notebook from its own directory (for example `cd epidemiology_systems_biology/HIV_Long_Term_Dynamics`), since notebooks load their data by relative path.

# License

Code (notebooks and `.py` files) is under the MIT License ([LICENSE](LICENSE)). Data are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)); credit the data sources named in each notebook.

# Acknowledgments

This material is based upon work supported by the U.S. Department of Energy Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC-0011077.

