# Notebooks by method

Every notebook, with its number of inputs, the dimension of the active subspace it finds, how it estimates the subspace, how it models the inputs, and what it does with the subspace beyond the eigenvalue, eigenvector and sufficient summary plots. [using-the-notebooks.md](using-the-notebooks.md) explains the terms. Written by `docs/make_notebook_index.py` from the notebooks' code, with the inputs and dimensions read from the notebooks' text and kept in the script; do not edit by hand.

- **Inputs (m)**: the number of inputs, the dimension of the input space the subspace is computed in. A list gives one value for each model, data set or parameterization in the notebook with a different number of inputs.
- **Dimension (n)**: the dimension of the active subspace the notebook settles on, from the eigenvalue gaps and the fits on the active variables. A list gives the different dimensions found for different outputs or models; *none* marks an output with no low-dimensional subspace.
- **Subspace from**: *gradients* (the gradient-based estimator, with gradients from files or the model); *local linear gradients* (gradients estimated from values by local linear fits); *finite differences*; *linear fit* or *quadratic fit* (a global least-squares model of the values); *SIR* (sliced inverse regression); *given matrix* or *given basis* (computed in the notebook, e.g. by quadrature).
- **Input distribution**: *uniform* on a box, *Gaussian* (possibly correlated), or *mixed* (a product of uniform and Gaussian blocks).

## Aerospace and CFD

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [AirfRANS_Airfoil_RANS_Forces](../aerospace_cfd/AirfRANS_Airfoil_RANS_Forces/AirfRANS_Airfoil_RANS_Forces.ipynb) | stored data | 5 | 1, 2 | linear fit, quadratic fit | uniform | surrogate |
| [DTMB_5415_Hull_Resistance](../aerospace_cfd/DTMB_5415_Hull_Resistance/DTMB_5415_Hull_Resistance.ipynb) | stored data | 14, 16 | 1, 3, none | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [HyShotII_Scramjet](../aerospace_cfd/HyShotII_Scramjet/HyShotII_Scramjet.ipynb) | stored data | 7 | 1 | linear fit | uniform | surrogate, prediction intervals |
| [LS89_Turbine_Blade_Envelopes](../aerospace_cfd/LS89_Turbine_Blade_Envelopes/LS89_Turbine_Blade_Envelopes.ipynb) | stored data | 20 | 1 | local linear gradients, quadratic fit | uniform | surrogate |
| [N2_Supersonic_Jet](../aerospace_cfd/N2_Supersonic_Jet/N2_Supersonic_Jet.ipynb) | stored data | 105 | 1, 2 | linear fit | uniform | surrogate |
| [NACA0012_Airfoil](../aerospace_cfd/NACA0012_Airfoil/NACA0012_Airfoil.ipynb) | stored data | 18 | 1, 2 | gradients | uniform | surrogate |
| [NACA2412_Flapped_Airfoil](../aerospace_cfd/NACA2412_Flapped_Airfoil/NACA2412_Flapped_Airfoil.ipynb) | stored data | 3, 6 | 1, 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [NACA4412_Shape_Lift_OpenFOAM](../aerospace_cfd/NACA4412_Shape_Lift_OpenFOAM/NACA4412_Shape_Lift_OpenFOAM.ipynb) | stored data | 10 | 2 | gradients, local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [ONERA_M6_Wing](../aerospace_cfd/ONERA_M6_Wing/ONERA_M6_Wing.ipynb) | stored data | 50 | 1, 2 | gradients | uniform | surrogate, active domain |
| [Pipe_Flow_Dimensional_Analysis](../aerospace_cfd/Pipe_Flow_Dimensional_Analysis/Pipe_Flow_Dimensional_Analysis.ipynb) | Python model | 5 | 1, 2 | gradients | uniform | surrogate |
| [RAE2822_SHAP_Design_Exploration](../aerospace_cfd/RAE2822_SHAP_Design_Exploration/RAE2822_SHAP_Design_Exploration.ipynb) | stored data | 20 | 1, 2 | gradients, linear fit | uniform | surrogate |
| [Stagnation_Temperature_Probe_Design](../aerospace_cfd/Stagnation_Temperature_Probe_Design/Stagnation_Temperature_Probe_Design.ipynb) | stored data | 7 | 1, 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Transonic_Airfoil_Shape_Parameterizations](../aerospace_cfd/Transonic_Airfoil_Shape_Parameterizations/Transonic_Airfoil_Shape_Parameterizations.ipynb) | stored data | 10, 11 | 1, 2 | quadratic fit | uniform | surrogate |
| [Transonic_Fan_Blade_Performance_Maps](../aerospace_cfd/Transonic_Fan_Blade_Performance_Maps/Transonic_Fan_Blade_Performance_Maps.ipynb) | stored data | 25 | 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate, active domain, inverse map |

## Hydrology, subsurface and geoscience

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Aquifer_Tracer_Advection_Diffusion](../hydrology_subsurface_geoscience/Aquifer_Tracer_Advection_Diffusion/Aquifer_Tracer_Advection_Diffusion.ipynb) | Python model | 10 | 3 | gradients | Gaussian | surrogate |
| [CMF_Soil_Moisture_SPOTPY](../hydrology_subsurface_geoscience/CMF_Soil_Moisture_SPOTPY/CMF_Soil_Moisture_SPOTPY.ipynb) | stored data | 4 | 1, 2 | local linear gradients, quadratic fit | Gaussian | surrogate |
| [CO2_Injection_Benchmark_Saturation](../hydrology_subsurface_geoscience/CO2_Injection_Benchmark_Saturation/CO2_Injection_Benchmark_Saturation.ipynb) | stored data | 3 | 1 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Earthquake_Rupture_Seismic_Moment](../hydrology_subsurface_geoscience/Earthquake_Rupture_Seismic_Moment/Earthquake_Rupture_Seismic_Moment.ipynb) | stored data | 3 | 1 | linear fit, quadratic fit | uniform | surrogate |
| [Freyberg_MODFLOW_Forecast](../hydrology_subsurface_geoscience/Freyberg_MODFLOW_Forecast/Freyberg_MODFLOW_Forecast.ipynb) | stored data | 71 | 2 | gradients | uniform | surrogate |
| [Geological_CO2_Storage_Leakage](../hydrology_subsurface_geoscience/Geological_CO2_Storage_Leakage/Geological_CO2_Storage_Leakage.ipynb) | Python model | 22, 30 | 2 | gradients | uniform | surrogate, activity scores |
| [HBV_SASK_Oldman_Streamflow](../hydrology_subsurface_geoscience/HBV_SASK_Oldman_Streamflow/HBV_SASK_Oldman_Streamflow.ipynb) | Python model | 12 | 1, 2, 3 | gradients, linear fit | uniform | surrogate, activity scores |
| [HyMod_Leaf_River_Rainfall_Runoff](../hydrology_subsurface_geoscience/HyMod_Leaf_River_Rainfall_Runoff/HyMod_Leaf_River_Rainfall_Runoff.ipynb) | Python model | 5 | 1, 2, none | gradients, local linear gradients | uniform | surrogate |
| [Kaesbach_Catchment_Subsurface_Flow](../hydrology_subsurface_geoscience/Kaesbach_Catchment_Subsurface_Flow/Kaesbach_Catchment_Subsurface_Flow.ipynb) | stored data | 32 | 1, 2, none | gradients, quadratic fit | uniform | surrogate, activity scores |
| [Karst_Aquifer_LuKARS_Calibration](../hydrology_subsurface_geoscience/Karst_Aquifer_LuKARS_Calibration/Karst_Aquifer_LuKARS_Calibration.ipynb) | Python model | 21 | 2 | gradients | uniform | surrogate |
| [Kurchatov_Sr90_Groundwater_Migration](../hydrology_subsurface_geoscience/Kurchatov_Sr90_Groundwater_Migration/Kurchatov_Sr90_Groundwater_Migration.ipynb) | stored data | 20 | 1 | linear fit, given matrix | uniform | surrogate |
| [Land_Surface_Energy_Fluxes](../hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/Land_Surface_Energy_Fluxes.ipynb) | stored data | 19 | 1 | linear fit | uniform, Gaussian, mixed | surrogate, prediction intervals |
| [Photosynthesis_Stomatal_Resistance](../hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/Photosynthesis_Stomatal_Resistance.ipynb) | stored data | 20 | 1 | linear fit | uniform |  |
| [PyGEM_Glacier_Mass_Balance](../hydrology_subsurface_geoscience/PyGEM_Glacier_Mass_Balance/PyGEM_Glacier_Mass_Balance.ipynb) | Python model | 3, 5 | 1 | gradients | uniform | surrogate, activity scores |
| [Raven_Salmon_River_Blended_Model](../hydrology_subsurface_geoscience/Raven_Salmon_River_Blended_Model/Raven_Salmon_River_Blended_Model.ipynb) | stored data | 43 | 1 | gradients, local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Runoff_Permeability_Field](../hydrology_subsurface_geoscience/Runoff_Permeability_Field/Runoff_Permeability_Field.ipynb) | stored data | 100 | 1 | linear fit, quadratic fit | Gaussian | surrogate |
| [Subglacial_Drainage_GlaDS](../hydrology_subsurface_geoscience/Subglacial_Drainage_GlaDS/Subglacial_Drainage_GlaDS.ipynb) | stored data | 8 | 1, 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate, activity scores |

## Climate, atmosphere and ocean

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Arctic_Snow_Depth_SMRT](../climate_atmosphere_ocean/Arctic_Snow_Depth_SMRT/Arctic_Snow_Depth_SMRT.ipynb) | stored data | 5 | 1, 2 | gradients | uniform | surrogate |
| [Atmospheric_Selenium_Lifetime](../climate_atmosphere_ocean/Atmospheric_Selenium_Lifetime/Atmospheric_Selenium_Lifetime.ipynb) | stored data | 34 | 2 | linear fit, SIR, given basis | uniform | surrogate |
| [FaIR_Climate_Response_Calibration](../climate_atmosphere_ocean/FaIR_Climate_Response_Calibration/FaIR_Climate_Response_Calibration.ipynb) | stored data | 45 | 1, 4, none | gradients, linear fit | Gaussian | surrogate |
| [Hector_Matilda_Warming](../climate_atmosphere_ocean/Hector_Matilda_Warming/Hector_Matilda_Warming.ipynb) | stored data | 6 | 1 | gradients, linear fit, quadratic fit | Gaussian | surrogate |
| [Ice_Sheet_Basal_Melt_Ensemble](../climate_atmosphere_ocean/Ice_Sheet_Basal_Melt_Ensemble/Ice_Sheet_Basal_Melt_Ensemble.ipynb) | stored data | 15 | 1, 2 | local linear gradients, quadratic fit | uniform | surrogate |
| [Lorenz96_Initial_State_Inference](../climate_atmosphere_ocean/Lorenz96_Initial_State_Inference/Lorenz96_Initial_State_Inference.ipynb) | Python model | 37 | 1, none | linear fit, quadratic fit, given basis | Gaussian | surrogate |
| [Marine_Carbonate_System_PyCO2SYS](../climate_atmosphere_ocean/Marine_Carbonate_System_PyCO2SYS/Marine_Carbonate_System_PyCO2SYS.ipynb) | stored data | 14 | 1, 2 | gradients | Gaussian | surrogate |
| [Parcel_Droplet_Activation](../climate_atmosphere_ocean/Parcel_Droplet_Activation/Parcel_Droplet_Activation.ipynb) | stored data | 8 | 1, 2 | gradients | uniform | surrogate |

## Ecology and agriculture

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [SIPNET_Ecosystem_Carbon_Flux](../ecology_agriculture/SIPNET_Ecosystem_Carbon_Flux/SIPNET_Ecosystem_Carbon_Flux.ipynb) | stored data | 8 | 1, 4 | gradients | uniform | surrogate, activity scores, MCMC |

## Energy

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [BISON_Fuel_Performance_Surrogates](../energy/BISON_Fuel_Performance_Surrogates/BISON_Fuel_Performance_Surrogates.ipynb) | stored data | 13 | 1, 2 | quadratic fit | Gaussian | surrogate |
| [European_LV_Feeder_Probabilistic_Power_Flow](../energy/European_LV_Feeder_Probabilistic_Power_Flow/European_LV_Feeder_Probabilistic_Power_Flow.ipynb) | stored data | 55, 110 | 1 | gradients | uniform, Gaussian, mixed | surrogate |
| [HTGR_Microreactor_Control_Drums](../energy/HTGR_Microreactor_Control_Drums/HTGR_Microreactor_Control_Drums.ipynb) | stored data | 8 | 1 | linear fit, quadratic fit | uniform, Gaussian | surrogate |
| [Lillgrund_FLORIS_Wake_Model](../energy/Lillgrund_FLORIS_Wake_Model/Lillgrund_FLORIS_Wake_Model.ipynb) | Python model | 6 | 3 | gradients | uniform | surrogate |
| [Lithium_Ion_Battery](../energy/Lithium_Ion_Battery/Lithium_Ion_Battery.ipynb) | stored data | 19 | 1 | quadratic fit | uniform |  |
| [Lithium_Ion_DFN_Drive_Cycle](../energy/Lithium_Ion_DFN_Drive_Cycle/Lithium_Ion_DFN_Drive_Cycle.ipynb) | stored data | 24 | 1, 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Lithium_Ion_Impedance_Calibration](../energy/Lithium_Ion_Impedance_Calibration/Lithium_Ion_Impedance_Calibration.ipynb) | stored data | 18 | 1 | gradients | uniform | surrogate, activity scores |
| [Offshore_Wind_Turbine_Fatigue](../energy/Offshore_Wind_Turbine_Fatigue/Offshore_Wind_Turbine_Fatigue.ipynb) | stored data | 5 | 1, 2, 3 | gradients, local linear gradients | uniform | surrogate |
| [PyPSA_Eur_Investment_Ranges](../energy/PyPSA_Eur_Investment_Ranges/PyPSA_Eur_Investment_Ranges.ipynb) | stored data | 5 | 1 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Residential_Building_Stock_Energy_Use](../energy/Residential_Building_Stock_Energy_Use/Residential_Building_Stock_Energy_Use.ipynb) | stored data | 15 | 1, 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Single_Diode_Solar_Cell](../energy/Single_Diode_Solar_Cell/Single_Diode_Solar_Cell.ipynb) | stored data | 5 | 1 | gradients | uniform | surrogate |

## Plasma and fusion

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [EU_DEMO_PROCESS_Major_Radius](../plasma_fusion/EU_DEMO_PROCESS_Major_Radius/EU_DEMO_PROCESS_Major_Radius.ipynb) | stored data | 6 | 1 | gradients, local linear gradients, linear fit, quadratic fit | uniform | surrogate, activity scores |
| [Hall_Thruster_Multi_Component](../plasma_fusion/Hall_Thruster_Multi_Component/Hall_Thruster_Multi_Component.ipynb) | stored data | 18 | 1, 2 | linear fit, quadratic fit | uniform | surrogate, activity scores |
| [JAG_ICF_Implosion_Scalars](../plasma_fusion/JAG_ICF_Implosion_Scalars/JAG_ICF_Implosion_Scalars.ipynb) | stored data | 5 | 1 | gradients, local linear gradients, quadratic fit | uniform | surrogate |
| [MHD_Generator](../plasma_fusion/MHD_Generator/MHD_Generator.ipynb) | stored data | 5 | 1, 2 | gradients | uniform | surrogate |
| [Vlasov_Poisson_Instability_Growth_Rate](../plasma_fusion/Vlasov_Poisson_Instability_Growth_Rate/Vlasov_Poisson_Instability_Growth_Rate.ipynb) | Python model | 3, 5, 6 | 1, 2 | gradients | uniform | surrogate |

## Epidemiology and systems biology

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [COVID19_Exit_Strategies_virsim](../epidemiology_systems_biology/COVID19_Exit_Strategies_virsim/COVID19_Exit_Strategies_virsim.ipynb) | stored data | 6, 7, 8 | 2, 3 | linear fit, quadratic fit | Gaussian | surrogate, activity scores |
| [COVID19_SEIR_Epidemic_Peak](../epidemiology_systems_biology/COVID19_SEIR_Epidemic_Peak/COVID19_SEIR_Epidemic_Peak.ipynb) | Python model | 3 | 1 | gradients | Gaussian | surrogate, activity scores |
| [CityCOVID_Chicago_Calibration](../epidemiology_systems_biology/CityCOVID_Chicago_Calibration/CityCOVID_Chicago_Calibration.ipynb) | stored data | 4 | 1 | linear fit, quadratic fit | uniform | surrogate, activity scores |
| [CovidSim_Cumulative_Deaths](../epidemiology_systems_biology/CovidSim_Cumulative_Deaths/CovidSim_Cumulative_Deaths.ipynb) | stored data | 51 | 2 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [EMT_Switch_ODE](../epidemiology_systems_biology/EMT_Switch_ODE/EMT_Switch_ODE.ipynb) | Python model | 11 | 1 | gradients | uniform | surrogate, activity scores |
| [Ebola_SEIR](../epidemiology_systems_biology/Ebola_SEIR/Ebola_SEIR.ipynb) | Python model | 8 | 2 | gradients, given matrix | uniform | surrogate, activity scores |
| [HIV_Long_Term_Dynamics](../epidemiology_systems_biology/HIV_Long_Term_Dynamics/HIV_Long_Term_Dynamics.ipynb) | Python model | 27 | 1, 2 | gradients, finite differences | uniform | surrogate |
| [HIV_Three_Stage_Reservoir](../epidemiology_systems_biology/HIV_Three_Stage_Reservoir/HIV_Three_Stage_Reservoir.ipynb) | Python model | 15 | 1 | gradients | uniform | surrogate |
| [MAPK_Cascade_Bistable_Oscillatory](../epidemiology_systems_biology/MAPK_Cascade_Bistable_Oscillatory/MAPK_Cascade_Bistable_Oscillatory.ipynb) | Python model | 7 | 1, none | gradients, local linear gradients | uniform | surrogate, activity scores |
| [Malaria_Bed_Net_Reproduction_Number](../epidemiology_systems_biology/Malaria_Bed_Net_Reproduction_Number/Malaria_Bed_Net_Reproduction_Number.ipynb) | Python model | 20 | 1 | gradients | uniform | surrogate, activity scores |
| [Minnesota_COVID19_Deaths](../epidemiology_systems_biology/Minnesota_COVID19_Deaths/Minnesota_COVID19_Deaths.ipynb) | stored data | 8 | 3 | gradients | uniform | surrogate, activity scores |
| [STAT5_Dimerization_Signaling](../epidemiology_systems_biology/STAT5_Dimerization_Signaling/STAT5_Dimerization_Signaling.ipynb) | Python model | 6 | 1 | gradients | uniform | surrogate |
| [Zika_SEIR_SEI_Brazil](../epidemiology_systems_biology/Zika_SEIR_SEI_Brazil/Zika_SEIR_SEI_Brazil.ipynb) | Python model | 10 | 2 | gradients | uniform | surrogate |

## Biomedical and cardiovascular

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Atrial_Cell_Models_Courtemanche_Maleckar](../biomedical_cardiovascular/Atrial_Cell_Models_Courtemanche_Maleckar/Atrial_Cell_Models_Courtemanche_Maleckar.ipynb) | stored data | 8, 10, 20, 22 | 1, 2, 3 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [CiPA_Ventricular_Proarrhythmia_Metrics](../biomedical_cardiovascular/CiPA_Ventricular_Proarrhythmia_Metrics/CiPA_Ventricular_Proarrhythmia_Metrics.ipynb) | stored data | 9 | 1, 2 | linear fit, quadratic fit | uniform | surrogate |
| [Cole_Cole_Cell_Monolayer_Impedance](../biomedical_cardiovascular/Cole_Cole_Cell_Monolayer_Impedance/Cole_Cole_Cell_Monolayer_Impedance.ipynb) | Python model | 4 | 1, 2 | gradients | Gaussian | surrogate |
| [Four_Chamber_Heart_Electromechanics](../biomedical_cardiovascular/Four_Chamber_Heart_Electromechanics/Four_Chamber_Heart_Electromechanics.ipynb) | stored data | 29, 45 | 1 | linear fit, quadratic fit | uniform | surrogate |
| [Hodgkin_Huxley_Action_Potential](../biomedical_cardiovascular/Hodgkin_Huxley_Action_Potential/Hodgkin_Huxley_Action_Potential.ipynb) | Python model | 11 | 1, 2 | gradients, given matrix | uniform | surrogate |
| [Luo_Rudy_Cardiac_Action_Potential](../biomedical_cardiovascular/Luo_Rudy_Cardiac_Action_Potential/Luo_Rudy_Cardiac_Action_Potential.ipynb) | stored data | 6 | 1, none | linear fit, quadratic fit | uniform | surrogate |
| [Pulmonary_Pulse_Wave_Network](../biomedical_cardiovascular/Pulmonary_Pulse_Wave_Network/Pulmonary_Pulse_Wave_Network.ipynb) | stored data | 9 | 1, 2 | gradients, quadratic fit | uniform | surrogate |
| [Pyloric_Circuit_Energy_Consumption](../biomedical_cardiovascular/Pyloric_Circuit_Energy_Consumption/Pyloric_Circuit_Energy_Consumption.ipynb) | stored data | 31 | 1 | linear fit, quadratic fit, given matrix | uniform | surrogate |
| [Rabbit_Ventricular_Action_Potential](../biomedical_cardiovascular/Rabbit_Ventricular_Action_Potential/Rabbit_Ventricular_Action_Potential.ipynb) | stored data | 15 | 1 | quadratic fit | uniform | surrogate, activity scores |
| [Tumour_Competition_ODE](../biomedical_cardiovascular/Tumour_Competition_ODE/Tumour_Competition_ODE.ipynb) | Python model | 6 | 4, none | gradients | uniform | surrogate, activity scores |

## Combustion and chemical kinetics

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Ammonia_Syngas_Optimized_Kinetics](../combustion_chemical_kinetics/Ammonia_Syngas_Optimized_Kinetics/Ammonia_Syngas_Optimized_Kinetics.ipynb) | stored data | 123 | 1 | gradients | uniform | surrogate |
| [Annular_Combustor_Thermoacoustics](../combustion_chemical_kinetics/Annular_Combustor_Thermoacoustics/Annular_Combustor_Thermoacoustics.ipynb) | Python model | 38 | 3 | gradients | uniform, Gaussian | surrogate |
| [H2_Flame_Kinetics_Bayesian_Calibration](../combustion_chemical_kinetics/H2_Flame_Kinetics_Bayesian_Calibration/H2_Flame_Kinetics_Bayesian_Calibration.ipynb) | stored data | 31 | 1, 2 | gradients | Gaussian | surrogate, activity scores |
| [H2_NH3_Combustor_CFD_NOx](../combustion_chemical_kinetics/H2_NH3_Combustor_CFD_NOx/H2_NH3_Combustor_CFD_NOx.ipynb) | stored data | 3 | 1, 2 | quadratic fit | uniform | surrogate |
| [H2_O2_Ignition_Delay_Kinetics](../combustion_chemical_kinetics/H2_O2_Ignition_Delay_Kinetics/H2_O2_Ignition_Delay_Kinetics.ipynb) | stored data | 19, 36 | 1 | gradients, local linear gradients | uniform | surrogate |
| [NH3_H2_Kinetics_Uncertainty](../combustion_chemical_kinetics/NH3_H2_Kinetics_Uncertainty/NH3_H2_Kinetics_Uncertainty.ipynb) | stored data | 52 | 1 | gradients | uniform | surrogate |
| [Shared_Subspace_H2_Ignition_Flame](../combustion_chemical_kinetics/Shared_Subspace_H2_Ignition_Flame/Shared_Subspace_H2_Ignition_Flame.ipynb) | stored data | 33 | 1 | gradients | Gaussian | surrogate |

## Structural, materials and additive manufacturing

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Epoxy_MD_Force_Field_Sensitivity](../structural_materials_manufacturing/Epoxy_MD_Force_Field_Sensitivity/Epoxy_MD_Force_Field_Sensitivity.ipynb) | stored data | 103 | 1 | linear fit | uniform | surrogate |
| [Powder_Bed_Fusion_Residual_Stress](../structural_materials_manufacturing/Powder_Bed_Fusion_Residual_Stress/Powder_Bed_Fusion_Residual_Stress.ipynb) | stored data | 6 | 1 | local linear gradients, linear fit, quadratic fit | uniform | surrogate |
| [Steel_Arch_Inplane_Buckling](../structural_materials_manufacturing/Steel_Arch_Inplane_Buckling/Steel_Arch_Inplane_Buckling.ipynb) | Python model | 7 | 1, 2 | gradients | uniform | surrogate |
| [Two_Material_Thermal_Compliance](../structural_materials_manufacturing/Two_Material_Thermal_Compliance/Two_Material_Thermal_Compliance.ipynb) | Python model | 512 | 1 | gradients | uniform | surrogate |
| [Wrench_Elasticity_Displacement](../structural_materials_manufacturing/Wrench_Elasticity_Displacement/Wrench_Elasticity_Displacement.ipynb) | Python model | 318 | 3 | gradients | Gaussian | surrogate |

## Electronics and circuits

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Cavity_Shielding_Effectiveness](../electronics_circuits/Cavity_Shielding_Effectiveness/Cavity_Shielding_Effectiveness.ipynb) | Python model | 6 | 1, 2 | gradients | uniform, Gaussian | surrogate, activity scores |
| [Two_Stage_CMOS_Op_Amp](../electronics_circuits/Two_Stage_CMOS_Op_Amp/Two_Stage_CMOS_Op_Amp.ipynb) | stored data | 7 | 1, 2, 3 | gradients | uniform | surrogate |

## UQ benchmark collections

| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |
|---|---|---|---|---|---|---|
| [Elliptic_PDE_Random_Coefficients](../uq_benchmarks/Elliptic_PDE_Random_Coefficients/Elliptic_PDE_Random_Coefficients.ipynb) | Python model | 100 | 1 | gradients, given basis | Gaussian | surrogate |
| [Fire_Satellite_Coupled_System](../uq_benchmarks/Fire_Satellite_Coupled_System/Fire_Satellite_Coupled_System.ipynb) | Python model | 9 | 1, 2 | gradients | Gaussian | surrogate |
| [Golinski_Speed_Reducer](../uq_benchmarks/Golinski_Speed_Reducer/Golinski_Speed_Reducer.ipynb) | Python model | 6 | 1 | gradients | uniform | surrogate |
| [Hat_Stiffened_Panel_Shear_Buckling](../uq_benchmarks/Hat_Stiffened_Panel_Shear_Buckling/Hat_Stiffened_Panel_Shear_Buckling.ipynb) | Python model | 10 | 1 | gradients | Gaussian | surrogate |
| [OpenAeroStruct_Wing_Aerostructural](../uq_benchmarks/OpenAeroStruct_Wing_Aerostructural/OpenAeroStruct_Wing_Aerostructural.ipynb) | stored data | 12 | 1, 2 | gradients | uniform | surrogate |
| [PDE_Inverse_Problem_MCMC](../uq_benchmarks/PDE_Inverse_Problem_MCMC/PDE_Inverse_Problem_MCMC.ipynb) | Python model | 2, 100 | 1, 2 | gradients | Gaussian | MCMC |
| [Point_Heat_Source_Consolidation](../uq_benchmarks/Point_Heat_Source_Consolidation/Point_Heat_Source_Consolidation.ipynb) | Python model | 8 | 1, 2 | gradients | Gaussian | surrogate |
| [River_Flood_Dyke_Overflow](../uq_benchmarks/River_Flood_Dyke_Overflow/River_Flood_Dyke_Overflow.ipynb) | Python model | 8 | 1 | gradients | Gaussian, mixed | surrogate, activity scores |
| [Simply_Supported_Tube_Deflection](../uq_benchmarks/Simply_Supported_Tube_Deflection/Simply_Supported_Tube_Deflection.ipynb) | Python model | 6 | 2 | gradients | Gaussian | surrogate |
| [Speed_Reducer_Shaft_Limit_State](../uq_benchmarks/Speed_Reducer_Shaft_Limit_State/Speed_Reducer_Shaft_Limit_State.ipynb) | Python model | 5 | 1 | gradients | Gaussian | surrogate |
| [Steel_Column_Reliability](../uq_benchmarks/Steel_Column_Reliability/Steel_Column_Reliability.ipynb) | Python model | 9 | 1 | gradients | Gaussian | surrogate |
| [Sulfur_Aerosol_Radiative_Forcing](../uq_benchmarks/Sulfur_Aerosol_Radiative_Forcing/Sulfur_Aerosol_Radiative_Forcing.ipynb) | Python model | 9 | 1 | gradients | Gaussian | surrogate |
| [Truss_23_Bar_Midspan_Deflection](../uq_benchmarks/Truss_23_Bar_Midspan_Deflection/Truss_23_Bar_Midspan_Deflection.ipynb) | Python model | 10 | 1 | gradients | Gaussian | surrogate, activity scores |
| [Two_DOF_Equipment_Structure_Oscillator](../uq_benchmarks/Two_DOF_Equipment_Structure_Oscillator/Two_DOF_Equipment_Structure_Oscillator.ipynb) | Python model | 7 | 2 | gradients | Gaussian | surrogate |
| [Undamped_Oscillator_Limit_State](../uq_benchmarks/Undamped_Oscillator_Limit_State/Undamped_Oscillator_Limit_State.ipynb) | Python model | 6 | 2 | gradients | Gaussian | surrogate |
