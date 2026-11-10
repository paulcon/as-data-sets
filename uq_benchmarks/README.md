# UQ benchmark collections

Standard test problems for uncertainty quantification and dimension reduction methods.

Data sets (descriptions and papers are in the [top-level README](../README.md#data-sets)):

- [Elliptic_PDE_Random_Coefficients](Elliptic_PDE_Random_Coefficients/)
- [Fire_Satellite_Coupled_System](Fire_Satellite_Coupled_System/)
- [Golinski_Speed_Reducer](Golinski_Speed_Reducer/)
- [Hat_Stiffened_Panel_Shear_Buckling](Hat_Stiffened_Panel_Shear_Buckling/)
- [OpenAeroStruct_Wing_Aerostructural](OpenAeroStruct_Wing_Aerostructural/)
- [PDE_Inverse_Problem_MCMC](PDE_Inverse_Problem_MCMC/)
- [Point_Heat_Source_Consolidation](Point_Heat_Source_Consolidation/)
- [River_Flood_Dyke_Overflow](River_Flood_Dyke_Overflow/)
- [Simply_Supported_Tube_Deflection](Simply_Supported_Tube_Deflection/)
- [Speed_Reducer_Shaft_Limit_State](Speed_Reducer_Shaft_Limit_State/)
- [Steel_Column_Reliability](Steel_Column_Reliability/)
- [Sulfur_Aerosol_Radiative_Forcing](Sulfur_Aerosol_Radiative_Forcing/)
- [Truss_23_Bar_Midspan_Deflection](Truss_23_Bar_Midspan_Deflection/)
- [Two_DOF_Equipment_Structure_Oscillator](Two_DOF_Equipment_Structure_Oscillator/)
- [Undamped_Oscillator_Limit_State](Undamped_Oscillator_Limit_State/)

Elliptic_PDE_Random_Coefficients and PDE_Inverse_Problem_MCMC use the same PDE solver, `pde_model.py`, as two byte-identical copies. After any change, copy it to the other directory and run `python uq_benchmarks/check_pde_model.py`.
