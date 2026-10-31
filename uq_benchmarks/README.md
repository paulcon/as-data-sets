# UQ benchmark collections

Standard test problems for uncertainty quantification and dimension reduction methods.

Data sets (descriptions and papers are in the [top-level README](../README.md#data-sets)):

- [Elliptic_PDE_Random_Coefficients](Elliptic_PDE_Random_Coefficients/)
- [Fire_Satellite_Coupled_System](Fire_Satellite_Coupled_System/)
- [Golinski_Speed_Reducer](Golinski_Speed_Reducer/)
- [PDE_Inverse_Problem_MCMC](PDE_Inverse_Problem_MCMC/)
- [River_Flood_Dyke_Overflow](River_Flood_Dyke_Overflow/)
- [Sulfur_Aerosol_Radiative_Forcing](Sulfur_Aerosol_Radiative_Forcing/)

Elliptic_PDE_Random_Coefficients and PDE_Inverse_Problem_MCMC use the same PDE solver, `pde_model.py`, as two byte-identical copies. After any change, copy it to the other directory and run `python uq_benchmarks/check_pde_model.py`.
