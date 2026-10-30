# UQ benchmark collections

Standard test problems for uncertainty quantification and dimension reduction methods.

Data sets (descriptions and papers are in the [top-level README](../README.md#data-sets)):

- [Elliptic_PDE_Random_Coefficients](Elliptic_PDE_Random_Coefficients/)
- [PDE_Inverse_Problem_MCMC](PDE_Inverse_Problem_MCMC/)

Both data sets use the same PDE solver, `pde_model.py`, as two byte-identical copies. After any change, copy it to the other directory and run `python uq_benchmarks/check_pde_model.py`.
