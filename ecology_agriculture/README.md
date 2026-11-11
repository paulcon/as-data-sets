# Ecology and agriculture

Population, ecosystem, crop and agricultural models.

Each data set below has a short summary of the science: what the model computes, what varies, and what the active subspace shows. The numbers come from the data set's notebook, which has the details and the comparison with the paper. The [top-level README](../README.md#data-sets) has the paper links and the kind of each data set (stored data or Python model).

### [SIPNET_Ecosystem_Carbon_Flux](SIPNET_Ecosystem_Carbon_Flux/)

Fer et al. calibrated eight physiological and soil parameters of the SIPNET ecosystem model against flux-tower data from Bartlett Experimental Forest. The notebook runs SIPNET at 1000 Latin hypercube samples of the paper's uniform priors on these eight parameters, with finite-difference gradients of three two-year totals: cumulative net ecosystem exchange (NEE), total evapotranspiration (ET) and total gross primary production (GPP). It is a substitute set-up, not the paper's runs: the meteorological driver is the one that ships with SIPNET (2004–2005, labelled, with a question mark, as Niwot Ridge, Colorado), and the parameters the paper did not calibrate take generic template values; in this colder, drier set-up 902 of the 1000 samples are net carbon sources. Total ET has a one-dimensional active subspace along the soil water holding capacity (λ₁/λ₂ = 51.9; a quartic in the active variable has leave-one-out R² = 0.929). Cumulative NEE has a dominant direction, almost the optimum photosynthesis temperature (λ₁/λ₂ = 3.67), but a cubic in it has leave-one-out R² of only 0.313, rising to 0.928 with four active variables, so NEE needs about four dimensions; GPP behaves like NEE.
