# Description

*Active subspaces* are part of an emerging set of tools for subspace-based dimension reduction in complex simulation models in science and engineering. This repository contains data sets and associated [Jupyter](http://jupyter.org) notebooks that apply active subspace methods to a wide range of science and engineering models. The Python implementations of the methods are available on [Github](http://github.com/paulcon/active_subspaces/). More information on active subspaces can be found at [activesubspaces.org](http://activesubspaces.org).

Each directory contains a data set as a CSV file and a Jupyter notebook (`.ipynb`). Github allows you view the notebook by simply clicking the `.ipynb` file. However, Github's rendering of mathematical expressions is not as robust as the rendering at [nbviewer.jupyter.org](http://nbviewer.jupyter.org/). We recommend you copy the link to the notebook you wish to view and paste it in the box at [nbviewer.jupyter.org](http://nbviewer.jupyter.org/). Each notebook also has links to associated publications for more details.

# Data sets

The repository contains Jupyter notebooks for the following data sets.

* **Ebola:** a 9-parameter dynamical system model for the geographic spread of Ebola. [[paper]](http://arxiv.org/abs/1603.04955)
* **HIV:** a 21-parameter dynamical system model for in-host HIV. [[paper]](http://arxiv.org/abs/1604.04588)
* **HyShotII:** a 7-parameter multiphysics model of a hypersonic scramjet. [[paper]](http://dx.doi.org/10.1016/j.jcp.2015.09.001)
* **Hydrology:** a 20-parameter integrated surface/subsurface hydrologic model. [[paper]](http://dx.doi.org/10.1016/j.cageo.2015.07.001)
* **Lithium_Battery:** an 18-parameter model of a lithium-ion battery. [[paper]](https://arxiv.org/abs/1606.08770)
* **MCMC:** exploiting active subspaces to apply MCMC in a 100-parameter PDE model. [[paper]](http://dx.doi.org/10.1137/15M1042127)
* **MHD:** a 5-parameter magnetohydrodynamics power generation model. [[paper]](https://arxiv.org/abs/1609.01255)
* **NACA0012:** an 18-parameter model for a two-spatial-dimensional airfoil. [[chapter 5]](http://dx.doi.org/10.1137/1.9781611973860)
* **NREL_Wind:** a 5-parameter model of off-shore wind turbine fatige. [[paper]](http://dx.doi.org/10.1002/we.1870)
* **ONERA-M6:** a 60-parameter model for a three-spatial-dimensional transonic wing. [[paper]](http://dx.doi.org/10.2514/6.2014-1171)
* **Reentry_freestream_conditions:** a 7-parameter model for an atmostpheric re-entry vehicle.
* **SingleDiodePV:** a 5-parameter model for a lumped-parameter single-diode photovoltaic solar cell. [[paper]](http://dx.doi.org/10.1002/sam.11281)
* **Stomatal_Resistance:** a 20-parameter model of stomatal resistance in an integrated surface/subsurface hydrologic model. [[thesis]](http://hdl.handle.net/11124/170080)
* **Subsurface_Permeability:** a 100-parameter model for subsurface permeability for groundwater flow. [[paper]](http://dx.doi.org/10.1016/j.advwatres.2016.03.020)

# Contributing

If you have a data set from a model, and you find evidence of an active subspace, feel free to contribute a notebook to our repository. Contact [Paul Constantine](http://inside.mines.edu/~pconstan) at Colorado School of Mines with questions or comments. Ryan Howard at Colorado School of Mines created the notebooks from the associated publications and data sets over the summer of 2016.

# Acknowledgments

This material is based upon work supported by the U.S. Department of Energy Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC-0011077.

