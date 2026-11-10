# Aerospace and CFD

Aerodynamic shape design, hypersonic propulsion and other computational fluid dynamics models.

Each data set below has a short summary of the science: what the model computes, what varies, and what the active subspace shows. The numbers come from the data set's notebook, which has the details and the comparison with the paper. The [top-level README](../README.md#data-sets) has the paper links and the kind of each data set (stored data or Python model).

### [AirfRANS_Airfoil_RANS_Forces](AirfRANS_Airfoil_RANS_Forces/)

AirfRANS is a benchmark for machine-learning surrogates: 1000 steady, incompressible RANS simulations (OpenFOAM, k–ω SST) of NACA 4- and 5-digit airfoils at Reynolds numbers from 2 to 6 million and angles of attack from −5° to 15°. The notebook studies the lift and drag of the 369 cambered 4-digit airfoils as functions of five inputs: inlet velocity, angle of attack, maximum camber, camber position and thickness. The simulations give no gradients, so the subspaces come from a global quadratic fit. Lift has a one-dimensional active subspace (λ₁/λ₂ = 405.6) whose direction agrees with thin-airfoil theory (|w₁ᵀw₁,theory| = 0.9973). Drag needs two directions, led by the angle of attack and then by thickness and camber (λ₁/λ₂ = 40.0, λ₂/λ₃ = 7.7); a cubic in the two active variables has leave-one-out R² 0.9787.

### [DTMB_5415_Hull_Resistance](DTMB_5415_Hull_Resistance/)

The L2-Sea potential-flow code computes the calm-water resistance of the DTMB 5415 destroyer hull from 16 inputs: Froude number, draft and 14 hull-shape modes. On 377 runs, the total resistance has a one-dimensional active subspace (λ₁/λ₂ = 21.6) along the Froude number and the draft. That direction is mostly the speed scaling of resistance: for the dimensionless coefficient C_T the gap disappears (λ₁/λ₂ = 1.87), and a design that varies speed and shape together cannot resolve the shape directions. A second study varies only the 14 shape modes, at the optimization benchmark's Froude number 0.28 and draft −6.16 m (256 runs). It has no dominant direction (λ₁/λ₂ = 1.92) but a three-dimensional subspace (λ₃/λ₄ = 5.10), on which a quadratic has leave-one-out R² 0.9405.

### [HyShotII_Scramjet](HyShotII_Scramjet/)

A scramjet burns fuel in a supersonic air stream. The simulations model the HyShot II vehicle in the ground tests of the Göttingen high-enthalpy shock tunnel, with seven uncertain inputs: angle of attack, turbulence intensity and length scale, the stagnation pressure and enthalpy of the flow, and two transition locations. The output is the pressure at the combustor exit, a proxy for unstart (the engine losing its thrust). With 50 runs at one fuel setting and 14 at another, and no gradients, a global linear fit gives one direction, and the exit pressure is a tight quadratic function of it (leave-one-out R² 0.9899 and 0.9821). The direction is led by the angle of attack, the turbulence intensity and the stagnation conditions. The notebook uses this one variable to bound the exit pressure, to find which inputs must be limited to keep it below 2.8 bar, and to estimate its distribution function.

### [LS89_Turbine_Blade_Envelopes](LS89_Turbine_Blade_Envelopes/)

The VKI LS89 is a transonic turbine blade. Its shape is perturbed by 20 free-form-deformation variables and the stagnation pressure loss coefficient Y_p is computed by 2-D RANS (SU2) for 1000 random designs. Both a global quadratic fit and local linear gradients find one dominant direction, a coordinated thickening and thinning of the blade carried mostly by the variables x₅–x₇ and x₁₁–x₁₄. The paper samples the 19 inactive directions to draw "blade envelopes": the range of blade shapes with nearly the same loss, used to set manufacturing tolerances.

### [N2_Supersonic_Jet](N2_Supersonic_Jet/)

The N+2 is a 70-passenger supersonic jet design, studied at Mach 1.7. Its shape is parameterized by 105 free-form-deformation variables on the wing, tail, aft deck and fuselage, and SU2 computes lift, drag and the equivalent area (a measure used to shape the sonic boom), with adjoint gradients. Lift (λ₁/λ₂ ≈ 1948) and the equivalent area (λ₁/λ₂ ≈ 315) have one-dimensional active subspaces, with close first directions: lift falls along a line and the equivalent area along a parabola. Drag needs two active variables (leave-one-out R² 0.748, against 0.494 with one). The notebook also measures the adjoint gradients against the data: the lift gradients are about two thirds of the true slope and 25° off in direction.

### [NACA0012_Airfoil](NACA0012_Airfoil/)

The shape of the NACA 0012 airfoil is deformed by 18 Hicks–Henne bumps, and SU2 computes the lift and drag and, with its adjoint solver, their gradients. Lift has a one-dimensional active subspace (λ₁/λ₂ ≈ 15.9; a one-variable quadratic has leave-one-out R² 0.934). Drag has a smaller gap (λ₁/λ₂ ≈ 5.2): its first direction leaves out 28.1 % of the eigenvalue sum and the first two leave out 14.4 %, so the notebook uses two. The first drag direction weights the last nine parameters about twice as much as the first nine; the first lift direction moves the two halves with opposite signs.

### [NACA2412_Flapped_Airfoil](NACA2412_Flapped_Airfoil/)

XFOIL computes the lift, drag and quarter-chord moment of the NACA 2412 airfoil with a 30 % flap over the Second AIAA UQ Challenge envelope: angle of attack −5° to 10°, flap −5° to 15° and Reynolds number 7×10⁵ ± 0.5 % (1000 Latin hypercube points). Lift and drag each have one dominant direction (λ₁/λ₂ ≈ 147 and 67), a fixed mix of incidence and flap with a flap-to-incidence ratio of 0.568, below the 0.661 of thin-airfoil theory. The moment is driven mainly by the flap but needs a second variable (leave-one-out R² 0.951 with one, 0.998 with two). The Reynolds number is inactive over its narrow range. An extension adds XFOIL's transition settings; the leading directions do not change, and the transition settings appear only in weaker second directions.

### [NACA4412_Shape_Lift_OpenFOAM](NACA4412_Shape_Lift_OpenFOAM/)

Tezzele et al. ran 170 unsteady RANS simulations (OpenFOAM, Spalart–Allmaras, 30 s of flow) of a NACA 4412 airfoil whose upper and lower surfaces are perturbed by ten shape weights. The notebook takes the mean lift over the last 5 s of each run as the output. Its eigenvalues have the largest gap after the second (λ₂/λ₃ = 11.14), and a quadratic in two active variables predicts it with leave-one-out R² 0.9708; the subspace is spanned mainly by the upper-surface weights c₃, c₄ and the lower-surface weights d₃, d₄. From the same runs the notebook also reproduces the paper's time-dependent active subspace of the lift at 6, 10, 14 and 18 s.

### [ONERA_M6_Wing](ONERA_M6_Wing/)

The ONERA M6 is a transonic wing, studied in Euler flow at Mach 0.8395 and 3.06° angle of attack. Its shape is set by 50 free-form-deformation parameters, and the data are the lift and drag of 297 designs with SU2 adjoint gradients. Lift has a one-dimensional active subspace (λ₁/λ₂ ≈ 115; leave-one-out R² 0.980 for one active variable). Drag needs at least two: the leave-one-out R² rises from 0.703 with one active variable to 0.884 with two and 0.945 with five, the number the paper uses for its drag surrogate. The notebook sketches how the two subspaces serve drag minimization with a lift constraint.

### [Pipe_Flow_Dimensional_Analysis](Pipe_Flow_Dimensional_Analysis/)

Any semi-empirical model is a ridge function of the logarithms of its dimensional inputs, and the eigenvectors of the active subspace in those log inputs define unique dimensionless groups, ranked by how much they change the output. The example is the pressure loss of viscous pipe flow (Colebrook equation) as a function of five dimensional inputs, computed in numpy with exact gradients. In laminar flow the active subspace is one-dimensional, the Reynolds number (λ₁/λ₂ = 8.0×10⁶); at high Reynolds number it is the relative roughness (λ₁/λ₂ = 7.5×10⁷). In turbulent flow it is two-dimensional (λ₁/λ₂ = 19.9), and the most relevant group is Re^0.307 (ε/D)^−0.425, a mix of the two classical groups.

### [RAE2822_SHAP_Design_Exploration](RAE2822_SHAP_Design_Exploration/)

Palar et al. ran SU2 RANS on 98 shapes of the RAE 2822 transonic airfoil, each perturbed by 20 Hicks–Henne bumps, at Mach 0.729 and a fixed lift coefficient of 0.723, and recorded the drag and the absolute quarter-chord moment |C_m|. With 98 runs for 20 inputs and no gradients, the notebook finds one direction per output by least squares and tests a second. |C_m| is almost exactly a linear function of one direction (leave-one-out R² 0.994). Drag has a leading direction on the upper surface near x₃, x₄ and a second led by the trailing-edge bump x₂₀, as the paper states, so the notebook takes two dimensions for drag.

### [Stagnation_Temperature_Probe_Design](Stagnation_Temperature_Probe_Design/)

A Kiel probe measures the stagnation temperature of a gas stream. Scillitoe et al. morph a single-shrouded probe with seven geometric parameters and run conjugate RANS simulations (ANSYS Fluent) of 128 designs at six Mach numbers. The stagnation pressure loss is almost a linear function of one direction, dominated by the Kiel outer diameter (λ₁/λ₂ = 84.5). The objective for temperature recovery is a valley along the bleed-hole diameter, with a weaker second direction that thins the shroud (λ₁/λ₂ = 8.2, λ₂/λ₃ = 6.94). Both subspaces match the paper's ridge approximations (cosines 0.9996 for the loss; 0.9972 and 0.9810 for recovery).

### [Transonic_Airfoil_Shape_Parameterizations](Transonic_Airfoil_Shape_Parameterizations/)

Does the active subspace depend on how the shape is parameterized? Grey and Constantine perturb a NACA 0012 airfoil by ±20 % in the parameters of two global parameterizations, PARSEC (11 parameters) and CST (10 parameters), and compute lift and drag with SU2's Euler solver at Mach 0.8 (about 6500 designs each). Under both, lift has a one-dimensional active subspace and drag a two-dimensional one. Under CST, lift and drag depend only on the two leading coefficients x₁ and x₆, and the lift direction lies in the drag subspace; PARSEC spreads the same structure over 7–8 of its 11 parameters.

### [Transonic_Fan_Blade_Performance_Maps](Transonic_Fan_Blade_Performance_Maps/)

Seshadri et al. perturb a transonic research fan blade with 25 design variables (five degrees of freedom at five spanwise sections) and run 3-D RANS simulations with the Rolls-Royce solver Hydra (548 designs). Cruise efficiency and cruise pressure ratio each have a two-dimensional active subspace, which the paper turns into contour maps of the design space ("performance maps"). Efficiency has a ridge with a high-efficiency corridor and is dominated by leading-edge recamber at 50–100 % span; the pressure ratio is almost linear in its first active variable and is dominated by trailing-edge recamber. The first directions match the paper's (|w₁·w₁,paper| = 0.992 and 0.988).
