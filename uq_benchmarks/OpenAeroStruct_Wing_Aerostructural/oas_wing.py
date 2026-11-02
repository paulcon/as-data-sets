"""OpenAeroStruct (v2) reconstruction of the psdr ``OpenAeroStruct`` test function.

The psdr test function (Hokanson, psdr/demos/oas.py and psdr/demos/oas/oas_driver.py)
runs OpenAeroStruct v1 (OpenMDAO 1, Hokanson's fork at commit 8ee82d3, June 2018) inside
a Docker image. This module rebuilds the same analysis with the maintained
OpenAeroStruct 2.x (pip ``openaerostruct``, Apache-2.0) on OpenMDAO 3:

* rectangular wing, span 10 m, base chord 1 m, num_y = 7 (4 spanwise nodes on the
  symmetric half), num_x = 2, uniform spanwise spacing, symmetry;
* tube spar at 35 % chord, outer radius fixed at 0.06 m (OAS v1 computes it once from
  the undeformed 1 m chord and t/c = 0.12 and does not update it when the chord changes);
* aluminium-like material (E, G, density are inputs), yield 500 MPa / 2.5;
* cruise M = 0.84, a = 295.4 m/s, rho = 0.38 kg/m^3, CT = 9.80665*17e-6 1/s,
  R = 11.165e6 m, W0 = 1000 kg, inviscid (no viscous or wave drag, CL0 = CD0 = 0);
* chord control points [taper, 1] * root chord (tip, root), linear B-spline;
* twist control points (deg) and thickness control points (m) with the v1 B-spline
  convention (3 control points, order 3, evaluated at equally spaced parameters from tip
  to root: at the 4 nodes for twist, at 3 points for the 3 elements for thickness).
  OAS 2.x evaluates its thickness spline at element midpoints instead, so the v1 control
  points are mapped exactly to OAS 2.x control points by a fixed 3x3 matrix.

Inputs x (12, physical units, psdr order):
  twist_1..3 [deg], thickness_1..3 [m], root_chord [-, scale on the 1 m chord],
  taper_ratio [-], alpha [deg], E [Pa], G [Pa], mrho [kg/m^3]
Outputs (psdr order; psdr reported the two weights in N, here they are masses in kg):
  fuelburn [kg], structural_mass [kg], lift [N], total_mass [kg] = W0 + structural_mass
  + fuelburn, failure [-] (KS aggregate, rho = 100, of von Mises stress / allowable - 1;
  < 0 means the spar is within the allowable stress).

Gradients: OpenMDAO total derivatives (direct linear solve of the coupled system) for the
9 geometric/flight inputs; E and G are OAS surface options (not OpenMDAO variables) and
are differentiated by central differences (relative step 1e-4) of solves converged to
an absolute residual of 1e-10 (OAS's NonlinearBlockGS with Aitken); mrho enters only the
structural mass (mass is linear in mrho, no weight relief), so its derivative is exact:
d m_s / d mrho = m_s / mrho and d fuelburn / d mrho = fuelburn / (W0 + m_s) * m_s / mrho.

Known difference from psdr: OAS 2.x transfers the aerodynamic moments to the beam nodes
with arms from each panel's aerodynamic centre to each adjacent node, OAS v1 with one arm
to the element midpoint. Fuel burn and lift differ from psdr's stored outputs by < 0.4 %,
the KS failure value by 0.07-0.14 (see DATA.md). psdr reported structural weight with
g = 9.81 (OAS v1); masses here need no g.

Held settings (HELD) can be changed with OASWing(settings={...}); the defaults are the
psdr values, and the shipped data use them. Mesh sizes are fixed.

Written for as-data-sets (MIT). No code from psdr (AGPL-3.0) or OAS v1 is copied; the
settings were read from psdr's oas_driver.py and OAS v1's run_classes.py.
"""
import contextlib
import io
import warnings

import numpy as np

G0 = 9.80665
NAMES = ["twist_1", "twist_2", "twist_3", "thickness_1", "thickness_2", "thickness_3",
         "root_chord", "taper_ratio", "alpha", "E", "G", "mrho"]
QOIS = ["fuelburn", "structural_mass", "lift", "total_mass", "failure"]
QOI_UNITS = ["kg", "kg", "N", "kg", "-"]
LOWER = np.array([-1, -1, -1, 0.005, 0.005, 0.005, 0.7, 0.75, 2.0, 0.8 * 70e9, 0.8 * 30e9, 0.8 * 3e3])
UPPER = np.array([1, 1, 1, 0.05, 0.05, 0.05, 1.3, 1.25, 5.0, 1.2 * 70e9, 1.2 * 30e9, 1.2 * 3e3])
NOMINAL = 0.5 * (LOWER + UPPER)

FLIGHT = dict(M=0.84, a=295.4, rho=0.38, CT=G0 * 17.0e-6, R=11.165e6, W0=1000.0, Re=1.0e6)
RADIUS = 0.12 * 1.0 / 2.0  # m, v1 radii(): t/c * chord / 2 on the undeformed mesh
#: Held settings (psdr / OAS v1 defaults); OASWing(settings=...) may override them.
HELD = dict(W0=FLIGHT["W0"], Mach_number=FLIGHT["M"], speed_of_sound=FLIGHT["a"], rho_air=FLIGHT["rho"],
            CT=FLIGHT["CT"], R=FLIGHT["R"], span=10.0, spar_radius=RADIUS, yield_stress=500.0e6,
            safety_factor=2.5, solver_atol=1e-10)


def _v1_bspline_matrix(num_cp, num_pt):
    """B-spline basis matrix with clamped uniform knots, order min(num_cp, 4),
    evaluated at num_pt equally spaced parameters in [0, 1] (Cox-de Boor)."""
    k = min(num_cp, 4)
    knots = np.concatenate([np.zeros(k - 1), np.linspace(0, 1, num_cp - k + 2), np.ones(k - 1)])
    t = np.linspace(0, 1, num_pt)

    def basis(i, p, x):
        if p == 0:
            if knots[i] <= x < knots[i + 1]:
                return 1.0
            if x == knots[-1] and knots[i] < x <= knots[i + 1]:
                return 1.0
            return 0.0
        out = 0.0
        d1 = knots[i + p] - knots[i]
        d2 = knots[i + p + 1] - knots[i + 1]
        if d1 > 0:
            out += (x - knots[i]) / d1 * basis(i, p - 1, x)
        if d2 > 0:
            out += (knots[i + p + 1] - x) / d2 * basis(i + 1, p - 1, x)
        return out

    return np.array([[basis(i, k - 1, x) for i in range(num_cp)] for x in t])


class OASWing:
    """Builds the OpenMDAO problem once per (E, G, mrho) and evaluates outputs and gradients."""

    def __init__(self, tol=None, settings=None):
        self.held = dict(HELD)
        if settings:
            unknown = set(settings) - set(HELD)
            if unknown:
                raise ValueError("unknown held settings: %s" % sorted(unknown))
            self.held.update({k: float(v) for k, v in settings.items()})
        if tol is not None:
            self.held["solver_atol"] = float(tol)
        self.tol = self.held["solver_atol"]
        self._prob = None
        self._mat = None
        self._T = None

    # ------------------------------------------------------------------ setup
    def _build(self, E, G, mrho):
        import openmdao.api as om
        from openaerostruct.meshing.mesh_generator import generate_mesh
        from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

        mesh = generate_mesh({"num_y": 7, "num_x": 2, "wing_type": "rect", "symmetry": True,
                              "span": self.held["span"], "root_chord": 1.0, "span_cos_spacing": 0.0})
        surface = {
            "name": "wing", "symmetry": True, "S_ref_type": "wetted", "fem_model_type": "tube",
            "mesh": mesh,
            "twist_cp": np.zeros(3), "thickness_cp": 0.01 * np.ones(3), "chord_cp": np.ones(2),
            "radius_cp": self.held["spar_radius"] * np.ones(2),
            "CL0": 0.0, "CD0": 0.0, "k_lam": 0.05, "t_over_c_cp": np.array([0.12]), "c_max_t": 0.303,
            "with_viscous": False, "with_wave": False,
            "E": float(E), "G": float(G), "yield": self.held["yield_stress"], "safety_factor": self.held["safety_factor"], "mrho": float(mrho),
            "fem_origin": 0.35, "wing_weight_ratio": 1.0, "struct_weight_relief": False,
            "distributed_fuel_weight": False, "exact_failure_constraint": False,
        }
        h = self.held
        prob = om.Problem(reports=False)
        iv = om.IndepVarComp()
        iv.add_output("v", val=h["Mach_number"] * h["speed_of_sound"], units="m/s")
        iv.add_output("alpha", val=3.5, units="deg")
        iv.add_output("Mach_number", val=h["Mach_number"])
        iv.add_output("re", val=FLIGHT["Re"], units="1/m")
        iv.add_output("rho", val=h["rho_air"], units="kg/m**3")
        iv.add_output("CT", val=h["CT"], units="1/s")
        iv.add_output("R", val=h["R"], units="m")
        iv.add_output("W0", val=h["W0"], units="kg")
        iv.add_output("speed_of_sound", val=h["speed_of_sound"], units="m/s")
        iv.add_output("load_factor", val=1.0)
        iv.add_output("empty_cg", val=np.zeros(3), units="m")
        prob.model.add_subsystem("prob_vars", iv, promotes=["*"])
        prob.model.add_subsystem("wing", AerostructGeometry(surface=surface))
        pt = "AS_point_0"
        prob.model.add_subsystem(pt, AerostructPoint(surfaces=[surface]),
                                 promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "CT", "R",
                                                  "W0", "speed_of_sound", "empty_cg", "load_factor"])
        perf = pt + ".wing_perf"
        c = prob.model.connect
        c("wing.local_stiff_transformed", pt + ".coupled.wing.local_stiff_transformed")
        c("wing.nodes", pt + ".coupled.wing.nodes")
        c("wing.mesh", pt + ".coupled.wing.mesh")
        c("wing.radius", perf + ".radius")
        c("wing.thickness", perf + ".thickness")
        c("wing.nodes", perf + ".nodes")
        c("wing.cg_location", pt + ".total_perf.wing_cg_location")
        c("wing.structural_mass", pt + ".total_perf.wing_structural_mass")
        c("wing.t_over_c", perf + ".t_over_c")
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            prob.setup(mode="rev", check=False)
            coupled = getattr(prob.model, pt).coupled
            nl = coupled.nonlinear_solver  # OAS default: NonlinearBlockGS with Aitken
            nl.options["atol"] = self.tol
            nl.options["rtol"] = 1e-13
            nl.options["maxiter"] = 200
            nl.options["iprint"] = -1
            nl.options["err_on_non_converge"] = True
            prob.final_setup()
        if self._T is None:
            self._T = self._thickness_map(prob)
        self._prob = prob
        self._mat = (float(E), float(G), float(mrho))

    def _thickness_map(self, prob):
        """Exact map from v1 thickness control points to OAS 2.x control points.

        The OAS 2.x thickness spline is linear in its control points; its matrix B2 is
        obtained by evaluating a copy of the same SplineComp on unit vectors."""
        import openmdao.api as om
        from openaerostruct.utils.interpolation import get_normalized_span_coords
        x_interp = get_normalized_span_coords({"mesh": prob.model.wing.options["surface"]["mesh"]}, mid_panel=True)
        sp = om.Problem(reports=False)
        comp = om.SplineComp(method="bsplines", x_interp_val=x_interp, num_cp=3,
                             interp_options={"order": 3, "x_cp_start": 0, "x_cp_end": 1})
        comp.add_spline(y_cp_name="thickness_cp", y_interp_name="thickness")
        sp.model.add_subsystem("s", comp)
        sp.setup()
        B2 = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = 1.0
            sp.set_val("s.thickness_cp", e)
            sp.run_model()
            B2[:, j] = sp.get_val("s.thickness")
        B1 = _v1_bspline_matrix(3, 3)
        self.B1, self.B2 = B1, B2
        return np.linalg.solve(B2, B1)

    def _ensure(self, E, G, mrho):
        """Build the problem once; later material changes are written in place.

        OAS 2.x reads E, G and mrho from the surface dictionaries at compute time, except
        the von Mises component, which caches E and G at setup; both are updated here.
        Checked during the build: in-place updates and a fresh build agree to rounding
        in all outputs and gradients (DATA.md, Accuracy)."""
        mat = (float(E), float(G), float(mrho))
        if self._prob is None:
            self._build(*mat)
            return
        if self._mat == mat:
            return
        for sysm in self._prob.model.system_iter(recurse=True, include_self=True):
            if "surface" in sysm.options:
                surf = sysm.options["surface"]
                surfs = surf if isinstance(surf, list) else [surf]
                for sd in surfs:
                    sd["E"], sd["G"], sd["mrho"] = mat
            if "surfaces" in sysm.options:
                for sd in sysm.options["surfaces"]:
                    sd["E"], sd["G"], sd["mrho"] = mat
            if type(sysm).__name__ == "VonMisesTube":
                sysm.E, sysm.G = mat[0], mat[1]
        self._mat = mat

    # --------------------------------------------------------------- evaluate
    def _set(self, x):
        p = self._prob
        p.set_val("wing.twist_cp", x[0:3], units="deg")
        p.set_val("wing.thickness_cp", self._T @ x[3:6], units="m")
        p.set_val("wing.geometry.chord_cp", np.array([x[7], 1.0]) * x[6])
        p.set_val("alpha", x[8], units="deg")

    def _outputs(self):
        p = self._prob
        fb = float(p.get_val("AS_point_0.fuelburn", units="kg")[0])
        ms = float(p.get_val("wing.structural_mass", units="kg")[0])
        L = float(p.get_val("AS_point_0.wing_perf.L", units="N")[0])
        fail = float(p.get_val("AS_point_0.wing_perf.failure")[0])
        return np.array([fb, ms, L, self.held["W0"] + ms + fb, fail])

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        self._ensure(*x[9:12])
        self._set(x)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            self._prob.run_model()
        y = self._outputs()
        if not np.all(np.isfinite(y)):
            raise RuntimeError("non-finite OpenAeroStruct output")
        return y

    def _totals(self):
        p = self._prob
        of = ["AS_point_0.fuelburn", "wing.structural_mass", "AS_point_0.wing_perf.L",
              "AS_point_0.wing_perf.failure"]
        wrt = ["wing.twist_cp", "wing.thickness_cp", "wing.geometry.chord_cp", "alpha"]
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            J = p.compute_totals(of=of, wrt=wrt, return_format="dict")
        return J

    def evaluate_with_gradient(self, x, rel_step=1e-4):
        """Return (y[5], dy/dx[5, 12]), both in physical units (rows: QOIS, cols: NAMES)."""
        x = np.asarray(x, dtype=float)
        y = self.evaluate(x)
        J = self._totals()
        ofs = ["AS_point_0.fuelburn", "wing.structural_mass", "AS_point_0.wing_perf.L",
               None, "AS_point_0.wing_perf.failure"]
        D = np.zeros((5, 12))
        for i, o in enumerate(ofs):
            if o is None:
                continue
            D[i, 0:3] = J[o]["wing.twist_cp"].ravel()
            D[i, 3:6] = J[o]["wing.thickness_cp"].ravel() @ self._T
            dc = J[o]["wing.geometry.chord_cp"].ravel()  # chord_cp = [taper * rc, rc]
            D[i, 6] = dc[0] * x[7] + dc[1]
            D[i, 7] = dc[0] * x[6]
            D[i, 8] = J[o]["alpha"].ravel()[0]
        # E and G: central differences with a relative step (OAS options, not variables).
        for k in (9, 10):
            h = rel_step * x[k]
            xp = x.copy(); xp[k] += h
            xm = x.copy(); xm[k] -= h
            D[:, k] = (self.evaluate(xp) - self.evaluate(xm)) / (2 * h)
        self.evaluate(x)  # leave the problem at the base point
        # mrho enters only m_s = mrho * V(geometry) (no weight relief): exact derivative.
        ms = y[1]
        D[:, 11] = 0.0
        D[1, 11] = ms / x[11]
        D[0, 11] = y[0] / (self.held["W0"] + ms) * ms / x[11]  # Breguet: fb proportional to W0 + m_s
        D[3] = D[0] + D[1]  # total mass = W0 + m_s + fuelburn
        return y, D
