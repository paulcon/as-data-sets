"""Provenance script: write kle_basis.csv from PyApprox 1.0.3 and compare tracer_model.py with it.

NOT needed to run the notebook or generate_data.py, and NOT runnable in the repository's
.venv: it needs PyApprox 1.0.3 (MIT) and its dependencies, e.g. a Python 3.11 venv with

    pip install torch==2.2.2 numpy==1.26.4 scipy==1.11.4 matplotlib sympy cython \
        scikit-learn networkx numba scikit-fem umbridge pytest pytest-cov coverage
    pip install pyapprox==1.0.3        # the cp311 manylinux wheel; the sdist does not build
                                       # with current Cython

(scipy < 1.12 is needed: PyApprox 1.0.3 calls scipy.linalg.eigh(..., turbo=False)).

    python pyapprox_reference.py --kle kle_basis_new.csv    # write the KLE basis
    python pyapprox_reference.py --compare                  # compare 3 random points and z = 0

The 2026-11-01 run that wrote the shipped kle_basis.csv used PyApprox 1.0.3, torch 2.2.2,
numpy 1.26.4, scipy 1.11.4, Python 3.11, Linux x86_64.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def benchmark(m=10):
    try:
        from pyapprox.benchmarks import setup_benchmark
    except ImportError:
        sys.exit("PyApprox 1.0.3 is needed and was not found. Install it in a separate Python 3.11 "
                 "venv as the docstring of this script describes (torch 2.2.2, numpy 1.26.4, "
                 "scipy 1.11.4, then pyapprox==1.0.3), and rerun this script with that venv's python.")
    b = setup_benchmark("multi_index_advection_diffusion", kle_nvars=m,
                        kle_length_scale=0.5, time_scenario=True)
    return b.model_ensemble.functions[0]      # highest fidelity f0: orders [11, 11], dt = 0.2/64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kle", help="write the KLE basis CSV to this path")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    hf = benchmark()
    if args.kle:
        kle = hf._kle._kle                    # MeshKLE on the f0 mesh
        pts = np.asarray(hf._fwd_solver.physics.mesh.mesh_pts)
        hdr = ("KLE basis of the log-diffusivity on the 12x12 Chebyshev-Gauss-Lobatto mesh of "
               "[0,1]^2 (columns phi_d = sqrt(lambda_d)*v_d, squared-exponential kernel, length "
               "scale 0.5, sigma 1), computed by pyapprox 1.0.3 MeshKLE\n"
               + "x (-), y (-), " + ", ".join(f"phi_{j + 1} (-)" for j in range(kle.nterms)))
        np.savetxt(args.kle, np.column_stack([pts.T, kle.eig_vecs]), delimiter=",",
                   fmt="%.17g", header=hdr)
        print("wrote", args.kle)
    if args.compare:
        sys.path.insert(0, HERE)
        import tracer_model as tm
        model = tm.TracerModel(order=tm.ORDER_F0)   # PyApprox's f0 mesh
        Z = np.column_stack([np.zeros(10), np.random.default_rng(7).standard_normal((10, 3))])
        ref = hf(Z).ravel()
        for j in range(Z.shape[1]):
            q = model.qoi(Z[:, j])
            print(f"pyapprox {ref[j]:.15f}  tracer_model {q:.15f}  rel diff "
                  f"{abs(q - ref[j]) / abs(ref[j]):.1e}")


if __name__ == "__main__":
    main()
