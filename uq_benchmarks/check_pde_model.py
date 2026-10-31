"""Acceptance check for the shared elliptic PDE solver (pde_model.py).

Runs the criteria in docs/port-2026/numpad_replacement.md section 8 on both copies:
Elliptic_PDE_Random_Coefficients/ and PDE_Inverse_Problem_MCMC/. Run it from
anywhere after any change to pde_model.py:

    python uq_benchmarks/check_pde_model.py

Exits nonzero if a check fails. Takes about a minute.
"""
import filecmp
import importlib.util
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {'elliptic': os.path.join(HERE, 'Elliptic_PDE_Random_Coefficients'),
        'mcmc': os.path.join(HERE, 'PDE_Inverse_Problem_MCMC')}
CASES = {'elliptic': ['short', 'long'], 'mcmc': ['short']}
failures = []


def report(ok, name, detail):
    print('{} {}: {}'.format('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        failures.append(name)


def load(name, d):
    spec = importlib.util.spec_from_file_location('pde_model_' + name, os.path.join(d, 'pde_model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fd_grad(f, x, h=1e-5):
    return np.array([(f(x + h*e) - f(x - h*e)) / (2*h) for e in np.eye(len(x))])


same = filecmp.cmp(os.path.join(DIRS['elliptic'], 'pde_model.py'),
                   os.path.join(DIRS['mcmc'], 'pde_model.py'), shallow=False)
report(same, 'copies identical', 'pde_model.py in both directories')

for name, d in DIRS.items():
    pdem = load(name, d)
    for case in CASES[name]:
        tag = '{} {}'.format(name, case)
        rng = np.random.default_rng(20261030)
        S = pdem.PDESolver(case)

        # legacy residual at the new solution
        res = 0.0
        for _ in range(5):
            u, a, _ = S.solve(rng.standard_normal(S.m))
            res = max(res, np.abs(S.residual(u, a)).max())
        report(res < 1e-10, tag + ' residual', 'max |residual| = {:.1e} (< 1e-10)'.format(res))

        q0 = S.qoi(np.zeros(S.m))
        report(round(q0, 5) == 0.07608, tag + ' q(0)', '{:.5f} (0.07608)'.format(q0))

        if case == 'long':
            t = np.array([-3, -1.5, 0, 1.5, 3])
            q = np.array([S.qoi(ti*np.eye(S.m)[0]) for ti in t])
            ref = np.array([0.0064, 0.0221, 0.0761, 0.2614, 0.8971])
            err = np.abs(q - ref).max()
            report(err <= 1e-4, tag + ' q(t e_0)', '{} (max diff {:.1e}, <= 1e-4)'.format(np.round(q, 4), err))

        x = rng.standard_normal(S.m)
        g = S.grad_qoi(x)
        rel = np.linalg.norm(g - fd_grad(S.qoi, x)) / np.linalg.norm(g)
        report(rel < 1e-6, tag + ' grad_qoi vs FD', 'relative error {:.1e} (< 1e-6)'.format(rel))

        q1, g1 = S.qoi_and_grad(x)
        dq, dg = abs(q1 - S.qoi(x)) / abs(q1), np.linalg.norm(g1 - g) / np.linalg.norm(g)
        report(max(dq, dg) < 1e-12, tag + ' qoi_and_grad',
               'vs qoi, grad_qoi: {:.1e}, {:.1e} (< 1e-12)'.format(dq, dg))

        M = pdem.PDEMisfit(case)
        s2 = (np.array(M.indz) + 1) * M.dx
        report(np.allclose(s2, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), tag + ' sensors',
               's2 = {}'.format(np.round(s2, 2)))
        obs = M.observations(rng.standard_normal(M.m))
        sig2 = 1e-4*np.linalg.norm(obs)**2
        M.set_data(obs + np.sqrt(sig2)*rng.standard_normal(obs.shape), sig2)
        x = rng.standard_normal(M.m)
        g = M.misfit_grad(x)
        rel = np.linalg.norm(g - fd_grad(M.misfit, x)) / np.linalg.norm(g)
        report(rel < 1e-6, tag + ' misfit_grad vs FD', 'relative error {:.1e} (< 1e-6)'.format(rel))

        X = rng.standard_normal((20, S.m))
        for fname, fn in [('qoi', S.qoi), ('grad_qoi', S.grad_qoi), ('misfit', M.misfit), ('misfit_grad', M.misfit_grad)]:
            fn(X[0])
            t0 = time.perf_counter()
            for xx in X:
                fn(xx)
            ms = 1e3*(time.perf_counter() - t0)/len(X)
            report(ms <= 30, '{} {} time'.format(tag, fname), '{:.1f} ms per call (<= 30)'.format(ms))

imp = re.compile(r'^\s*(import|from)\s+numpad\b', re.M)
hits = [os.path.join(r, f) for r, _, fs in os.walk(HERE) for f in fs
        if f.endswith('.py') and imp.search(open(os.path.join(r, f)).read())]
report(not hits, 'no numpad import in .py files', ', '.join(os.path.relpath(h, HERE) for h in hits) or 'none')

print('{} failed'.format(len(failures)) if failures else 'all passed')
sys.exit(1 if failures else 0)
