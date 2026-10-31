"""Scratch prototype: numpad-free re-implementation of pde_model.PDESolver (residual identical)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

class PDESolver:
    def __init__(self, klfile):
        kl = np.load(klfile); K, lam = kl['K'], kl['lam']
        self.B = K*np.sqrt(lam)[None, :]
        self.nx = int(np.sqrt(self.B.shape[0])); self.dx = 1./self.nx; self.m = K.shape[1]; self.f = 1.0
        n = self.nx; self.N = (n-1)**2
    def _node(self, r, s):
        """padded node (r,s) -> unknown index or -1 (Dirichlet)."""
        n = self.nx; s = np.where(s == n, n-1, s)   # Neumann ghost copy
        ok = (r >= 1) & (r <= n-1) & (s >= 1) & (s <= n-1)
        return np.where(ok, (r-1)*(n-1) + (s-1), -1)
    def matrix(self, a):
        n, dx = self.nx, self.dx
        ahor = 0.5*(a[1:, :] + a[:-1, :]) / dx**2      # (n-1, n): edge (i+1,k)-(i+1,k+1)
        aver = 0.5*(a[:, 1:] + a[:, :-1]) / dx**2      # (n, n-1): edge (k,j+1)-(k+1,j+1)
        rows, cols, vals = [], [], []
        def edge(c, p0, p1):   # flux c*(u[p1]-u[p0]); +flux into eq at p0, -flux into eq at p1
            e0, e1 = self._node(*p0), self._node(*p1)
            for eq, sgn, owner in ((p0, 1., True), (p1, -1., False)):
                ie = self._node(*eq) if True else None
                # only true (non-ghost) interior nodes own an equation
                r, s = eq; own = (r >= 1) & (r <= n-1) & (s >= 1) & (s <= n-1)
                for node_idx, w in ((e1, 1.), (e0, -1.)):
                    msk = own & (node_idx >= 0)
                    rows.append(ie[msk]); cols.append(node_idx[msk]); vals.append((sgn*w*c)[msk])
        I, Kk = np.meshgrid(np.arange(n-1), np.arange(n), indexing='ij')
        edge(ahor, (I+1, Kk), (I+1, Kk+1))
        Kk, J = np.meshgrid(np.arange(n), np.arange(n-1), indexing='ij')
        edge(aver, (Kk, J+1), (Kk+1, J+1))
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.N, self.N))
    def coeff(self, x):
        return np.exp(self.B @ np.ravel(x)).reshape(self.nx, self.nx)
    def solve(self, x):
        a = self.coeff(x); L = self.matrix(a)
        lu = spla.splu(L)
        u = lu.solve(-self.f*np.ones(self.N)).reshape(self.nx-1, self.nx-1)
        return a, lu, u
    def _grad_from_adjoint(self, a, u, lam):
        n, dx = self.nx, self.dx
        P = np.pad(u, ((1, 1), (1, 0))); P = np.hstack([P, P[:, -1:]]); P[0, :] = 0; P[-1, :] = 0
        Dy = (P[1:-1, 1:] - P[1:-1, :-1])/dx; Dx = (P[1:, 1:-1] - P[:-1, 1:-1])/dx
        Lp = np.pad(lam, ((0, 0), (1, 1))); gFy = (Lp[:, :-1] - Lp[:, 1:])/dx
        Lq = np.pad(lam, ((1, 1), (0, 0))); gFx = (Lq[:-1, :] - Lq[1:, :])/dx
        gh, gv = gFy*Dy, gFx*Dx
        ga = np.zeros_like(a); ga[1:, :] += .5*gh; ga[:-1, :] += .5*gh; ga[:, 1:] += .5*gv; ga[:, :-1] += .5*gv
        return self.B.T @ (ga*a).ravel()      # = lam^T dR/dx
    # Elliptic_PDE QoI (identical formula to legacy: trapezoid over interior boundary-column nodes)
    def qoi_and_grad(self, x):
        a, lu, u = self.solve(x); n, dx = self.nx, self.dx
        c = np.zeros((n-1, n-1)); c[:, -1] = dx; c[0, -1] = c[-1, -1] = .5*dx
        q = np.sum(c*u)
        lam = lu.solve(c.ravel(), trans='T').reshape(n-1, n-1)
        return q, -self._grad_from_adjoint(a, u, lam)
    # MCMC QoI
    indz = [20, 30, 40, 50, 60, 70, 80]
    def observations(self, x):
        return self.solve(x)[2][self.indz, -1]
    def misfit_and_grad(self, x, d, sig2):
        a, lu, u = self.solve(x); n = self.nx
        r = u[self.indz, -1] - d; mf = .5*r@r/sig2
        c = np.zeros((n-1, n-1)); c[self.indz, -1] = r/sig2
        lam = lu.solve(c.ravel(), trans='T').reshape(n-1, n-1)
        return mf, -self._grad_from_adjoint(a, u, lam)
    def residual(self, u, a):   # legacy residual, numpy version, for verification
        n, dx = self.nx, self.dx
        u = np.hstack([np.zeros([n-1, 1]), u, u[:, -1:]]); u = np.vstack([np.zeros([1, n+1]), u, np.zeros([1, n+1])])
        ah = .5*(a[1:, :]+a[:-1, :]); av = .5*(a[:, 1:]+a[:, :-1])
        ady = ah*(u[1:-1, 1:]-u[1:-1, :-1])/dx; adx = av*(u[1:, 1:-1]-u[:-1, 1:-1])/dx
        return (ady[:, 1:]-ady[:, :-1])/dx + (adx[1:, :]-adx[:-1, :])/dx + self.f
