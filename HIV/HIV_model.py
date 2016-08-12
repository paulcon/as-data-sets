from scipy.integrate import ode
import numpy as np

def Tcells(p, times):
    """
    This takes the unnormalized parameters, p = (s1, s2, s3, p1, C1, K1, K2, K3, K4, K5, K6, K7, K8, 
    K9, K10, K11, K12, K13, d1, d2, d3, d4, d5, d6, d7, a1, psy), and returns the Tcell count at points
    in the 'times' input.
    """
    
    s1 = p[:,0]; s2 = p[:,1]; s3 = p[:,2]; p1 = p[:,3]; C1 = p[:,4]; K1 = p[:,5]; K2 = p[:,6]; K3 = p[:,7]
    K4 = p[:,8]; K5 = p[:,9]; K6 = p[:,10]; K7 = p[:,11]; K8 = p[:,12]; K9 = p[:,13]; K10 = p[:,14]; K11 = p[:,15]
    K12 = p[:,16]; K13 = p[:,17]; d1 = p[:,18]; d2 = p[:,19]; d3 = p[:,20]; d4 = p[:,21]; d5 = p[:,22]; d6 = p[:,23]
    d7 = p[:,24]; a1 = p[:,25]; psy = p[:,26]
    
    def ode_system(x, t, i):
        T = x[0]; TI = x[1]; TL = x[2]; M = x[3]; MI = x[4]; CTL = x[5]; V = x[6]
        
        if T < .1: return np.zeros(7)
        
        dTdt = s1[i] + p1[i]*T*V/(C1[i] + V) - d1[i]*T - (K1[i]*V + K2[i]*MI)*T
        dTIdt = psy[i]*(K1[i]*V + K2[i]*MI)*T + a1[i]*TL - d2[i]*TI - K3[i]*TI*CTL
        dTLdt = (1 - psy[i])*(K1[i]*V + K2[i]*MI)*T - a1[i]*TL - d3[i]*TL
        dMdt = s2[i] + K4[i]*M*V - K5[i]*M*V - d4[i]*M
        dMIdt = K5[i]*M*V - d5[i]*MI - K6[i]*MI*CTL
        dCTLdt = s3[i] + (K7[i]*TI + K8[i]*MI)*CTL - d6[i]*CTL
        dVdt = K9[i]*TI + K10[i]*MI - K11[i]*T*V - (K12[i] + K13[i])*M*V - d7[i]*V
        
        return np.array((dTdt, dTIdt, dTLdt, dMdt, dMIdt, dCTLdt, dVdt))
        
    def jac(x, t, i):
        T = x[0]; TI = x[1]; TL = x[2]; M = x[3]; MI = x[4]; CTL = x[5]; V = x[6]
        
        if T < .1: return np.zeros((7, 7))
        
        return np.array([[p1[i]*V/(C1[i] + V) - d1[i] - (K1[i]*V + K2[i]*MI), psy[i]*(K1[i]*V + K2[i]*MI), \
            (1 - psy[i])*(K1[i]*V + K2[i]*MI), 0, 0, 0, -K11[i]*V],\
        [0, -d2[i] - K3[i]*CTL, 0, 0, 0, K7[i]*CTL, K9[i]],\
        [0, a1[i], -a1[i] - d3[i], 0, 0, 0, 0],\
        [0, 0, 0, K4[i]*V - K5[i]*V - d4[i], K5[i]*V, 0, -(K12[i] + K13[i])*V],\
        [-K2[i]*T, psy[i]*K2[i]*T, (1 - psy[i])*K2[i]*T, 0, -d5[i] - K6[i]*CTL, K8[i]*CTL, K10[i]],\
        [0, -K3[i]*TI, 0, 0, -K6[i]*MI, K7[i]*TI + K8[i]*MI - d6[i], 0],\
        [p1[i]*T/(C1[i] + V) - p1[i]*T*V/(C1[i] + V)**2 - K1[i]*T, psy[i]*K1[i]*T, (1 - psy[i])*K1[i]*T,
            K4[i]*M - K5[i]*M, K5[i]*M, 0, -K11[i]*T - (K12[i] + K13[i])*M - d7[i]]])
    
    ans = np.empty((p.shape[0], len(times)))
    
    for i in range(p.shape[0]):        
        T0 = 1000
        M0 = .15/.005
        CTL0 = 5/.015
        
        y0 = np.array((T0, 0, 0, M0, 0, CTL0, 1))
        
        solver = ode(lambda t, x: ode_system(x, t, i), lambda t, x: jac(x, t, i).T)
        solver.set_integrator('vode', nsteps=1500, method='bdf', max_step=1)
        solver.set_initial_value(y0, 0)
        
        for j in range(len(times)):
            ans[i,j] = solver.integrate(times[j])[0]      
    
    return ans
