# Exact evaluation (quadrature) of E[max(M, m+D3)] for 3 jobs, p=1, P2, lam=mu=1
import numpy as np
from scipy import integrate, special
lam=mu=1.0
a0=np.exp(-lam)                      # P(R=0)
def f(r):                            # density of R on (0,inf)
    return np.exp(-lam-mu*r)*np.sqrt(lam*mu/r)*special.i1(2*np.sqrt(lam*mu*r)) if r>0 else lam*mu*np.exp(-lam)
def F(r):                            # CDF of R
    if r<0: return 0.0
    return a0+integrate.quad(f,0,r,limit=200)[0]
# E[max(X,Y)] for X=max(R1,R2) style via E[C]=1+E[max(Rmax, Rmin+R3)+1]-1 ... do by survival function:
# C-2 = max(max(R1,R2)-1, min(R1,R2)+R3) is messy; use C = max(M, m+D3) with D=1+R:
# C = 1 + max(Rmax, Rmin + 1 + R3).  P(C-1 > x) = 1 - P(Rmax<=x, Rmin+1+R3<=x)
grid=np.linspace(0,60,6001)
Fg=np.array([F(x) for x in grid])
# P(Rmax<=x, Rmin + 1 + R3 <= x) = P(R1<=x,R2<=x, min<=x-1-R3)
# = E_R3[ G(x, x-1-R3) ], G(x,y)=P(both<=x) - P(both in (y,x]) for y<x  (y<0 => P(min<=y)=0 -> G = 0 when y<0)
def Fi(x): return np.interp(x,grid,Fg,left=0.0,right=1.0)
def G(x,y):
    if y<0: return 0.0
    return Fi(x)**2-(Fi(x)-Fi(y))**2
def joint(x):
    # integrate over R3: atom at 0 plus density
    val=a0*G(x,x-1)
    if x-1>0:
        val+=integrate.quad(lambda r:f(r)*G(x,x-1-r),0,x-1,limit=200)[0]
    return val
xs=np.linspace(0,60,3001)
surv=np.array([1-joint(x) for x in xs])
EC=1+np.trapezoid(surv,xs)
print("E[C_adaptive] =",EC," bound 2(1+lam/mu) =",2*(1+lam/mu))
