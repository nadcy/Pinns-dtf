a = 5
import numpy as np
from scipy import integrate
from user_fun.field import Field

def solve(x,t,mu=0.01 / np.pi):
    if np.abs(x) == 0 :
        return 0
    if np.abs(t) == 0:
        return -np.sin(np.pi*x)
    else:
        f = lambda y:np.exp(-np.cos(np.pi*y)/(2*np.pi*mu))
        g = lambda y,t:np.exp((-y**2)/(4*mu*t))
        fun = lambda eta:np.sin(np.pi*(x-eta)) * f(x-eta) * g(eta,t)
        uxt = -integrate.quad(fun,-np.inf,np.inf)[0]
        fun = lambda eta:f(x-eta)*g(eta,t)
        return uxt / integrate.quad(fun,-np.inf,np.inf)[0]
