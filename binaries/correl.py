import numpy as np
import scipy.special as sp

c = 299792458
h = 6.62607015e-34
k = 1.380649e-23
    
T_a = 25e3
T_b = 21e3
r_a = 0.22*8e-9
r_b = 0.12*8e-9

def visib(X,Y,lam,dis,phi):
    
    nu = c/lam

    cs,sn = np.cos(phi),np.sin(phi)
    X,Y = cs*X + sn*Y, sn*X - cs*Y
    
    r = np.sqrt(X**2 + Y**2) + 1e-12
    v = 2*np.pi*r/c

    a1 = np.pi * (nu * r_a / c)**2
    a2 = h*nu/(k*(T_a + 2.725))
    a2 = np.clip(a2,0,10)
    I_a = a1 / (np.exp(a2) - 1)
    u_a = v * nu *  r_a 

    b1 = np.pi * (nu * r_b / c)**2
    b2 = h*nu/(k*(T_b + 2.725))
    b2 = np.clip(b2,0,10)
    I_b = b1 / (np.exp(b2) - 1)
    u_b = v * nu * r_b

    v_a = (I_a * 2 * sp.j1(u_a)/u_a)
    v_b = (I_b * 2 * sp.j1(u_b)/u_b)
    v_ab = 2 * v_a * v_b * np.cos(v * nu * dis * X/r)

    return (I_a + I_b)**(-1) * (v_a**2 + v_b**2 + v_ab)

