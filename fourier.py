import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def fourier(f):
    return fftshift(fft2(fftshift(f)))

def ifourier(f):
    return fftshift(ifft2(fftshift(f)))

# Generates sky and ground grids
# ds is grid spacing on the sky
# N is grid size (equal on ground and sky)
def grids(ds,N,lam):
    dx = lam/(N*ds)
    sx = (np.arange(N) - N//2) * ds
    sx,sy = np.meshgrid(sx,sx)
    x = (np.arange(N) - N//2) * dx
    x,y = np.meshgrid(x,x)
    return sx,sy,x,y

