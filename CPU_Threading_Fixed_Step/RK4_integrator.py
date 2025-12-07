# rk4.py
from numba import njit
import numpy as np
from geodesicSystem import geodesicSystem

@njit
def rk4(r0, phi_range, h, M, b):
    """
    Integrate a photon geodesic in Schwarzschild spacetime using 
    the fixed-step Runge-Kutta 4th order (RK4) method.

    This function computes only the final radial position 'r' and 
    azimuthal angle 'phi' for a photon ray, given its impact parameter 'b'.

    Parameters
    ----------
    r0 : float
        Initial radial distance of the photon from the black hole.
    phi_range : tuple of float
        Start and end azimuthal angles (phi_start, phi_end) in radians.
    h : float
        Fixed step size for the RK4 integration.
    M : float
        Mass of the Schwarzschild black hole.
    b : float
        Impact parameter of the photon (angular momentum per unit energy).

    Returns
    -------
    phi : float
        Final azimuthal angle of the photon after integration.
    r : float
        Final radial distance of the photon after integration.

    Notes
    -----
    * The integration stops early if the photon either falls into the 
      black hole (r < 2M) or escapes beyond a safe limit, as indicated
      by the 'geodesicSystem'.
    * The direction variable tracks inward/outward motion.
    * Designed for use in parallel execution where each photon ray is
      integrated independently.
    """
    # Runge-Kutta 4th order integrator returning only the last phi and r.
    phi_vals = np.arange(phi_range[0], phi_range[1], h, dtype=np.float32)
    phi = phi_range[0]
    r = r0
    # Used to keep track of how many steps were actually computed
    direction = -1 # Start with backward / inward direction

    for i in range(1, len(phi_vals)):        
        k1, stop, temp = geodesicSystem(r, b, direction, M)
        k1 = h * k1
        
        k2, stop, temp = geodesicSystem(r + 0.5 * k1, b, direction, M)
        k2 = h * k2
        
        k3, stop, temp = geodesicSystem(r + 0.5 * k2, b, direction, M)
        k3 = h * k3
        
        k4, stop, temp = geodesicSystem(r + k3, b, direction, M)
        k4 = h * k4
        
        if stop==1: 
            break
        
        r += (k1 + 2*k2 + 2*k3 + k4) / 6
        phi += h
        direction = temp

    return phi, r
