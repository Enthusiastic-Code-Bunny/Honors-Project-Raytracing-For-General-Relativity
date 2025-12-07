# rk4.py
from numba import njit
import numpy as np
from geodesicSystem import geodesicSystem

@njit
def rk4(r0, phi_range, h, M, b):  
    """
    Adaptive Runge-Kutta 4th order integration for a Schwarzschild geodesic.

    This function integrates the radial position r as a function of azimuthal angle phi
    for a light ray with impact parameter b near a black hole of mass M. It uses an
    RK4 estimate to adjust the step size dynamically for accuracy.

    Parameters
    ----------
    r0 : float
        Initial radial distance from the black hole.
    phi_range : tuple of floats
        (phi_start, phi_end) defining the azimuthal angle range for integration.
    h : float
        Initial step size for the integration.
    M : float
        Black hole mass.
    b : float
        Impact parameter of the light ray.

    Returns
    -------
    phi : float
        Final azimuthal angle reached by the integration.
    r : float
        Final radial distance of the ray.
    """ 
    
    # Initializing the integration variables 
    phi = phi_range[0]      # current azimuthal position
    phi_end = phi_range[1]
    r = r0                  # current radial position
    direction = -1          # initial direction (inward)
    
    # Parameters controlling the adaptive stepping
    tol = 1e-5          # target relative error per step
    h_min = 1e-6        # smallest allowed step to prevent infinite loop
    h_max = 0.01        # largest allowed step for accuracy

    # Main integration loop
    while phi < phi_end:        
        # Performs one full step of size h
        k1, stop, temp = geodesicSystem(r, b, direction, M)
        k2, _, _    = geodesicSystem(r + 0.5*h*k1, b, direction, M)
        k3, _, _    = geodesicSystem(r + 0.5*h*k2, b, direction, M)
        k4, _, _    = geodesicSystem(r + h*k3, b, direction, M)
        r_big = r + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Performs the first half step of size h/2
        h2 = 0.5*h
        k1s, _, _ = geodesicSystem(r, b, direction, M)
        k2s, _, _ = geodesicSystem(r + 0.5*h2*k1s, b, direction, M)
        k3s, _, _ = geodesicSystem(r + 0.5*h2*k2s, b, direction, M)
        k4s, _, _ = geodesicSystem(r + h2*k3s, b, direction, M)
        r_half = r + (h2/6.0)*(k1s + 2*k2s + 2*k3s + k4s)

        # Performs the second half step
        k1s, _, _ = geodesicSystem(r_half, b, direction, M)
        k2s, _, _ = geodesicSystem(r_half + 0.5*h2*k1s, b, direction, M)
        k3s, _, _ = geodesicSystem(r_half + 0.5*h2*k2s, b, direction, M)
        k4s, stop, temp = geodesicSystem(r_half + h2*k3s, b, direction, M)
        r_small = r_half + (h2/6.0)*(k1s + 2*k2s + 2*k3s + k4s)

        # Estimates the error between the two step sizes
        error = abs(r_small - r_big)
        # Adaptive step size adjustment based on the estimated error
        if error > tol and h > h_min:
            # error is too big, reduce step size and retry
            h *= 0.5
            continue  # redo this step with smaller h
        elif error < tol*0.25 and h < h_max:
            # error is very small, can increase step size
            h *= 1.5

        # Accept step
        r = r_small         # update radial coordinate
        phi += h            # update azimuthal angle
        direction = temp    # update direction in case the ray turned

        # Stop the integration early if geodesicSystem signals to stop
        # i.e., if the photon escapes or falls into the black hole
        if stop:
            break
    
    # Return final results for this ray
    return phi, r
