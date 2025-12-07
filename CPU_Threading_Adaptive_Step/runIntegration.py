from numba import njit, prange
import numpy as np
from RK4_integrator import rk4 

@njit(parallel=True)
def runIntegration(numRays, M, b_vals, r0):
    """
    Integrate multiple light rays in parallel using CPU cores.

    Each ray is propagated from the initial radius 'r0' through a full
    azimuthal range [0, 2pi] using an adaptive RK4 integrator. Only the 
    final radial and angular positions are returned.

    Parameters
    ----------
    numRays : int
        Number of rays (pixels) to integrate.
    M : float
        Black hole mass.
    b_vals : 1D array of floats
        Impact parameters for each ray.
    r0 : float
        Initial radial distance of the camera from the black hole.

    Returns
    -------
    phi_out : 1D array of floats
        Final azimuthal angles for each ray.
    r_out : 1D array of floats
        Final radial distances for each ray.
    """

    # Allocate output arrays
    phi_out = np.zeros(numRays, dtype=np.float32)
    r_out = np.zeros(numRays, dtype=np.float32)

    # Integration parameters
    phi_range = (0.0, 2.0 * np.pi)
    h = 0.0001          # Initial step size

    # Parallel loop over all rays
    for i in prange(numRays):
        b = b_vals[i]

        # Integrate the ray using adaptive RK4
        phi, r = rk4(r0, phi_range, h, M, b)

        # Store results
        phi_out[i] = phi
        r_out[i] = r

    return phi_out, r_out
