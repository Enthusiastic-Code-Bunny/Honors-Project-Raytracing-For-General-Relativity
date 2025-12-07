# rk4_parallel.py
from numba import cuda
from geodesicSystem import geodesicSystem

@cuda.jit(fastmath=True)
def rk4_kernel(M, r0, phi_start, phi_end, h, b_vals, phi_final_out, r_final_out, maxSteps):
    """
    GPU-accelerated Runge-Kutta 4 (RK4) kernel for integrating photon geodesics 
    in Schwarzschild spacetime using a fixed step size.

    Each CUDA thread traces one photon ray characterized by its impact parameter 'b',
    integrating the null geodesic equation in the equatorial plane. 

    Parameters
    ----------
    mass : float
        Mass of the central Schwarzschild black hole.
    r0 : float
        Initial radial position of the photon (starting distance).
    phi_start : float
        Starting azimuthal angle (usually 0).
    phi_end : float
        Ending azimuthal angle for the integration.
    h : float
        Step size for the Runge-Kutta integrator.
    b_vals : 1D device array of floats
        Impact parameters for each photon (one per thread).
    phi_final_out : 1D device array of floats
        Output array storing the final azimuthal angle for each photon.
    r_final_out : 1D device array of floats
        Output array storing the final radial coordinate for each photon.
    maxSteps : int
        Maximum number of integration steps to perform.
    
    Notes
    -----
    * The integration is stopped early if the photon escapes (r > 15M)
      or falls into the black hole (r < 1.99M), as flagged by 'geodesicSystem'.
    * Designed for parallel GPU execution: each thread integrates one geodesic independently.
    """
    
    # Thread identification and bounds check
    i = cuda.grid(1)
    if i >= b_vals.shape[0]:
        return      # Skips the out-of-range threads
    
    # Initializing the integration variables
    r = r0              # current radial position
    b = b_vals[i]       # impact parameter for this photon
    phi = phi_start     # current azimuthal position
    direction = -1      # initial direction (inward)
    
    # Main integration loop
    for _ in range(maxSteps):
         # Stop if we've reached or exceeded the specified end angle
        if phi >= phi_end:
            break
        
        # k1: slope at current point
        k1, stop, temp = geodesicSystem(r, b, direction, M)
        k1 = h * k1
        
        # k2: slope at midpoint using k1
        k2, stop, temp = geodesicSystem(r + 0.5*k1, b, direction, M)
        k2 = h * k2
        
        # k3: slope at midpoint using k2  
        k3, stop, temp = geodesicSystem(r + 0.5*k2, b, direction, M)
        k3 = h * k3
        
        # k4: slope at end using k3
        k4, stop, temp = geodesicSystem(r + k3, b, direction, M)
        k4 = h * k4
        
        # Stop the integration early if geodesicSystem signals to stop
        # i.e., if the photon escapes or falls into the black hole
        if stop == 1:
            break
        
        # Update for next iteration
        r += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0  # update radial coordinate
        phi += h            # update azimuthal angle
        direction = temp    # Update direction based on geodesic system
    
    # Store final results for this ray
    phi_final_out[i] = phi
    r_final_out[i] = r