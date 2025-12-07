# rk4_parallel.py
from numba import cuda
from geodesicSystem import geodesicSystem

@cuda.jit(fastmath=True)
def rk4_kernel(mass, r0, phi_start, phi_end, h_init, b_vals, phi_final_out, r_final_out, maxSteps):
    '''
    GPU-accelerated Runge-Kutta 4 (RK4) kernel for integrating photon geodesics 
    in Schwarzschild spacetime using adaptive step size control.

    Each CUDA thread traces one photon ray characterized by its impact parameter 'b',
    integrating the null geodesic equation in the equatorial plane. 
    The kernel uses a pair of embedded RK4 integrations (one large step vs. two half steps)
    to estimate local truncation error and adaptively adjust the integration step size.

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
    h_init : float
        Initial step size for the Runge-Kutta integrator.
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
    * Adaptive step sizing ensures both efficiency and accuracy.
    * Designed for parallel GPU execution: each thread integrates one geodesic independently.
    '''
    
    # Thread identification and bounds check
    i = cuda.grid(1)
    if i >= b_vals.shape[0]:
        return  # Skips the out-of-range threads

    # Initializing the integration variables
    r = r0              # current radial position
    M = mass            # black hole mass
    h = h_init          # initial step size
    phi = phi_start     # current azimuthal position
    direction = -1      # initial direction (inward)
    b = b_vals[i]       # impact parameter for this photon

    # Parameters controlling the adaptive stepping
    tol = 1e-6          # target relative error per step
    h_min = 1e-4        # smallest allowed step to prevent infinite loop
    h_max = 0.01        # largest allowed step for accuracy

    # Main integration loop
    for _ in range(maxSteps):
        # Stop if we've reached or exceeded the specified end angle
        if phi >= phi_end:
            break

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
 
    # Store final results for this ray
    phi_final_out[i] = phi
    r_final_out[i] = r
