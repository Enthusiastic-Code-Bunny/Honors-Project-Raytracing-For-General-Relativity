# geodesicSystem.py
from numba import cuda
from numba.cuda import libdevice

@cuda.jit(device=True, inline=True, fastmath=True)
def geodesicSystem(r, b, direction, M):
    '''
    Compute the differential equation for null geodesics in Schwarzschild spacetime.

    This CUDA device function evaluates the rate of change of the radial coordinate
    for a photon traveling in the equatorial plane of a Schwarzschild metric.
    It uses the impact parameter 'b' and the black hole mass 'M' to determine the
    trajectory direction (inward or outward).

    Parameters:
    -----------
    r : float
        Current radial coordinate.
    b : float
        Impact parameter of the photon (b = L/E, where L is angular momentum and E is energy).
    direction : int
        Sign indicating motion direction: +1 for outward, -1 for inward.
    M : float
        Mass of the central Schwarzschild black hole.
    
    Returns:
    --------
    dr_dphi : float
        The derivative of the radial coordinate with respect to phi (d_phi being the azimuthal angle).
    flag : int
        returns 1 if integration stopped (escaping to flat spacetime or falling into the black hole).
        returns 0 otherwise.
    direction : int
        Potentially updated motion direction if a turning point is encountered during integration.
    
    Notes:
    ------
    * The function is designed for GPU use with Numba's CUDA JIT compiler.
    * For stability, it returns zeros when 'r' is outside the integration domain
      (inside the event horizon, ~2M, or where space time is essentially flat).
    '''
    # Terminate if the radius of the photon is too close to the black hole or too far away
    if r < 1.99 * M  or r > 15.0:
        return 0.0, 1, direction

    # Precomputed the square terms for efficiency (used more than once)
    r_squared = r * r
    b_squared = b * b
    
    # Calculate the term under the square root in the null geodesic equation
    radical = (r_squared - b_squared * (1 - (2*M)/r)) / (b_squared * r_squared)
    
    # If the radical is negative, we have reached a turning point; reverse direction
    if radical < 0.0:
        radical *= -1
        direction *= -1
    
    # compute the rate of change of the radial coordinate with respect to phi
    dr_dphi = direction * (r_squared) * libdevice.sqrt(radical)
    
    # Return the derivative, the continuation flag, and the direction
    return dr_dphi, 0, direction
