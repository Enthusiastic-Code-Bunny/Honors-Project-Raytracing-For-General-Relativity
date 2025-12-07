# angleImpact.py
import numpy as np
from numba import njit

@njit
def angle_to_impact(theta, r0=10, M=1.0):
    '''
    Converts angle theta (radians) from the image centerline to impact parameter b.
    Assumes flat space locally at r0.
    
    Parameters:
    -----------
        theta : float
            Angle between the outgoing photon and the radial axis.
            (theta = 0 means straight toward the black hole center.)

        r0 : float, optional (default = 10)
            Initial radial coordinate of the camera (in units of M).
            Represents how far the observer is from the black hole.

    Returns:
    --------
        b : float
            The photon's impact parameter b = L / E,
            where L is angular momentum per unit energy,
            and E is total energy per unit mass.
    
    Description:
    ------------
        In the Schwarzschild spacetime, a photon's conserved quantities
        satisfy the relation:
        
                b = L / E
        
        At the radius r0, under the locally flat-space, the photon leaves 
        at an angle theta with respect to the radial direction. Thus:
        
                L = r0 * sin(theta)
                E = sqrt(1 - 2M / r0)

        Therefore:
        
                b = (r0 * sin(theta)) / sqrt(1 - 2M / r0)
    '''
    # Avoid division by zero or horizon crossing
    if r0 <= 2.0:
        return 0.0
    
    L = r0 * np.sin(theta)      # angular momentum for photon at angle theta
    E = np.sqrt(1.0 - 2.0*M / r0) # energy per unit mass at radius r0
    
    return L/E