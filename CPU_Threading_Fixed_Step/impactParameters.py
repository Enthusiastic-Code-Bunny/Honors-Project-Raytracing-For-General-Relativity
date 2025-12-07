import numpy as np
from numba import njit
from angleImpact import angle_to_impact

@njit
def impactParameters(r0, screen_x, screen_y, screenWidth, screenHeight, \
    camDir, camUp, camRight, fov, numRays, M):
    """
    Compute impact parameters and azimuthal angles for each screen pixel.

    For each pixel, a ray is traced from the camera and its direction is
    converted to spherical coordinates. The impact parameter 'b' is then
    computed based on the angle from the camera centerline, which determines
    the bending of light around a Schwarzschild black hole.
 
    Parameters
    ----------
    r0 : float
        Radial distance of the camera from the black hole center.
    screen_x : 1D array
        Horizontal coordinates of screen pixels in camera space.
    screen_y : 1D array
        Vertical coordinates of screen pixels in camera space.
    screenWidth : int
        Number of horizontal pixels.
    screenHeight : int
        Number of vertical pixels.
    camDir : array_like
        Camera forward direction (3D vector).
    camUp : array_like
        Camera up direction (3D vector).
    camRight : array_like
        Camera right direction (3D vector).
    fov : float
        Field of view in radians.
    numRays : int
        Total number of rays/pixels.
    M : float
        Black hole mass (used in impact parameter calculation).

    Returns
    -------
    b_vals : 1D array of floats
        Impact parameter for each ray.
    alphaVals : 1D array of floats
        Azimuthal angle (phi) in the camera's xy-plane for each ray.
    """
    
    # Allocate arrays for results
    b_vals = np.empty(numRays, dtype=np.float32)
    alphaVals = np.empty(numRays, dtype=np.float32)
    
    # Loop over each pixel
    for i in range(screenHeight):      # Row index (vertical)
        for j in range(screenWidth):   # Column index (horizontal)
            idx = i * screenWidth + j  # Linear index for flattened arrays
            
            # Start with the camera's forward vector and offset it by a small amount
            # horizontally (camRight) and vertically (camUp) based on pixel position
            # tan(fov/2) just scales how big the left/right/up offsets should be
            rayDir = (camDir + 
                      screen_x[j] * camRight * np.tan(fov/2) + 
                      screen_y[i] * camUp * np.tan(fov/2))
            
            # Normalize the ray direction to unit length
            rayDir = rayDir / np.linalg.norm(rayDir)
            
            # Convert to spherical coordinates:
            # theta: how high or low it points (polar angle)
            # alpha: which way around the circle (azimuthal angle)
            theta = np.arccos(rayDir[2])              
            alpha = np.arctan2(rayDir[1], rayDir[0]) 
            
            # Remember the azimuthal angle for this pixel to use
            # later when choosing a color from the background
            alphaVals[idx] = alpha
            
            # Determin how strongly the ray bends by computing its impact parameter 'b'
            b_vals[idx] = angle_to_impact(theta, r0, M)
    
    return b_vals, alphaVals