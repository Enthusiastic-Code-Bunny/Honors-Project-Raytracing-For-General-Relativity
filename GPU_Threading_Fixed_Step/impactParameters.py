import numpy as np
from numba import njit
from angleImpact import angle_to_impact

@njit
def impactParameters(r0, screen_x, screen_y, screenWidth, screenHeight, \
    camDir, camUp, camRight, fov, numRays, M):
    """
    Computes the impact parameter 'b' and azimuthal angle 'alpha'
    for every pixel on the simulated camera screen (i.e., for all camera
    rays projected across the image plane).
    
    Each ray originates from the camera position at radius r0 and points 
    toward a direction determined by the pixel location on the screen. The
    resulting impact parameter determines how much light from that ray will
    bend due to the black hole's gravity.

    Parameters:
    -----------
        r0 : float
            Radial distance of the camera from the black hole.

        screen_x, screen_y : np.ndarray
            1D coordinate arrays defining the horizontal and vertical
            pixel positions across the camera's screen.

        screenWidth, screenHeight : int
            Resolution of the simulated screen (in pixels).

        camDir, camUp, camRight : np.ndarray
            Unit vectors defining the camera's forward, upward, and
            rightward directions in 3D space.

        fov : float
            Field of view (in radians) controlling how wide the camera
            "looks" into space.

        numRays : int
            Total number of rays to trace (screenWidth x screenHeight).

        M : float
            Black hole mass parameter (used by angle_to_impact).

    Returns:
    --------
        b_vals : np.ndarray (float32)
            Array of impact parameters, one per pixel.
        
        alphaVals : np.ndarray (float32)
            Array of azimuthal angles (alpha), used later for
            background color mapping.
    
    Description:
    ------------
        For each pixel:
          1. Construct a ray direction vector based on camera
             orientation and pixel position on the screen.
          2. Convert the direction to spherical coordinates
             (theta, alpha) to determine where the ray points.
          3. Compute the impact parameter 'b' from theta using 
             the angle_to_impact function.

        The resulting b-values are later passed to the GPU
        integrator (rk4_kernel) to compute geodesic trajectories.
    """
    
    # Initialize output arrays
    b_vals = np.empty(numRays, dtype=np.float32)    # impact parameters
    alphaVals = np.empty(numRays, dtype=np.float32) # azimuthal angles
    
    # Loop through every pixel on the screen
    for i in range(screenHeight):      # row index (vertical)
        for j in range(screenWidth):   # column index (horizontal)
            idx = i * screenWidth + j  # linear index for flattened arrays
            
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