import numpy as np
from numba import njit

@njit
def cameraSetup(r0):
    """
    Defines the camera setup and screen configuration
    for raytraced visualization around a Schwarzschild black hole.

    It constructs:
    * The pixel grid (screen_x, screen_y) on the camera's image plane.
    * The camera's position, direction, and orientation vectors.
    
    The output is used by the rest of the simulation to compute
    ray trajectories and image formation.

    Parameters:
    -----------
        r0 : float
            The radial distance of the camera from the black hole
            (i.e., how far the observer is from the origin).

    Returns: 
    --------
        screenWidth : int
            Horizontal resolution (number of pixels per row).

        screenHeight : int
            Vertical resolution (number of pixels per column).

        screen_x : ndarray of shape (screenWidth,)
            Horizontal pixel coordinates on the virtual screen
            (spanning from -aspectRatio to +aspectRatio).

        screen_y : ndarray of shape (screenHeight,)
            Vertical pixel coordinates on the virtual screen
            (spanning from +aspectRatio to -aspectRatio).

        numRays : int
            Total number of rays/pixels = screenWidth x screenHeight.

        camDir : ndarray, shape (3,)
            Normalized vector pointing from the camera toward
            the black hole (camera's forward direction).

        camUp : ndarray, shape (3,)
            Normalized vector representing the upward direction
            in the camera's coordinate frame.

        camRight : ndarray, shape (3,)
            Normalized vector representing the rightward direction
            in the camera's coordinate frame.
    """
    # Set the desired screen resolution for the simulated image
    screenWidth = 1024  
    screenHeight = 1024
    
    # Determine the aspect ratio to preserve the proportions of the image.
    aspectRatio = screenWidth / screenHeight
    
    # Screen coordinates in the camera space:
    # screen_x: Make lists of horizontal positions across the camera screen
    screen_x = np.linspace(-aspectRatio, aspectRatio, screenWidth)
    # screen_y: Make lists of vertical positions across the camera screen
    screen_y = np.linspace(aspectRatio, -aspectRatio, screenHeight) 
    
    # Total rays needed = the number of pixels on the screen
    numRays = screenWidth * screenHeight
    
    # Define the camera's position and orientation relative to the black hole
    camPos = np.array([0.0, 0.0, r0])       # Camera position on z-axis
    camTarget = np.array([0.0, 0.0, 0.0])   # Camera points toward the origin (black hole)
    camUp = np.array([0.0, 1.0, 0.0])       # Camera vertical up direction
    
    # Normalize the direction vectors:
    # Forward (view) direction: from camera to target
    camDir = camPos - camTarget
    camDir = camDir / np.linalg.norm(camDir)
    
    # Right direction: perpendicular to both camDir and camUp
    camRight = np.cross(camDir, camUp)
    camRight = camRight / np.linalg.norm(camRight)
    
    # Recompute the true up direction to make sure of orthogonality
    camUp = np.cross(camRight, camDir) 
    
    return screenWidth, screenHeight,\
        screen_x, screen_y, numRays, \
        camDir, camUp, camRight
