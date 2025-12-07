import numpy as np
import math
from numba import njit, prange
from backgroundColor import backgroundColor

@njit(parallel=True)
def createImage(screenWidth, screenHeight, numRays, phi_out, r_out, alphaVals, M):
    """
    Generate the final image from raytracing results using a simple
    checkered background pattern or a real image.

    Each pixel color is determined by tracing the corresponding ray,
    computing its lensed direction based on the Schwarzschild metric,
    and selecting a color based on where it points.

    Parameters
    ----------
    screenWidth : int
        Width of the screen (number of pixels horizontally).
    screenHeight : int
        Height of the screen (number of pixels vertically).
    numRays : int
        Total number of rays (should equal screenWidth * screenHeight).
    phi_out : 1D np.ndarray of floats
        Final azimuthal angles for each ray after integration.
    r_out : 1D np.ndarray of floats
        Final radial distances for each ray after integration.
    alphaVals : 1D np.ndarray of floats
        Azimuthal angles in the camera plane for each ray.
    M : float
        Black hole mass.

    Returns
    -------
    image : 3D np.ndarray of floats
        Array of shape (screenHeight, screenWidth, 3) containing
        RGB values for each pixel.
    """
    # PART 1: UNCOMMENT TO USE BACKGROUND IMAGE
    # background_img = np.array(Image.open("starImage.jpg").convert("RGB")) / 255.0
    
    # Create empty image array (height × width × RGB channels)
    image = np.zeros((screenHeight, screenWidth, 3), dtype=np.float32)
    
    # Loop over all rays in parallel
    for idx in prange(numRays):
        # Convert linear index back to 2D image coordinates
        i = idx // screenWidth  # Row index
        j = idx % screenWidth   # Column index

        # Retrieve final ray properties
        final_r = r_out[idx]
        final_phi = phi_out[idx]
        alpha = alphaVals[idx]
        
        if final_r > 2.0 * M:  # Ray escaped
            # Calculate lensed direction in 3D space
            x = final_r * math.sin(final_phi) * math.cos(alpha)
            y = final_r * math.sin(final_phi) * math.sin(alpha)
            z = final_r * math.cos(final_phi)
            lensed_phi = math.atan2(y, x)
            lensed_theta = math.acos(z/final_r)
            
            # PART 2: UNCOMMENT TO USE BACKGROUND IMAGE
            # Pick a color from the picture based on the lensed direction
            # color = backgroundImage(lensed_phi, lensed_theta, background_img)
            
            # COMMENT OUT AND UNCOMMENT ABOVE TO USE BACKGROUND IMAGE INSTEAD OF CHECKERED PATTERN
            # Alternatively, use the checkered pattern:
            color = backgroundColor(lensed_phi, lensed_theta)
        else:  # Ray hit black hole
            color = np.array([0.0, 0.0, 0.0])
        
        # Store color in image array
        image[i, j] = color
    
    return image