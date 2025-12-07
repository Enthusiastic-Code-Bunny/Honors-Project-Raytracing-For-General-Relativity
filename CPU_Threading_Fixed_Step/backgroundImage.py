from numba import njit
import math

# Picks a color from a real picture (background_image) 
# using the directions phi, theta
@njit
def backgroundImage(phi, theta, background_img):
    """
    This module defines a function that samples color data
    from a background image based on the direction angles (phi, theta) of 
    an incoming light ray.
    
    It maps each ray direction to normalized texture coordinates (u, v),
    converts these to pixel indices in the image, and retrieves the
    corresponding RGB color values.

    Parameters:
    -----------
        phi : float
            Azimuthal angle (in radians) around the z-axis.
            Corresponds to left-right direction in the image.

        theta : float
            Polar angle (in radians) from the z-axis.
            Corresponds to up-down direction in the image.

        background_img : 3D float32 array
            The RGB background texture of shape (height, width, 3),
            with each channel normalized to [0,1].

    Returns:
    --------
        np.ndarray, shape (3,)
            Array containing the RGB values sampled
            from the corresponding pixel of the image.
    
    Description:
    ------------
        The function performs the following steps:
          1. Convert the spherical coordinates into
             normalized texture coordinates.
          2. Map (u, v) onto pixel indices (x, y) in the image.
          3. Read and return the RGB values from that pixel.

        This acts as a simple environment map lookup, effectively
        wrapping the background image onto a sphere surrounding
        the camera for rays that never fall into the black hole.
    """
    # Reads the pic's height and width (i.e., how many pixels it has)
    img_h, img_w, _ = background_img.shape
    
    # Normalize spherical coords into [0,1] converts direction into two 
    # numbers between 0 and 1 so we know where on the picture to look.
    u = (phi % (2*math.pi)) / (2*math.pi)
    v = (theta % math.pi) / math.pi
    
    # Convert to pixel indices
    x = int(u * (img_w - 1))
    y = int(v * (img_h - 1))
    
    # Return the color found at that little square in the picture 
    # (note: images index rows first, that's why y then x)
    return background_img[y, x] 