import numpy as np
from numba import njit

@njit
def backgroundColor(phi, theta, scale=10):
    """
    defines a CUDA device function for GPU execution
    that computes the background color of a pixel based on the
    direction (phi, theta) of the corresponding light ray.

    The color pattern is a blue-and-white checkered texture, providing 
    a simple visual background to reflect the gravitational lensing effects.

    Parameters:
    -----------
        phi : float
            Azimuthal angle in spherical coordinates (radians).
            Defines the horizontal direction around the black hole.

        theta : float
            Polar angle in spherical coordinates (radians).
            Defines the vertical direction from the “north pole”.

        scale : float
            Controls the density of the checker pattern.
            Larger values produce smaller, more frequent squares.

    Returns:
    --------
        color : 1D np.ndarray of floats
            RGB color for the given direction. Blue ([0.2, 0.4, 1.0])
            if the checker index sum is even, otherwise white ([1.0, 1.0, 1.0]).

    Description:
    ------------
        1. Converts the input spherical angles (phi, theta) into
           normalized texture coordinates (u, v) in [0,1].
           * u corresponds to longitude
           * v corresponds to latitude
           
        2. Multiplies (u, v) by 'scale' to define the check size.
        
        3. Uses integer parts of the scaled (u, v) to determine which
           "square" of the checkerboard the point falls into.

        4. Alternates between two colors (blue and white) depending
           on whether the sum of the square indices is even or odd.
    """
    # Converts the input spherical angles (phi, theta) into 
    # normalized texture coordinates (u, v) in [0,1]
    u = (phi % (2 * np.pi)) / (2 * np.pi)
    v = (theta % np.pi) / np.pi

    # Scale (u,v) and take integer parts to find row/column indices.
    check_u = int(u * scale)
    check_v = int(v * scale)

    # If the square is even (sum of row + col is even) return blue
    # otherwise, return white
    if (check_u + check_v) % 2 == 0:
        return np.array([0.2, 0.4, 1.0])  # blue
    else:
        return np.array([1.0, 1.0, 1.0])  # white