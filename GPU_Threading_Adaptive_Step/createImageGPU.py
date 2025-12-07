from numba import cuda
import math
import numpy as np
from backgroundColor import backgroundColorDevice
from backgroundImage import backgroundImage
import PIL.Image as Image

def createImageGPU(screenWidth, screenHeight, phi_out, r_out, alphaVals, M):
    """
    This module generates the final rendered image of the scene on the GPU. 
    It maps each ray's integration result to a pixel color based on whether
    the ray escaped or fell into the black hole. Rays that escape can sample 
    a background image or a checkered pattern based on their lensed direction.

    Parameters:
    -----------
        screenWidth, screenHeight : int
            Dimensions of the output image in pixels.
        
        phi_out, r_out : np.ndarray
            Arrays containing the final phi and r values for each ray.
        
        alphaVals : np.ndarray
            Azimuthal angles (alpha) for each ray, used for lensed direction.
        
        M : float
            Black hole mass.

    Returns:
        image : np.ndarray
            RGB image array of shape (screenHeight, screenWidth, 3)
            with values in [0,1].
    """

    # Set up CUDA grid dimensions
    threadsperblock = (16, 16)  # 16x16 threads per block
    blockspergrid_x = (screenWidth + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (screenHeight + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Copy data to device
    phi_out_gpu = cuda.to_device(phi_out.astype(np.float32))
    r_out_gpu = cuda.to_device(r_out.astype(np.float32))
    alphaVals_gpu = cuda.to_device(alphaVals.astype(np.float32))
    
    # Open a background image and normalize RGB values to [0, 1] and copy to GPU
    host_image = np.array(Image.open("starImage.jpg").convert("RGB"), dtype=np.float32) / 255.0
    backgroundImage_gpu = cuda.to_device(host_image) 

    # Allocate output image on device
    image_gpu = cuda.device_array((screenHeight, screenWidth, 3), dtype=np.float32)

    # Launch kernel
    createImageKernal[blockspergrid, threadsperblock](
        screenWidth, screenHeight, phi_out_gpu, r_out_gpu, alphaVals_gpu, M, image_gpu, backgroundImage_gpu
    )
    
    # Wait for all threads to finish
    cuda.synchronize()
    
    # Copy image back to host and return it
    return image_gpu.copy_to_host()

@cuda.jit(fastmath=True)
def createImageKernal(screenWidth, screenHeight, phi_out, r_out, alphaVals, M, image, background_img):
    """
    CUDA kernel to map each ray's final position to a pixel color.
    Rays that escape sample a background pattern; rays that fall
    into the black hole are black.
    """
    
    # Compute 2D thread indices
    x, y = cuda.grid(2)
    
    # Check the bounds
    if x < screenWidth and y < screenHeight:
        idx = y * screenWidth + x  # Linear index for flattened arrays

        # Retrieve final r and phi for this ray
        final_r = r_out[idx]
        final_phi = phi_out[idx]
        alpha = alphaVals[idx]
        
        if final_r > 2.0 * M:  # Ray escaped
            # Convert final spherical cooridinates to 3D Cartesian
            lensed_x = final_r * math.sin(final_phi) * math.cos(alpha)
            lensed_y = final_r * math.sin(final_phi) * math.sin(alpha)
            lensed_z = final_r * math.cos(final_phi)
            
            # Convert back to spherical to get angles for background sampling
            lensed_phi = math.atan2(lensed_y, lensed_x)
            lensed_theta = math.acos(lensed_z/final_r)
            
            # UNCOMMENT TO USE BACKGROUND IMAGE
            # Pick a color from the image based on the lensed direction
            # r_col, g_col, b_col = backgroundImage(lensed_phi, lensed_theta, background_img)
            
            # COMMENT OUT AND UNCOMMENT ABOVE TO USE BACKGROUND IMAGE INSTEAD OF CHECKERED PATTERN
            # Pick a color from the checkered background based on the lensed direction
            r_col, g_col, b_col = backgroundColorDevice(lensed_phi, lensed_theta, 10)
        else:  # Ray fell into black hole
            r_col, g_col, b_col = 0.0, 0.0, 0.0
        
        # Store RBG values into output image
        image[y, x, 0] = r_col
        image[y, x, 1] = g_col
        image[y, x, 2] = b_col
