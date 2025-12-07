# main.py
import numpy as np
from PIL import Image
import time
from cameraSetup import cameraSetup
from impactParameters import impactParameters
from createImageGPU import createImageGPU
from runIntegrationGPU import runGPUIntegration   

# image: NumPy array of shape (height, width, 3) with values in [0,1]
def saveImage(image, filename="blackHole.png"):
    '''
    Convert a floating-point RGB image (values in [0, 1]) to 8-bit format,
    save it as a PNG, and display it.

    Parameters
    ----------
    image : numpy.ndarray
        Image array of shape (height, width, 3) with float values in [0, 1].
    filename : str, optional
        Output filename (default is "blackHole.png").

    Notes
    -----
    * Converts to 8-bit color depth for saving.
    * Uses the Pillow (PIL) library to handle file output and display.
    '''
    # Convert floar image to unit8 (0 - 255 range) for saving
    img_uint8 = (image * 255).astype(np.uint8) 
    # Create a PIL image object from the NumPy array and save/display
    pil_img = Image.fromarray(img_uint8)
    pil_img.save(filename)
    pil_img.show() # Commented out to prevent automatic display during batch runs

def main():
    '''
    Main control routine for the Schwarzschild raytracing simulation.

    This script manages the end-to-end process of generating an image
    of a gravitationally lensed background as viewed near a Schwarzschild
    black hole. It performs camera setup, computes ray directions and
    impact parameters, runs GPU-based geodesic integrations, and assembles
    the resulting pixel data into a final image.

    Workflow
    --------
    1. Initialize simulation parameters (camera radius, FOV, mass).
    2. Set up the virtual camera orientation and screen geometry.
    3. Compute impact parameters and azimuthal angles for each pixel.
    4. Integrate null geodesics in parallel on the GPU.
    5. Construct the final rendered image and save it to disk.

    Notes
    -----
    * The computation is GPU-accelerated for the geodesic integration step.
    * The field of view (FOV) is specified in radians.
    * The observer is placed at a fixed radius 'r0' from the black hole.
    '''
    totalTime = 1.0            # Adds up the total simulation time over multiple iterations
    iters = 1                  # Number of iterations to average timing
    for _ in range(iters):
        start = time.perf_counter()     # Start timing for this iteration
        
        # Parameters for simulation
        r0 = 10.0                       # Initial radial position of the camera
        fov = np.radians(90.0)          # Field of view in radians
        M = 0.0                         # Black hole mass (geometric units)
        
        # Set up camera and screen
        screenWidth, screenHeight, \
            screen_x, screen_y, numRays, \
            camDir, camUp, camRight = cameraSetup(r0)

        # Calculate the imapct parameters and ray directions for each pixel
        print("Calculating impact parameters...")
        b_vals, alphaVals = impactParameters(r0, screen_x, screen_y, \
            screenWidth, screenHeight, \
            camDir, camUp, camRight, fov, numRays, M
        )
        
        # Integrate the geodesics on the GPU
        print("Running ray integrations...")
        phi_out, r_out = runGPUIntegration(numRays, M, b_vals, r0)
        
        # Create final image from the integration results
        print("Creating final image...")
        image = createImageGPU(screenWidth, screenHeight, phi_out, r_out
                            , alphaVals, M)
        
        # Save the final image
        saveImage(image, "hubble1Flat.png")
        print("Image saved as finalImage.png")
        
        end = time.perf_counter()           # End timing for this iteration
        totalTime += (end - start)          # Add to total time
    
    # Display the average timing results
    avgTime = totalTime / iters
    print(f"Average integration time over {iters} iterations: {avgTime:.4f} seconds")


if __name__ == "__main__":
    main()
