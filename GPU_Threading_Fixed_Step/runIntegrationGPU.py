import numpy as np
from RK4_parallel import rk4_kernel
from numba import cuda

def runGPUIntegration(numRays, M, b_vals, r0):
    """
    Handles GPU-accelerated ray integration using the RK4 method. Each thread
    integrates a single ray from the camera until it either escapes or hits the
    black hole. Outputs are the final phi and radial distance for each ray.

    Parameters:
    -----------
        numRays : int
            Total number of rays to integrate (screenWidth x screenHeight).

        M : float
            Black hole mass.

        b_vals : np.ndarray
            Array of impact parameters for each ray.

        r0 : float
            Initial radial position of the camera.

    Returns:
    --------
        phi_out : np.ndarray
            Final azimuthal angles (phi) for each ray.

        r_out : np.ndarray
            Final radial distances for each ray.
    """
    
    # Integration parameters
    phi_range = (0.0, 2.0 * np.pi)  # angular range to integrate over
    h = 0.0001                      # initial step size
    max_steps = 100000              # maximum number of integration steps allowed
    
    # Ensure input arrays are contiguous and of type float32
    b_vals = np.ascontiguousarray(b_vals, dtype=np.float32)
        
    # Allocate GPU memory
    phi_out_d = cuda.device_array(numRays, dtype=np.float32)
    r_out_d = cuda.device_array(numRays, dtype=np.float32)
    
    # Copy data to GPU
    d_b_vals = cuda.to_device(b_vals)

    # Launch kernel
    threadsPerBlock = 256
    blocks = (numRays + threadsPerBlock - 1) // threadsPerBlock
    
    print(f"Launching kernel with {blocks} blocks, {threadsPerBlock} threads per block") 
    
    rk4_kernel[blocks, threadsPerBlock](
        np.float32(M),                  # Black hole mass
        np.float32(r0),                 # Initial radii
        np.float32(phi_range[0]),       # Start angle
        np.float32(phi_range[1]),       # End angle
        np.float32(h),                  # Step size
        d_b_vals,                       # Impact parameters
        phi_out_d,                      # Output: phi values
        r_out_d,                        # Output: radial distances
        np.float32(max_steps)           # Maximum steps allowed
    )
    
    # Wait for all threads to finish
    cuda.synchronize()
    
    # Copy results from GPU back to CPU memory
    phi_out = phi_out_d.copy_to_host()  # Phi values for all rays
    r_out = r_out_d.copy_to_host()      # Radial distances for all rays
        
    return phi_out, r_out