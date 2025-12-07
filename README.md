________________________________________________________________

**GPU-Based Null Geodesic Integration in Schwarzschild Spacetime**
________________________________________________________________

This project implements a GPU-accelerated solver for tracing null geodesics (light paths) in Schwarzschild spacetime.
The integration is performed using an adaptive Runge–Kutta 4 (RK4) method, where each GPU thread computes the trajectory of a single photon ray characterized by its impact parameter.

________________________________________________________________

**Overview**
________________________________________________________________

The code simulates how photons travel near a black hole by utilizing the symmetric nature of Schwarzschild black holes to map their paths using the geodesic equation.
The program runs efficiently on the GPU by leveraging Numba's CUDA JIT compilation, allowing for large-scale parallel computation of photon trajectories.

________________________________________________________________

**File Structure**
________________________________________________________________

All code is found on the website in the "May_The_Geodesics_Be_With_You" folder.
There are four different versions of the code as mentioned within the thesis contained in this folder, namely:
*  The CPU multithreading fixed-step approach.
*  The CPU multithreading adaptive-step approach.
*  The GPU parallelized fixed-step approach.
*  The GPU parallelized adaptive-step approach.

Chapter 5 of the thesis examines the differences between the fixed-step and adaptive-step approaches.
________________________________________________________________

**Running the Code**
________________________________________________________________

Ensure that you have the following installed on your device:
*  Python 3.10+
*  CUDA Toolkit 12.8+
*  Numba, Numpy, matplotlib, Pillow
*  An NVIDIA driver that is CUDA-enabled

The simulation can be run for either of the code versions by calling running:

*  python main.py

This initializes the black hole parameters, defines photon impact parameters, and integrates each trajectory using the GPU RK4 kernel.

________________________________________________________________

**Output**
________________________________________________________________

The output consists of raytraced black hole images, showing either:
*  A distorted background grid, illustrating gravitational lensing geometry, or
*  A starfield, showing light deflection and photon capture around the event horizon.

The background type can be changed by changing two lines in the createImageGPU.py file (for the adaptive GPU approach). These two lines are located at lines 96 and 100.

*  To view a starfield distorted image, uncomment line 96 and comment out line 100.
*  To view a checkered grid distorted image, comment out line 96 and uncomment line 100.

Similar adjustments are made for all versions of the code. Line numbers might differ, so observe the comments left in the respective createImage files.

________________________________________________________________

**Notes**
________________________________________________________________

*  Integration stops if a photon escapes the system (r > 15M) or falls into the black hole (r < 1.99M).
*  The adaptive step size improves performance while maintaining accuracy when compared to fixed-step integration.


