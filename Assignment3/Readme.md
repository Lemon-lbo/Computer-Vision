# Computer Vision – Assignment 3

The folder covers spatial and frequency-domain operations, edge detection, and segmentation.

## Tasks

1. **Fourier analysis**
   - Central white rectangle (size configurable) in a 100×100 binary image.
   - Show magnitude, phase, and power spectra (DC at centre); study effect of rectangle size.

2. **Convolution: spatial vs frequency**
   - On `cameraman.jpg`, apply an 11×11 average filter:
     - Directly in the spatial domain.
     - Via frequency domain (padding, DFT, multiply, inverse DFT, crop).
   - Compare runtimes of both implementations.

3. **Edge detection**
   - On `building.jpg` and `objects.png`, after Gaussian smoothing:
     - Sobel + threshold, LoG zero-crossing, Canny, and Hough transform.
   - Compare results for mostly straight edges vs curved/irregular scenes.

4. **Segmentation**
   - Images in `pizza`, `road`, `furniture`.
   - Apply K-means (k≈3), SLIC superpixels, GrabCut, and Watershed.
   - For each image, note which method works best and how this relates to scene structure.
