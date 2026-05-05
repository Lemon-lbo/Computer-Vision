# Computer Vision – Assignment 1

This repository contains implementations of key image processing techniques, including histogram analysis, intensity transformations, bit-plane slicing, and channel alignment.

### 1. Histogram Analysis and Matching

* Extracted RGB channels from two images and plotted their histograms.
* Performed histogram matching across corresponding channels using `cv2.equalizeHist()`.
* Reconstructed the color image and compared results with the originals.

### 2. Intensity Slicing

* Applied intensity slicing on a grayscale image:

  * Highlighted selected intensity ranges with background suppression.
  * Highlighted selected ranges while preserving the remaining intensities.
* Plotted the corresponding intensity transformation functions and compared outputs.

### 3. Bit-Plane Slicing

* Decomposed a grayscale image into 8 bit planes (bitplane0 to bitplane7).
* Visualized selected bit-plane transformations.
* Reconstructed the image using reduced bit planes and analyzed quality degradation.

### 4. Channel Alignment (Prokudin-Gorskii Collection)

* Extracted RGB channels from *windmills.jpg*.
* Aligned green and red channels to the blue channel using SSD over a displacement window.
* Computed optimal displacement vectors and reconstructed the final color image.


If you want, I can make this sound **more “GitHub polished” (with badges, structure, or recruiter-friendly tone)** or **more technical (with equations and metrics)**.
