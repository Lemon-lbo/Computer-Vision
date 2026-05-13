# Spatial and Frequency-Domain Image Processing: Filtering, Edges, and Segmentation

This repository contains code and reports for **Computer Vision – Assignment 3**. The focus is on understanding how operations in the spatial domain relate to the frequency domain, and how different edge detection and segmentation methods behave on varied images.

## Contents

1. **Fourier analysis of synthetic rectangles**
   - Generate a 100×100 binary image with a filled white rectangle at the centre (size is parameterised).
   - Compute and display **magnitude**, **phase**, and **power** spectra with DC shifted to the centre.
   - Experiment with different rectangle sizes to study how object size and shape affect the Fourier domain.

2. **Convolution: spatial vs frequency domain (cameraman.jpg)**
   - (a) Apply an **11×11 average filter** directly in the spatial domain.
   - (b) Implement the same filtering via the **frequency domain**:
     - pad image and kernel, take DFTs, multiply in frequency, inverse DFT, and crop.
   - (c) Compare **runtime** of the two implementations and comment on when frequency-domain convolution becomes advantageous.

3. **Edge detection on structured vs curved scenes**
   - Apply, after Gaussian smoothing and appropriate thresholding:
     - (a) **Sobel** gradient + threshold
     - (b) **LoG** with zero-crossing detection
     - (c) **Canny** edge detector
     - (d) **Hough transform** for line detection
   - Run on **building.jpg** (mostly straight edges) and **objects.png** (curved/irregular shapes).
   - Compare edge maps and explain how scene geometry (straight vs curved edges) affects each method’s performance.

4. **Segmentation methods on multiple image categories**
   - Images grouped into three folders: **pizza**, **road**, and **furniture**.
   - Apply:
     - (a) **K-means clustering** in RGB (with k chosen by a simple elbow heuristic, typically k = 3).
     - (b) **SLIC superpixels** (visualised via mean-colour patches and/or boundary overlays).
     - (c) **GrabCut** with a central rectangle prior.
     - (d) **Watershed** segmentation with marker-based initialisation.
   - For each image, discuss which method works best and relate this to scene structure (compact foreground objects vs extended scenes).
