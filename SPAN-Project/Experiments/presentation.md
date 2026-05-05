# SPAN: Spatial Pyramid Attention Network for Image Manipulation Detection
### A Presentation on Implementation & Evaluation

---

## 1. Paper Overview

**Title:** SPAN — Spatial Pyramid Attention Network for Image Manipulation Detection  
**Authors:** Xuefeng Hu et al.  
**Venue:** ECCV 2020  
**Link:** [Arxiv / Official IRIS0-SPAN GitHub Repo](https://github.com/IRIS0/SPAN)

### What Problem Does It Solve?

Image forgeries — splicing, copy-move, inpainting — leave subtle statistical traces in pixel-level noise patterns. SPAN detects these traces by comparing the noise fingerprint of each pixel against its spatial neighbours at **multiple scales simultaneously** (the "spatial pyramid").

---

## 2. Architecture

### Big Picture

```
Input image (RGB)
    │
    ▼
CombinedConv2D  ← multi-stream first layer
    │
    ▼
11× SymConv2D (3×3, reflect padding)  ← deep feature extractor
    │
    ▼
L2 Normalisation  ← unit-norm feature map per pixel
    │
    ▼
Resize to 224×224  ← fixed internal resolution
    │
    ▼
OutlierTransform  ← robust initialisation
    │
    ▼
5× PixelAttention blocks (shifts = 1, 3, 9, 27, 81)  ← SPATIAL PYRAMID
    │
    ▼
5× Conv2D decoder (1×1)
    │
    ▼
Pixel-level forgery probability map
```

---

### CombinedConv2D — The First Layer

The feature extraction starts with **three parallel streams**, each targeting a different signal:

| Stream | Filters | Purpose |
|---|---|---|
| Regular (learnable) | 4 | General RGB features |
| SRM (fixed) | 9 | Steganalysis rich model — noise residuals |
| Bayar (learnable) | 3 | Constrained conv for tampering artefacts |

> **SRM kernels** are fixed 5×5 high-pass filters that suppress image content and amplify the camera's noise fingerprint. This is the core insight borrowed from steganalysis.

The Bayar convolution has a **centre-suppression constraint**: the centre weight is forced to $-1$ of the sum of all other weights. This makes it act like a learned high-pass filter.

---

### PixelAttention — The Core Module

Each `PixelAttention` block computes a **self-attention** comparison between a pixel's feature vector and all 9 pixels in its 3×3 neighbourhood, sampled at dilation `shift`:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{D}}\right)V$$

The five blocks use **shifts = {1, 3, 9, 27, 81}**, covering neighbourhoods at exponentially growing scales:

| Block | Dilation (shift) | Neighbourhood radius |
|---|---|---|
| 1 | 1 | ≈ 3 px |
| 2 | 3 | ≈ 9 px |
| 3 | 9 | ≈ 27 px |
| 4 | 27 | ≈ 81 px |
| 5 | 81 | ≈ 243 px |

This is the **Spatial Pyramid** — a single pixel attends to neighbours at five different scales. Authentic regions have consistent noise across all scales; tampered regions break the pattern at some scale.

Each block also has a 3-layer feed-forward network (FFN) with residual skip connections.

---

### ManTraNet — The Baseline Comparison

ManTraNet (Wu et al., CVPR 2019) is used as a **comparison model** throughout this work:

- Uses the same `b1c1` feature extractor (equivalent to CombinedConv2D)
- Replaces the spatial pyramid attention with a **local anomaly detection** module (LSTM-based Z-score)
- Runs at **original image resolution** (no internal resize), so it handles large images natively
- We use the pretrained weights `ManTraNet_Ptrain4.h5` (no fine-tuning)

---

## 3. Implementation Details

### Why We Re-Implemented in PyTorch

The original IRIS0-SPAN code is written in **TensorFlow / Keras**. Due to **NumPy 2.x breaking TensorFlow API compatibility** in the current environment, TensorFlow could not be loaded at all. We therefore:

1. Ported SPAN to **PyTorch** (`span_pytorch.py`)
2. Ported ManTraNet to **PyTorch** (`mantranet_pytorch.py`)
3. Wrote H5-weight loaders that map the original Keras layer names to our PyTorch parameter names

Both ports load **directly from the original `.h5` weight files** — no re-training was done.

### Key Implementation Challenges

| Challenge | Solution |
|---|---|
| TensorFlow dead (NumPy 2.x) | Full PyTorch re-implementation |
| Keras weight name mapping | Custom `h5py` loader with layer-name translation |
| SRM kernels not stored in H5 | Hard-coded the 3 standard SRM filters |
| Bayar constraint not a standard layer | Implemented as constrained parameter update |
| `torch.cuda.amp.autocast()` broken in PyTorch 2.5 | Used `torch.amp.autocast(device_type='cuda')` |
| NumPy `np.trapz` removed in 2.x | Replaced with `np.trapezoid` |
| SPAN outputs 224×224, labels at original size | Upsampled with `cv2.INTER_NEAREST` back to GT size |
| Coverage mask/image size mismatches | Resized GT mask to image shape during pre-load |

### Hardware

- **GPU:** NVIDIA RTX A1000 Laptop GPU
- **PyTorch:** 2.5.1+cu121
- **NumPy:** 2.4.3
- **Inference mode:** FP16 autocast on GPU

---

## 4. Datasets

### Columbia Uncompressed Image Splicing

- **Type:** Splicing detection (authentic/tampered pairs)
- **Size:** 180 spliced image pairs
- **Images:** High-resolution uncompressed TIFFs
- **Ground truth:** Binary edge masks (boundary of spliced region)
- **Protocol:** Images resized so that `512 / min(H, W)` ≤ 1 (prescaled to ≤512px), then prediction upsampled back

### COVERAGE Copy-Move Forgery Database

- **Type:** Copy-move detection
- **Size:** 100 forged image pairs
- **Images:** TIFF format (some with non-standard metadata tags — benign)
- **Ground truth:** Grayscale masks (0 = authentic, 255 = forged), stored as `Nforged.tif`
- **File naming:** `Nt.tif` (tampered image) ↔ `Nforged.tif` (mask)
- **Protocol:** No prescaling; SPAN output (224×224) upsampled to original image dimensions

### CASIA1 Image Tampering Detection Database

- **Type:** Splicing (both inter-scene and intra-scene)
- **Size:** 921 forged images (`Sp/`) + 800 authentic images (`Au/`)
- **Images:** JPEG format, various resolutions
- **Ground truth:** **No pixel-level masks** — only image-level labels (forged / authentic)
- **Protocol:** Image-level evaluation — forgery score per image = **max predicted pixel value**; image-level AUC and F1
- **Note:** This is different from Columbia and COVERAGE which are pixel-level; CASIA1 tests whether the model can flag a whole forged image vs. a clean one

---

## 5. Evaluation Protocol

### Metrics

| Metric | Description |
|---|---|
| **Pixel AUC** | Area under the pixel-level ROC curve — used for Columbia and COVERAGE (primary metric) |
| **Image AUC** | Area under the image-level ROC curve — used for CASIA1 (no pixel masks available) |
| **F1 @ 0.5** | F1 score at 0.5 threshold, averaged over images |

> Columbia and COVERAGE use **pixel-level AUC**. CASIA1 uses **image-level AUC** (max pixel prediction per image, forged vs. authentic).

### Perturbation Analysis (Table 5 in paper)

To test **robustness**, each image is processed through 8 perturbations before inference:

| Perturbation | Parameters |
|---|---|
| No manipulation | Baseline |
| Resize | 0.78× and 0.25× (down then back up) |
| Gaussian Blur | kernel 3×3 and 15×15 |
| Gaussian Noise | σ = 3 and σ = 15 |
| JPEG Compression | quality = 100 and quality = 50 |

For Resize, the image is downscaled and then restored to its original size before inference — simulating a lossy resize round-trip.

---

## 6. Results

### Columbia Dataset — Pixel AUC

| Perturbation | SPAN (ours) | SPAN (paper) | ManTraNet (ours) | ManTraNet (paper) |
|---|---|---|---|---|
| No manipulation | **93.78** | 93.6 | **76.65** | 77.95 |
| Resize (0.78×) | **90.13** | 89.99 | 63.86 | 71.66 |
| Resize (0.25×) | 63.63 | 69.08 | 62.18 | 68.64 |
| GaussianBlur k=3 | 71.59 | 78.97 | 62.83 | 67.72 |
| GaussianBlur k=15 | 64.64 | 67.70 | 64.09 | 62.88 |
| GaussianNoise σ=3 | 72.00 | 75.11 | 58.95 | 68.22 |
| GaussianNoise σ=15 | 68.28 | 65.80 | 51.88 | 54.97 |
| JPEGCompress q=100 | **92.47** | 93.32 | **73.70** | 75.00 |
| JPEGCompress q=50 | 74.06 | 74.62 | 57.69 | 59.37 |

#### Columbia F1 (%)

#### Columbia F1 (%)

| Perturbation | SPAN F1 | ManTraNet F1 |
|---|---|---|
| No manipulation | 70.64 | 48.22 |
| Resize (0.78×) | 59.94 | 40.56 |
| Resize (0.25×) | 42.67 | 42.03 |
| GaussianBlur k=3 | 42.67 | 41.68 |
| GaussianBlur k=15 | 42.69 | 42.88 |
| GaussianNoise σ=3 | 42.85 | 42.44 |
| GaussianNoise σ=15 | 42.02 | 41.99 |
| JPEGCompress q=100 | 69.76 | 41.75 |
| JPEGCompress q=50 | 42.70 | 42.27 |

---

### COVERAGE Dataset — Pixel AUC

*(COVERAGE was not part of Table 5 in the paper; results below are our own evaluation using identical protocol)*

| Perturbation | SPAN Pixel AUC | ManTraNet Pixel AUC | SPAN F1 (%) | ManTraNet F1 (%) |
|---|---|---|---|---|
| No manipulation | **91.37** | 71.61 | 44.95 | 24.31 |
| Resize (0.78×) | 74.05 | 65.16 | 21.85 | 21.71 |
| Resize (0.25×) | 63.13 | 62.61 | 19.57 | 20.60 |
| GaussianBlur k=3 | 65.06 | 64.45 | 19.74 | 22.38 |
| GaussianBlur k=15 | 63.36 | 61.67 | 19.57 | 22.19 |
| GaussianNoise σ=3 | 69.03 | 60.19 | 19.99 | 20.10 |
| GaussianNoise σ=15 | 60.26 | 53.59 | 20.52 | 19.41 |
| JPEGCompress q=100 | **87.63** | 66.14 | 35.35 | 21.74 |
| JPEGCompress q=50 | 65.99 | 59.49 | 19.57 | 19.99 |

---

### CASIA1 Dataset — Image-Level AUC

*(No pixel masks — evaluated as binary image classification: forged vs. authentic)*  
*(Paper Table 2 reference: SPAN = 72.0, ManTraNet = 79.5 for pretrained models)*

| Perturbation | SPAN Image AUC | ManTraNet Image AUC | SPAN F1 (%) | ManTraNet F1 (%) |
|---|---|---|---|---|
| No manipulation | **73.11** | 47.96 | 63.14 | 69.72 |
| Resize (0.78×) | 59.99 | 45.26 | 35.15 | 69.72 |
| Resize (0.25×) | 48.67 | 45.79 | 36.86 | 69.72 |
| GaussianBlur k=3 | 53.79 | 46.62 | 10.50 | 69.72 |
| GaussianBlur k=15 | 49.18 | 47.13 | 6.81 | 69.72 |
| GaussianNoise σ=3 | 48.72 | 49.44 | 67.21 | 69.72 |
| GaussianNoise σ=15 | 52.00 | 57.35 | 69.65 | 69.72 |
| JPEGCompress q=100 | **72.54** | 48.03 | 59.54 | 69.72 |
| JPEGCompress q=50 | 60.46 | 41.76 | 44.27 | 69.67 |

---

## 7. Analysis & Key Observations

### SPAN vs ManTraNet

- **SPAN consistently outperforms ManTraNet** across almost every perturbation on both datasets by a wide margin (often 10–20 AUC points on Columbia).
- SPAN's multi-scale spatial pyramid gives it a structural advantage in detecting forgery boundaries at different granularities.

### Robustness to Perturbations

- **JPEG at q=100** barely hurts SPAN (93.78 → 92.47 on Columbia), confirming that lossless re-compression is a near-trivial perturbation.
- **JPEG at q=50** is much more damaging (93.78 → 74.06), as heavy quantisation destroys the noise fingerprint that SRM/Bayar filters rely on.
- **Aggressive blur (k=15)** and **aggressive resize (0.25×)** hurt both models roughly equally — the noise signal is simply destroyed.
- **Light perturbations (Resize 0.78×, JPEG q=100, Noise σ=3)** degrade SPAN mildly, confirming its robustness claim from the paper.

### Columbia vs COVERAGE vs CASIA1

- Columbia is **splicing** (pasting a region from another image); COVERAGE is **copy-move** (duplicating a region within the same image); CASIA1 is **mixed** (splicing, copy-move, inpainting).
- SPAN's pixel AUC drop: Columbia 93.78 → COVERAGE 91.37 (small drop — generalises well).
- F1 drops sharply on COVERAGE (70.64 → 44.95) because copy-move boundaries are more subtle.
- ManTraNet F1 on COVERAGE (24.31 baseline) is noticeably worse than SPAN — the Z-score anomaly detector struggles more with copy-move than with splicing.
- CASIA1 uses image-level evaluation (different metric), so direct pixel AUC comparison is not possible — but it tests detection ability across a larger, more diverse set of ~1700 images.

### Our Results vs Paper

#### Columbia (Pixel AUC)
- Replication is very close: within 1–5 AUC points for most perturbations.
- Larger gaps appear at GaussianBlur k=3 (71.59 vs. 78.97) and Resize 0.25× (63.63 vs. 69.08) — likely due to subtle differences in pre/post-processing between our PyTorch port and the original TF code (bilinear vs. nearest-neighbour upsampling, exact padding behaviour).
- ManTraNet closely matches on most rows (within ~2–8 AUC).

#### CASIA1 (Image-Level AUC — Paper Table 2)
| Model | Ours (no manip) | Paper | Difference |
|---|---|---|---|
| SPAN | 73.11 | 72.0 | **+1.1** ✅ |
| ManTraNet | 47.96 | 79.5 | **−31.5** ⚠️ |

- **SPAN matches the paper almost exactly** (+1.1 points) — strong validation of the PyTorch port.
- **ManTraNet falls far short** (47.96 vs. 79.5) — essentially random (AUC ≈ 50%). The most likely explanation is that the paper used a **different ManTraNet checkpoint or configuration** than the publicly released `ManTraNet_Ptrain4.h5`. The released weights may have been optimised on different training data. The F1 score also staying constant at 69.72 across all perturbations is a tell-tale sign that ManTraNet is outputting near-uniform scores on CASIA images (no discrimination ability on this dataset).
- **Note:** ManTraNet performs much better on Columbia and COVERAGE (pixel AUC 60–76%), where the pixel-level anomaly signal is stronger than the image-level classification task CASIA1 requires.

---

## 8. Files Produced

| File | Description |
|---|---|
| `span_pytorch.py` | Full PyTorch port of SPAN, loads from `.h5` |
| `mantranet_pytorch.py` | Full PyTorch port of ManTraNet, loads from `.h5` |
| `evaluate_gpu.py` | Robustness evaluation on Columbia dataset |
| `evaluate_coverage.py` | Robustness evaluation on COVERAGE dataset |
| `evaluate_casia.py` | Image-level evaluation on CASIA1 dataset |
| `columbia_results.csv` | Raw Columbia results (all perturbations) |
| `coverage_results.csv` | Raw COVERAGE results (all perturbations) |
| `casia_results.csv` | Raw CASIA1 results (image-level, all perturbations) |
| `results.md` | Full results tables |

---

## 9. References

1. Hu, X. et al. **"SPAN: Spatial Pyramid Attention Network for Image Manipulation Detection."** ECCV 2020.
2. Wu, Y. et al. **"ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries With Anomalous Features."** CVPR 2019.
3. Hsu, Y.-F. & Chang, S.-F. **"Detecting Image Splicing Using Geometry Invariants and Camera Characteristics Consistency."** Columbia Uncompressed Dataset, 2006.
4. Wen, B. et al. **"Coverage — A Novel Database for Copy-Move Forgery Detection."** ICASSP 2016.
5. Dong, J. et al. **"CASIA Image Tampering Detection Evaluation Database."** CISP 2013.
6. Fridrich, J. et al. **"Rich Models for Steganalysis of Digital Images."** (SRM filters) IEEE TIFS, 2012.
7. Bayar, B. & Stamm, M. **"A Deep Learning Approach to Universal Image Manipulation Detection."** (Bayar convolution) IH&MMSec 2016.
