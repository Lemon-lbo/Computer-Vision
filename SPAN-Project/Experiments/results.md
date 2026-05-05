# SPAN + ManTraNet Evaluation Results — Columbia Uncompressed Dataset

**Models:** SPAN (`PixelAttention32.h5`) and ManTraNet (`ManTraNet_Ptrain4.h5`) — pretrained only  
**Dataset:** Columbia Uncompressed Image Splicing Detection Dataset (180 spliced)  
**Reference:** Hu et al., ECCV 2020 — Table 5 (Robustness Analysis)

---

## Results Table

| Perturbation | SPAN Pixel AUC | ManTraNet Pixel AUC | SPAN F1 (%) | ManTraNet F1 (%) |
|---|---|---|---|---|
| No manipulation | 93.78 | 76.65 | 70.64 | 48.22 |
| Resize (0.78x) | 90.13 | 63.86 | 59.94 | 40.56 |
| Resize (0.25x) | 63.63 | 62.18 | 42.67 | 42.03 |
| No manipulation | 93.78 | 76.65 | 70.64 | 48.22 |
| GaussianBlur (kernal size=3) | 71.59 | 62.83 | 42.67 | 41.68 |
| GaussianBlur (kernal size=15) | 64.64 | 64.09 | 42.69 | 42.88 |
| No manipulation | 93.78 | 76.65 | 70.64 | 48.22 |
| GaussianNoise (sigma=3) | 72.00 | 58.95 | 42.85 | 42.44 |
| GaussianNoise (sigma=15) | 68.28 | 51.88 | 42.02 | 41.99 |
| No manipulation | 93.78 | 76.65 | 70.64 | 48.22 |
| JPEGCompress (quality=100) | 92.47 | 73.70 | 69.76 | 41.75 |
| JPEGCompress (quality=50) | 74.06 | 57.69 | 42.70 | 42.27 |

---

## Paper Reference Values (Table 5 — Columbia Pixel AUC)

| Perturbation | SPAN (paper) | ManTraNet (paper) |
|---|---|---|
| No manipulation | 93.6 | 77.95 |
| Resize (0.78x) | 89.99 | 71.66 |
| Resize (0.25x) | 69.08 | 68.64 |
| GaussianBlur (kernal size=3) | 78.97 | 67.72 |
| GaussianBlur (kernal size=15) | 67.70 | 62.88 |
| GaussianNoise (sigma=3) | 75.11 | 68.22 |
| GaussianNoise (sigma=15) | 65.80 | 54.97 |
| JPEGCompress (quality=100) | 93.32 | 75.00 |
| JPEGCompress (quality=50) | 74.62 | 59.37 |

---

## Notes

- Pixel AUC is the pixel-level ROC-AUC averaged over all spliced images (paper's primary metric)
- F1 is at 0.5 threshold, best polarity, averaged (not reported in Table 5 but included for completeness)
- Paper (Table 2) reports Columbia AUC = 93.6, F1 = 81.5 for SPAN pretrained; ManTraNet (GitHub model) = 77.95
- ManTraNet note: the released `ManTraNet_Ptrain4.h5` shares an identical feature extractor architecture to SPAN (16-filter b1c1)

---

# SPAN + ManTraNet Evaluation Results — COVERAGE Dataset

**Models:** SPAN (`PixelAttention32.h5`) and ManTraNet (`ManTraNet_Ptrain4.h5`) — pretrained only  
**Dataset:** COVERAGE Copy-Move Forgery Database (100 forged images)  
**Reference:** Hu et al., ECCV 2020 — Table 5 (Robustness Analysis, same perturbations)

---

## Results Table

| Perturbation | SPAN Pixel AUC | ManTraNet Pixel AUC | SPAN F1 (%) | ManTraNet F1 (%) |
|---|---|---|---|---|
| No manipulation | 91.37 | 71.61 | 44.95 | 24.31 |
| Resize (0.78x) | 74.05 | 65.16 | 21.85 | 21.71 |
| Resize (0.25x) | 63.13 | 62.61 | 19.57 | 20.60 |
| GaussianBlur (kernel size=3) | 65.06 | 64.45 | 19.74 | 22.38 |
| GaussianBlur (kernel size=15) | 63.36 | 61.67 | 19.57 | 22.19 |
| GaussianNoise (sigma=3) | 69.03 | 60.19 | 19.99 | 20.10 |
| GaussianNoise (sigma=15) | 60.26 | 53.59 | 20.52 | 19.41 |
| JPEGCompress (quality=100) | 87.63 | 66.14 | 35.35 | 21.74 |
| JPEGCompress (quality=50) | 65.99 | 59.49 | 19.57 | 19.99 |

---

# SPAN + ManTraNet Evaluation Results — CASIA1 Dataset

**Models:** SPAN (`PixelAttention32.h5`) and ManTraNet (`ManTraNet_Ptrain4.h5`) — pretrained only  
**Dataset:** CASIA1 Image Tampering Detection Evaluation Database (921 forged + 800 authentic)  
**Reference:** Hu et al., ECCV 2020 — Table 2 (Image-level detection)

> **Note:** CASIA1 does not include pixel-level ground truth masks.  
> Evaluation is therefore at **image level**: each image's forgery score = max predicted pixel value.  
> Metrics are image-level AUC and F1 (forged vs. authentic images), not pixel-level.

---

## Results Table

| Perturbation | SPAN Image AUC | ManTraNet Image AUC | SPAN F1 (%) | ManTraNet F1 (%) |
|---|---|---|---|---|
| No manipulation | 73.11 | 47.96 | 63.14 | 69.72 |
| Resize (0.78x) | 59.99 | 45.26 | 35.15 | 69.72 |
| Resize (0.25x) | 48.67 | 45.79 | 36.86 | 69.72 |
| GaussianBlur (kernel size=3) | 53.79 | 46.62 | 10.50 | 69.72 |
| GaussianBlur (kernel size=15) | 49.18 | 47.13 | 6.81 | 69.72 |
| GaussianNoise (sigma=3) | 48.72 | 49.44 | 67.21 | 69.72 |
| GaussianNoise (sigma=15) | 51.998 | 57.35 | 69.65 | 69.72 |
| JPEGCompress (quality=100) | 72.54 | 48.03 | 59.54 | 69.72 |
| JPEGCompress (quality=50) | 60.46 | 41.76 | 44.27 | 69.67 |

---

## Notes

- Image AUC: ROC AUC using image-level forgery scores (max pixel prediction) over all 1721 images
- F1 is at 0.5 score threshold
- Paper (Table 2) reports CASIA1 image-level AUC: SPAN=72.0, ManTraNet=79.5 (pretrained)
