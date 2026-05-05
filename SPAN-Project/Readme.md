# SPAN: Spatial Pyramid Attention Network for Image Manipulation Localization

Image manipulation detection has become increasingly important with the rise of digitally altered media, where identifying tampered regions is critical for ensuring authenticity and trust. This project explores the **Spatial Pyramid Attention Network (SPAN)** architecture for localizing manipulated regions in images.

### Overview

The primary focus of this work is to understand the SPAN architecture and its components, followed by a partial implementation and experimentation.

* Studied the SPAN architecture, including spatial pyramid attention mechanisms for multi-scale feature extraction
* Implemented a selected subset of the experiments
* Compared performance with ManTra-Net

### Experiments

Experiments were conducted to evaluate and compare models on standard image manipulation datasets:

* **CASIA V1 Dataset** – contains tampered images with splicing and copy-move manipulations
* **Columbia Dataset** – focuses on splicing-based forgeries
* **Coverage Dataset** – designed for copy-move manipulation detection

The results provide a comparative understanding  and robustness of SPAN against existing methods in detecting manipulated regions.

### Resources

* SPAN Paper: [https://arxiv.org/abs/2009.00726](https://arxiv.org/abs/2009.00726)
* SPAN Repository: [https://github.com/ZhiHanZ/IRIS0-SPAN](https://github.com/ZhiHanZ/IRIS0-SPAN)
* ManTra-Net: [https://github.com/ISICV/ManTraNet](https://github.com/ISICV/ManTraNet)

### Datasets

* Columbia Dataset: [http://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm](http://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm)
* Coverage: [https://github.com/wenbihan/coverage](https://github.com/wenbihan/coverage)
* CASIA: [http://forensics.idealtest.org/casiav2/](http://forensics.idealtest.org/casiav2/)

