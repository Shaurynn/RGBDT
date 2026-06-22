# RGBDT
A computer vision POC evaluating the increased semantic segmentation accuracy of tri-modal RGBDT sensor fusion over standard RGB, RGBD, and RGBT baselines using the MM5 dataset.

# Multimodal RGB-D-T Sensor Fusion: Structural Inspection POC

## Overview
This repository contains a Proof of Concept (POC) evaluating a 5-channel multi-modal AI pipeline `(5, 480, 640)`. The primary objective is to prove the viability and quantify the increased accuracy of utilizing tri-modal RGB-D-T (Vision, Depth, and Thermal) data over traditional single-modal (RGB) or bi-modal (RGB-D, RGB-T) approaches.
 
The architecture targets pixel-accurate semantic segmentation for structural anomaly detection—specifically evaluating environmental wear and tear on rapid transit infrastructure. The pipeline is structured to validate the software architecture prior to deployment on autonomous heavy-lift UAV platforms utilizing companion edge computers (e.g., Jetson Orin Nano / Pixhawk flight controllers).

## POC Methodology & Experimental Design
To empirically validate the performance gains of the tri-modal architecture without incurring the cost of initial physical data collection, this POC leverages the academic **MM5 Dataset**. 

The training loop and evaluation scripts are structured to benchmark four distinct network configurations against each other:
1. **Baseline 1 (Single-Modal):** RGB only `(3, 480, 640)`
2. **Baseline 2 (Bi-Modal):** RGB + Depth `(4, 480, 640)`
3. **Baseline 3 (Bi-Modal):** RGB + Thermal (LWIR) `(4, 480, 640)`
4. **Primary POC (Tri-Modal):** RGB + Depth + Thermal `(5, 480, 640)`

By comparing metric scores across these configurations, this repository serves as the empirical justification for developing a custom, physical sensor fusion payload.

### Dataset Engineering
* **Native LWIR:** The MM5 dataset provides raw 16-bit Thermal (LWIR) data, allowing the network to learn true emissive heat signatures natively.
* **Native Segmentation:** The dataset provides pixel-wise annotations, which are directly ingested by the PyTorch `DataLoader` to train the segmentation backbone.
* **Modality Filtering:** While MM5 provides 5 distinct sensor cues, the `DataLoader` explicitly extracts only the RGB, Depth, and Thermal directories. 
 
## Environment Setup
This pipeline requires a dedicated Python 3.12 virtual environment optimized for CUDA 13.0.
 
1. Clone the repository.
2. Install the core PyTorch binaries:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
