# 🌊 Positive-Unlabeled (PU) Learning for Marine Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Pipeline](https://github.com/user/pu-marine-detection/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

A modular, production-ready PyTorch implementation of **Positive-Unlabeled (PU) Semi-Supervised Learning for Underwater Marine Object Detection**.

---

## 🌟 Key Features

- **2-Stage Training Pipeline**:
  - **Stage 1 (Supervised Baseline)**: Detector trained on available partially labeled underwater images.
  - **Stage 2 (Teacher-Student PU Retraining)**: Semi-supervised learning with high-confidence pseudo-labeling and Exponential Moving Average (EMA) teacher updates.
- **Non-Negative PU Risk Estimator (`nnPU Loss`)**: Treats unlabeled image regions as unknown samples rather than negative background, eliminating false negative gradients.
- **Underwater Visual Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE), color jittering, dehazing, and underwater noise robust augmentations.
- **32 Marine Species Support**: Taxon-aware detection for species such as *pyrosome, larvacean, urchin, crab, soft coral, octopus, jellyfish, sea star, sea squirt, benthic worm*, and more.
- **Class-Imbalance Sampler**: Weighted sampling to boost detection accuracy on rare marine species with limited training samples.
- **Complete Evaluation Suite**: Automatic calculation of `mAP50`, `mAP50-95`, `Precision`, `Recall`, `F1-Score`, and per-class AP breakdown.
- **Interactive Gradio Web Application**: Upload underwater images, tune detection confidence thresholds, and visualize predicted bounding boxes in real-time.

---

## 🔬 Performance Benchmark Results

| Stage | Training Strategy | mAP50 | mAP50-95 | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Supervised Baseline | 0.4279 | 0.3246 | **0.6283** | 0.4565 | 0.5288 |
| **Stage 2** | **Teacher-Student PU (Ours)** | **0.5943** | **0.4366** | 0.5526 | **0.7047** | **0.6194** |
| **Delta** | **Improvement** | <font color="green">**+16.64%**</font> | <font color="green">**+11.20%**</font> | -7.57% | <font color="green">**+24.82%**</font> | <font color="green">**+9.06%**</font> |

### Top Performing Species (Stage 2 AP50)
- **Barnacle**: 0.9950
- **Pyrosome**: 0.9950
- **Sea Pen**: 0.9950
- **Sea Slug**: 0.9950
- **Octopus**: 0.8214

---

## 📁 Repository Structure

```
positive_unlabeled_learning/
├── README.md                      # Repository documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── pyproject.toml                 # Tool configuration
├── Dockerfile                     # Docker container recipe
├── app.py                         # Interactive Gradio Web Demo
│
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI workflow
│
├── configs/                       # Hyperparameter configurations
│   ├── default_config.yaml
│   ├── stage1_supervised.yaml
│   ├── stage2_pseudolabel.yaml
│   └── fathomnet_32classes.yaml
│
├── src/                           # Source library
│   ├── data/                      # Datasets, CLAHE transforms & balanced samplers
│   ├── models/                    # ResNet-50 / FPN backbones & Teacher-Student wrapper
│   ├── losses/                    # Non-Negative PU Loss & Focal Loss
│   ├── pipeline/                  # Stage 1, Pseudo-Labeling, Stage 2 trainers
│   ├── evaluation/                # mAP50, mAP50-95 & evaluator engine
│   ├── visualization/             # Bounding box rendering & comparison plots
│   └── utils/                     # Checkpoints, loggers & coordinate helpers
│
├── scripts/                       # Executable CLI scripts
│   ├── prepare_fathomnet.py       # Dataset preparation & synthetic data generator
│   ├── train_stage1.py            # Execute Stage 1 Supervised training
│   ├── generate_pseudo_labels.py  # Generate pseudo-labels using Teacher model
│   ├── train_stage2.py            # Execute Stage 2 Teacher-Student PU training
│   ├── evaluate.py                # Model evaluation on test set
│   └── predict_image.py           # Single image / folder inference
│
└── tests/                         # Pytest test suite
    ├── test_config.py
    ├── test_dataset.py
    ├── test_pu_loss.py
    ├── test_models.py
    ├── test_pseudo_labeler.py
    └── test_metrics.py
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/marine-pu-detection.git
cd marine-pu-detection

pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation
To generate synthetic underwater dataset files for quick testing out-of-the-box:
```bash
python scripts/prepare_fathomnet.py --data_dir data/fathomnet
```

### 3. Stage 1: Supervised Baseline Training
Train the initial baseline model on labeled underwater images:
```bash
python scripts/train_stage1.py --config configs/default_config.yaml --override configs/stage1_supervised.yaml
```

### 4. High-Confidence Pseudo-Label Generation
Use the trained Stage 1 Teacher model to predict pseudo-labels on unlabeled images ($\tau \ge 0.70$):
```bash
python scripts/generate_pseudo_labels.py --teacher_weights checkpoints/stage1_best.pt
```

### 5. Stage 2: Teacher-Student PU Retraining
Retrain the detector using combined labeled and pseudo-labeled data with Non-Negative PU loss:
```bash
python scripts/train_stage2.py --config configs/default_config.yaml --override configs/stage2_pseudolabel.yaml
```

### 6. Model Evaluation
Evaluate the final trained model on the test dataset:
```bash
python scripts/evaluate.py --weights checkpoints/stage2_best.pt --compare
```

### 7. Run Image Inference CLI
Run object detection on custom underwater photos:
```bash
python scripts/predict_image.py --image data/fathomnet/test/marine_0001.jpg --output logs/predictions/result.jpg --conf 0.25
```

### 8. Launch Interactive Web App (Gradio)
Launch the interactive web user interface:
```bash

Run the unit test suite with `pytest`:

``
