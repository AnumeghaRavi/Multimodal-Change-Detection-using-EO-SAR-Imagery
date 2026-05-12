# EO-SAR Change Detection using Deep Learning

## Overview

This project focuses on multimodal change detection using Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery for building damage assessment.

The objective is to identify regions of structural change between pre-event and post-event imagery using semantic segmentation techniques.

This project was developed as part of an AI internship assignment focused on remote sensing and computer vision.

---

# Dataset

The dataset contains:

```text
train/
├── pre-event/
├── post-event/
└── target/

val/
├── pre-event/
├── post-event/
└── target/

test/
├── pre-event/
├── post-event/
└── target/
```

## Data Characteristics

- EO imagery: RGB optical satellite images
- SAR imagery: grayscale radar images
- Target masks: binary building damage/change masks

Image resolution:

```text
1024 × 1024
```

---

# Approach

## Preprocessing

- TIFF image loading
- Dataset inspection and visualization
- Patch extraction for efficient training
- Data augmentation using random flips

## Model

The project uses a U-Net inspired segmentation pipeline with a pretrained ResNet encoder backbone.

Input representation combines:

- 3 EO channels
- 1 SAR channel

The network predicts binary segmentation masks representing regions of structural change.

## Loss Functions

To address class imbalance, the training pipeline combines:

- Dice Loss
- Cross Entropy Loss

---

# Evaluation

The notebook includes:

- Prediction visualization
- Full-image inference
- Validation pipeline
- IoU and Dice Score evaluation

---

# Tools & Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebooks

---

# Challenges

Some key challenges in this project include:

- Cross-modal EO/SAR feature learning
- Sparse segmentation masks
- High class imbalance
- Large satellite image resolution

---

# Current Status

Completed:

- Dataset exploration
- Data preprocessing pipeline
- Model implementation
- Training and evaluation setup
- Visualization pipeline

Future improvements may include:

- Siamese dual-encoder architectures
- Attention-based fusion
- Transformer-based segmentation methods

---

# Repository Structure

```text
EO-SAR-Change-Detection/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
└── results/
```

---

# Author

Anumegha Ravi
