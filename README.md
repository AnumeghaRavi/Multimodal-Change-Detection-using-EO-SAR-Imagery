# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space — AI Research Intern Technical Assignment

A deep learning pipeline for pixel-level binary change detection using co-registered
Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite image pairs.
Built with a pretrained ResNet34 encoder and UNet-style decoder, trained with
combined Dice + Weighted Cross-Entropy loss to handle severe class imbalance.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (trained on Kaggle T4 x2)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/AnumeghaRavi/Multimodal-Change-Detection-using-EO-SAR-Imagery.git
cd Multimodal-Change-Detection-using-EO-SAR-Imagery

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

```
data/
├── train/
│   ├── pre-event/      ← EO RGB images (.tif)
│   ├── post-event/     ← SAR grayscale images (.tif)
│   └── target/         ← Binary change masks (.tif)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Update `config.yaml` with the correct dataset path.

---

## Training

```bash
python train.py --config config.yaml
```

Model checkpoints are saved to `checkpoints/best_model.pth` (best validation F1).

---

## Evaluation

```bash
python eval.py \
  --data_path data/test \
  --weights checkpoints/best_model.pth \
  --config config.yaml \
  --output_dir results
```

Outputs: Precision, Recall, F1, IoU printed to console + confusion matrix saved to `results/`.

---

## Model Weights

Pre-trained model weights (best checkpoint, epoch 12):

**[Download best_model.pth from Google Drive](#)**
> *(https://drive.google.com/file/d/10O00obZcub8o2MiHxbNvEsq_GEeE63Kb/view?usp=sharing)*

---

## Results

| Split | Precision | Recall | F1 Score | IoU |
|-------|-----------|--------|----------|-----|
| Validation (334 samples) | 0.2653 | 0.5147 | 0.3501 | 0.2122 |
| Test (77 samples, 50% blind) | 0.0319 | 0.3629 | 0.0587 | 0.0302 |

Metrics computed on full images (resized to 512x512) for the **change class (label=1)**.

---

## Model Architecture

- **Encoder**: ResNet34 pretrained on ImageNet, adapted for 4-channel input (3ch EO + 1ch SAR)
- **Decoder**: UNet-style with skip connections at each encoder stage
- **Loss**: Dice Loss + Weighted Cross-Entropy (weights: [1.0, 20.0])
- **Parameters**: ~24.5M

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Patch size 256x256 | Full 1024x1024 images exceed GPU memory for reasonable batch sizes |
| 4-channel early fusion | Allows joint cross-modal learning from first layer |
| Pretrained ResNet34 backbone | Leverages ImageNet features, reduces training time |
| Combined Dice + Weighted CE loss | Addresses 95%/5% class imbalance |
| Label remapping in dataloader | 25% of masks retained original 4-class values; remapping applied programmatically |

---

## Citation / References

```
He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
Daudt et al. (2018). Fully Convolutional Siamese Networks for Change Detection. ICIP.
Chen & Shi (2021). A Spatial-Temporal Attention-Based Method for Remote Sensing Change Detection.
Milletari et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.
```
