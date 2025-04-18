## Part B: Fine-Tuning a Pretrained ResNet50 on iNaturalist Dataset

This project fine-tunes a pretrained `ResNet50` model on the iNaturalist dataset for classifying images into 10 natural categories. The experiment involves adjusting the final classification layer, selecting a tuning strategy (`head`, `partial`, or `full`), conducting hyperparameter sweeps using Weights & Biases (Wandb), and comparing performance with **Part A** (CNN from scratch).

---

### 1. Data Preprocessing

**Dataset:**

- **Training data:** 12,000 images from 10 classes.
- **Test data:** Reserved for final evaluation (not used during training/validation).

**Image Resizing:**

- All images are resized to **224 x 224** pixels to match the input size for `ResNet50`.

**Normalization:**

- Pixel values are standardized:
  - **Mean:** `[0.485, 0.456, 0.406]`
  - **Std:** `[0.229, 0.224, 0.225]`

**Validation Split:**

- **20%** of the training data is used for validation.
- **Stratified sampling** ensures equal class distribution across train/val sets.

### 2. Model Architecture and Fine-Tuning Modes

**Base Model:**

- **ResNet50** from `torchvision.models` with pretrained weights from **ImageNet**.

**Final Layer Modification:**

- The original classifier (1000-class) is replaced with a new dense layer for **10-class** classification:
```python
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(in_features, 10)
)
```

**Fine-Tuning Strategies Implemented:**

| Strategy        | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **Head Tuning** | All layers frozen except the final classification head.                     |
| **Partial Tuning** | Unfreezes later ResNet layers (e.g., `layer3`, `layer4`) for selective fine-tuning. |
| **Full Tuning** | The entire network is unfrozen and retrained end-to-end.                    |

