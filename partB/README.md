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
