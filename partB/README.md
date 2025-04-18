## Part B: Fine-Tuning a Pretrained ResNet50 on iNaturalist Dataset

This project fine-tunes a pretrained `ResNet50` model on the iNaturalist dataset for classifying images into 10 natural categories. The experiment involves adjusting the final classification layer, selecting a tuning strategy (`head`, `partial`, or `full`), conducting hyperparameter sweeps using Weights & Biases (Wandb), and comparing performance with **Part A** (CNN from scratch).

---
### 1. Model Adaptation

**Pre-trained Model:**

- Loaded **ResNet50** from `torchvision.models` with pretrained weights.
```python
  model = torchvision.models.resnet50(weights="IMAGENET1K_V2")  
```

### 2. Data Preprocessing

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

### 3. Model Architecture and Fine-Tuning Modes

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
```python
    if mode == "head":
        # Freeze all layers except the final classification head
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    elif mode == "partial":
        # Partially unfreeze the model, training layers from 'unfreeze_from' onwards
        freeze = True
        for name, module in model.named_children():
            if name == unfreeze_from:
                freeze = False
            for param in module.parameters():
                param.requires_grad = not freeze

    elif mode == "full":
        # Unfreeze the entire model, training all layers
        for param in model.parameters():
            param.requires_grad = True
```

### 4. Hyperparameter Tuning with Wandb

A **Bayesian hyperparameter sweep** was performed (22 runs) to determine optimal settings for **head tuning**.

**Parameters Explored:**

| Hyperparameter   | Values               |
|------------------|----------------------|
| **Epochs**       | 5, 10                |
| **Dropout Rate** | 0.0, 0.2, 0.3        |
| **Batch Size**   | 32, 64               |
| **Learning Rate**| 1e-3, 1e-4           |

**Early Stopping:**

- Enabled to halt training after 3 epochs without validation improvement.

**Best Configuration Identified:**

- **Epochs:** 5
- **Dropout:** 0.3
- **Batch Size:** 64
- **Learning Rate:** 0.0001
- **Tuning Mode:** `'head'` Tuning

### 5. Training and Evaluation

**Loss Function:**  
- **CrossEntropyLoss**

**Optimizer:**  
- **Adam**

**Training Strategy:**
- Only the newly added final layer is trained in **head tuning**.
- **Wandb** is used to log loss and accuracy for each epoch.

**Validation:**
- Best model is chosen based on **validation accuracy**.
- Metrics are logged and visualized using **Wandb**.

**Test Evaluation:**
- **Best Validation Accuracy:** 0.8109
- **Corresponding Test Accuracy:** 0.8075

The model outperformed the custom CNN from **Part A** on this dataset.

#### Comparison with Part A:

- **Validation Accuracy:**  from 0.3932 (Part A) to 0.8109 (Part B)
- **Test Accuracy:**  from 0.4150 (Part A) to 0.8075 (Part B)

### GitHub Repository
The full implementation, including fine-tuning code, sweep configuration, and logging utilities, is available here:  
[GitHub Link](https://github.com/dassrikrishna/da6401_assignment2/tree/main/partB)



