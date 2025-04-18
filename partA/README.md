# Part A: Training a CNN from Scratch on iNaturalist Dataset
 
This project trains a convolutional neural network (CNN) from scratch to classify images from a subset of the iNaturalist dataset into 10 classes. The workflow includes data preprocessing, model architecture design, hyperparameter tuning with Weights & Biases (Wandb), and evaluation on a test set.  

### 1. Data Preprocessing  
**Dataset:**  
- Training data: 12,000 images (10 classes).  
- Test data: Reserved for final evaluation (not used during training/validation).  

**Image Resizing:**  
All images resized to 224 x 224 pixels for compatibility with the CNN input layer.  

**Normalization:**  
Pixel values standardized using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).  

**Validation Split:**  
- 20% of training data reserved for validation.  
- Stratified sampling ensured equal class representation.  

**Data Augmentation:** Optional (tested during hyperparameter sweeps).  

### 2. Model Architecture  
A 5-layer CNN was implemented with the following structure:

**Convolutional Blocks:**  
Each block contains:  
- Conv2D Layer: Kernel size = 3 x 3, stride = 1.  
- Activation: `ReLU`, `GELU`, `SiLU`, or `Mish` (configurable).  
- Max-Pooling: Kernel size = 2 x 2.  

**Filter Organization:**  
- `Same`: Fixed filters (e.g., 32 or 64 in all layers).  
- `doubling`: Filters doubled in subsequent layers (e.g., 32 → 64 → 128).  
- `halving`: Filters halved in subsequent layers (e.g., 64 → 32 → 16).  

**Post-Convolution Layers:**  
- Flattening: Output from convolutional blocks flattened.  
- Dense Layer: 128 neurons.  
- Output Layer: 10 neurons (Softmax activation for class probabilities).  

**Regularization:**  
- Batch Normalization: Optional after convolutional layers.  
- Dropout: Applied before the dense layer (rate = 0.2 or 0.3).  

### 3. Hyperparameter Tuning with Wandb  
A Bayesian optimization sweep (29 runs) was conducted to identify optimal hyperparameters:

**Parameters Explored:**

| Hyperparameter       | Values Tested                       |
|----------------------|-------------------------------------|
| Epocs                | 5, 10                               |
| Activation Function  | `ReLU`, `GELU`, `SiLU`, `Mish`      |
| Filters per Layer    | 32, 64                              |
| Filter Organization  | `same`, `doubling`, `halving`       |
| Batch Size           | 32, 64                              |
| Learning Rate        | 1e-3, 1e-4                          |
| Dropout Rate         | 0.2, 0.3                            |
| Batch Normalization  | True/False                          |
| Data Augmentation    | True/False                          |

**Early Stopping:** Triggered if no improvement after 3 epochs.

**Best Configuration:**  
- Epochs: 5
- Activation: `SiLU` (achieved highest validation accuracy).  
- Filters: `64` with `doubling` strategy.  
- Batch Normalization: True.
- Data Augmentation: False.
- Dropout: 0.3.  
- Learning Rate: 0.0001.  
- Batch Size: 64.  

### 4. Training and Evaluation  
**Optimizer:** Adam with cross-entropy loss.  

**Training:**  
- 5 epochs (optimal for convergence without overfitting).  
- Batch-wise loss and accuracy logged to Wandb.  

**Validation:**  
- Monitored validation accuracy/loss after each epoch.  
- Best model saved based on validation performance.  

**Test Evaluation:**
- Achieved Validation Accuracy: 0.8109 and Test Accuracy: 0.8075  
- Sample Predictions: Generated a 10 x 3 grid of test images with true vs. predicted labels.  

## Key Results  

**Wandb Plots:**  
- Accuracy vs. Epochs: Showed rapid convergence with SiLU and doubling filters.  
- Parallel Coordinates Plot: Highlighted strong correlation between SiLU, doubling filters, and high accuracy.  
- Correlation Summary: Identified dropout (0.3) and batch normalization as critical for reducing overfitting.

## GitHub Repository  
The complete code, including data preprocessing, model architecture, training loops, and Wandb integration, is available here:  
**[GitHub Link]()**
