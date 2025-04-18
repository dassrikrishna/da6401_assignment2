# DA6401 Assignment-2
#### Detailed Weights & Biases Report for My Project: [Click Here](https://wandb.ai/ma24m025-indian-institute-of-technology-madras/iNaturalist-CNN/reports/MA24M025_DA6401-Assignment-2-Report--VmlldzoxMjI2NzYyOQ?accessToken=o2eqcgzy3d6q1lswatnsd3sad5dkg2ijmxa3xbzw5mxn7nxix8eyc9ffv57jm5ry)
#### Github Link: [Click Here](https://github.com/dassrikrishna/da6401_assignment2)
#### part A: [Click Here](https://github.com/dassrikrishna/da6401_assignment2/tree/main/partA)
#### part B: [Click Here](https://github.com/dassrikrishna/da6401_assignment2/tree/main/partB)

## DEEP LEARNING
#### ```SRIKRISHNA DAS (MA24M025)```
#### `M.Tech (Industrial Mathematics and Scientific Computing) IIT Madras`
 

## [Problem Statement](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-2--VmlldzoxMjAyNjgyNw)

# CNN-Based Image Classification on iNaturalist Dataset

## Part A: Training from Scratch

This project involves building, training, and evaluating a Convolutional Neural Network (CNN) from scratch to classify images into 10 classes using [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). The workflow covers model design, hyperparameter tuning, and performance evaluation.

### 1. Model Architecture
- Designed a small CNN with 5 convolutional layers.
- Each convolutional layer is followed by an activation function and a max-pooling layer.
- After the 5 blocks, added one dense (fully connected) layer.
- The output layer has 10 neurons (one for each class).
- The input layer is configured to match the image dimensions in the iNaturalist dataset.

### 2. Data Preparation & Hyperparameter Tuning
- Used the provided train and test folders from the dataset.
- Set aside 20% of the training data for validation, ensuring equal class representation.
- The test data was kept strictly for final evaluation.
- Used the Weights & Biases (wandb) sweep feature to find the best hyperparameter configuration.

**Hyperparameters explored:**
- Number of filters per layer: 32, 64
- Activation functions: `ReLU`, `GELU`, `SiLU`, `Mish`
- Filter organization: `same` number in all layers, `doubling`, or `halving` across layers
- Data augmentation: Yes, No
- Batch normalization: Yes, No
- Dropout: 0.2, 0.3

- Smart strategies were used to reduce the number of runs while maintaining high accuracy, such as conditional sweeps and early stopping.

### 3. Hyperparameter Sweep Results
**Plots generated by wandb:**
- Accuracy vs. run plot
- Parallel coordinates plot
- Correlation summary table

The sweep results and plots provide insights into which hyperparameters most influence model performance.

### 4. Test Set Evaluation
- The best model (selected based on validation performance) was evaluated on the test set.
- Reported the final test accuracy.
- Presented a 10 x 3 grid of test images with predicted labels, displayed creatively for clarity and visual appeal.

### 5. Code Repository
The complete code for Part A, including data preprocessing, model building, training, hyperparameter sweeps, and evaluation, is available on GitHub.
[Link Part A](https://github.com/dassrikrishna/da6401_assignment2/tree/main/partA)


## Part B: Fine-tuning a Pre-trained Model

This project explores transfer learning by fine-tuning a model pre-trained on ImageNet for image classification on a subset of the iNaturalist dataset. The workflow covers adapting pre-trained models, experimenting with different fine-tuning strategies, and comparing results with training from scratch.

### 1. Adapting a Pre-trained Model
- Loaded a pre-trained model from the torchvision library (e.g., `ResNet50`, `VGG`, `EfficientNetV2`, etc.), originally trained on the ImageNet dataset.
- Adjusted input image dimensions to match the requirements of the chosen pre-trained model by resizing and normalizing the iNaturalist images accordingly.
- Modified the final classification layer of the pre-trained model to output 10 classes, matching the number of classes in the iNaturalist subset.

### 2. Fine-tuning Strategies
- To keep training tractable, experimented with the following strategies:
  - Freezing all layers except the last layer and training only the final classification layer.
  - Freezing up to a certain number of layers (e.g., up to `layer3`), and fine-tuning the remaining layers.
  - Unfreezing and fine-tuning all layers of the model.
- Compared these approaches to find a balance between computational efficiency and model performance.

### 3. Fine-tuning and Insights
- Selected one fine-tuning strategy and trained the adapted model on the iNaturalist dataset.
- Compared results from fine-tuning a large pre-trained model to training a small CNN from scratch.
- Documented key inferences, such as differences in training speed, accuracy, and generalization, highlighting the benefits and trade-offs of transfer learning.

### 4. Code Repository
The complete code for Part B, including model adaptation, fine-tuning strategies, and evaluation, is available on GitHub.
[Link Part B](https://github.com/dassrikrishna/da6401_assignment2/tree/main/partB)

