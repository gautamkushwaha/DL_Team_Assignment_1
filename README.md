# Stress-Testing of Convolutional Neural Networks on CIFAR-100

## Group Members
1. **Gautam Kumar Kushwaha** (M25CSA037)
2. **Aryan Baranwal** (M25CSE035)
3. **Parth Pitrubhakta** (M25CSE022)

---

## 📑 **Contents**
- [Stress-Testing of Convolutional Neural Networks on CIFAR-100](#stress-testing-of-convolutional-neural-networks-on-cifar-100)
  - [Group Members](#group-members)
  - [📑 **Contents**](#-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Results Summary](#results-summary)
  - [Project Structure](#project-structure)
  - [Key Visualizations](#key-visualizations)
  - [How to Run](#how-to-run)
  - [Dependencies](#dependencies)

---

## Overview

This project presents a comprehensive pipeline for training and evaluating a ResNet-18 model on the CIFAR-100 dataset. The primary focus is on "stress-testing" the model by analyzing its performance through various lenses: standard metrics (accuracy, precision, recall, F1-score), per-class analysis, identification of high-confidence failure cases, and explainability via Grad-CAM visualizations.

The entire experiment is designed to be reproducible, leveraging fixed random seeds and structured logging.

---

## Key Features

- **Model:** ResNet-18, modified for 32x32 CIFAR images (removed initial max-pool, adjusted first convolution layer).
- **Dataset:** CIFAR-100 (50k train, 10k test images). A 20% validation split is created from the training set.
- **Training:**
  - Optimizer: SGD with Nesterov momentum.
  - Learning Rate Schedule: Cosine annealing with a linear warmup.
  - Regularization: Label smoothing, weight decay, gradient clipping.
  - Automatic Mixed Precision (AMP) for faster training.
  - Early stopping based on validation accuracy.
- **Evaluation:**
  - Accuracy, Macro-Precision, Macro-Recall, Macro-F1 Score.
  - Confusion Matrix (Normalized).
  - Per-Class Accuracy analysis.
  - Discovery and storage of high-confidence misclassifications.
- **Explainability:** Grad-CAM visualizations generated for identified failure cases to understand model focus.

---

## Results Summary

After training for 40 epochs with early stopping, the model achieved the following performance on the held-out test set:

| Metric      | Value      |
|-------------|------------|
| **Accuracy**    | 65.96%     |
| **Precision**   | 66.11% (macro avg) |
| **Recall**      | 65.96% (macro avg) |
| **F1-Score**    | 65.88% (macro avg) |

The model contains **11.2 million** trainable parameters. The best validation accuracy of **64.96%** was achieved at epoch 40.

---

## Project Structure

The notebook is organized into the following logical steps:

1.  **Setup & Configuration:** Installing libraries, setting random seeds, defining hyperparameters.
2.  **Data Loading & Exploration:** Loading CIFAR-100, creating train/val/test splits, and visualizing samples.
3.  **Model Definition:** Modifying a standard ResNet-18 for the CIFAR-100 task.
4.  **Training Loop:** Implementing the training, validation, and early stopping logic.
5.  **Evaluation & Analysis:**
    - Final test metrics (Accuracy, Precision, Recall, F1).
    - Plotting training/validation curves.
    - Generating and saving Grad-CAM visualizations for failure cases.
    - Creating a normalized confusion matrix.
    - Analyzing per-class accuracy and identifying the worst-performing classes.
    - Reporting model complexity (total parameters).

---

## Key Visualizations

The analysis generates several key plots, saved in the `plots/` directory:

- **Training Curves:** Plots of accuracy and loss over epochs for both training and validation sets.
- **Metrics Bar Plot:** A bar chart comparing final test accuracy, precision, recall, and F1-score.
- **Confusion Matrix:** A normalized heatmap showing prediction patterns across all 100 classes.
- **Worst-Performing Classes:** A bar chart highlighting the 10 classes with the lowest individual accuracy.
- **Grad-CAM Overlays:** Visual explanations for high-confidence misclassifications, showing which image regions the model focused on.

---

## How to Run

1.  Open the provided `DL_Team_Assignment_1.ipynb` notebook in Google Colab.
2.  Ensure a GPU runtime is selected (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3.  Run all cells sequentially. The notebook will automatically install required dependencies and download the CIFAR-100 dataset.
4.  The training process and all subsequent analyses will execute. Results and plots will be saved to the `checkpoints/`, `plots/`, `failures/`, `gradcam/`, and `logs/` directories.

---

## Dependencies

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- NumPy
- scikit-learn
- tqdm
- Grad-CAM (`pytorch_grad_cam`)

All dependencies are installed automatically within the notebook.