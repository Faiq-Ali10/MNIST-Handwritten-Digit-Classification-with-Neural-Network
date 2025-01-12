# MNIST Handwritten Digit Classification with Neural Network and Confusion Matrix Visualization

This project demonstrates how to train a neural network using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model is trained for 10 epochs, evaluated on a test dataset, and the confusion matrix is computed and visualized to assess the classification performance.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Confusion Matrix Visualization](#confusion-matrix-visualization)
7. [License](#license)

## Overview

In this project, the MNIST dataset is loaded, preprocessed, and used to train a neural network model for handwritten digit classification. The model is trained for 10 epochs, and the accuracy and loss are tracked during the training. After training, the model is evaluated on the test data. Finally, a confusion matrix is generated and visualized to understand the model's classification results.

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- numpy
- matplotlib
- seaborn

You can install the required dependencies by running:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
