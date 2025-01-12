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

## Model Training

Epoch 1/10

1875/1875 [==============================] - 11s 5ms/step - accuracy: 0.8789 - loss: 0.4480

Epoch 2/10

1875/1875 [==============================] - 7s 4ms/step - accuracy: 0.9597 - loss: 0.1373

Epoch 3/10

1875/1875 [==============================] - 9s 5ms/step - accuracy: 0.9758 - loss: 0.0830

Epoch 4/10

1875/1875 [==============================] - 9s 5ms/step - accuracy: 0.9808 - loss: 0.0637

Epoch 5/10

1875/1875 [==============================] - 7s 4ms/step - accuracy: 0.9859 - loss: 0.0483

Epoch 6/10

1875/1875 [==============================] - 8s 4ms/step - accuracy: 0.9881 - loss: 0.0393

Epoch 7/10

1875/1875 [==============================] - 8s 4ms/step - accuracy: 0.9915 - loss: 0.0292

Epoch 8/10

1875/1875 [==============================] - 8s 4ms/step - accuracy: 0.9919 - loss: 0.0261

Epoch 9/10

1875/1875 [==============================] - 8s 4ms/step - accuracy: 0.9933 - loss: 0.0212

Epoch 10/10

1875/1875 [==============================] - 8s 4ms/step - accuracy: 0.9960 - loss: 0.0148

## Model Evaluation

313/313 [==============================] - 1s 2ms/step - accuracy: 0.9722 - loss: 0.1025



You can install the required dependencies by running:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
