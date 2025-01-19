# UrbanSounds
Classification of urban sounds using machine learning algorithms.



This repository contains the implementation of a deep learning project focused on classifying urban sound data using the UrbanSound8K dataset. The project was developed as part of the Machine Learning II course (2024/2025) at the Faculty of Sciences, University of Porto.

# Project Overview
The objective of this project is to build and evaluate deep learning classifiers capable of identifying sound samples from one of the following 10 urban sound categories:

-Air conditioner

-Car horn

-Children playing

-Dog bark

-Drilling

-Engine idling

-Gunshot

-Jackhammer

-Siren

-Street music

The project implements and compares two classifiers from the following options:

-Multilayer Perceptron (MLP)
-Convolutional Neural Network (CNN)
-Recurrent Neural Network (RNN)

# Dataset
The UrbanSound8K dataset is composed of 8,732 labeled audio excerpts, each with a duration of 4 seconds or less. The dataset is organized into 10 categories and is accessible through the official dataset webpage:



# Implementation Details
The project workflow is divided into four main steps:

  1. Data Pre-processing
Uniformizing and normalizing audio input.
Extracting features from audio signals using techniques such as Mel-frequency cepstral coefficients (MFCCs).
Managing class labels using the provided UrbanSound8K.csv file.

  3. Model Architecture
MLP: Customizable layers, neurons, and activation functions.
CNN: Includes options for 1D or 2D architectures based on raw signal processing or time-frequency representations.
RNN: Implements LSTM or GRU networks with options for unidirectional or bidirectional connections.

  5. Training Strategy
Optimizers: Configurable options such as Adam or SGD.
Hyperparameters: Adjustable learning rate, batch size, and epochs.
Regularization: Includes dropout, weight regularization, and early stopping.
Cross-validation: Uses a 10-fold strategy to ensure robust performance evaluation.

  7. Performance Evaluation
Confusion matrix computation.
Average classification accuracy and standard deviation over 10 experiments.



# Requirements
Python 3.x
TensorFlow or PyTorch (based on implementation)
Librosa for audio processing
NumPy, Matplotlib, and other standard libraries
