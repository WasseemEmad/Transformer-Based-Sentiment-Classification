# Transformer-Based-Sentiment-Classification
This project implements a transformer-based model for binary sentiment analysis using PyTorch. The model leverages attention mechanisms for efficient text representation and achieves impressive performance on a sentiment dataset.

# Project Overview
## This repository contains code for:
Preprocessing text data.
Implementing a transformer-based model for sentiment classification.
Training, evaluating, and visualizing model performance metrics.

# Performance Metrics
Training Accuracy: 93%
Validation Accuracy: 90%
Test Accuracy: 88%
F1 Score: 0.8870
Precision: 0.8740
Recall: 0.9003

# Evaluating the Model
## The script evaluates the model on the test dataset and outputs:
Accuracy, Precision, Recall, and F1 Score.
Confusion matrix.
Visualizations such as training loss over epochs.

# Model Inference
## To use the trained model for predictions:
1 Load the saved model weights:
  1.1 model.load_state_dict(torch.load("sentiment_model.pth"))
  1.2 model.eval()
2 Pass your input text to the preprocessing pipeline and feed it to the model.

# Key Components of a Transformer
1 Self-Attention Mechanism
2 Multi-Head Attention
3 Positional Encoding
4 Feed-Forward Neural Networks
5 Layer Normalization and Residual Connections
