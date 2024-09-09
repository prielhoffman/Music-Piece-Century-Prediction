# Music Piece Century Prediction
This repository contains Assignment 2 from the "Introduction to Machine Learning" course, completed during my degree. The assignment focuses on predicting the century in which a music piece was released using various machine learning models. The task utilizes the "YearPredictionMSD Data Set," which is derived from the Million Song Dataset. This dataset provides valuable features extracted from audio tracks that help in predicting the music's release century.

# Overview
The assignment is divided into two main tasks:
1) Regression Model: Predict the exact year in which a music piece was released.
2) Classification Model: Classify the century of a music piece's release based on the provided audio features.

# Dataset
The dataset used in this assignment is the "YearPredictionMSD Data Set." It contains numerous audio features from songs that are part of the Million Song Dataset. The target variable is the release year of each song, which needs to be converted into centuries for the classification task.
Relevant links for additional reading about the dataset:
* [YearPredictionMSD Data Set - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
* [Million Song Dataset - Year Recognition Task](http://millionsongdataset.com/pages/tasks-demos/#yearrecognition)

# Files in the Repository
1) ML_DL_Assignment2.ipynb: A Jupyter Notebook containing the implementation of both regression and classification models.
2) ML_DL_Functions2.py: A Python utility functions file to be completed. This file contains essential functions for data preprocessing, model building, and evaluation.
3) assignment2_submission_optimal_weights.npy: A NumPy file containing the output of the optimal weights of the model.
4) assignment2_submission_optimal_bias.npy: A NumPy file containing the output of the optimal bias of the model.

# Project Structure
1) Data Loading and Preparation: Load the dataset from Google Drive, preprocess the data by extracting and normalizing features, and transform the target variable from years to centuries.
2) Regression Model: Implement and train a regression model to predict the exact release year of a music piece.
3) Classification Model: Implement and train a classification model to predict the century of release.
4) Model Optimization: Tune the models using techniques such as hyperparameter optimization to achieve the best performance.
5) Evaluation and Saving Results: Evaluate both models using appropriate metrics (e.g., Mean Absolute Error for regression and Accuracy for classification) and save the optimal model weights and biases.

# Requirements
To run the notebook and scripts in this repository, you will need Python and several libraries installed. The following are the primary libraries required:
* pandas
* numpy
* matplotlib
* pytorch
* jupyter

# Acknowledgments
The "YearPredictionMSD Data Set" is sourced from the UCI Machine Learning Repository.
