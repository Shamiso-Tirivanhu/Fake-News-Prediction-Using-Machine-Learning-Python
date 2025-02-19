# Fake-News-Prediction-Using-Machine-Learning-Python

## Introduction

This project focuses on detecting fake news using machine learning techniques in Python. It involves data preprocessing, feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency), and training various machine learning models to classify news articles as real or fake.

## Features

- Data preprocessing using Pandas and NumPy

- Text processing with NLTK and Scikit-learn

- Feature extraction using TF-IDF Vectorizer

- Machine learning models: Logistic Regression, Decision Tree Classifier, and Support Vector Machine (SVM)

- Model evaluation using accuracy metrics

## Usage 

- Load the dataset containing labeled news articles.

- Preprocess the text data (removal of stopwords, stemming, etc.).

- Convert text data into numerical format using TF-IDF.

- Train machine learning models and evaluate their performance.

- Predict the authenticity of news articles using the trained model.

## Dataset

The dataset should contain two key columns:

- text: The content of the news article.

- label: Binary classification (1 for fake news, 0 for real news).

## Model Evaluation & Training

- The dataset is split into training and testing sets.

- Various models are trained and compared based on accuracy and precision-recall metrics.

- The best-performing model is selected for final predictions.

## Performance

- Training Accuracy: ~99.27%

- Testing Accuracy: ~53.77%

## Results

The results include accuracy scores and confusion matrices to evaluate model performance. The Logistic Regression model typically performs well for text classification tasks. However, it did not perform well. Nonetheless, l have deciced to post it as l have learnt so much from this project.

This project was unsuccessful as there is a massive difference between accuracy scores of the training and testing data

## Contribution

Free free to correct me were l went wrong.
