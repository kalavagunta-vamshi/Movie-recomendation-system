# Netflix Movie-recomendation-system



# Movie Recommendation System using Collaborative Filtering

## Overview

This project implements collaborative filtering (CF) techniques to build a movie recommendation system. CF predicts user preferences based on similarities among users or items and is widely used in recommendation systems to provide personalized suggestions.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
3. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
4. [Model Selection and Implementation](#model-selection-and-implementation)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Conclusion](#conclusion)
7. [Future Enhancements](#future-enhancements)


## Introduction

This project aims to predict movie ratings using various CF algorithms and evaluate their performance using RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error). The goal is to recommend movies based on user preferences inferred from historical ratings data.

## Setup Instructions

### Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Surprise
- XGBoost

### Installation

Clone the repository:

```bash
https://github.com/kalavagunta-vamshi/Movie-recomendation-system.git

```


### Dataset

The dataset used contains user ratings for movies. Ensure the dataset is placed in a directory accessible by the scripts.

## Data Collection and Preprocessing

### Steps

1. **Data Loading**: Load the dataset into a Pandas DataFrame.
2. **Handling Missing Values**: Impute missing ratings or remove incomplete entries.
3. **Normalization**: Scale ratings to a common range (e.g., 1-5).
4. **Train-Test Split**: Divide data into training and testing sets.

## Model Selection and Implementation

### CF Algorithms Implemented

1. **Baseline Models**:
   - BaselineOnly
   - Baseline with Stochastic Gradient Descent (SGD)

2. **Matrix Factorization Models**:
   - Singular Value Decomposition (SVD)
   - SVD++
   
3. **Neighborhood Models**:
   - User-Based KNN (knn_bsl_u)
   - Item-Based KNN (knn_bsl_m)

4. **Gradient Boosting Models**:
   - XGBoost with CF features
   - Hybrid XGBoost with KNN predictions

## Evaluation Metrics

- **RMSE**: Measures the deviation of predicted ratings from actual ratings.
- **MAPE**: Calculates the percentage difference between predicted and actual ratings.



### Performance Comparison

- Tabular comparison of RMSE and MAPE for each model.
- Visualization of results using bar charts or line plots.

### Insights

- Discuss the best-performing models based on evaluation metrics.
- Interpret strengths and weaknesses of each CF technique.



## Conclusion

This project demonstrates the implementation of collaborative filtering techniques for movie recommendation. It highlights the importance of model selection, evaluation metrics, and data preprocessing in building effective recommendation systems.

## Future Enhancements

- Implement hybrid models combining CF with content-based filtering.
- Incorporate deep learning approaches for improved prediction accuracy.
- Deploy the system as a web application using Flask or Django.


