# International E-Commerce Customer Analysis

**Course:** STA 6636  
**Group 5:** Brandon Rodriguez, Gabriel Ruiz, Jorge Corcino, Ronaldo Martinez Frias

This repository contains the documents, datasets, and codebase for our Large Data Analysis project. 

## Project Overview

This project analyzes a customer dataset from an international e-commerce company that sells electronic products. The dataset contains 10,999 records and 12 features, providing insights into customer satisfaction and the factors that influence it. By evaluating these variables, we aim to understand which factors carry the greatest weight in determining a customer's rating and whether their packages arrive on time.

## Dataset Features

The dataset consists of the following features:
- **ID**: Numerical
- **Warehouse block**: Nominal
- **Mode of shipment**: Nominal 
- **Customer care calls**: Numerical
- **Customer rating**: Numerical (1-5 ranking)
- **Cost of the product**: Numerical
- **Prior purchases**: Numerical
- **Product importance**: Ordinal
- **Gender**: Boolean
- **Discount offered**: Numerical
- **Weight in grams**: Numerical
- **Reached on time**: Boolean (Target Variable)

## Goals

1. **Classification Performance**:
   - Predict the **customer rating** based on all other features.
   - Predict if the customer received the package **on time** based on all other features.
2. **Correlation Analysis**: Identify if there is any correlation between the dataset's variables.
3. **Variable Selection**: Determine the most relevant features for our classification problems.

## Methodology

- **Classification Approach**: We will employ Logistic Regression for both target variables. Additionally, we will implement gradient boosting classifiers (such as Random Forest or XGBoost) to capture robust, non-linear relationships within the data.
- **Ordinal Regression**: Since the customer rating is a ranked scale from lowest (1) to highest (5), we will explore methods such as cumulative logit regression. This method computes the probability of a rating being at or below a specific number based on the given features.
- **Addressing Challenges**: We have identified a slight data imbalance in the "Reached on time" target variable. We will assess the F1-Score and ROC-AUC before performing any transformations.

## Evaluation

To evaluate our classification approach, we will use a **k-fold cross-validation** split. Models will be measured using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Repository Structure

- `assets/` - Contains the dataset (`Train.csv`).
- `doc/` - Contains project documentation and proposals (`Project_Proposal.pdf`, `STA6636_Project_Proposal.docx`).
- `notebook/` - Contains Jupyter notebooks for our analysis (`shipping.ipynb`).
