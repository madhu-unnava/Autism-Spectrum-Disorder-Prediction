# 🧠 Autism Spectrum Disorder Detection Using Machine Learning

This project focuses on the detection of Autism Spectrum Disorder (ASD) by analyzing demographic and behavioral survey data using multiple machine learning models. The goal is to predict ASD with high accuracy and to identify the most significant predictors through different classification techniques.

## 📝 Project Overview

Autism Spectrum Disorder (ASD) affects social communication and behavior, and early detection is crucial for timely interventions. This project applies various machine learning algorithms, including K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), and Random Forests to predict the likelihood of ASD based on a dataset of behavioral attributes.

### Objectives:
- Build machine learning models to predict the likelihood of ASD.
- Analyze the key behavioral and demographic factors influencing ASD prediction.
- Compare the performance of different models using metrics like accuracy, sensitivity, and specificity.

## 📊 Dataset Summary
- **Size**: 800 rows and 22 columns.
- **Features**: Includes survey responses (A1 to A10), demographic variables like age, gender, ethnicity, and other attributes like family history of ASD and jaundice.
- **Target**: Class/ASD (binary variable indicating the presence or absence of ASD).

## 🧑‍💻 Methodology

### Data Preprocessing:
- **Handling Missing Values**: Missing data points were imputed using median imputation for numerical values and mode imputation for categorical variables to ensure consistency across the dataset.
- **Encoding Categorical Variables**: Categorical features like gender, ethnicity, and family history were converted into numerical format using one-hot encoding, allowing the models to process these features.
- **Outlier Detection and Treatment**: Outliers were identified through interquartile range (IQR) analysis and treated by capping values to reduce their effect on model training.
- **Feature Engineering**: Relevant features were derived from existing ones, such as combining lower-frequency categories into an “Other” category to reduce model complexity and enhance performance.
- **Train-Test Split**: The dataset was split into training and testing sets in an 80-20 ratio to evaluate model performance effectively.

### Models Used:
- **K-Nearest Neighbors (KNN)**: The optimal K value was determined to be 5, with an accuracy of ~87%.
- **CART (Classification and Regression Tree)**: Both minimal error (ME) and best pruned (BP) trees were used, achieving accuracies of 88.17% and 84.02%, respectively.
- **Random Forest**: Achieved an accuracy of **84.6%**, with robust performance across the board by averaging results from multiple decision trees, enhancing model stability and reducing overfitting.

### Model Results:
- **Forward Selection Model**: Key predictors included A3_Score, A4_Score, A6_Score, and ethnicity (Middle Eastern and White-European).
- **Backward Elimination Model**: Identified A2_Score, A3_Score, A4_Score, jaundice history, and ethnicity as the most significant predictors, balancing sensitivity and specificity.
- **Stepwise Model**: The stepwise regression identified additional features like age and family relation (parent) to the subject as critical contributors to ASD prediction.
- **KNN**: The optimal K value was 5, resulting in a stable accuracy of 87%.
- **CART**: The minimal error tree achieved an accuracy of 88.17%, while the best pruned tree reached 84.02%.
- **Random Forest**: Provided robust predictions with **84.6% accuracy**, averaging multiple decision trees to improve classification reliability and reduce overfitting.

### Performance Metrics:
- **Accuracy**: Up to 88.17% using CART, ~87% using KNN, and **84.6%** using Random Forest.
- **Sensitivity**: The best model exhibited high sensitivity (72.3%) in detecting true positives.
- **Specificity**: Achieved up to 95% specificity, minimizing false positives.

## 🛠️ Technologies Used:
- **R**: For data processing, modeling, and analysis.
- **R Libraries**: `rpart`, `randomForest`, `ggplot2`, `dplyr`, `caret`.

## 🚀 Future Work:
- Integrating more complex models such as ensemble learning to further improve prediction accuracy.
- Expanding the dataset with more diverse demographics for better generalization.
- Testing the models on real-world clinical data to assess their practical applicability.
