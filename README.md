# Machine Learning Assignment 2
## Adult Income Prediction using Machine Learning

---

## Project Overview

This project predicts whether an individual's annual income is **greater than \$50K** or **less than or equal to \$50K** using the Adult Income (Census Income) dataset. Multiple machine learning classification algorithms are implemented, evaluated, and compared based on various performance metrics.

The objective is to analyze the dataset, preprocess the data, build multiple classification models, compare their performance, and identify the best-performing model.

---

## Dataset

**Dataset Name:** Adult Income Dataset (Census Income Dataset)

Files used:

- adult.data

The dataset contains demographic and employment-related information such as:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

Target Variable:

- Income
    - <=50K
    - >50K

---

## Machine Learning Workflow

The project follows the standard Machine Learning pipeline:

1. Data Loading
2. Data Exploration
3. Data Cleaning
4. Handling Missing Values
5. Encoding Categorical Variables
6. Feature Engineering
7. Train-Test Split
8. Model Building
9. Model Evaluation
10. Model Comparison
11. Selection of Best Model

---

## Machine Learning Algorithms Implemented

The following classification algorithms were implemented:

### 1. Logistic Regression

- Simple linear classifier
- Fast training
- Strong baseline model

---

### 2. Decision Tree Classifier

- Tree-based supervised learning algorithm
- Easy to interpret
- Handles nonlinear relationships

---

### 3. K-Nearest Neighbors (KNN)

- Distance-based classifier
- Classifies using nearest neighbors
- Performs well with scaled features

---

### 4. Naive Bayes

- Probabilistic classifier
- Based on Bayes' Theorem
- Fast and efficient

---

### 5. Random Forest Classifier

- Ensemble learning method
- Uses multiple decision trees
- Produces robust predictions
- Best performing model in this project

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Classification Report

---

## Model Performance Summary

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 82.78% |
| Decision Tree | 81.42% |
| K-Nearest Neighbors | 83.50% |
| Naive Bayes | 81.02% |
| Random Forest | **87.31%** |

---

## Best Model

The **Random Forest Classifier** achieved the highest overall performance.

Performance Metrics:

- Accuracy : 87.31%
- Precision : 78.58%
- Recall : 65.05%
- F1 Score : 71.17%
- ROC-AUC : 79.71%
- MCC : 63.95%

The Random Forest model outperformed all other classifiers in terms of overall accuracy, F1 Score, and Matthews Correlation Coefficient, making it the most reliable model for this classification task.

---

## Repository Structure

```
Machine-Learning-Assignment-2/

│── model.ipynb
│── app.py
│── requirements.txt
│── test_data.csv
│── README.md
│── models/
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    random_forest.pkl
    scaler.pkl
    encoders.pkl
    features.pkl
    metrics.pkl
    classification_reports.pkl
```

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## How to Run

### Clone Repository

```bash
git clone <repository_link>
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

Open:

```
model.ipynb
```

Run all cells.

---

## Streamlit Application

To launch the web application:

```bash
streamlit run app.py
```

---

## Results

The project successfully demonstrates the complete machine learning workflow from preprocessing to model comparison.

Among all implemented models, the Random Forest Classifier achieved the highest performance and was selected as the final prediction model.

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Feature selection
- Ensemble stacking methods
- XGBoost implementation
- LightGBM implementation
- Model deployment on cloud platforms

---
