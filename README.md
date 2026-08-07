# Machine Learning Assignment 2
# Adult Income Prediction using Multiple Machine Learning Models

---

## Student Details

- **Name:** DEBJIT BISWAS
- **BITS ID:** 2025DA04158
- **Programme:** M.Tech (DSE)
- **Course:** Machine Learning
- **Assignment:** Assignment 2

---

# Problem Statement

The objective of this project is to predict whether an individual's annual income is greater than \$50K or less than or equal to \$50K using supervised machine learning classification algorithms. Multiple classification models are implemented, evaluated, compared, and deployed through a Streamlit web application.

---

# Dataset Description

**Dataset Name:** Adult Income Dataset (Census Income Dataset)

**Source:** UCI Machine Learning Repository

Dataset Characteristics:

- Number of Features: 14
- Training Dataset: 32,561 instances
- Classification Type: Binary Classification

Target Variable:

- <=50K
- >50K

Important Features:

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

---

# GitHub Repository

https://github.com/debjitbits98/ML_Assignment_2

---

# Streamlit Application

https://mlassignment2-jgx6tc89gxy39zqlcu4opa.streamlit.app/

---

# Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Cleaning
4. Missing Value Handling
5. Feature Encoding
6. Feature Selection
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Streamlit Deployment

---

# Machine Learning Models Implemented

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

---

# Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------:|----:|----------:|-------:|---------:|----:|
| Logistic Regression | 82.78% | 85.67% | 72.42% | 46.05% | 56.30% | 48.06% |
| Decision Tree | 81.42% | 74.68% | 61.40% | 61.67% | 61.53% | 49.29% |
| KNN | 83.50% | 75.63% | 67.62% | 60.46% | 63.84% | 53.33% |
| Naive Bayes | 81.02% | 65.86% | 70.43% | 36.61% | 48.17% | 41.00% |
| Random Forest | **87.31%** | 79.71% | 78.58% | 65.05% | 71.17% | 63.60% |

---

# Model Observations

### Logistic Regression
- Good baseline classifier.
- Performs well on linearly separable patterns.
- Fast training and prediction.

### Decision Tree
- Easy to interpret.
- Captures nonlinear relationships.
- Can overfit if not controlled.

### K-Nearest Neighbors
- Performs well after feature scaling.
- Sensitive to the value of K.
- Better than Logistic Regression on this dataset.

### Naive Bayes
- Fast probabilistic classifier.
- Assumes feature independence.
- Lower overall accuracy than KNN and Random Forest.

### Random Forest
- Highest overall accuracy.
- Strongest F1 Score.
- Best MCC.
- Most robust model for this dataset.

---

# Overall Winner

**Random Forest Classifier**

Reasons:

- Highest Accuracy
- Highest F1 Score
- Best MCC
- Strong overall classification performance

---

# Evaluation Metrics Used

- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient
- Classification Report
- Confusion Matrix

---

# Repository Structure

```text
ML_Assignment_2/
│
├── app.py
├── model.ipynb
├── README.md
├── requirements.txt
├── test_data.csv
├── adult.data
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── features.pkl
    ├── metrics.pkl
    └── classification_reports.pkl
```

---

# Technologies Used

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

# How to Run

```bash
git clone https://github.com/debjitbits98/ML_Assignment_2
cd ML_Assignment_2
pip install -r requirements.txt
streamlit run app.py
```

---

# Results

The project successfully demonstrates an end-to-end machine learning workflow, from preprocessing and model training to deployment through Streamlit.

Among all implemented models, **Random Forest** achieved the best performance and was selected as the final model.

The Streamlit web application allows users to upload test data, select any of the five implemented models, view evaluation metrics and classification reports, and download prediction results.

---

# Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Feature selection
- XGBoost implementation
- LightGBM implementation
- Ensemble stacking
- Docker deployment
