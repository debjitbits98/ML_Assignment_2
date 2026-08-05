import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("💰 Adult Income Prediction System")

st.markdown("""
This application predicts whether an individual's annual income is:

- **<=50K**
- **>50K**

using Machine Learning models trained on the Adult Income Dataset.
""")

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    models = {

        "Random Forest":
            joblib.load("models/random_forest.pkl"),

        "Logistic Regression":
            joblib.load("models/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("models/decision_tree.pkl"),

        "KNN":
            joblib.load("models/knn.pkl"),

        "Naive Bayes":
            joblib.load("models/naive_bayes.pkl")

    }

    scaler = joblib.load("models/scaler.pkl")

    encoders = joblib.load("models/encoders.pkl")

    features = joblib.load("models/features.pkl")

    return models, scaler, encoders, features


try:

    models, scaler, encoders, features = load_models()
    
    metrics = joblib.load("models/metrics.pkl")
    reports = joblib.load("models/classification_reports.pkl")

except Exception as e:

    st.error("Unable to load model files.")

    st.exception(e)

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Settings")

selected_model = st.sidebar.selectbox(

    "Choose Classification Model",

    [

        "Random Forest",

        "Logistic Regression",

        "Decision Tree",

        "KNN",

        "Naive Bayes"

    ]

)
st.sidebar.success(f"Selected Model\n\n{selected_model}")

uploaded_file = st.sidebar.file_uploader(

    "Upload CSV File",

    type=["csv"]

)

st.sidebar.markdown("---")

st.sidebar.success("Notebook Verified ✔")

st.sidebar.info(
"""
Models Available

• Logistic Regression

• Decision Tree

• KNN

• Naive Bayes

• Random Forest
"""
)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def preprocess(df):

    df = df.copy()

    if "income" in df.columns:
        df = df.drop(columns=["income"])

    df = df[features]

    return df
# ==========================================================
# MAIN APPLICATION
# ==========================================================

if uploaded_file is None:

    st.info("👈 Upload test_data.csv from the sidebar to begin.")

    st.stop()


try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read uploaded CSV.")

    st.exception(e)

    st.stop()


st.header("Dataset Preview")

st.dataframe(data.head(), use_container_width=True)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric("Rows", data.shape[0])

with col2:

    st.metric("Columns", data.shape[1])

with col3:

    st.metric("Selected Model", selected_model)


# ==========================================================
# PREPROCESS
# ==========================================================

try:

    X = preprocess(data)

except Exception as e:

    st.error(e)

    st.stop()


# ==========================================================
# MODEL PREDICTION
# ==========================================================

model = models[selected_model]

scaled_models = [

    "Logistic Regression",

    "Decision Tree",

    "KNN",

    "Naive Bayes"

]


if selected_model in scaled_models:

    X_input = scaler.transform(X)

else:

    X_input = X


    prediction = model.predict(X_input)

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(X_input)

    else:

        probability = None

result = data.copy()

if "income" in result.columns:
    result = result.drop(columns=["income"])

result["Prediction"] = np.where(

    prediction == 1,

    ">50K",

    "<=50K"

)
# ==========================================================
# DISPLAY RESULTS
# ==========================================================

st.markdown("---")
st.header("Prediction Results")

st.dataframe(result, use_container_width=True)

# ==========================================================
# PREDICTION SUMMARY
# ==========================================================

st.subheader("Prediction Summary")

summary = result["Prediction"].value_counts()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted >50K",
        int(summary.get(">50K", 0))
    )

with col2:

    st.metric(
        "Predicted <=50K",
        int(summary.get("<=50K", 0))
    )

# ==========================================================
# PREDICTION PROBABILITIES
# ==========================================================

if probability is not None:

    st.subheader("Prediction Probability")

    if probability.shape[1] == 2:

        result["Probability_<=50K"] = probability[:, 0]

        result["Probability_>50K"] = probability[:, 1]

        st.dataframe(

            result[
                [
                    "Prediction",
                    "Probability_<=50K",
                    "Probability_>50K"
                ]
            ].head(),

            use_container_width=True

        )

# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

csv = result.to_csv(index=False).encode("utf-8")

st.download_button(

    label="📥 Download Prediction Results",

    data=csv,

    file_name="predictions.csv",

    mime="text/csv"

)

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown("---")

st.header("Selected Model")

st.subheader("Evaluation Metrics")

metric_df = pd.DataFrame(
    metrics[selected_model],
    index=["Value"]
).T

st.table(metric_df)

st.subheader("Classification Report")

st.text(reports[selected_model])

model_info = {

    "Random Forest":
        """
        Ensemble model using multiple decision trees.
        Best performing model in the notebook.
        """,

    "Logistic Regression":
        """
        Linear classifier suitable for binary classification.
        Uses StandardScaler.
        """,

    "Decision Tree":
        """
        Tree-based supervised classifier.
        Uses StandardScaler as in the notebook.
        """,

    "KNN":
        """
        Distance-based classifier.
        Uses StandardScaler.
        """,

    "Naive Bayes":
        """
        Gaussian Naive Bayes classifier.
        Uses StandardScaler.
        """

}

st.info(model_info[selected_model])
# ==========================================================
# MODEL FILES STATUS
# ==========================================================

st.markdown("---")
st.header("Project Information")

required_files = [
    "models/logistic_regression.pkl",
    "models/decision_tree.pkl",
    "models/knn.pkl",
    "models/naive_bayes.pkl",
    "models/random_forest.pkl",
    "models/scaler.pkl",
    "models/encoders.pkl",
    "models/features.pkl"
]

status = []

for file in required_files:
    if os.path.exists(file):
        status.append([os.path.basename(file), "Available"])
    else:
        status.append([os.path.basename(file), "Missing"])

status_df = pd.DataFrame(
    status,
    columns=["File", "Status"]
)

st.dataframe(status_df, use_container_width=True)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.subheader("Dataset")

st.write("Adult Income Dataset (UCI Machine Learning Repository)")

st.write("Target Variable : income")

st.write("Classification : Binary")

st.write("Total Features : 14")

# ==========================================================
# MODELS USED
# ==========================================================

st.subheader("Models Implemented")

st.markdown("""
- Logistic Regression

- Decision Tree

- K-Nearest Neighbors

- Gaussian Naive Bayes

- Random Forest
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.success("Application Loaded Successfully")

st.caption(
"""
Machine Learning Assignment 2

Income Classification using Multiple Machine Learning Models

Developed using

• Python

• Scikit-Learn

• Pandas

• Streamlit
"""
)

