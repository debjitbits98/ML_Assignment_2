import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("💰 Adult Income Prediction using Machine Learning")

st.markdown("""
This application predicts whether a person's annual income is:

- **<=50K**
- **>50K**

using the **Random Forest Classifier** trained on the Adult Income Dataset.
""")

# ---------------------------------------------------
# Expected Columns
# ---------------------------------------------------
expected_columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

# ---------------------------------------------------
# Upload CSV
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Encoded test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        # ---------------------------------------
        # Validate Columns
        # ---------------------------------------
        if list(df.columns) != expected_columns:

            st.error("❌ Uploaded CSV does not match the required format.")

            st.write("Expected Columns:")

            st.write(expected_columns)

            st.stop()

        st.success("Dataset uploaded successfully.")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.write("**Total Records:**", len(df))

        # ---------------------------------------
        # Predict Button
        # ---------------------------------------
        if st.button("Predict"):

            X = df.drop("income", axis=1)

            prediction = model.predict(X)

            output = df.copy()

            output["Predicted Income"] = prediction

            output["Predicted Income"] = output["Predicted Income"].replace({
                0: "<=50K",
                1: ">50K"
            })

            st.success("Prediction Completed Successfully!")

            st.subheader("Prediction Results")

            st.dataframe(output)

            # ---------------------------------------
            # Prediction Summary
            # ---------------------------------------
            st.subheader("Prediction Summary")

            st.write(
                output["Predicted Income"].value_counts()
            )

            # ---------------------------------------
            # Download CSV
            # ---------------------------------------
            csv = output.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction CSV",
                data=csv,
                file_name="prediction_output.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Error : {e}")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.header("Project Information")

st.markdown("""
### Dataset
Adult Income Dataset (UCI Machine Learning Repository)

### Machine Learning Models Implemented
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

### Best Performing Model
**Random Forest Classifier**

### Performance

- Accuracy : **86.28%**
- Precision : **74.83%**
- Recall : **64.86%**
- F1 Score : **69.49%**
- ROC-AUC : **78.97%**
- MCC : **0.6096**

This application was developed as part of the Machine Learning Assignment.
""")
