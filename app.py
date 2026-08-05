import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Adult Income Prediction using Machine Learning")

st.write("""
This application predicts whether a person's annual income is:

- <=50K
- >50K

using the trained Random Forest Classifier.
""")

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# -------------------------------------------------------
# Upload CSV
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Encoded test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Drop target column if present
    if "income" in df.columns:
        df.drop(columns=["income"], inplace=True)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Total Records:", len(df))

    if st.button("Predict"):

        prediction = model.predict(df)

        output = df.copy()

        output["Predicted Income"] = [
            ">50K" if x == 1 else "<=50K"
            for x in prediction
        ]

        st.success("Prediction Completed Successfully!")

        st.subheader("Prediction Results")
        st.dataframe(output)

        st.subheader("Prediction Summary")

        summary = (
            output["Predicted Income"]
            .value_counts()
            .rename_axis("Predicted Income")
            .reset_index(name="Count")
        )

        st.table(summary)

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction CSV",
            data=csv,
            file_name="prediction_output.csv",
            mime="text/csv"
        )
