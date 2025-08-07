
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Analyst AI Agent", layout="wide")
st.title("📊 Data Analyst AI Agent")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    st.subheader("🧹 Cleaned Data")
    df_cleaned = df.drop_duplicates().fillna(method='ffill')
    st.dataframe(df_cleaned)

    st.subheader("📈 Basic Statistics")
    st.write(df_cleaned.describe())

    st.subheader("📊 Correlation Heatmap")
    if not df_cleaned.select_dtypes(include='number').empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")
else:
    st.info("Please upload a CSV file to begin analysis.")
