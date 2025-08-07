import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import io
import os
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("')
# Streamlit config
st.set_page_config(page_title="📊 Data Analyst AI Agent", layout="wide")
st.title("📊 Data Analyst AI Agent")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# === Functions ===

# Load CSV/Excel
def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# AI Summary Function
def generate_ai_summary(df):
    prompt = f"""
You are a data analyst. Provide a detailed summary of this dataset.

Summary Statistics:
{df.describe().to_string()}

Column Types:
{df.dtypes.to_string()}

Sample Rows:
{df.head(3).to_string()}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ OpenAI API Error: {e}"

# AI Q&A Function
def answer_user_question(df, question):
    prompt = f"""
You are a data analyst. Based on the following dataset, answer the user's question.

Dataset Columns and Types:
{df.dtypes.to_string()}

First few rows of data:
{df.head(10).to_string()}

User's Question:
{question}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ OpenAI API Error: {e}"

# === Main App Logic ===
if uploaded_file:
    try:
        df = read_file(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Header with file name
        st.header(f"📄 Data from: {uploaded_file.name}")

        # Clean data
        df_cleaned = df.drop_duplicates().fillna(method='ffill')
        st.subheader("🧹 Cleaned Data")
        st.dataframe(df_cleaned)

        # Column selector
        selected_columns = st.multiselect(
            "📌 Select columns for analysis",
            options=df_cleaned.columns.tolist(),
            default=df_cleaned.columns.tolist()
        )
        df_filtered = df_cleaned[selected_columns]

        # Optional chart
        show_chart = st.checkbox("📊 Generate a Chart", value=False)
        if show_chart:
            st.sidebar.markdown("### Chart Options")
            numeric_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_columns = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

            chart_type = st.sidebar.selectbox("Select Chart Type", ["Scatter", "Line", "Bar"])

            if chart_type == "Bar":
                x_axis = st.sidebar.selectbox("X-axis (Categorical)", options=categorical_columns)
                y_axis = st.sidebar.selectbox("Y-axis (Numeric)", options=numeric_columns)
            else:
                x_axis = st.sidebar.selectbox("X-axis", options=numeric_columns)
                y_axis = st.sidebar.selectbox("Y-axis", options=numeric_columns)

            st.subheader("📈 Data Visualization")
            fig, ax = plt.subplots()
            try:
                if chart_type == "Scatter":
                    sns.scatterplot(data=df_filtered, x=x_axis, y=y_axis, ax=ax)
                elif chart_type == "Line":
                    sns.lineplot(data=df_filtered, x=x_axis, y=y_axis, ax=ax)
                elif chart_type == "Bar":
                    sns.barplot(data=df_filtered, x=x_axis, y=y_axis, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Chart Error: {e}")

        # === AI Summary ===
        st.subheader("🧠 AI-Generated Summary")
        summary_text = generate_ai_summary(df_filtered)
        st.write(summary_text)

        st.download_button(
            label="📄 Download Summary (TXT)",
            data=summary_text,
            file_name="ai_summary.txt",
            mime="text/plain"
        )

        # === Chat-style Q&A ===
        st.subheader("💬 Ask Questions About Your Data")
        with st.expander("Open Chat with Your Dataset"):
            user_question = st.text_input("Type your question here...")
            if user_question:
                answer = answer_user_question(df_filtered, user_question)
                st.success("Answer:")
                st.write(answer)

        # === Download Cleaned Data ===
        st.subheader("📥 Download Cleaned Data")
        csv_buffer = io.StringIO()
        df_filtered.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV or Excel file to begin.")
