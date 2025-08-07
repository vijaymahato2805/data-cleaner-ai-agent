import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Set Streamlit config
st.set_page_config(page_title="📊 Data Analyst AI Agent", layout="wide")

# Title
st.title("📊 Data Analyst AI Agent")

# Sidebar: OpenAI API Key
openai_api_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password")

# Sidebar: Upload file (CSV/Excel)
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Function to read file
def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Function to generate AI summary
def generate_ai_summary(df, api_key):
    openai.api_key = api_key

    prompt = f"""
You are a data analyst. Provide a professional summary of the dataset below.

Summary Statistics:
{df.describe().to_string()}

Column Types:
{df.dtypes.to_string()}

Sample Rows:
{df.head(3).to_string()}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Function to answer natural language questions about the data
def answer_question(question, df, api_key):
    openai.api_key = api_key
    prompt = f"""
You are a data expert. Answer the question based on this dataset.

Dataset Sample:
{df.head(10).to_string()}

Question:
{question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Function to export summary to PDF
def export_summary_to_pdf(summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    flowables = [Paragraph(summary_text, styles["Normal"])]
    doc.build(flowables)
    buffer.seek(0)
    return buffer

# If file is uploaded
if uploaded_file:
    try:
        # Load file
        df = read_file(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Clean data
        df_cleaned = df.drop_duplicates().fillna(method='ffill')

        # Preview
        st.subheader("📄 Cleaned Data")
        st.dataframe(df_cleaned)

        # Column selector
        selected_columns = st.multiselect("📌 Select columns for analysis", df_cleaned.columns.tolist(), default=df_cleaned.columns.tolist())
        df_filtered = df_cleaned[selected_columns]

        # Chart options
        st.sidebar.markdown("### 📊 Chart Options")
        numeric_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
        chart_type = st.sidebar.selectbox("Select Chart Type", ["Scatter", "Line", "Bar"])

        if chart_type == "Bar":
            x_axis = st.sidebar.selectbox("X-axis (Categorical)", categorical_columns)
            y_axis = st.sidebar.selectbox("Y-axis (Numeric)", numeric_columns)
        else:
            x_axis = st.sidebar.selectbox("X-axis", numeric_columns)
            y_axis = st.sidebar.selectbox("Y-axis", numeric_columns)

        # Render chart
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
            st.error(f"Chart Error: {e}")

        # AI summary
        st.subheader("🧠 AI Summary")
        if openai_api_key:
            summary_text = generate_ai_summary(df_filtered, openai_api_key)
            st.write(summary_text)

            # Export to text
            st.download_button("📄 Download Summary (TXT)", summary_text, file_name="ai_summary.txt", mime="text/plain")

            # Export to PDF
            pdf_file = export_summary_to_pdf(summary_text)
            st.download_button("📄 Download Summary (PDF)", pdf_file, file_name="ai_summary.pdf", mime="application/pdf")
        else:
            st.info("Enter your OpenAI API key to get AI summary.")

        # Natural language question
        st.subheader("💬 Ask a Question About Your Data")
        if openai_api_key:
            user_question = st.text_input("Ask your question (e.g., 'What is the average revenue?')")
            if user_question:
                answer = answer_question(user_question, df_filtered, openai_api_key)
                st.success("Answer:")
                st.write(answer)
        else:
            st.info("Enter your OpenAI API key to ask questions.")

        # Download cleaned data
        st.subheader("📥 Download Cleaned Data")
        csv_buffer = io.StringIO()
        df_filtered.to_csv(csv_buffer, index=False)
        st.download_button("Download CSV", csv_buffer.getvalue(), file_name="cleaned_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a CSV or Excel file to begin.")
