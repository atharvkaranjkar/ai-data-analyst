import streamlit as st
import pandas as pd
import numpy as np
import os
import analyst
import json
import re

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Personal AI Data Analyst",
    layout="wide"
)

# --------------------------------------------------
# Environment detection
# --------------------------------------------------
IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_dataset_context(df: pd.DataFrame) -> str:
    return f"""
Rows: {len(df)}
Columns: {list(df.columns)}

Column types:
{df.dtypes.to_string()}

Sample rows:
{df.head(5).to_string(index=False)}
"""

def detect_intent(prompt: str) -> str:
    p = prompt.lower()

    visual_keywords = [
        "plot", "chart", "graph", "visualize",
        "distribution", "trend", "anomaly",
        "outlier", "compare", "breakdown"
    ]

    analysis_keywords = [
        "recommend", "suggest", "improve",
        "action", "insight", "explain",
        "why", "analyze"
    ]

    if any(k in p for k in visual_keywords):
        return "visual"

    if any(k in p for k in analysis_keywords):
        return "analysis"

    return "simple"

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🧠 Personal AI Data Analyst")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")

if IS_CLOUD:
    st.sidebar.info("☁️ Using AWS / Cloud LLM")
    use_llm = True
else:
    use_llm = st.sidebar.checkbox(
        "Use local LLM (Ollama)",
        value=False
    )

llm_model = st.sidebar.text_input("LLM model", value="llama3.1")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "xls", "json"]
)

use_sample = st.sidebar.button("Use sample dataset")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "llm_output" not in st.session_state:
    st.session_state.llm_output = None

# --------------------------------------------------
# Load data
# --------------------------------------------------
if use_sample:
    st.session_state.df = pd.DataFrame({
        "department": ["HR", "IT", "IT", "Finance", "HR", "Finance"],
        "job_role": ["Manager", "Engineer", "Engineer", "Analyst", "Executive", "Analyst"],
        "gender": ["Female", "Male", "Female", "Male", "Female", "Female"],
        "age": [34, 29, 41, 36, 28, 45]
    })

elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            st.session_state.df = pd.read_excel(uploaded_file)
        else:
            st.session_state.df = pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")

if st.session_state.df is None:
    st.warning("⚠️ No dataset loaded.")
    st.info("Please upload a dataset or click **Use sample dataset**.")
    st.stop()

df = st.session_state.df
st.success("Dataset loaded successfully")

with st.expander("Preview data"):
    st.dataframe(df.head(100))

# --------------------------------------------------
# Prompt input
# --------------------------------------------------
suggestions = [
    "Summarize the dataset",
    "What is the gender distribution across departments and job roles?",
    "Analyze trends in the data",
    "Detect and visualize anomalies",
    "Recommend actions to improve system reliability"
]

prompt = st.selectbox("Choose an analysis", suggestions)
custom_prompt = st.text_area("Or write your own prompt", height=80)
final_prompt = custom_prompt.strip() if custom_prompt.strip() else prompt

st.markdown("### Prompt")
st.write(final_prompt)

# --------------------------------------------------
# Run analysis
# --------------------------------------------------
if st.button("Run analysis"):
    with st.spinner("Running analysis..."):

        intent = detect_intent(final_prompt)
        st.session_state.llm_output = None

        # -----------------------------
        # SIMPLE
        # -----------------------------
        if intent == "simple":
            st.json({
                "rows": len(df),
                "columns": list(df.columns),
            })

        # -----------------------------
        # VISUAL (LLM → JSON PLAN)
        # -----------------------------
        elif intent == "visual":

            dataset_context = build_dataset_context(df)

            plan_prompt = f"""
You are a data analyst.

Based ONLY on the dataset and question, return a visualization plan.

Dataset:
{dataset_context}

User question:
{final_prompt}

Return ONLY valid JSON in this format:
{{
  "chart": "bar | line",
  "x": "column_name",
  "y": "column_name",
  "group_by": "column_name_or_null",
  "aggregation": "count | mean | sum"
}}
"""

            llm_out = analyst.ask_llm(plan_prompt, model=llm_model)

            # ---- SAFE JSON EXTRACTION ----
            match = re.search(r"\{.*\}", llm_out or "", re.DOTALL)
            if not match:
                st.error("LLM did not return a valid JSON plan.")
                st.code(llm_out)
                st.stop()

            try:
                plan = json.loads(match.group())
            except Exception as e:
                st.error("Failed to parse visualization plan.")
                st.code(llm_out)
                st.stop()

            x = plan.get("x")
            y = plan.get("y")
            group = plan.get("group_by")
            agg = plan.get("aggregation", "count")

            if x not in df.columns or y not in df.columns:
                st.error("Invalid columns in visualization plan.")
                st.json(plan)
                st.stop()

            # ---- SMART AGGREGATION LOGIC ----
            if df[y].dtype == "object":
                plot_df = df.groupby(x)[y].count()
                st.bar_chart(plot_df)

            elif group and group != x:
                plot_df = (
                    df.groupby([x, group])[y]
                    .agg(agg)
                    .reset_index()
                )
                st.bar_chart(plot_df, x=x, y=y)

            else:
                plot_df = df.groupby(x)[y].agg(agg)
                st.bar_chart(plot_df)

            st.caption("📊 Auto-generated chart based on dataset semantics")

        # -----------------------------
        # ANALYSIS / RECOMMENDATIONS
        # -----------------------------
        else:
            dataset_context = build_dataset_context(df)

            llm_prompt = f"""
You are a senior data analyst.

Use ONLY the dataset below.

{dataset_context}

Question:
{final_prompt}

Provide clear, data-backed insights.
"""

            llm_out = analyst.ask_llm(llm_prompt, model=llm_model)

            if llm_out:
                st.session_state.llm_output = llm_out
            else:
                st.warning("No response from LLM.")

# --------------------------------------------------
# Persisted output
# --------------------------------------------------
if st.session_state.llm_output:
    st.subheader("🔍 AI Recommendations")
    st.caption("☁️ Response from AWS")
    st.markdown(st.session_state.llm_output)
