import streamlit as st
import pandas as pd
import numpy as np
import os
import analyst
import json
import re
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Personal AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Environment detection
# --------------------------------------------------
IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# Load AWS credentials from Streamlit secrets if available
if "AWS_ACCESS_KEY_ID" in st.secrets:
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
if "AWS_SECRET_ACCESS_KEY" in st.secrets:
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
if "AWS_REGION" in st.secrets:
    os.environ["AWS_REGION"] = st.secrets["AWS_REGION"]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_dataset_context(df: pd.DataFrame) -> str:
    """Build context string with dataset information."""
    return f"""
Rows: {len(df)}
Columns: {list(df.columns)}

Column types:
{df.dtypes.to_string()}

Sample rows:
{df.head(5).to_string(index=False)}
"""

def detect_intent(prompt: str) -> str:
    """Detect user intent from prompt."""
    p = prompt.lower()

    visual_keywords = [
        "plot", "chart", "graph", "visualize",
        "distribution", "trend", "anomaly",
        "outlier", "compare", "breakdown", "show"
    ]

    analysis_keywords = [
        "recommend", "suggest", "improve",
        "action", "insight", "explain",
        "why", "analyze", "identify",
        "gap", "difference", "bias",
        "inequality", "disparity"
    ]

    if any(k in p for k in visual_keywords):
        return "visual"

    if any(k in p for k in analysis_keywords):
        return "analysis"

    return "analysis"

def get_llm_status() -> str:
    """Get LLM status for display."""
    has_creds, status = analyst.check_aws_credentials()
    if has_creds:
        return f"☁️ AWS Bedrock (Amazon Nova Lite)"
    else:
        return "🖥️ Local Ollama"

def generate_visualization(plan: dict, df: pd.DataFrame) -> bool:
    """
    Generate visualization based on LLM plan.
    Returns True if successful, False otherwise.
    """
    try:
        chart_type = plan.get("chart", "bar").lower()
        x = plan.get("x")
        y = plan.get("y")
        group_by = plan.get("group_by")
        agg = plan.get("aggregation", "count").lower()

        # Prepare data
        if group_by and group_by != x:
            plot_df = df.groupby([x, group_by], as_index=False)[y].agg(agg)
        else:
            plot_df = df.groupby([x], as_index=False)[y].agg(agg)

        # Generate chart
        plt.figure(figsize=(10, 6))

        if chart_type == "bar":
            if group_by and group_by != x:
                pivot = plot_df.pivot(index=x, columns=group_by, values=y)
                pivot.plot(kind="bar", ax=plt.gca())
            else:
                plt.bar(plot_df[x].astype(str), plot_df[y])
            plt.xticks(rotation=45, ha='right')

        elif chart_type == "line":
            if group_by and group_by != x:
                for group_val in plot_df[group_by].unique():
                    subset = plot_df[plot_df[group_by] == group_val]
                    plt.plot(subset[x].astype(str), subset[y], marker='o', label=str(group_val))
                plt.legend()
            else:
                plt.plot(plot_df[x].astype(str), plot_df[y], marker='o')
            plt.xticks(rotation=45, ha='right')

        elif chart_type == "scatter":
            if group_by:
                for group_val in df[group_by].unique():
                    subset = df[df[group_by] == group_val]
                    plt.scatter(subset[x], subset[y], label=str(group_val), alpha=0.6)
                plt.legend()
            else:
                plt.scatter(df[x], df[y], alpha=0.6)

        elif chart_type == "histogram":
            plt.hist(df[y].dropna(), bins=30, edgecolor='black')

        elif chart_type == "box":
            if group_by:
                df.boxplot(column=y, by=group_by)
            else:
                plt.boxplot(df[y].dropna())

        plt.title(f"{y} {agg} by {x}")
        plt.xlabel(x)
        plt.ylabel(f"{y} ({agg})")
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.close('all')
        return True

    except Exception as e:
        st.error(f"❌ Failed to generate visualization: {str(e)}")
        return False


# --------------------------------------------------
# Sidebar & Settings
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

# Show LLM status
llm_status = get_llm_status()
st.sidebar.info(f"**LLM Source:** {llm_status}")

# LLM model selection
if "☁️ AWS" in llm_status:
    st.sidebar.caption("🔐 Using AWS Bedrock Amazon Nova Lite (amazon.nova-lite-v1:0)")
    llm_model = "amazon.nova-lite-v1:0"
else:
    llm_model = st.sidebar.text_input("Local LLM model", value="llama3.1", key="llm_model")
    st.sidebar.caption(f"Using: {llm_model}")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Data (CSV / Excel / JSON)",
    type=["csv", "xlsx", "xls", "json"]
)

# Sample dataset button
use_sample = st.sidebar.button("📊 Use Sample Dataset")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --------------------------------------------------
# Load data
# --------------------------------------------------
if use_sample:
    st.session_state.df = pd.DataFrame({
        "department": ["HR", "IT", "IT", "Finance", "HR", "Finance", "IT", "HR"],
        "job_role": ["Manager", "Engineer", "Engineer", "Analyst", "Executive", "Analyst", "Developer", "Recruiter"],
        "gender": ["Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female"],
        "salary": [85000, 95000, 92000, 75000, 120000, 72000, 98000, 68000],
        "age": [34, 29, 41, 36, 28, 45, 31, 26]
    })

elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            st.session_state.df = pd.read_excel(uploaded_file)
        else:
            st.session_state.df = pd.read_json(uploaded_file)
        st.success("✅ File loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

# Check if data is loaded
if st.session_state.df is None:
    st.warning("⚠️ No dataset loaded.")
    st.info("👉 Upload a file or click **Use Sample Dataset** in the sidebar.")
    st.stop()

df = st.session_state.df

# --------------------------------------------------
# Data Preview
# --------------------------------------------------
st.title("🧠 Personal AI Data Analyst")
st.success(f"✅ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")

with st.expander("📋 Preview Data", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
    with col2:
        st.write("**Data Info:**")
        st.write(df.dtypes.astype(str))

# --------------------------------------------------
# Analysis Input
# --------------------------------------------------
st.markdown("---")
st.subheader("🔍 Ask Your Data")

suggestions = [
    "Summarize the dataset in 5 bullet points",
    "What is the gender distribution across departments?",
    "Show salary trends by job role",
    "Analyze correlations between numeric columns",
    "Find anomalies in the dataset",
    "Recommend actionable insights"
]

col1, col2 = st.columns([1, 2])
with col1:
    use_preset = st.checkbox("Use preset prompt", value=True, key="preset_check")

if use_preset:
    prompt = st.selectbox("Choose an analysis:", suggestions, key="preset_select")
else:
    prompt = st.text_area("Or write your own prompt:", height=100, key="custom_prompt")
    if not prompt.strip():
        prompt = suggestions[0]

st.markdown(f"**Selected Prompt:** {prompt}")

# --------------------------------------------------
# Run Analysis
# --------------------------------------------------
if st.button("▶️ Run Analysis"):
    with st.spinner("🔄 Analyzing..."):
        intent = detect_intent(prompt)
        st.session_state.analysis_results = None

        # ============================================
        # ANALYSIS INTENT
        # ============================================
        if intent == "analysis":
            dataset_context = build_dataset_context(df)

            llm_prompt = f"""You are a senior data analyst.

Analyze ONLY the dataset below. Provide clear, data-backed insights.

Dataset:
{dataset_context}

Question:
{prompt}

Format your response with:
- Clear bullet points or sections
- Specific numbers/percentages from the data
- Actionable recommendations where applicable
"""

            response = analyst.ask_llm(llm_prompt, model=llm_model)

            # Check for errors
            if analyst.is_error_response(response):
                st.error("❌ LLM Error")
                st.code(response, language="plaintext")
            else:
                st.session_state.analysis_results = ("text", response)
                st.subheader("📊 Analysis Results")
                st.markdown(response)

        # ============================================
        # VISUAL INTENT
        # ============================================
        elif intent == "visual":
            dataset_context = build_dataset_context(df)

            plan_prompt = f"""You are a data visualization expert.

Based ONLY on the dataset and question, determine the best visualization.

Dataset:
{dataset_context}

User question:
{prompt}

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "chart": "bar|line|scatter|histogram|box",
  "x": "column_name",
  "y": "column_name",
  "group_by": "column_name_or_null",
  "aggregation": "count|sum|mean|median|min|max|std",
  "reasoning": "brief explanation"
}}
"""

            response = analyst.ask_llm(plan_prompt, model=llm_model)

            # Check for errors
            if analyst.is_error_response(response):
                st.error("❌ LLM Error")
                st.code(response, language="plaintext")
                st.stop()

            # Extract JSON
            plan = analyst.extract_json_from_response(response)
            if not plan:
                st.error("❌ Could not parse visualization plan")
                st.info("**LLM Response:**")
                st.code(response, language="plaintext")
                st.stop()

            # Validate plan
            is_valid, error_msg = analyst.validate_visualization_plan(plan, df)
            if not is_valid:
                st.error(f"❌ Invalid visualization plan: {error_msg}")
                st.json(plan)
                st.stop()

            # Generate visualization
            st.subheader("📈 Visualization")
            st.caption(f"Reasoning: {plan.get('reasoning', 'N/A')}")
            
            success = generate_visualization(plan, df)
            
            if success:
                st.session_state.analysis_results = ("visual", plan)
                
                # Show data used
                with st.expander("📋 Data Used", expanded=False):
                    x = plan.get("x")
                    y = plan.get("y")
                    group = plan.get("group_by")
                    agg = plan.get("aggregation", "count")
                    
                    if group and group != x:
                        show_df = df.groupby([x, group], as_index=False)[y].agg(agg)
                    else:
                        show_df = df.groupby([x], as_index=False)[y].agg(agg)
                    
                    st.dataframe(show_df)

# --------------------------------------------------
# History Section
# --------------------------------------------------
st.markdown("---")

if st.session_state.analysis_results:
    result_type, result_data = st.session_state.analysis_results
    
    if result_type == "text":
        st.subheader("💾 Last Analysis")
        st.markdown(result_data)
    elif result_type == "visual":
        st.subheader("💾 Last Visualization")
        st.json(result_data)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔐 Secure & Privacy-first")
with col2:
    st.caption("⚡ Powered by AWS Bedrock + Nova")
with col3:
    st.caption("📈 AI Data Analysis Made Easy")
