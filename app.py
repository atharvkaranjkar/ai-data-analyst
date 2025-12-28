import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import analyst

# --------------------------------------------------
# Basic Streamlit setup
# --------------------------------------------------
st.set_page_config(
    page_title="Personal AI Data Analyst",
    layout="wide"
)

st.title("🧠 Personal AI Data Analyst")

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")

if IS_CLOUD:
    st.sidebar.info("Local LLM (Ollama) is disabled on Streamlit Cloud.")
    use_llm = False
else:
    use_llm = st.sidebar.checkbox(
        "Use local LLM (Ollama) for custom prompts",
        value=False
    )

llm_model = st.sidebar.text_input(
    "LLM model name",
    value="llama3.1"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "xls", "json"]
)

use_sample = st.sidebar.button("Use sample dataset")

# --------------------------------------------------
# Load data
# --------------------------------------------------
if use_sample:
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10),
        "category": ["A", "B", "A", "C", "B", "A", "B", "C", "A", "B"],
        "value": [10, 15, 7, 12, 20, 5, 8, 9, 14, 11]
    })
else:
    if uploaded_file is None:
        st.info("Upload a dataset or click **Use sample dataset**.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

st.success("Dataset loaded successfully")

with st.expander("Preview data"):
    st.dataframe(df.head(100))

# --------------------------------------------------
# Suggested prompts (simple + safe)
# --------------------------------------------------
suggestions = [
    "Summarize the dataset",
    "Show summary statistics for numeric columns",
    "Show top 10 counts for a categorical column",
    "Create a histogram of a numeric column"
]

prompt = st.selectbox(
    "Choose an analysis",
    suggestions
)

custom_prompt = st.text_area(
    "Or write your own prompt (advanced)",
    height=80
)

final_prompt = custom_prompt.strip() if custom_prompt.strip() else prompt

st.markdown("### Prompt")
st.write(final_prompt)

# --------------------------------------------------
# Run analysis
# --------------------------------------------------
if st.button("Run analysis"):
    with st.spinner("Running analysis..."):

        # 1️⃣ Try deterministic (rule-based) analysis
        code = None
        if "summarize" in final_prompt.lower():
            result = {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Numeric columns": df.select_dtypes(include="number").columns.tolist()
            }
            st.json(result)
            st.stop()

        if "summary statistics" in final_prompt.lower():
            st.dataframe(df.describe())
            st.stop()

        # 2️⃣ If LLM requested
        if use_llm:
            raw_prompt = (
                "Return ONLY Python code inside a ```python``` block.\n"
                "DataFrame is named df.\n\n"
                f"User prompt: {final_prompt}"
            )

            llm_out = analyst.ask_llm(raw_prompt, model=llm_model)

            if llm_out is None:
                st.warning(
                    "Local LLM is not available. "
                    "Run the app locally with Ollama installed."
                )
                st.stop()

            if llm_out.startswith("[LLM-"):
                st.error("LLM error:")
                st.write(llm_out)
                st.stop()

            if "```python" not in llm_out:
                st.error("LLM did not return Python code.")
                st.write(llm_out)
                st.stop()

            try:
                code = llm_out.split("```python")[1].split("```")[0]
                res = analyst.run_code(df, code)
            except Exception as e:
                st.error(f"Failed to execute LLM code: {e}")
                st.stop()

        else:
            st.error(
                "This prompt requires an LLM. "
                "Enable **Use local LLM** when running locally."
            )
            st.stop()

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------
    if res["type"] == "text":
        st.text(res["output"])
    elif res["type"] == "dataframe":
        st.dataframe(res["df"])
    elif res["type"] == "image":
        st.image(res["path"])
    else:
        st.write(res)
