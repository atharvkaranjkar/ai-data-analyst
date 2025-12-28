import streamlit as st
import analyst
import importlib
import traceback
import textwrap
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("🧠 Personal AI Data Analyst — Interactive Dashboard")

st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Use local LLM (ollama) for custom prompts", value=False)
llm_model = st.sidebar.text_input("LLM model name (ollama)", value="llama3.1")
st.sidebar.markdown("If you don't have `ollama` installed, leave this off and use built-in prompts.")

# File uploader (moved to sidebar) + sample dataset button for quick demos
uploaded = st.sidebar.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xls","xlsx","json"])
use_sample = st.sidebar.button("Use sample dataset")

if use_sample:
    # small example dataset so the UI can be explored without uploading files
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "category": ["A","B","A","C","B","A","B","C","A","B"],
        "value": [10,15,7,12,20,5,8,9,14,11]
    })
else:
    if uploaded is None:
        st.info("Upload a CSV / XLSX / JSON to get started. Suggestions will appear automatically.")
        st.stop()
    # Load data (use analyst.load_data if available, otherwise fallback)
    def _safe_load(fp):
        loader = getattr(analyst, "load_data", None)
        if callable(loader):
            return loader(fp)
        # fallback: try to read with pandas directly
        try:
            # if fp is a path string
            if isinstance(fp, (str,)):
                return pd.read_csv(fp)
            # if it's a Streamlit UploadedFile or file-like
            name = getattr(fp, "name", None)
            raw = fp.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            bio = io.BytesIO(raw)
            # try csv then json
            try:
                bio.seek(0)
                return pd.read_csv(bio)
            except Exception:
                bio.seek(0)
                return pd.read_json(bio)
        except Exception:
            raise

    try:
        df = _safe_load(uploaded)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

st.success("File loaded.")
with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))

# simple fallback prompt->code mapper (used if analyst.prompt_to_code missing)
def _prompt_to_code(prompt: str, df: pd.DataFrame):
    p = prompt.strip().lower()
    # Summary
    if p.startswith("summarize the dataset"):
        return 'result = "Rows: %d, Columns: %d" % (len(df), len(df.columns))'
    # top counts for categorical
    if "top 10 counts" in p and "'" in prompt:
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            return textwrap.dedent(f"""
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)
    # numeric describe
    if "summary statistics" in p or "describe" in p:
        return "result = df.select_dtypes(include=['number']).describe().T"
    return None

# Generate suggestions (use fallback if analyst.suggest_prompts is unavailable)
try:
    importlib.reload(analyst)
except Exception:
    st.warning("Reloading `analyst` module failed; proceeding with available functions and fallbacks.")

# choose prompt_to_code implementation
prompt_to_code = getattr(analyst, 'prompt_to_code', _prompt_to_code)

if hasattr(analyst, "suggest_prompts"):
    # simple fallback suggestions so the UI still works
    def _simple_suggest(df):
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        categorical = [c for c in df.columns if c not in numeric + datetime_cols and df[c].nunique(dropna=True) <= 50]
        s = ["Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical)."]
        if categorical:
            s.append(f"Show the top 10 counts for the categorical column '{categorical[0]}'.")
        if numeric:
            s.append("Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
            s.append(f"Create a histogram of the numeric column '{numeric[0]}'.")
            if len(numeric) >= 2:
                s.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        if datetime_cols:
            s.append(f"Show counts per month using the datetime column '{datetime_cols[0]}'.")
        s.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
        return s[:8]

    suggestions = _simple_suggest(df)
else:
    # if analyst.suggest_prompts missing, use same simple fallback
    def _simple_suggest(df):
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        categorical = [c for c in df.columns if c not in numeric + datetime_cols and df[c].nunique(dropna=True) <= 50]
        s = ["Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical)."]
        if categorical:
            s.append(f"Show the top 10 counts for the categorical column '{categorical[0]}'.")
        if numeric:
            s.append("Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
            s.append(f"Create a histogram of the numeric column '{numeric[0]}'.")
            if len(numeric) >= 2:
                s.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        if datetime_cols:
            s.append(f"Show counts per month using the datetime column '{datetime_cols[0]}'.")
        s.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
        return s[:8]

    suggestions = _simple_suggest(df)
st.markdown("## Suggested analyses (pick one or write your own)")
col1, col2 = st.columns([3,1])
with col1:
    selected = st.selectbox("Choose a suggested prompt", options=suggestions)
    custom = st.text_area("Or write a custom prompt (leave blank to use the selected suggestion)", height=80)
with col2:
    st.markdown("**Quick actions**")
    if st.button("Show suggestions again"):
        st.write(suggestions)

# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

st.markdown("### Final prompt")
st.write(final_prompt)

# Run button
if st.button("Run analysis"):
    with st.spinner("Running..."):
        # First try deterministic conversion
        code = prompt_to_code(final_prompt, df)
        if code:
            runner = getattr(analyst, 'run_code', None)
            if callable(runner):
                res = runner(df, code)
            else:
                # basic exec fallback
                local_ns = {"pd": pd, "np": np, "df": df, "plt": __import__('matplotlib.pyplot')}
                try:
                    exec(code, {}, local_ns)
                    if 'result' in local_ns and isinstance(local_ns['result'], pd.DataFrame):
                        res = {"type": "dataframe", "df": local_ns['result']}
                    else:
                        res = {"type": "text", "output": str(local_ns.get('result', 'No result'))}
                except Exception as e:
                    res = {"type": "text", "output": f"Execution error: {e}"}
        else:
            # No deterministic code found. If user requested LLM, send the prompt.
            if use_llm:
                # craft a system instruction that asks for python in a ```python``` block that uses df, pd, plt
                system = (
                    "You are a helpful data analyst and will respond with Python code only.\n"
                    "You must return code inside a ```python ... ``` block. The DataFrame is named `df`.\n"
                    "Use pandas for data manipulation and matplotlib for charts. Do not import heavy libs.\n"
                    "If returning a chart, produce matplotlib code that draws the figure (no show()) and nothing else.\n"
                )
                raw = system + "\n# User prompt: " + final_prompt
                llm_out = analyst.ask_llm(raw, model=llm_model)
                if llm_out.startswith("[LLM-missing]") or llm_out.startswith("[LLM-"):
                    st.warning("LLM unavailable or returned an error. Falling back to built-in behavior is not possible for this custom prompt.")
                    st.write(llm_out)
                    st.stop()
                # attempt to extract python block
                if "```python" in llm_out:
                    try:
                        code = llm_out.split("```python")[1].split("```")[0]
                        res = analyst.run_code(df, code)
                    except Exception as e:
                        st.error(f"Failed to execute code from LLM: {e}")
                        st.write(llm_out)
                        st.stop()
                else:
                    st.error("LLM did not return a python code block. Showing raw LLM output:")
                    st.write(llm_out)
                    st.stop()
            else:
                st.error("This is a custom prompt that the app cannot deterministically convert to code. Enable 'Use local LLM' in the sidebar to let a local model generate Python, or edit your prompt to match one of the suggested patterns.")
                st.stop()

    # Display result
    if res["type"] == "text":
        st.markdown("#### Output (text)")
        st.text(res["output"])
    elif res["type"] == "dataframe":
        st.markdown("#### Output (table)")
        st.dataframe(res["df"])
        # Provide CSV download
        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download result as CSV", data=csv, file_name="result.csv", mime="text/csv")
    elif res["type"] == "image":
        st.markdown("#### Output (chart)")
        st.image(res["path"], use_column_width=True)
    else:
        st.write("Unknown result type", res)