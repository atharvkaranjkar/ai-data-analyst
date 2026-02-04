import io
import tempfile
import subprocess
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os
from botocore.exceptions import ClientError
import json
import aws_llm
import subprocess
import re
# Step 1: Robust Data Loading
#-------------- Load Data--------------

def _looks_like_csv(raw_bytes:bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(error="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample

def load_data(file_or_path) -> pd.DataFrame:
    """
    Accepts Streamlit UploadedFile, path string/Path, or file-like object.
    Returns pandas DataFrame.
    """
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)  # default to csv
    
    # file-like (UploadedFile)
    name = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    bio = io.BytesIO(raw)

    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
    # fallback
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)
    

# Step 2: Deterministic Intelligence (Prompt Suggestions)

def _detect_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = []
        
    # try to infer datetime columns   
# this is more robust code wrt NumPy
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            datetime.append(c)
        else:
            # try to parse small sample as date
            try:
                sample = df[c].dropna().astype(str).iloc[:20]
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, min(5, len(sample)//2)):
                    datetime.append(c)
            except Exception:
                pass
    # categoricals: low cardinality non-numeric
    categorical = [c for c in df.columns if c not in numeric + datetime and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "datetime": datetime, "categorical": categorical}

def suggest_prompts(df: pd.DataFrame, max_suggestions: int = 8):
    """
    Return a list of helpful, ready-to-run prompt strings for the dataset.
    Deterministic and works without any LLM.
    """
    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime = types["datetime"]
    categorical = types["categorical"]
    
    suggestions = []
    # Basic summary
    suggestions.append("Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical).")
    # Top value queries
    if categorical:
        col = categorical[0]
        suggestions.append(f"Show the top 10 counts for the categorical column '{col}'.")
    # Numeric summaries
    if numeric:
        suggestions.append(f"Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
        col = numeric[0]
        suggestions.append(f"Create a histogram of the numeric column '{col}'.")
        if len(numeric) >= 2:
            suggestions.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        suggestions.append(f"Show the top 10 rows sorted by '{col}' descending.")
    # Time series
    if datetime:
        dcol = datetime[0]
        # choose a numeric for aggregation if exists
        ag = numeric[0] if numeric else None
        if ag:
            suggestions.append(f"Create a time series of monthly sum of '{ag}' using the datetime column '{dcol}'.")
        else:
            suggestions.append(f"Show counts per month using the datetime column '{dcol}'.")
    # Correlation
    if len(numeric) >= 2:
        suggestions.append("Show the correlation matrix heatmap for numeric columns.")
    # Generic top-k
    suggestions.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
    # limit suggestions
    return suggestions[:max_suggestions]

# Step 3: The Translator
def prompt_to_code(prompt: str, df: pd.DataFrame):
    """
    Convert known prompt templates into runnable python code strings.
    If the prompt is custom/unrecognized, return None (so UI can send to LLM instead).
    """
    p = prompt.strip().lower()

   # Summary
    if p.startswith("summarize the dataset"):
        code = textwrap.dedent("""
            # produce a short summary as printed text
            info = []
            info.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            info.append("Column types: " + ", ".join([f\"{c}:{str(df[c].dtype)[:10]}\" for c in df.columns[:10]]))
            miss = df.isnull().sum().sort_values(ascending=False).head(10)
            info.append("Top missing: " + ", ".join([f\"{idx}:{val}\" for idx,val in miss.items() if val>0]))
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            info.append(f\"Numeric columns count: {len(numeric)}\")
            # print concise bullets
            result = \"\\n\".join([\"- \"+i for i in info])
        """)
        return code



    # Top counts for categorical
    if "top 10 counts for the categorical column" in p or "top 10 counts" in p and "'" in p:
        # try to extract column name between quotes
        import re
        m = re.search(r"'([^']+)'", prompt)
        if not m:
            m = re.search(r'"([^"]+)"', prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # top 10 counts for '{col}'
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)
            return code

 # Summary statistics for numeric
    if "summary statistics" in p or "describe" in p:
        code = textwrap.dedent("""
            result = df.select_dtypes(include=['number']).describe().T
        """)
        return code

# Histogram
    if p.startswith("create a histogram of the numeric column") or "histogram of the numeric column" in p:
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # histogram for '{col}'
                plt.figure(figsize=(6,4))
                df['{col}'].dropna().astype(float).hist(bins=30)
                plt.title('Histogram of {col}')
                plt.xlabel('{col}')
                plt.ylabel('count')
                # produce an image by saving to result_img_path variable
                result_img_path = None
            """)
            # We'll return plotting code that uses plt; execution will save figure
            return code
    
    # Scatter plot
    if "scatter plot comparing" in p and "vs" in p:
        import re
        m = re.search(r"'([^']+)' \\(x\\) vs '([^']+)' \\(y\\)", prompt)
        if m:
            xcol, ycol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df.plot.scatter(x='{xcol}', y='{ycol}')
                plt.title('{ycol} vs {xcol}')
                result_img_path = None
            """)
            return code
        
    # Top N rows sorted by col
    if p.startswith("show the top 10 rows sorted by"):
        import re
        m = re.search(r"by '([^']+)'", prompt)
        if m:
            col = m.group(1)
            code = textwrap.dedent(f"""
                result = df.sort_values('{col}', ascending=False).head(10).reset_index(drop=True)
            """)
            return code

    # Time series monthly sum
    if "monthly sum" in p and "using the datetime column" in p:
        import re
        m = re.search(r"sum of '([^']+)' using the datetime column '([^']+)'", prompt)
        if m:
            ag, dcol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}'])
                res = res.set_index('{dcol}').resample('M')['{ag}'].sum().reset_index()
                result = res
            """)
            return code

    # Counts per month (datetime only)
    if "counts per month using the datetime column" in p:
        import re
        m = re.search(r"datetime column '([^']+)'", prompt)
        dcol = m.group(1) if m else None
        if dcol:
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}']).set_index('{dcol}').resample('M').size().reset_index(name='count')
                result = res
            """)
            return code

    # Correlation heatmap
    if "correlation matrix heatmap" in p or "correlation heatmap" in p:
        code = textwrap.dedent("""
            corr = df.select_dtypes(include=['number']).corr()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,5))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation matrix')
            result_img_path = None
        """)
        return code

    # Anomaly detection using z-score
    if "anomalies" in p and "z-score" in p:
        code = textwrap.dedent("""
            from scipy import stats
            num = df.select_dtypes(include=['number']).dropna()
            if num.shape[1]==0:
                result = pd.DataFrame()
            else:
                z = np.abs(stats.zscore(num.select_dtypes(include=['number'])))
                mask = (z > 3).any(axis=1)
                result = df.loc[mask].head(20).reset_index(drop=True)
        """)
        return code

    # Unknown / custom prompts -> return None
    return None


# Step 4: The Execution Engine
def run_code(df: pd.DataFrame, code: str):
    """
    Execute code string in a restricted local namespace.
    Returns a dict:
      - {"type":"text","output":...}
      - {"type":"dataframe","df": pandas.DataFrame}
      - {"type":"image","path": path_to_png}
    Execution conventions:
      - If code sets a variable `result` to a DataFrame or string, we return it.
      - If code uses matplotlib to plot, we save the current figure to a temp PNG and return image.
      - If code raises, return error text.
    """
    # prepare namespace
    local_ns = {"pd": pd, "np": np, "df": df, "plt": plt}
    # capture prints
    old_stdout = sys.stdout
    stdout_buf = io.StringIO()
    sys.stdout = stdout_buf
    try:
        # run
        exec(code, {}, local_ns)
        # first, if plotting occurred (plt has a current figure), save it
        # If code created result_img_path variable, prefer it
        if "result_img_path" in local_ns and local_ns["result_img_path"]:
            path = local_ns["result_img_path"]
            return {"type": "image", "path": path}
        # check for figure in plt
        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, bbox_inches="tight", dpi=150)
                plt.close("all")
                return {"type": "image", "path": f.name}
        # check for result variable
        if "result" in local_ns:
            res = local_ns["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            else:
                return {"type": "text", "output": str(res)}
        # otherwise, return captured stdout
        out = stdout_buf.getvalue().strip()
        if out:
            return {"type": "text", "output": out}
        return {"type": "text", "output": "Execution finished. No result produced."}
    except Exception as e:
        return {"type": "text", "output": f"Execution error: {e}"}
    finally:
        sys.stdout = old_stdout


# Step 5: Helper Functions for Robust LLM Handling

def is_error_response(response: str) -> bool:
    """Check if LLM response indicates an error."""
    if not response:
        return True
    error_patterns = [
        "[LLM-AWS-error]",
        "[LLM-missing]",
        "[LLM-local-error]",
        "ClientError",
        "Connection refused"
    ]
    return any(pattern in response for pattern in error_patterns)

def extract_json_from_response(response: str) -> Optional[dict]:
    """
    Extract JSON from LLM response.
    Handles:
    - Plain JSON objects
    - JSON wrapped in markdown code blocks
    - JSON with surrounding text
    """
    if not response:
        return None
    
    # Try markdown code blocks first
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*?\})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    return None

def validate_visualization_plan(plan: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate visualization plan returned by LLM.
    Returns (is_valid, error_message)
    """
    required_fields = ["chart", "x", "y", "aggregation"]
    
    # Check required fields
    for field in required_fields:
        if field not in plan:
            return False, f"Missing required field: '{field}'"
    
    chart_type = plan.get("chart", "").lower()
    if chart_type not in ["bar", "line", "scatter", "histogram", "box"]:
        return False, f"Unsupported chart type: '{chart_type}'"
    
    x = plan.get("x")
    y = plan.get("y")
    group_by = plan.get("group_by")
    agg = plan.get("aggregation", "").lower()
    
    # Validate columns exist
    if x not in df.columns:
        return False, f"Column '{x}' not found in dataset"
    if y not in df.columns:
        return False, f"Column '{y}' not found in dataset"
    if group_by and group_by not in df.columns:
        return False, f"Group column '{group_by}' not found in dataset"
    
    # Validate aggregation
    valid_aggs = ["count", "sum", "mean", "median", "min", "max", "std"]
    if agg not in valid_aggs:
        return False, f"Unsupported aggregation: '{agg}'. Use one of {valid_aggs}"
    
    return True, ""

def check_aws_credentials() -> tuple[bool, str]:
    """
    Check if AWS credentials are properly configured.
    Returns (has_credentials, status_message)
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    
    if access_key and secret_key and region:
        return True, f"✅ AWS configured (region: {region})"
    
    missing = []
    if not access_key:
        missing.append("AWS_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not region:
        missing.append("AWS_REGION")
    
    return False, f"⚠️ AWS not configured. Missing: {', '.join(missing)}"

# Step 6: The LLM Connector (Enhanced)
def ask_llm(prompt: str, model: str = "llama3.1", timeout: int = 180) -> str:
    """
    Unified LLM interface.
    - Uses AWS Bedrock (Claude) if credentials exist
    - Falls back to local Ollama
    - Returns error message if both fail
    """

    # Check AWS environment
    has_aws, aws_status = check_aws_credentials()

    # -------------------------
    # AWS Bedrock (Claude)
    # -------------------------
    if has_aws:
        try:
            result = aws_llm.ask_aws_llm(prompt)
            return result["text"]
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            return f"[LLM-AWS-error] {error_code}: {str(e)}"
        except Exception as e:
            return f"[LLM-AWS-error] {type(e).__name__}: {str(e)}"

    # -------------------------
    # Local Ollama fallback
    # -------------------------
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        output = proc.stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return f"[LLM-local-error] Ollama returned empty response"
        return output

    except FileNotFoundError:
        return "[LLM-missing] Ollama is not installed or not in PATH. Install from https://ollama.ai"

    except subprocess.TimeoutExpired:
        return f"[LLM-local-error] Ollama request timed out after {timeout}s"

    except Exception as e:
        return f"[LLM-local-error] {type(e).__name__}: {str(e)}"
