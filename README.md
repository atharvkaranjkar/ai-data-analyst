# AI Data Analyst

Small Streamlit app + helpers to explore CSV/Excel/JSON datasets and optionally call a local LLM (Ollama).

**Prerequisites**

- Python 3.8+ installed
- (Optional) Ollama installed and on PATH for LLM features: https://ollama.com/

**Setup (PowerShell)**

```powershell
# create and activate virtualenv
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip

# install runtime dependencies
pip install pandas numpy matplotlib streamlit scipy duckdb
```

If you plan to use the LLM connector (`ask_llm()`), install and run Ollama following their docs and ensure the `ollama` CLI is on PATH.

**Run the app**

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

**Quick checks**

```powershell
# ensure the helper module imports
python -c "import analyst; print('analyst OK')"

# try loading a CSV using the helper
python - <<'PY'
from analyst import load_data
print(load_data('path/to/file.csv').shape)
PY
```

**Files of interest**

- `app.py` — Streamlit UI and entrypoint
- `analyst.py` — data loaders, prompt templates, execution engine, and LLM connector

**Notes**

- The LLM integration uses the `ollama` CLI; if not installed `ask_llm()` returns a `[LLM-missing]` message.
- The project makes simple assumptions about categorical thresholds (<=50 uniques). Adjust `_detect_column_types()` as needed for your dataset.

If you want, I can also add a `requirements.txt` and a short `.gitignore` next.```