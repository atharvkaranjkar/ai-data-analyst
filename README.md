# 🧠 Personal AI Data Analyst

An AI-powered data analysis application built with **Streamlit** that allows users to upload datasets and ask **natural language questions** to generate **insights, visualizations, and recommendations** using **AWS Bedrock LLMs**.

This project demonstrates how to combine **rule-based analytics**, **semantic intent detection**, and **LLM-powered reasoning** in a production-ready, cloud-deployed application.

---

## 🚀 Key Features

- Upload **CSV / Excel / JSON** datasets
- Automatic dataset understanding (schema, types, samples)
- Natural language queries for:
  - Summaries
  - Trends
  - Distributions
  - Anomaly detection
  - Business recommendations
- **Auto-generated visualizations** based on user intent
- **AWS Bedrock LLM integration** (Amazon Nova Lite)
- Optional **local LLM support** (Ollama) for development
- Cloud-ready deployment on **Streamlit Cloud**

---

## 🧠 How the App Works (Architecture)

The app follows a **three-layer decision pipeline**:

### 1️⃣ Intent Detection
User prompts are classified into:
- **Simple** → deterministic stats (rows, columns)
- **Visual** → charts & distributions
- **Analysis** → insights & recommendations

### 2️⃣ Data-Aware Reasoning
- Dataset schema and samples are passed to the LLM
- The LLM returns either:
  - A **structured visualization plan (JSON)**, or
  - A **textual analytical response**

### 3️⃣ Safe Execution
- The app validates LLM output
- Charts are rendered using Streamlit primitives
- No unsafe code execution (`exec` avoided)

---

## 📊 Example Use Cases

- *“What is the gender distribution across departments and job roles?”*
- *“Analyze salary trends over time”*
- *“Detect and visualize anomalies in sensor data”*
- *“Recommend actions to improve system reliability”*

---

## ☁️ LLM & Cloud Support

### Cloud (Production)
- **AWS Bedrock**
  - Current model: **Amazon Nova Lite** (`amazon.nova-lite-v1:0`)
  - Region: `us-east-1` (configurable via `AWS_REGION`)

### Local (Development)
- **Ollama**
- Disabled automatically on Streamlit Cloud

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **AWS Bedrock (boto3)**
- **JSON-based LLM planning**
- **GitHub + Streamlit Cloud**

---

## 📦 Requirements

See `requirements.txt` for complete dependencies. Key packages:
- `streamlit` - UI framework
- `pandas` / `numpy` - Data processing
- `boto3` - AWS SDK for Bedrock access

---

## 🔐 Secrets Configuration

### Local Development
Create `.streamlit/secrets.toml` in your project root:

```toml
AWS_ACCESS_KEY_ID = "your_access_key_here"
AWS_SECRET_ACCESS_KEY = "your_secret_key_here"
AWS_REGION = "us-east-1"
```

A template file (`.streamlit/secrets.example.toml`) is provided for reference.

### Cloud Deployment (Streamlit Cloud)
Set these in **Streamlit Cloud → App Settings → Secrets** using the same format as above.

⚠️ **Never commit `secrets.toml` to GitHub** — it's already in `.gitignore`

---

## ▶️ Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup credentials:**
   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
   - Add your AWS credentials

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Access:** Open `http://localhost:8501` in your browser

(Optional) Install Ollama for local LLM fallback testing.

---

## 📁 Project Structure

```
├── app.py                        # Streamlit UI & routing logic
├── analyst.py                    # LLM abstraction layer & analytics
├── aws_llm.py                    # AWS Bedrock integration
├── .streamlit/
│   ├── secrets.example.toml      # Template for local secrets
│   └── secrets.toml              # Local credentials (not in git)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🧪 Design Principles

- **Intent-driven analytics**, not prompt-specific hacks
- **LLM as planner**, not executor
- **Rule-based first**, LLM only when needed
- **Cloud-first**, stateless execution
- **Safe parsing & validation** of LLM output

---

## 📄 Resume Description (2–3 Lines)

> Built a cloud-deployed AI data analysis tool using Streamlit and AWS Bedrock that interprets natural language queries to generate insights and visualizations. Implemented semantic intent detection and LLM-driven visualization planning for scalable, production-safe analytics.

---

## 🏁 Status

✅ Feature-complete  
✅ Cloud deployed  
✅ Resume & interview ready  

---

## 📜 License

MIT License
