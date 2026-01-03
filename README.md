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
- **AWS Bedrock LLM integration** (Claude / Titan)
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
  - Recommended model: **Claude 3 Sonnet**
  - Region: `us-east-1` or `ap-south-1`

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

```txt
streamlit==1.29.0
pandas>=1.5,<2.1
numpy>=1.24,<1.27
boto3>=1.34.0
```

---

## 🔐 Environment Variables (Cloud)

Set these in **Streamlit Cloud → App Settings → Secrets**:

```toml
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
AWS_REGION = "us-east-1"
```

⚠️ Never commit credentials to GitHub.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

(Optional) Install Ollama for local LLM testing.

---

## 📁 Project Structure

```
├── app.py          # Streamlit UI & routing logic
├── analyst.py      # LLM abstraction layer
├── aws_llm.py      # AWS Bedrock integration
├── requirements.txt
└── README.md
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
