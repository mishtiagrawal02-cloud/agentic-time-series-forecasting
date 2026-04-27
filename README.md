# Agentic Time Series Forecasting (sktime)

An intelligent **agentic forecasting system** built using `sktime` that understands natural language prompts, selects models, compares them, and explains predictions.

---

## 🚀 What this does

You can give prompts like:

> "Predict next 6 months using best model and explain trend"

And the system will:
- Parse the request
- Extract forecasting horizon
- Select or compare models
- Generate predictions
- Explain results

---

## 🧠 Why is this agentic?

This system demonstrates **agent-like decision making**:

- Interprets natural language prompts
- Dynamically selects forecasting models
- Compares multiple models using error metrics (MAE)
- Chooses the best-performing model
- Generates human-readable explanations

👉 Unlike static pipelines, this system **adapts based on user intent**

---

## ✨ Features

- Prompt-based forecasting
- Model selection (Theta / Naive / Exponential Smoothing)
- Model comparison (automatic best model selection)
- Error-based evaluation (MAE)
- Trend explanation
- Supports natural language:
  - "next 6 months"
  - "next year"
  - "compare models"

---

## 📊 Example

### Input


### Output


---

## 📦 Installation

```bash
pip install -r requirements.txt