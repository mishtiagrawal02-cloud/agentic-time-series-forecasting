# Agentic Time Series Forecasting (sktime)

This project demonstrates a simple **agentic forecasting system** using `sktime`.

It allows users to give natural language prompts like:

"Predict next 6 months and explain trend"

and automatically:
- extracts forecasting horizon
- runs a time series model
- returns predictions + explanation

## Why is this agentic?

This system demonstrates agent-like behavior by:

- Interpreting natural language prompts
- Dynamically selecting forecasting models (Theta / ExponentialSmoothing)
- Extracting forecasting horizon from text
- Generating human-readable explanations

Unlike static pipelines, this system adapts its behavior based on user intent.


## 🚀 Features

- Prompt-based forecasting
- Automatic step extraction from text
- Trend explanation
- Built using `sktime`

---

## 📦 Installation

```bash
pip install sktime statsmodels pandas