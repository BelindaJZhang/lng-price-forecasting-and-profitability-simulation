# LNG Price Forecasting & Trade Profitability Simulation
*A data-driven tool to support smarter LNG trading decisions.*

## Author
**Juan Zhang**  
LinkedIn: https://www.linkedin.com/in/juan-zhang-finance-professional/

---

## Overview
This project combines deep-learning price forecasts with an interactive LNG trade profitability simulator.  
It helps LNG analysts and traders explore price scenarios, compare route economics, and evaluate expected voyage margins.

The application integrates:

- Multi-horizon LNG benchmark price forecasts (TTF, PVB, Henry Hub, JKM)  
- Voyage profitability calculations (netback, shipping assumptions, regas fees)  
- Interactive visualisations and scenario testing  

Originally developed during the Constructor Academy Data Science & AI Bootcamp, this project has since been expanded into a standalone forecasting and simulation engine.

---

## The Problem
LNG traders must quickly assess trade opportunities under significant uncertainty:

- Volatile spot benchmarks  
- Changing freight costs and regas fees  
- Terminal-specific spreads  
- Fuel loss and boil-off  
- Geopolitical shocks  

Manual spreadsheets make this process slow, error-prone, and difficult to scenario-test.

---

## The Solution

### 1. Deep Learning Forecasting Engine (Conv1D + LSTM)
The price forecasting module provides 30-, 60-, and 90-day horizon predictions for:

- TTF (EU)  
- PVB (Spain)  
- Henry Hub (US)  
- JKM (Asia)  

Features and methodology:

- Rolling feature engineering (returns, Bollinger width, z-scores, etc.)  
- Horizon-aware windowing  
- Conv1D → LSTM stacked architecture  
- Metrics: MAE, RMSE, R²  
- Validation split based on historical cutoff date  

Forecasts are precomputed offline and loaded by the Streamlit app for fast rendering.

---

### 2. Voyage Profitability Simulation (Netback Analysis)
The simulation module calculates:

- Netback at origin  
- Delivered price at destination  
- Voyage profitability  
- Impact of regas fees, spreads, freight costs, and fuel loss  

Users can adjust:

- Regas fee  
- Shipping cost  
- Margin assumptions  
- Fuel loss / boil-off  
- Benchmark spreads  

Outputs include:

- Profitability tables  
- Waterfall charts  
- Route comparison visuals  

---

## Illustrations

### LNG Trading Flow
![LNG_flow](images/trade_flow.png)

### Forecast Panels (30/60/90 days)
![Forecast_panels](images/forecast_panels.png)

---

## Repository Structure
```
.
├── app.py                         # Main Streamlit application
├── Forecast_Plotting_Functions.py # Forecast visualisation helpers
├── requirements.txt
├── Dockerfile                     # Deployment config (HuggingFace)
│
├── data/
│   └── processed/                 # Netback & route datasets
│
├── reports/                       # Precomputed forecast CSVs
├── images/                        # Illustrations
│
├── notebooks/                     # Model development notebooks
├── models/                        # Saved LSTM models (optional)
└── README.md
```

---

## Data
Stored under `data/` and includes:

- Historical LNG price benchmarks  
- Technical indicators  
- Route-specific cost assumptions  
- Weather data where applicable  

Raw data comes from public sources and internal market reports.

---

## Model Development
The workflow in `notebooks/` includes:

- Data cleaning and exploration  
- Feature engineering  
- Model training (Conv1D + LSTM)  
- Hyperparameter tuning  
- Model validation  
- Horizon comparison  
- Forecast export to CSV  

---

## Running Locally
```
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment (HuggingFace Spaces – Docker)

This project uses a custom Docker image to ensure compatible versions of:

- Python 3.10  
- pandas 2.2  
- numpy 1.26  
- scikit-learn 1.3  
- Streamlit 1.32  

**Dockerfile:**
```
FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential gcc g++ libffi-dev libssl-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

---

## Live Demo
HuggingFace App: https://huggingface.co/spaces/JUANGE1014/lng_trade_profitability_simulation?logs=container
GitHub Repo: https://github.com/BelindaJZhang/lng-price-forecasting-and-profitability-simulation

---

## About the Author
I’m **Juan Zhang**, a finance professional and and trained data scientist passionate about AI-driven forecasting, and process optimisation.  
LinkedIn: https://www.linkedin.com/in/juan-zhang-finance-professional/
