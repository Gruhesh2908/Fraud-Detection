# Fraud Detection (Streamlit App)

A simple Streamlit web application for transaction fraud prediction using three ML models:
- Logistic Regression (with scaler)
- Random Forest
- XGBoost

The app supports:
- ✅ Single transaction prediction (form input)
- ✅ Batch CSV prediction (upload CSV + download results)
- ✅ Model selection and threshold tuning

---

## Project Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `Fraud_Detection.ipynb` — training/evaluation notebook (optional)

> Note: Trained model files and datasets are not included in the repository (by design).  
> You must place trained model artifacts inside a `models/` folder before running the app.

---

## 1) Setup (Windows / VS Code)

### Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Upgrade pip tools

python -m pip install --upgrade pip setuptools wheel

3) Install dependencies
pip install -r requirements.txt


4) Run the App

streamlit run app.py
