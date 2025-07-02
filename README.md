# AutoRisk+: End-to-End Insurance Pricing with Machine Learning

AutoRisk+ is a machine learning project that simulates a real-world auto insurance pricing system. It covers the full ML pipeline: data processing, model training, evaluation, and deployment. This project is inspired by modern pricing analytics roles in the insurance industry and demonstrates practical experience in applied ML, DevOps practices, and cloud-aware architecture.

---

##  Project Goals

- Build an end-to-end machine learning pipeline to predict insurance claim cost and risk.
- Simulate real-world insurance pricing data and extract meaningful features.
- Evaluate and compare several ML models (GLM, XGBoost, MLP).
- Create a deployment-ready model inference API.
- Develop a dashboard to monitor pricing risk and visualize predictions.
- Incorporate best practices in reproducibility (Docker, version control, modular code).

---

## Use Case

This project simulates how an insurance company like TD Insurance might use machine learning to:

- Predict the risk or cost of claims based on customer and vehicle attributes.
- Develop fair and accurate premium pricing.
- Automate and scale pricing insights across large portfolios.

---

##  Tech Stack

| Layer              | Tools Used |
|-------------------|------------|
| Language           | Python 3.10 |
| Data Handling      | Pandas, NumPy, SQL |
| ML Models          | scikit-learn, XGBoost, PyTorch |
| Pipelines & Infra  | MLflow, FastAPI, Docker |
| Visualization      | Streamlit, Matplotlib, Seaborn |
| DevOps Practices   | Git, CI-ready structure, Dockerfile |
| Optional Scaling   | PySpark (local), Databricks (mocked) |

---

##  Project Structure
AutoRiskPlus/
├── data/ # Raw or simulated insurance data
├── notebooks/ # Jupyter notebooks for exploration
│ ├── 01_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
├── src/ # Modular code
│ ├── preprocess.py
│ ├── train_model.py
│ ├── inference.py
│ └── utils.py
├── api/ # FastAPI server for inference
│ └── app.py
├── dashboard/ # Streamlit dashboard
│ └── app.py
├── Dockerfile # For containerization
├── requirements.txt
├── README.md
└── report/
└── AutoRisk_Report.pdf # Business-style insights and summary

 Installation

Clone the repository:
git clone https://github.com/yourusername/AutoRiskPlus.git
cd AutoRiskPlus
(Optional) Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run notebooks or scripts in src/ to generate the model.
Launch the API:
uvicorn api.app:app --reload
Launch the dashboard:
streamlit run dashboard/app.py

Report

Find a business-facing PDF report with insights, model performance, and pricing strategy rationale under /report/AutoRisk_Report.pdf.



