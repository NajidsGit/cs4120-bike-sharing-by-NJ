# cs4120-bike-sharing-by-NJ
CS-4120 Machine Learning Course Project – Bike Sharing Demand.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# OR: source venv/bin/activate  # Mac/Linux

Install dependencies:

bash
pip install -r requirements.txt
Download data (see data/README.md)

Running the Project
Train Baseline Models
bash
python -m src.train_baselines
Generate Evaluation Plots & Tables
bash
python -m src.evaluate
View MLflow Results
bash
mlflow ui
Then open http://127.0.0.1:5000