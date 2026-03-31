# ToxPredict — Drug Toxicity Prediction

## Problem
Drug development fails due to unexpected toxicity.
Early AI-based prediction can reduce costs and improve safety.

## Solution
Ensemble ML model predicting 12 toxicity endpoints
from molecular structure using the Tox21 dataset.

## Models
| Model         | Val AUC | Test AUC |
|---------------|---------|----------|
| XGBoost       | 0.7741  | 0.7430   |
| DNN           | 0.7789  | 0.7450   |
| Random Forest | 0.7805  | 0.7534   |
| Ensemble      | 0.8068  | 0.7628   |

## Key Findings
- Ring count and aromatic structure are top toxicity predictors
- One scaffold showed 85% toxicity rate in SR-ARE assay
- Ensemble outperforms any single model by 3-4% AUC

## How to Run
# Install
conda activate toxpred

# Preprocess
python src/preprocess.py

# Train
python src/train.py

# Evaluate
python src/evaluate.py

# Explain
python src/explain.py

# App
streamlit run app.py

## Dataset
Tox21 — 7831 compounds, 12 toxicity assays