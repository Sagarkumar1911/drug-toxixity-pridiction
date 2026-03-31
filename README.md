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

### Prerequisites
- All dependencies are pre-installed in the `toxpred` conda environment
- Python packages: numpy, pandas, matplotlib, streamlit, scikit-learn, xgboost, torch, shap, scipy, rdkit, dgl, tqdm

## Deployment

### Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to `app.py`
5. The app will automatically install dependencies from `requirements.txt`

### Local Development
```bash
# Using conda environment (recommended)
C:\Users\dell\anaconda3\envs\toxpred\python.exe app.py

# Or double-click run_app.bat
```

### Troubleshooting
If deployment fails:
1. Check that all model files exist in the `models/` directory
2. Ensure `data/processed/` contains the required data files
3. Try the minimal requirements.txt versions for better compatibility
```bash
# Preprocess data
C:\Users\dell\anaconda3\envs\toxpred\python.exe src/preprocess.py

# Train models
C:\Users\dell\anaconda3\envs\toxpred\python.exe src/train.py

# Evaluate models
C:\Users\dell\anaconda3\envs\toxpred\python.exe src/evaluate.py

# Generate explanations
C:\Users\dell\anaconda3\envs\toxpred\python.exe src/explain.py

# Run Streamlit app
C:\Users\dell\anaconda3\envs\toxpred\python.exe -m streamlit run app.py
```

## Dataset
Tox21 — 7831 compounds, 12 toxicity assays