
"""
app/app.py
Purpose : Streamlit prediction app for drug toxicity
Run     : streamlit run app/app.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, MACCSkeys,
    rdMolDescriptors, Draw
)
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

# PyTorch
import torch
import torch.nn as nn

# SHAP
import shap

# ── Config ────────────────────────────────────────────────────────
MODELS_DIR    = 'models/'
PROCESSED_DIR = 'data/processed/'
DEVICE        = torch.device('cuda' if torch.cuda.is_available()
                              else 'cpu')

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

TARGET_NAMES = {
    'NR-AR':        'Androgen Receptor',
    'NR-AR-LBD':    'Androgen Receptor LBD',
    'NR-AhR':       'Aryl Hydrocarbon Receptor',
    'NR-Aromatase': 'Aromatase',
    'NR-ER':        'Estrogen Receptor',
    'NR-ER-LBD':    'Estrogen Receptor LBD',
    'NR-PPAR-gamma':'PPAR Gamma',
    'SR-ARE':       'Antioxidant Response',
    'SR-ATAD5':     'DNA Damage (ATAD5)',
    'SR-HSE':       'Heat Shock Response',
    'SR-MMP':       'Mitochondrial Membrane',
    'SR-p53':       'Tumor Suppressor p53'
}

# ── DNN Architecture ──────────────────────────────────────────────
class MultiTaskToxNet(nn.Module):
    def __init__(self, input_dim, n_tasks=12, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        h = self.encoder(x)
        return torch.cat(
            [head(h) for head in self.heads], dim=1
        )


# ── Feature Extraction ────────────────────────────────────────────
RDKIT_DESCS = [
    ('MolWt',              Descriptors.MolWt),
    ('MolLogP',            Descriptors.MolLogP),
    ('TPSA',               Descriptors.TPSA),
    ('NumHDonors',         Descriptors.NumHDonors),
    ('NumHAcceptors',      Descriptors.NumHAcceptors),
    ('NumRotatableBonds',  Descriptors.NumRotatableBonds),
    ('NumAromaticRings',   rdMolDescriptors.CalcNumAromaticRings),
    ('NumAliphaticRings',  rdMolDescriptors.CalcNumAliphaticRings),
    ('RingCount',          Descriptors.RingCount),
    ('FractionCSP3',       Descriptors.FractionCSP3),
    ('HeavyAtomCount',     Descriptors.HeavyAtomCount),
    ('NumHeteroatoms',     rdMolDescriptors.CalcNumHeteroatoms),
    ('MolMR',              Descriptors.MolMR),
    ('LabuteASA',          Descriptors.LabuteASA),
    ('BalabanJ',           Descriptors.BalabanJ),
    ('BertzCT',            Descriptors.BertzCT),
    ('Chi0',               Descriptors.Chi0),
    ('Chi1',               Descriptors.Chi1),
    ('Kappa1',             Descriptors.Kappa1),
    ('Kappa2',             Descriptors.Kappa2),
    ('Kappa3',             Descriptors.Kappa3),
    ('MaxPartialCharge',   Descriptors.MaxPartialCharge),
    ('MinPartialCharge',   Descriptors.MinPartialCharge),
    ('NHOHCount',          Descriptors.NHOHCount),
    ('NOCount',            Descriptors.NOCount),
    ('NumValenceElectrons',Descriptors.NumValenceElectrons),
    ('PEOE_VSA1',          Descriptors.PEOE_VSA1),
    ('PEOE_VSA2',          Descriptors.PEOE_VSA2),
    ('SMR_VSA1',           Descriptors.SMR_VSA1),
    ('SlogP_VSA1',         Descriptors.SlogP_VSA1),
]


def extract_features(mol):
    """Extract all features for one molecule"""
    morgan = np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, useChirality=True
        ), dtype=np.float32
    )
    maccs = np.array(
        MACCSkeys.GenMACCSKeys(mol),
        dtype=np.float32
    )
    rdkit = []
    for name, fn in RDKIT_DESCS:
        try:
            v = float(fn(mol))
            rdkit.append(
                0.0 if (np.isnan(v) or np.isinf(v)) else v
            )
        except Exception:
            rdkit.append(0.0)
    rdkit = np.array(rdkit, dtype=np.float32)
    return np.concatenate([morgan, maccs, rdkit])


def apply_variance_selector(features):
    """Apply saved variance selector"""
    try:
        selector = pickle.load(
            open(f'{MODELS_DIR}var_selector.pkl', 'rb')
        )
        return selector.transform(features.reshape(1, -1))[0]
    except Exception:
        return features


# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all models — cached so only runs once"""
    xgb_models  = pickle.load(
        open(f'{MODELS_DIR}xgb_models.pkl', 'rb')
    )
    rf_models   = pickle.load(
        open(f'{MODELS_DIR}rf_models.pkl', 'rb')
    )
    meta_models = pickle.load(
        open(f'{MODELS_DIR}meta_models.pkl', 'rb')
    )

    with open(f'{PROCESSED_DIR}feature_names.json') as f:
        feature_names = json.load(f)

    # DNN
    sample_features = apply_variance_selector(
        np.zeros(2245)
    )
    input_dim = len(sample_features)
    dnn_model = MultiTaskToxNet(input_dim).to(DEVICE)
    dnn_model.load_state_dict(
        torch.load(f'{MODELS_DIR}dnn_best.pt',
                   map_location=DEVICE)
    )
    dnn_model.eval()

    return (xgb_models, dnn_model,
            rf_models, meta_models, feature_names)


# ── Predict ───────────────────────────────────────────────────────
def predict_toxicity(mol, xgb_models, dnn_model,
                      rf_models, meta_models):
    """Get ensemble toxicity predictions for one molecule"""
    # Extract features
    raw_features = extract_features(mol)
    features     = apply_variance_selector(raw_features)
    X            = features.reshape(1, -1)

    # XGBoost predictions
    xgb_probs = np.array([
        xgb_models[col].predict_proba(X)[0, 1]
        for col in TARGET_COLS
    ])

    # DNN predictions
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        dnn_out = torch.sigmoid(
            dnn_model(X_tensor)
        ).cpu().numpy()[0]
    dnn_probs = dnn_out

    # RF predictions
    rf_probs = np.array([
        rf_models[col].predict_proba(X)[0, 1]
        for col in TARGET_COLS
    ])

    # Ensemble predictions
    ens_probs = np.array([
        meta_models[col].predict_proba(
            np.array([[xgb_probs[i],
                       dnn_probs[i],
                       rf_probs[i]]])
        )[0, 1]
        for i, col in enumerate(TARGET_COLS)
    ])

    return xgb_probs, dnn_probs, rf_probs, ens_probs, features


# ── SHAP for Single Molecule ──────────────────────────────────────
def get_shap_for_molecule(features, xgb_models,
                           feature_names, task='SR-ARE'):
    """Get SHAP values for one molecule"""
    try:
        xgb_clf   = xgb_models[task]
        explainer = shap.Explainer(xgb_clf,
                                    features.reshape(1, -1))
        shap_obj  = explainer(features.reshape(1, -1))
        shap_vals = shap_obj.values[0]

        # Top 10 features
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = shap_vals[top_idx]

        return top_names, top_vals

    except Exception as e:
        # Fallback to feature importance
        imp     = xgb_models[task].feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        return ([feature_names[i] for i in top_idx],
                imp[top_idx])


# ── Draw Molecule ─────────────────────────────────────────────────
def draw_molecule(mol, size=(400, 300)):
    """Draw molecule as PNG"""
    try:
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════════════
def main():
    # Page config
    st.set_page_config(
        page_title="ToxPredict — Drug Toxicity Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .toxic-badge {
        background: #E24B4A;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .safe-badge {
        background: #1D9E75;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div class="main-title">🧬 ToxPredict</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-powered drug toxicity '
        'prediction across 12 biological assays</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **ToxPredict** uses an ensemble of 3 ML models:
        - XGBoost
        - Deep Neural Network
        - Random Forest

        Trained on **Tox21 dataset**
        (~7,800 compounds, 12 assays)

        **Test AUC: 0.7628**
        """)

        st.header("Example SMILES")
        examples = {
            "Paracetamol":   "CC(=O)Nc1ccc(O)cc1",
            "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "Ibuprofen":     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Bisphenol A":   "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
        }
        for name, smi in examples.items():
            if st.button(name, use_container_width=True):
                st.session_state['smiles'] = smi

        st.header("Settings")
        threshold = st.slider(
            "Toxicity threshold", 0.1, 0.9, 0.5, 0.05
        )
        focus_task = st.selectbox(
            "SHAP analysis task", TARGET_COLS, index=7
        )

    # Load models
    with st.spinner("Loading models..."):
        try:
            (xgb_models, dnn_model,
             rf_models, meta_models,
             feature_names) = load_models()
            st.success("Models loaded successfully")
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return

    # SMILES Input
    st.subheader("Enter a SMILES string")
    default_smiles = st.session_state.get(
        'smiles', 'CC(=O)Nc1ccc(O)cc1'
    )
    smiles = st.text_input(
        "SMILES",
        value=default_smiles,
        placeholder="e.g. CC(=O)Nc1ccc(O)cc1",
        label_visibility="collapsed"
    )

    if not smiles:
        st.info("Enter a SMILES string above to get predictions")
        return

    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check and try again.")
        return

    # Get predictions
    with st.spinner("Computing predictions..."):
        (xgb_probs, dnn_probs,
         rf_probs, ens_probs,
         features) = predict_toxicity(
            mol, xgb_models, dnn_model,
            rf_models, meta_models
        )

    # ── Layout ────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    # Column 1 — Molecule + Properties
    with col1:
        st.subheader("Molecule")

        img = draw_molecule(mol, size=(300, 250))
        if img:
            st.image(img, use_column_width=True)

        st.subheader("Properties")
        props = {
            "Mol. Weight":  f"{Descriptors.MolWt(mol):.1f}",
            "LogP":         f"{Descriptors.MolLogP(mol):.2f}",
            "TPSA":         f"{Descriptors.TPSA(mol):.1f}",
            "HBD":          f"{Descriptors.NumHDonors(mol)}",
            "HBA":          f"{Descriptors.NumHAcceptors(mol)}",
            "Rings":        f"{Descriptors.RingCount(mol)}",
            "Heavy atoms":  f"{Descriptors.HeavyAtomCount(mol)}",
        }
        for prop, val in props.items():
            col_a, col_b = st.columns([1.5, 1])
            col_a.write(prop)
            col_b.write(f"**{val}**")

        # Overall risk
        n_toxic  = (ens_probs >= threshold).sum()
        risk_pct = n_toxic / len(TARGET_COLS) * 100
        st.subheader("Overall Risk")
        if risk_pct >= 50:
            st.error(f"HIGH RISK — {n_toxic}/12 assays toxic")
        elif risk_pct >= 25:
            st.warning(
                f"MODERATE RISK — {n_toxic}/12 assays toxic"
            )
        else:
            st.success(f"LOW RISK — {n_toxic}/12 assays toxic")

    # Column 2 — Toxicity Predictions
    with col2:
        st.subheader("Toxicity Predictions (Ensemble)")

        for i, col in enumerate(TARGET_COLS):
            prob  = float(ens_probs[i])
            toxic = prob >= threshold
            color = "#E24B4A" if toxic else "#1D9E75"
            label = "TOXIC" if toxic else "SAFE"
            name  = TARGET_NAMES[col]

            st.markdown(
                f"**{col}** — {name}",
                unsafe_allow_html=False
            )

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(prob)
            with col_b:
                st.markdown(
                    f'<span style="color:{color};'
                    f'font-weight:bold">'
                    f'{prob:.1%} {label}</span>',
                    unsafe_allow_html=True
                )

    # Column 3 — SHAP + Model Comparison
    with col3:
        st.subheader(f"SHAP Analysis — {focus_task}")

        top_names, top_vals = get_shap_for_molecule(
            features, xgb_models,
            feature_names, task=focus_task
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        colors  = ['#E24B4A' if v > 0 else '#378ADD'
                   for v in top_vals]
        short   = [n[:20]+'...' if len(n)>20 else n
                   for n in top_names]

        ax.barh(range(len(top_names)),
                top_vals[::-1],
                color=colors[::-1],
                edgecolor='none', alpha=0.85)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(short[::-1], fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP value', fontsize=9)
        ax.set_title(
            f'Feature impact for {focus_task}',
            fontsize=10, fontweight='bold'
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Model comparison
        st.subheader("Model Agreement")
        comp_df = pd.DataFrame({
            'Model':    ['XGBoost', 'DNN', 'RF', 'Ensemble'],
            'Prediction': [
                f"{xgb_probs[TARGET_COLS.index(focus_task)]:.1%}",
                f"{dnn_probs[TARGET_COLS.index(focus_task)]:.1%}",
                f"{rf_probs[TARGET_COLS.index(focus_task)]:.1%}",
                f"{ens_probs[TARGET_COLS.index(focus_task)]:.1%}",
            ]
        })
        st.dataframe(
            comp_df,
            hide_index=True,
            use_container_width=True
        )

    # ── Full Results Table ─────────────────────────────────────
    st.subheader("Full Prediction Table")

    results_df = pd.DataFrame({
        'Task':        TARGET_COLS,
        'Assay':       [TARGET_NAMES[c] for c in TARGET_COLS],
        'XGBoost':     [f"{p:.3f}" for p in xgb_probs],
        'DNN':         [f"{p:.3f}" for p in dnn_probs],
        'RF':          [f"{p:.3f}" for p in rf_probs],
        'Ensemble':    [f"{p:.3f}" for p in ens_probs],
        'Result':      ['TOXIC' if p >= threshold else 'SAFE'
                        for p in ens_probs]
    })

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True
    )

    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name=f"toxicity_predictions_{smiles[:20]}.csv",
        mime="text/csv"
    )


if __name__ == '__main__':
    main()