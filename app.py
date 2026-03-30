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

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

# RDKit — safe imports only
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, MACCSkeys,
    rdMolDescriptors
)

# Draw — optional, handle DLL error
RDKIT_DRAW = False
try:
    from rdkit.Chem import Draw
    RDKIT_DRAW = True
except Exception:
    Draw = None

# PyTorch
import torch
import torch.nn as nn

# SHAP
import shap

# ── Config ────────────────────────────────────────────────────────
MODELS_DIR    = 'models/'
PROCESSED_DIR = 'data/processed/'
DEVICE        = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

TARGET_NAMES = {
    'NR-AR':         'Androgen Receptor',
    'NR-AR-LBD':     'Androgen Receptor LBD',
    'NR-AhR':        'Aryl Hydrocarbon Receptor',
    'NR-Aromatase':  'Aromatase',
    'NR-ER':         'Estrogen Receptor',
    'NR-ER-LBD':     'Estrogen Receptor LBD',
    'NR-PPAR-gamma': 'PPAR Gamma',
    'SR-ARE':        'Antioxidant Response',
    'SR-ATAD5':      'DNA Damage (ATAD5)',
    'SR-HSE':        'Heat Shock Response',
    'SR-MMP':        'Mitochondrial Membrane',
    'SR-p53':        'Tumor Suppressor p53'
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
    ('MolWt',               Descriptors.MolWt),
    ('MolLogP',             Descriptors.MolLogP),
    ('TPSA',                Descriptors.TPSA),
    ('NumHDonors',          Descriptors.NumHDonors),
    ('NumHAcceptors',       Descriptors.NumHAcceptors),
    ('NumRotatableBonds',   Descriptors.NumRotatableBonds),
    ('NumAromaticRings',    rdMolDescriptors.CalcNumAromaticRings),
    ('NumAliphaticRings',   rdMolDescriptors.CalcNumAliphaticRings),
    ('RingCount',           Descriptors.RingCount),
    ('FractionCSP3',        Descriptors.FractionCSP3),
    ('HeavyAtomCount',      Descriptors.HeavyAtomCount),
    ('NumHeteroatoms',      rdMolDescriptors.CalcNumHeteroatoms),
    ('MolMR',               Descriptors.MolMR),
    ('LabuteASA',           Descriptors.LabuteASA),
    ('BalabanJ',            Descriptors.BalabanJ),
    ('BertzCT',             Descriptors.BertzCT),
    ('Chi0',                Descriptors.Chi0),
    ('Chi1',                Descriptors.Chi1),
    ('Kappa1',              Descriptors.Kappa1),
    ('Kappa2',              Descriptors.Kappa2),
    ('Kappa3',              Descriptors.Kappa3),
    ('MaxPartialCharge',    Descriptors.MaxPartialCharge),
    ('MinPartialCharge',    Descriptors.MinPartialCharge),
    ('NHOHCount',           Descriptors.NHOHCount),
    ('NOCount',             Descriptors.NOCount),
    ('NumValenceElectrons', Descriptors.NumValenceElectrons),
    ('PEOE_VSA1',           Descriptors.PEOE_VSA1),
    ('PEOE_VSA2',           Descriptors.PEOE_VSA2),
    ('SMR_VSA1',            Descriptors.SMR_VSA1),
    ('SlogP_VSA1',          Descriptors.SlogP_VSA1),
]


def extract_features(mol):
    # Fix Deprecation Warning: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
    # Using the standard Morgan Fingerprint call for now to ensure all versions work
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
    return np.concatenate(
        [morgan, maccs, np.array(rdkit, dtype=np.float32)]
    )


def apply_variance_selector(features):
    try:
        selector = pickle.load(
            open(f'{MODELS_DIR}var_selector.pkl', 'rb')
        )
        return selector.transform(
            features.reshape(1, -1)
        )[0]
    except Exception:
        return features


# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
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

    sample_features = apply_variance_selector(
        np.zeros(2245)
    )
    input_dim = len(sample_features)
    dnn_model = MultiTaskToxNet(input_dim).to(DEVICE)
    dnn_model.load_state_dict(
        torch.load(
            f'{MODELS_DIR}dnn_best.pt',
            map_location=DEVICE
        )
    )
    dnn_model.eval()

    return (xgb_models, dnn_model,
            rf_models, meta_models, feature_names)


# ── Predict ───────────────────────────────────────────────────────
def predict_toxicity(mol, xgb_models, dnn_model,
                     rf_models, meta_models):
    raw_features = extract_features(mol)
    features     = apply_variance_selector(raw_features)
    X            = features.reshape(1, -1)

    xgb_probs = np.array([
        xgb_models[col].predict_proba(X)[0, 1]
        for col in TARGET_COLS
    ])

    X_tensor = torch.tensor(
        X, dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        dnn_probs = torch.sigmoid(
            dnn_model(X_tensor)
        ).cpu().numpy()[0]

    rf_probs = np.array([
        rf_models[col].predict_proba(X)[0, 1]
        for col in TARGET_COLS
    ])

    ens_probs = np.array([
        meta_models[col].predict_proba(
            np.array([[xgb_probs[i],
                       dnn_probs[i],
                       rf_probs[i]]])
        )[0, 1]
        for i, col in enumerate(TARGET_COLS)
    ])

    return xgb_probs, dnn_probs, rf_probs, ens_probs, features


# ── SHAP ──────────────────────────────────────────────────────────
def get_shap_for_molecule(features, xgb_models,
                           feature_names, task='SR-ARE'):
    try:
        xgb_clf   = xgb_models[task]
        explainer = shap.Explainer(
            xgb_clf, features.reshape(1, -1)
        )
        shap_obj  = explainer(features.reshape(1, -1))
        shap_vals = shap_obj.values[0]
        top_idx   = np.argsort(
            np.abs(shap_vals)
        )[::-1][:10]
        return ([feature_names[i] for i in top_idx],
                shap_vals[top_idx])
    except Exception:
        imp     = xgb_models[task].feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        return ([feature_names[i] for i in top_idx],
                imp[top_idx])


# ── Draw Molecule ─────────────────────────────────────────────────
def draw_molecule(mol, size=(300, 250)):
    if not RDKIT_DRAW or Draw is None:
        return None
    try:
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════════════
def get_app_styles():
    return """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    .stApp { background-color: #f1f5f9; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, b, strong { color: #0f172a !important; font-family: 'Outfit', sans-serif; }
    .glass-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px 0 rgb(0 0 0/0.1); margin-bottom: 20px; }
    .hero-box { background: #1e293b; padding: 30px; border-radius: 15px; color: white !important; text-align: center; margin-bottom: 30px; }
    .hero-box h1 { color: white !important; }
    .hero-box p { color: #94a3b8 !important; }
    .prop-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; }
    .prop-box { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px; }
    .prop-box .lbl { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 700; margin-bottom: 2px; }
    .prop-box .val { font-size: 1rem; color: #0f172a; font-weight: 800; }
    .res-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .pill { padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 800; }
    .pill-safe { background: #dcfce7; color: #166534; }
    .pill-toxic { background: #fee2e2; color: #991b1b; }
    section[data-testid="stSidebar"] { background-color: #0f172a !important; }
    section[data-testid="stSidebar"] * { color: #f8fafc !important; }
    .stProgress > div > div > div > div { background-color: #3b82f6; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

def main():
    st.set_page_config(
        page_title="ToxPredict Intelligence",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Style Injection ───────────────────────────────────────────────
    st.markdown(get_app_styles(), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Neural Engine")
        st.caption("Active Stack: XGB + DNN + RF")
        st.markdown("---")
        st.markdown("### 🧪 Quick-Launch SMILES")
        examples = {
            "💊 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
            "🩹 Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
            "☕ Caffeine":    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "🏃 Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "☣️ Bisphenol A": "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
        }
        for name, smi in examples.items():
            if st.button(name, use_container_width=True):
                st.session_state['smiles'] = smi

        st.markdown("---")
        st.markdown("### ⚙️ Engine Parameters")
        threshold = st.slider("Conf. Threshold", 0.1, 0.9, 0.5, 0.05)
        focus_task = st.selectbox("Intelligence Focus", TARGET_COLS, index=7)
        st.markdown("---")
        st.markdown("**Version 2.3.0**")

    # Hero
    st.markdown("""
        <div class="hero-box">
            <h1>ToxPredict Engine</h1>
            <p>High-Fidelity Toxicology Prediction Framework</p>
        </div>
    """, unsafe_allow_html=True)

    # Core Loading
    with st.spinner("Calibrating Decision Matrix..."):
        try:
            (xgb_models, dnn_model, rf_models, meta_models, feature_names) = load_models()
        except Exception as e:
            st.error(f"Engine Fault: {e}")
            return

    # User Input
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Molecular Sequence Analysis")
    default_smiles = st.session_state.get('smiles', 'CC(=O)Nc1ccc(O)cc1')
    smiles = st.text_input("Enter sequence", value=default_smiles, placeholder="e.g. CC(=O)Nc1ccc(O)cc1", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if not smiles:
        st.info("Awaiting input to begin simulation.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Simulation Aborted: Invalid SMILES detected.")
        return

    # Simulation
    with st.spinner("Running bio-toxicological simulations..."):
        (xgb_probs, dnn_probs, rf_probs, ens_probs, features) = predict_toxicity(mol, xgb_models, dnn_model, rf_models, meta_models)

    # Interface
    c_left, c_right = st.columns([1, 1.8], gap="small")

    with c_left:
        st.markdown("#### 🧪 Molecular Topology")
        img = draw_molecule(mol, size=(600, 500))
        if img:
            st.image(img, use_container_width=True)
        else:
            st.code(smiles)

        st.markdown("#### 🔬 Physicochemical Profile")
        props = {
            "Weight": f"{Descriptors.MolWt(mol):.1f}",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "TPSA": f"{Descriptors.TPSA(mol):.1f}",
            "Donor": f"{Descriptors.NumHDonors(mol)}",
            "Acceptor": f"{Descriptors.NumHAcceptors(mol)}",
            "Rings": f"{Descriptors.RingCount(mol)}",
        }
        
        # Fixed HTML Rendering: No Indentation
        prop_html = '<div class="prop-row">'
        for p, v in props.items():
            prop_html += f'<div class="prop-box"><div class="lbl">{p}</div><div class="val">{v}</div></div>'
        prop_html += '</div>'
        st.markdown(prop_html, unsafe_allow_html=True)

        # Score
        n_tox = int((ens_probs >= threshold).sum())
        score = n_tox / len(TARGET_COLS) * 100
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Final Tox Index</p>
                <h2 style="font-size: 3rem; margin: 5px 0;">{score:.0f}%</h2>
                <p style="color: #94a3b8; font-size: 0.8rem;">Detected in {n_tox}/12 active assays</p>
            </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("#### 📊 Decision Intelligence Logs")
        for i, col in enumerate(TARGET_COLS):
            prob = float(ens_probs[i])
            tox = prob >= threshold
            pill = "pill-toxic" if tox else "pill-safe"
            text = "TOXIC" if tox else "SAFE"
            st.markdown(f"""
                <div class="res-card">
                    <div style="font-weight: 600; font-size: 0.9rem;">{col} <span style="font-weight: 300; color: #94a3b8; font-size: 0.75rem;">— {TARGET_NAMES[col]}</span></div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 800;">{prob:.1%}</span>
                        <span class="pill {pill}">{text}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(prob)

    # Explainer
    st.markdown("---")
    st.markdown(f"#### 📈 Attribution Analysis ({focus_task})")
    a1, a2 = st.columns([2, 1])
    with a1:
        top_names, top_vals = get_shap_for_molecule(features, xgb_models, feature_names, task=focus_task)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(top_names)), top_vals[::-1], color=['#f43f5e' if v > 0 else '#3b82f6' for v in top_vals[::-1]], alpha=0.9)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels([n[:30] for n in top_names[::-1]], fontsize=9)
        ax.set_xlabel('Score Impact')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with a2:
        st.markdown("##### Consensus Logs")
        idx = TARGET_COLS.index(focus_task)
        st.dataframe(pd.DataFrame({
            'Engine': ['XGB', 'DNN', 'RF', 'Meta'],
            'Confidence': [f"{xgb_probs[idx]:.1%}", f"{dnn_probs[idx]:.1%}", f"{rf_probs[idx]:.1%}", f"{ens_probs[idx]:.1%}"]
        }), hide_index=True, use_container_width=True) # Note: user version may require width="stretch" but 1.34.0+ uses width
        st.info("Aggregated confidence values represent the weighted average of the neural stack.")

if __name__ == '__main__':
    main()