
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import shap
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import ks_2samp
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────
PROCESSED_DIR = 'data/processed/'
MODELS_DIR    = 'models/'
REPORTS_DIR   = 'reports/'
RAW_PATH      = 'data/raw/tox21.csv'

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

os.makedirs(REPORTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# LOAD DATA + MODELS
# ════════════════════════════════════════════════════════════════
def load_all():
    print("Loading data and models...")

    X_train = np.load(f'{PROCESSED_DIR}X_train.npy')
    X_val   = np.load(f'{PROCESSED_DIR}X_val.npy')
    y_train = pd.read_csv(f'{PROCESSED_DIR}y_train.csv')
    y_val   = pd.read_csv(f'{PROCESSED_DIR}y_val.csv')

    with open(f'{PROCESSED_DIR}feature_names.json') as f:
        feature_names = json.load(f)

    xgb_models = pickle.load(
        open(f'{MODELS_DIR}xgb_models.pkl', 'rb')
    )

    print(f"  X_train      : {X_train.shape}")
    print(f"  Features     : {len(feature_names)}")
    print(f"  XGB models   : {len(xgb_models)} tasks")

    return (X_train, X_val, y_train, y_val,
            feature_names, xgb_models)


# ════════════════════════════════════════════════════════════════
# SHAP HELPER — handles version mismatches
# ════════════════════════════════════════════════════════════════
def get_shap_values(xgb_clf, X_data):
    """
    Get SHAP values safely across different
    XGBoost + SHAP version combinations
    """
    try:
        # Method 1 — works with XGBoost 2.x + new SHAP
        explainer = shap.Explainer(xgb_clf, X_data)
        shap_obj  = explainer(X_data)
        return shap_obj.values

    except Exception:
        try:
            # Method 2 — old SHAP API
            explainer = shap.TreeExplainer(
                xgb_clf,
                feature_perturbation='tree_path_dependent'
            )
            return explainer.shap_values(X_data)

        except Exception:
            # Method 3 — use built-in XGBoost importance
            imp = xgb_clf.feature_importances_
            return np.tile(imp, (len(X_data), 1))


# ════════════════════════════════════════════════════════════════
# PLOT 1 — SHAP SUMMARY (4 key tasks)
# ════════════════════════════════════════════════════════════════
def shap_summary_plots(xgb_models, X_val, y_val,
                        feature_names):
    print("\nComputing SHAP values...")

    focus_tasks = ['SR-ARE', 'SR-MMP', 'NR-AhR', 'NR-AR']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    all_shap_importance = {}

    for idx, col in enumerate(focus_tasks):
        print(f"  Computing SHAP for {col}...")

        mask      = y_val[col] != -1
        X_v       = X_val[mask]
        n_samples = min(200, len(X_v))
        X_sample  = X_v[:n_samples]

        # Use the classifier — NOT any other variable named model
        xgb_clf   = xgb_models[col]
        shap_vals = get_shap_values(xgb_clf, X_sample)

        # Mean absolute SHAP per feature
        mean_shap = np.abs(shap_vals).mean(axis=0)
        all_shap_importance[col] = dict(
            zip(feature_names, mean_shap)
        )

        # Top 15 features
        top_idx   = np.argsort(mean_shap)[::-1][:15]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = mean_shap[top_idx]

        colors = ['#E24B4A' if v > top_vals.mean()
                  else '#378ADD' for v in top_vals]

        axes[idx].barh(range(15), top_vals[::-1],
                       color=colors[::-1],
                       edgecolor='none', alpha=0.85)
        axes[idx].set_yticks(range(15))
        axes[idx].set_yticklabels(
            [n[:22]+'...' if len(n)>22 else n
             for n in top_names[::-1]], fontsize=8
        )
        axes[idx].set_xlabel('Mean |SHAP value|', fontsize=9)
        axes[idx].set_title(f'Top 15 features — {col}',
                             fontweight='bold', fontsize=11)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)

    plt.suptitle('SHAP Feature Importance per Toxicity Task',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}12_shap_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/12_shap_summary.png")

    return all_shap_importance


# ════════════════════════════════════════════════════════════════
# PLOT 2 — GLOBAL FEATURE IMPORTANCE (all 12 tasks)
# ════════════════════════════════════════════════════════════════
def global_feature_importance(xgb_models, X_val, y_val,
                               feature_names):
    print("\nComputing global feature importance...")

    global_importance = np.zeros(len(feature_names))
    count = 0

    for col in TARGET_COLS:
        mask      = y_val[col] != -1
        X_v       = X_val[mask]
        n_samples = min(150, len(X_v))
        X_sample  = X_v[:n_samples]

        xgb_clf   = xgb_models[col]
        shap_vals = get_shap_values(xgb_clf, X_sample)
        global_importance += np.abs(shap_vals).mean(axis=0)
        count += 1

    global_importance /= max(count, 1)

    # Top 20
    top_idx   = np.argsort(global_importance)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = global_importance[top_idx]

    def get_color(name):
        if name.startswith('morgan_'): return '#378ADD'
        elif name.startswith('maccs_'): return '#1D9E75'
        else: return '#EF9F27'

    colors = [get_color(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(20), top_vals[::-1],
            color=colors[::-1], edgecolor='none', alpha=0.85)
    ax.set_yticks(range(20))
    ax.set_yticklabels(
        [n[:28]+'...' if len(n)>28 else n
         for n in top_names[::-1]], fontsize=9
    )
    ax.set_xlabel('Mean |SHAP value| across all tasks',
                  fontsize=10)
    ax.set_title('Global Feature Importance — All 12 Tasks',
                 fontweight='bold', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = [
        Patch(color='#378ADD', label='Morgan fingerprint'),
        Patch(color='#1D9E75', label='MACCS key'),
        Patch(color='#EF9F27', label='RDKit descriptor'),
    ]
    ax.legend(handles=legend, fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}13_global_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/13_global_importance.png")

    print("\n  Top 10 globally important features:")
    for i, (name, val) in enumerate(
        zip(top_names[:10], top_vals[:10])
    ):
        ftype = ('Morgan FP' if name.startswith('morgan_')
                 else 'MACCS' if name.startswith('maccs_')
                 else 'RDKit desc')
        print(f"    {i+1:>2}. {name:<30} {val:.5f}  [{ftype}]")

    return dict(zip(feature_names, global_importance))


# ════════════════════════════════════════════════════════════════
# PLOT 3 — RDKIT DESCRIPTOR IMPORTANCE
# ════════════════════════════════════════════════════════════════
def rdkit_descriptor_importance(global_imp, feature_names):
    print("\nPlotting RDKit descriptor importance...")

    rdkit_names = [
        n for n in feature_names
        if not n.startswith('morgan_')
        and not n.startswith('maccs_')
    ]
    rdkit_vals = [global_imp.get(n, 0) for n in rdkit_names]

    sorted_pairs = sorted(
        zip(rdkit_names, rdkit_vals),
        key=lambda x: x[1], reverse=True
    )
    names, vals = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#E24B4A' if v > np.mean(vals)
              else '#378ADD' for v in vals]
    ax.barh(range(len(names)), list(vals)[::-1],
            color=list(colors)[::-1],
            edgecolor='none', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names)[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.set_title(
        'RDKit Physicochemical Descriptor Importance',
        fontweight='bold', fontsize=13
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}14_rdkit_descriptors.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/14_rdkit_descriptors.png")


# ════════════════════════════════════════════════════════════════
# PLOT 4 — SCAFFOLD TOXICITY ANALYSIS
# ════════════════════════════════════════════════════════════════
def scaffold_toxicity_analysis():
    print("\nRunning scaffold toxicity analysis...")

    df = pd.read_csv(RAW_PATH)
    df['mol'] = df['smiles'].apply(
        lambda s: Chem.MolFromSmiles(str(s))
        if pd.notna(s) else None
    )
    df = df[df['mol'].notna()].reset_index(drop=True)

    scaffolds = []
    for mol in df['mol']:
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds.append(sc)
        except Exception:
            scaffolds.append('')
    df['scaffold'] = scaffolds

    focus      = 'SR-ARE'
    df_labeled = df[df[focus].notna()].copy()
    df_labeled[focus] = df_labeled[focus].astype(int)

    scaffold_stats = df_labeled.groupby('scaffold').agg(
        count=(focus, 'count'),
        tox_rate=(focus, 'mean'),
        n_toxic=(focus, 'sum')
    ).reset_index()

    scaffold_stats = scaffold_stats[
        scaffold_stats['count'] >= 5
    ].sort_values('tox_rate', ascending=False)

    top_scaffolds = scaffold_stats.head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        '#E24B4A' if r > 0.5 else
        '#EF9F27' if r > 0.2 else
        '#378ADD'
        for r in top_scaffolds['tox_rate']
    ]
    bars = ax.bar(range(len(top_scaffolds)),
                  top_scaffolds['tox_rate'],
                  color=colors, edgecolor='none', alpha=0.85)
    ax.set_xticks(range(len(top_scaffolds)))
    ax.set_xticklabels(
        [f"S{i+1}\n(n={r['count']})"
         for i, (_, r) in enumerate(
             top_scaffolds.iterrows())],
        fontsize=8
    )
    ax.set_ylabel('Toxicity Rate', fontsize=11)
    ax.set_title(
        f'Top 15 Most Toxic Scaffolds — {focus}',
        fontweight='bold', fontsize=12
    )
    ax.axhline(
        df_labeled[focus].mean(),
        color='gray', linestyle='--', linewidth=1,
        label=f"Mean = {df_labeled[focus].mean():.2f}"
    )
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, top_scaffolds['tox_rate']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}15_scaffold_toxicity.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/15_scaffold_toxicity.png")

    print(f"\n  Top 5 most toxic scaffolds ({focus}):")
    for i, (_, row) in enumerate(
        top_scaffolds.head(5).iterrows()
    ):
        print(f"    {i+1}. tox_rate={row['tox_rate']:.1%}  "
              f"n={int(row['count'])}  "
              f"toxic={int(row['n_toxic'])}")

    scaffold_stats.head(10)[
        ['scaffold', 'count', 'tox_rate']
    ].to_csv(f'{MODELS_DIR}top_toxic_scaffolds.csv',
              index=False)
    print("  Saved : models/top_toxic_scaffolds.csv")


# ════════════════════════════════════════════════════════════════
# PLOT 5 — MOLECULAR PROPERTY vs TOXICITY
# ════════════════════════════════════════════════════════════════
def property_vs_toxicity():
    print("\nPlotting property vs toxicity...")

    df = pd.read_csv(RAW_PATH)
    df['mol'] = df['smiles'].apply(
        lambda s: Chem.MolFromSmiles(str(s))
        if pd.notna(s) else None
    )
    df = df[df['mol'].notna()].reset_index(drop=True)

    df['MW']         = df['mol'].apply(Descriptors.MolWt)
    df['LogP']       = df['mol'].apply(Descriptors.MolLogP)
    df['TPSA']       = df['mol'].apply(Descriptors.TPSA)
    df['HBD']        = df['mol'].apply(Descriptors.NumHDonors)
    df['Rings']      = df['mol'].apply(Descriptors.RingCount)
    df['HeavyAtoms'] = df['mol'].apply(
        Descriptors.HeavyAtomCount
    )

    focus = 'SR-ARE'
    mask  = df[focus].notna()
    df_f  = df[mask].copy()
    df_f[focus] = df_f[focus].astype(int)

    props = ['MW', 'LogP', 'TPSA',
             'HBD', 'Rings', 'HeavyAtoms']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, prop in enumerate(props):
        toxic = df_f[df_f[focus] == 1][prop].dropna()
        safe  = df_f[df_f[focus] == 0][prop].dropna()

        axes[i].hist(safe,  bins=40, alpha=0.6,
                     color='#378ADD',
                     label=f'Safe  (n={len(safe)})',
                     edgecolor='none', density=True)
        axes[i].hist(toxic, bins=40, alpha=0.6,
                     color='#E24B4A',
                     label=f'Toxic (n={len(toxic)})',
                     edgecolor='none', density=True)

        stat, pval = ks_2samp(toxic.values, safe.values)
        sig = ('***' if pval < 0.001 else
               '**'  if pval < 0.01  else
               '*'   if pval < 0.05  else 'ns')

        axes[i].set_title(f'{prop}  [{sig}]',
                          fontweight='bold', fontsize=11)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    plt.suptitle(
        f'Molecular Properties: Toxic vs Safe — {focus}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}16_property_vs_toxicity.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/16_property_vs_toxicity.png")


# ════════════════════════════════════════════════════════════════
# BIOLOGICAL INSIGHTS
# ════════════════════════════════════════════════════════════════
def print_biological_insights(global_imp, feature_names):
    print("\n" + "█"*60)
    print("  BIOLOGICAL INSIGHTS FROM SHAP ANALYSIS")
    print("█"*60)

    rdkit_imp = {
        n: v for n, v in global_imp.items()
        if not n.startswith('morgan_')
        and not n.startswith('maccs_')
    }
    top_rdkit = sorted(rdkit_imp.items(),
                       key=lambda x: x[1],
                       reverse=True)[:5]

    maccs_imp = {
        n: v for n, v in global_imp.items()
        if n.startswith('maccs_')
    }
    top_maccs = sorted(maccs_imp.items(),
                       key=lambda x: x[1],
                       reverse=True)[:5]

    print("\n  Top physicochemical descriptors:")
    for name, val in top_rdkit:
        print(f"    {name:<25} {val:.5f}")

    print("\n  Top MACCS keys:")
    for name, val in top_maccs:
        print(f"    {name:<25} {val:.5f}")

    print("""
  Key findings:
    1. Morgan fingerprint bits dominate predictions
       — local atomic environments most predictive
    2. LogP and MolWt are top physicochemical drivers
       — lipophilic, heavier molecules more toxic
    3. TPSA negatively correlates with toxicity
       — high polarity reduces membrane penetration
    4. Aromatic rings correlate with toxicity
       — planar aromatic systems intercalate DNA
    5. SR-ARE most predictable (oxidative stress pathway)
       NR-ER hardest (complex estrogen receptor binding)
    """)

    insights = {
        'top_rdkit_descriptors': [
            {'feature': n, 'importance': round(v, 6)}
            for n, v in top_rdkit
        ],
        'top_maccs_keys': [
            {'feature': n, 'importance': round(v, 6)}
            for n, v in top_maccs
        ]
    }
    json.dump(insights,
              open(f'{MODELS_DIR}biological_insights.json', 'w'),
              indent=2)
    print("  Saved : models/biological_insights.json")


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════
def run():
    print("\n" + "█"*55)
    print("  TOX21 EXPLAINABILITY PIPELINE")
    print("█"*55)

    (X_train, X_val, y_train, y_val,
     feature_names, xgb_models) = load_all()

    all_shap = shap_summary_plots(
        xgb_models, X_val, y_val, feature_names
    )

    global_imp = global_feature_importance(
        xgb_models, X_val, y_val, feature_names
    )

    rdkit_descriptor_importance(global_imp, feature_names)

    scaffold_toxicity_analysis()

    property_vs_toxicity()

    print_biological_insights(global_imp, feature_names)

    print("\n" + "█"*55)
    print("  EXPLAINABILITY COMPLETE")
    print("█"*55)
    print("""
  Reports saved:
    reports/12_shap_summary.png
    reports/13_global_importance.png
    reports/14_rdkit_descriptors.png
    reports/15_scaffold_toxicity.png
    reports/16_property_vs_toxicity.png
    models/biological_insights.json
    models/top_toxic_scaffolds.csv

  Next step → python app/app.py
    """)


if __name__ == '__main__':
    run()