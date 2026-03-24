

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────
PROCESSED_DIR = 'data/processed/'
MODELS_DIR    = 'models/'
REPORTS_DIR   = 'reports/'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────
COLORS = {
    'xgb':      '#378ADD',
    'dnn':      '#E24B4A',
    'rf':       '#1D9E75',
    'ensemble': '#EF9F27'
}


# ════════════════════════════════════════════════════════════════
# DNN Architecture (must match train.py)
# ════════════════════════════════════════════════════════════════
class ToxDataset(Dataset):
    def __init__(self, X, y_df):
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.y    = torch.tensor(
            y_df[TARGET_COLS].values, dtype=torch.float32
        )
        self.mask = torch.tensor(
            y_df[TARGET_COLS].values != -1, dtype=torch.bool
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.mask[i]


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
        return torch.cat([head(h) for head in self.heads], dim=1)


# ════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ════════════════════════════════════════════════════════════════
def load_all():
    print("Loading data and models...")

    # Data
    X_test = np.load(f'{PROCESSED_DIR}X_test.npy')
    y_test = pd.read_csv(f'{PROCESSED_DIR}y_test.csv')

    # XGBoost
    xgb_models = pickle.load(
        open(f'{MODELS_DIR}xgb_models.pkl', 'rb')
    )

    # DNN
    input_dim  = X_test.shape[1]
    dnn_model  = MultiTaskToxNet(input_dim).to(DEVICE)
    dnn_model.load_state_dict(
        torch.load(f'{MODELS_DIR}dnn_best.pt',
                   map_location=DEVICE)
    )
    dnn_model.eval()

    # Random Forest
    rf_models = pickle.load(
        open(f'{MODELS_DIR}rf_models.pkl', 'rb')
    )

    # Ensemble meta-learner
    meta_models = pickle.load(
        open(f'{MODELS_DIR}meta_models.pkl', 'rb')
    )

    print(f"  X_test  : {X_test.shape}")
    print(f"  Device  : {DEVICE}")
    print(f"  Models  : XGBoost + DNN + RF + Ensemble loaded")

    return (X_test, y_test,
            xgb_models, dnn_model,
            rf_models, meta_models)


# ════════════════════════════════════════════════════════════════
# GET PREDICTIONS ON TEST SET
# ════════════════════════════════════════════════════════════════
def get_all_predictions(X_test, y_test,
                         xgb_models, dnn_model,
                         rf_models, meta_models):
    print("\nGetting predictions on test set...")
    n = len(X_test)
    n_tasks = len(TARGET_COLS)

    # XGBoost predictions
    xgb_probs = np.zeros((n, n_tasks))
    for i, col in enumerate(TARGET_COLS):
        xgb_probs[:, i] = xgb_models[col].predict_proba(
            X_test
        )[:, 1]
    print("  XGBoost predictions done")

    # DNN predictions
    test_ds = ToxDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=512,
                         shuffle=False, num_workers=0)
    dnn_preds_list = []
    with torch.no_grad():
        for X_b, _, _ in test_dl:
            p = torch.sigmoid(
                dnn_model(X_b.to(DEVICE))
            ).cpu().numpy()
            dnn_preds_list.append(p)
    dnn_probs = np.vstack(dnn_preds_list)
    print("  DNN predictions done")

    # RF predictions
    rf_probs = np.zeros((n, n_tasks))
    for i, col in enumerate(TARGET_COLS):
        rf_probs[:, i] = rf_models[col].predict_proba(
            X_test
        )[:, 1]
    print("  RF predictions done")

    # Ensemble predictions
    ens_probs = np.zeros((n, n_tasks))
    for i, col in enumerate(TARGET_COLS):
        meta_X = np.column_stack([
            xgb_probs[:, i],
            dnn_probs[:, i],
            rf_probs[:, i]
        ])
        ens_probs[:, i] = meta_models[col].predict_proba(
            meta_X
        )[:, 1]
    print("  Ensemble predictions done")

    return xgb_probs, dnn_probs, rf_probs, ens_probs


# ════════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ════════════════════════════════════════════════════════════════
def compute_metrics(y_test, xgb_probs, dnn_probs,
                    rf_probs, ens_probs):
    print("\nComputing metrics on test set...")

    results = {}

    for i, col in enumerate(TARGET_COLS):
        mask   = y_test[col] != -1
        y_true = y_test.loc[mask, col].values

        row = {}
        for name, probs in [('xgb', xgb_probs),
                             ('dnn', dnn_probs),
                             ('rf',  rf_probs),
                             ('ens', ens_probs)]:
            p = probs[mask, i]
            row[f'{name}_roc_auc'] = round(roc_auc_score(y_true, p), 4)
            row[f'{name}_pr_auc']  = round(
                average_precision_score(y_true, p), 4
            )

        row['n_test']    = int(mask.sum())
        row['n_pos']     = int(y_true.sum())
        row['pos_rate']  = round(float(y_true.mean()), 4)
        results[col]     = row

    # Print table
    print(f"\n  {'Task':<20} {'XGB':>8} {'DNN':>8} "
          f"{'RF':>8} {'ENS':>8}")
    print("  " + "-"*55)

    for col in TARGET_COLS:
        r = results[col]
        print(f"  {col:<20} "
              f"{r['xgb_roc_auc']:>8.4f} "
              f"{r['dnn_roc_auc']:>8.4f} "
              f"{r['rf_roc_auc']:>8.4f} "
              f"{r['ens_roc_auc']:>8.4f}")

    print("  " + "-"*55)

    for name in ['xgb', 'dnn', 'rf', 'ens']:
        mean = np.mean([results[c][f'{name}_roc_auc']
                        for c in TARGET_COLS])
        label = {'xgb': 'XGBoost', 'dnn': 'DNN',
                 'rf': 'RF', 'ens': 'Ensemble'}[name]
        print(f"  {label:<20} {mean:>8.4f} (mean ROC-AUC)")

    # Save
    json.dump(results,
              open(f'{MODELS_DIR}test_metrics.json', 'w'),
              indent=2)
    print(f"\n  Saved : models/test_metrics.json")

    return results


# ════════════════════════════════════════════════════════════════
# PLOT 1 — ROC CURVES (all tasks, ensemble only)
# ════════════════════════════════════════════════════════════════
def plot_roc_curves(y_test, ens_probs):
    print("\nPlotting ROC curves...")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(TARGET_COLS):
        mask   = y_test[col] != -1
        y_true = y_test.loc[mask, col].values
        probs  = ens_probs[mask, i]

        fpr, tpr, _ = roc_curve(y_true, probs)
        auc         = roc_auc_score(y_true, probs)

        axes[i].plot(fpr, tpr, color=COLORS['ensemble'],
                     linewidth=2, label=f'AUC = {auc:.3f}')
        axes[i].plot([0,1], [0,1], 'k--',
                     linewidth=0.8, alpha=0.5)
        axes[i].fill_between(fpr, tpr, alpha=0.08,
                              color=COLORS['ensemble'])
        axes[i].set_title(col, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('False Positive Rate', fontsize=8)
        axes[i].set_ylabel('True Positive Rate', fontsize=8)
        axes[i].legend(fontsize=9)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    plt.suptitle('ROC Curves — Ensemble Model (Test Set)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}07_roc_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/07_roc_curves.png")


# ════════════════════════════════════════════════════════════════
# PLOT 2 — MODEL COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════
def plot_model_comparison(results):
    print("\nPlotting model comparison...")

    tasks = TARGET_COLS
    x     = np.arange(len(tasks))
    w     = 0.2

    xgb_aucs = [results[c]['xgb_roc_auc'] for c in tasks]
    dnn_aucs = [results[c]['dnn_roc_auc'] for c in tasks]
    rf_aucs  = [results[c]['rf_roc_auc']  for c in tasks]
    ens_aucs = [results[c]['ens_roc_auc'] for c in tasks]

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - 1.5*w, xgb_aucs, w, label='XGBoost',
           color=COLORS['xgb'],      alpha=0.85, edgecolor='none')
    ax.bar(x - 0.5*w, dnn_aucs, w, label='DNN',
           color=COLORS['dnn'],      alpha=0.85, edgecolor='none')
    ax.bar(x + 0.5*w, rf_aucs,  w, label='Random Forest',
           color=COLORS['rf'],       alpha=0.85, edgecolor='none')
    ax.bar(x + 1.5*w, ens_aucs, w, label='Ensemble',
           color=COLORS['ensemble'], alpha=0.85, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('ROC-AUC', fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.set_title('Model comparison — ROC-AUC per task (Test Set)',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(0.8, color='gray', linestyle='--',
               linewidth=0.8, alpha=0.7, label='AUC=0.8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}08_model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/08_model_comparison.png")


# ════════════════════════════════════════════════════════════════
# PLOT 3 — PR CURVES (Precision-Recall)
# ════════════════════════════════════════════════════════════════
def plot_pr_curves(y_test, ens_probs):
    print("\nPlotting PR curves...")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(TARGET_COLS):
        mask      = y_test[col] != -1
        y_true    = y_test.loc[mask, col].values
        probs     = ens_probs[mask, i]
        pos_rate  = y_true.mean()

        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_auc       = average_precision_score(y_true, probs)

        axes[i].plot(rec, prec, color=COLORS['ensemble'],
                     linewidth=2, label=f'AP = {pr_auc:.3f}')
        axes[i].axhline(pos_rate, color='gray', linestyle='--',
                        linewidth=0.8,
                        label=f'Baseline = {pos_rate:.3f}')
        axes[i].fill_between(rec, prec, alpha=0.08,
                              color=COLORS['ensemble'])
        axes[i].set_title(col, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Recall', fontsize=8)
        axes[i].set_ylabel('Precision', fontsize=8)
        axes[i].legend(fontsize=8)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    plt.suptitle('Precision-Recall Curves — Ensemble (Test Set)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}09_pr_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/09_pr_curves.png")


# ════════════════════════════════════════════════════════════════
# PLOT 4 — MEAN AUC SUMMARY
# ════════════════════════════════════════════════════════════════
def plot_summary(results):
    print("\nPlotting summary...")

    models   = ['XGBoost', 'DNN', 'Random Forest', 'Ensemble']
    keys     = ['xgb', 'dnn', 'rf', 'ens']
    colors   = [COLORS['xgb'], COLORS['dnn'],
                COLORS['rf'],  COLORS['ensemble']]

    mean_aucs = [
        np.mean([results[c][f'{k}_roc_auc'] for c in TARGET_COLS])
        for k in keys
    ]
    pr_aucs = [
        np.mean([results[c][f'{k}_pr_auc'] for c in TARGET_COLS])
        for k in keys
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC-AUC bars
    bars = axes[0].bar(models, mean_aucs, color=colors,
                       edgecolor='none', alpha=0.85, width=0.5)
    axes[0].set_ylabel('Mean ROC-AUC', fontsize=11)
    axes[0].set_title('Mean ROC-AUC — all 12 tasks',
                      fontweight='bold')
    axes[0].set_ylim([0.6, 0.9])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for bar, val in zip(bars, mean_aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{val:.4f}', ha='center',
                     fontsize=10, fontweight='bold')

    # PR-AUC bars
    bars2 = axes[1].bar(models, pr_aucs, color=colors,
                        edgecolor='none', alpha=0.85, width=0.5)
    axes[1].set_ylabel('Mean PR-AUC', fontsize=11)
    axes[1].set_title('Mean PR-AUC — all 12 tasks',
                      fontweight='bold')
    axes[1].set_ylim([0.0, 0.6])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    for bar, val in zip(bars2, pr_aucs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center',
                     fontsize=10, fontweight='bold')

    plt.suptitle('Model Performance Summary (Test Set)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}10_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/10_summary.png")


# ════════════════════════════════════════════════════════════════
# PLOT 5 — CONFUSION MATRIX (best task)
# ════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_test, ens_probs, results):
    print("\nPlotting confusion matrices...")

    # Pick top 4 tasks by ensemble AUC
    top4 = sorted(TARGET_COLS,
                  key=lambda c: results[c]['ens_roc_auc'],
                  reverse=True)[:4]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, col in zip(axes, top4):
        i      = TARGET_COLS.index(col)
        mask   = y_test[col] != -1
        y_true = y_test.loc[mask, col].values
        probs  = ens_probs[mask, i]
        preds  = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_true, preds)
        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Safe', 'Toxic'])
        ax.set_yticklabels(['Safe', 'Toxic'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        auc = results[col]['ens_roc_auc']
        ax.set_title(f'{col}\nAUC={auc:.3f}', fontweight='bold')

        for r in range(2):
            for c_ in range(2):
                ax.text(c_, r, str(cm[r, c_]),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[r,c_] > cm.max()/2
                        else 'black')

    plt.suptitle('Confusion Matrices — Top 4 Tasks (threshold=0.5)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}11_confusion_matrices.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved : reports/11_confusion_matrices.png")


# ════════════════════════════════════════════════════════════════
# FINAL REPORT TABLE
# ════════════════════════════════════════════════════════════════
def print_final_report(results):
    print("\n" + "█"*60)
    print("  FINAL EVALUATION REPORT — TEST SET")
    print("█"*60)

    xgb_mean = np.mean([results[c]['xgb_roc_auc']
                        for c in TARGET_COLS])
    dnn_mean = np.mean([results[c]['dnn_roc_auc']
                        for c in TARGET_COLS])
    rf_mean  = np.mean([results[c]['rf_roc_auc']
                        for c in TARGET_COLS])
    ens_mean = np.mean([results[c]['ens_roc_auc']
                        for c in TARGET_COLS])

    ens_pr = np.mean([results[c]['ens_pr_auc']
                      for c in TARGET_COLS])

    best_task  = max(TARGET_COLS,
                     key=lambda c: results[c]['ens_roc_auc'])
    worst_task = min(TARGET_COLS,
                     key=lambda c: results[c]['ens_roc_auc'])

    print(f"""
  Test set size  : {results[TARGET_COLS[0]]['n_test']} compounds

  ROC-AUC (mean across 12 tasks):
    XGBoost      : {xgb_mean:.4f}
    DNN          : {dnn_mean:.4f}
    Random Forest: {rf_mean:.4f}
    Ensemble     : {ens_mean:.4f}  ← best

  PR-AUC (ensemble mean) : {ens_pr:.4f}

  Best task  : {best_task}
               AUC = {results[best_task]['ens_roc_auc']:.4f}
  Worst task : {worst_task}
               AUC = {results[worst_task]['ens_roc_auc']:.4f}
               (only {results[worst_task]['n_pos']} positives
                in test set)

  Reports saved:
    reports/07_roc_curves.png
    reports/08_model_comparison.png
    reports/09_pr_curves.png
    reports/10_summary.png
    reports/11_confusion_matrices.png
    models/test_metrics.json

  Next step → python src/explain.py
    """)


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════
def run():
    print("\n" + "█"*55)
    print("  TOX21 EVALUATION PIPELINE")
    print("█"*55)

    # Load
    (X_test, y_test,
     xgb_models, dnn_model,
     rf_models, meta_models) = load_all()

    # Predictions
    (xgb_probs, dnn_probs,
     rf_probs, ens_probs) = get_all_predictions(
        X_test, y_test,
        xgb_models, dnn_model,
        rf_models, meta_models
    )

    # Metrics
    results = compute_metrics(
        y_test, xgb_probs, dnn_probs,
        rf_probs, ens_probs
    )

    # Plots
    plot_roc_curves(y_test, ens_probs)
    plot_model_comparison(results)
    plot_pr_curves(y_test, ens_probs)
    plot_summary(results)
    plot_confusion_matrix(y_test, ens_probs, results)

    # Final report
    print_final_report(results)


if __name__ == '__main__':
    run()