

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# XGBoost
from xgboost import XGBClassifier

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────
PROCESSED_DIR = 'data/processed/'
MODELS_DIR    = 'models/'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────
def timer(start):
    elapsed = int(time.time() - start)
    m, s    = divmod(elapsed, 60)
    return f"{m}m {s}s"

def print_header(title):
    print("\n" + "█"*55)
    print(f"  {title}")
    print("█"*55)

def print_section(title):
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)


# ════════════════════════════════════════════════════════════════
# LOAD PROCESSED DATA
# ════════════════════════════════════════════════════════════════
def load_data():
    print_section("Loading processed data")

    X_train = np.load(f'{PROCESSED_DIR}X_train.npy')
    X_val   = np.load(f'{PROCESSED_DIR}X_val.npy')
    X_test  = np.load(f'{PROCESSED_DIR}X_test.npy')

    y_train = pd.read_csv(f'{PROCESSED_DIR}y_train.csv')
    y_val   = pd.read_csv(f'{PROCESSED_DIR}y_val.csv')
    y_test  = pd.read_csv(f'{PROCESSED_DIR}y_test.csv')

    with open(f'{PROCESSED_DIR}class_weights.json') as f:
        class_weights = json.load(f)

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {vram:.1f} GB")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            class_weights)


# ════════════════════════════════════════════════════════════════
# MODEL 1 — XGBOOST
# ════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_val, y_val, class_weights):
    print_header("MODEL 1 — XGBoost")
    start  = time.time()
    models = {}
    aucs   = {}

    for col in TARGET_COLS:
        tr_mask  = y_train[col] != -1
        val_mask = y_val[col]   != -1

        X_tr = X_train[tr_mask]
        y_tr = y_train.loc[tr_mask, col].values
        X_v  = X_val[val_mask]
        y_v  = y_val.loc[val_mask, col].values

        spw = class_weights.get(col, 1.0)

        clf = XGBClassifier(
            n_estimators          = 500,
            max_depth             = 6,
            learning_rate         = 0.05,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 3,
            gamma                 = 0.1,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            scale_pos_weight      = spw,
            tree_method           = 'hist',
            device                = 'cuda' if torch.cuda.is_available() else 'cpu',
            eval_metric           = 'auc',
            early_stopping_rounds = 40,
            random_state          = 42,
            verbosity             = 0
        )

        clf.fit(
            X_tr, y_tr,
            eval_set = [(X_v, y_v)],
            verbose  = False
        )

        preds = clf.predict_proba(X_v)[:, 1]
        auc   = roc_auc_score(y_v, preds)
        models[col] = clf
        aucs[col]   = round(auc, 4)
        print(f"  {col:<20} AUC: {auc:.4f}  "
              f"(trees: {clf.best_iteration})")

    mean_auc = round(np.mean(list(aucs.values())), 4)
    print(f"\n  Mean AUC : {mean_auc}")
    print(f"  Time     : {timer(start)}")

    pickle.dump(models, open(f'{MODELS_DIR}xgb_models.pkl', 'wb'))
    json.dump(aucs,     open(f'{MODELS_DIR}xgb_aucs.json', 'w'),
              indent=2)
    print(f"  Saved    : models/xgb_models.pkl")

    return models, aucs


# ════════════════════════════════════════════════════════════════
# MODEL 2 — MULTI-TASK DNN
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
        return torch.cat(
            [head(h) for head in self.heads], dim=1
        )


def masked_bce_loss(preds, labels, mask):
    loss = nn.BCEWithLogitsLoss(reduction='none')(
        preds, labels.clamp(0, 1)
    )
    return (loss * mask.float()).sum() / \
           (mask.float().sum() + 1e-8)


def get_dnn_aucs(model, loader):
    model.eval()
    all_preds, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for X_b, y_b, mask_b in loader:
            preds = torch.sigmoid(
                model(X_b.to(DEVICE))
            ).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_b.numpy())
            all_masks.append(mask_b.numpy())

    preds_arr  = np.vstack(all_preds)
    labels_arr = np.vstack(all_labels)
    masks_arr  = np.vstack(all_masks)

    aucs = []
    for i, col in enumerate(TARGET_COLS):
        m = masks_arr[:, i]
        if m.sum() > 10 and labels_arr[m, i].sum() > 0:
            aucs.append(
                roc_auc_score(labels_arr[m, i], preds_arr[m, i])
            )
    return (np.mean(aucs) if aucs else 0.0), preds_arr


def train_dnn(X_train, y_train, X_val, y_val,
              epochs=80, batch_size=256, lr=3e-4):
    print_header("MODEL 2 — Multi-Task DNN")
    start = time.time()

    input_dim = X_train.shape[1]
    print(f"  Input dim  : {input_dim}")
    print(f"  Batch size : {batch_size}")
    print(f"  Epochs     : {epochs}")
    print(f"  Device     : {DEVICE}")

    train_ds = ToxDataset(X_train, y_train)
    val_ds   = ToxDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=512,
                          shuffle=False, num_workers=0)

    model     = MultiTaskToxNet(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_auc     = 0
    patience_ctr = 0
    patience     = 20

    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Val AUC':>10}")
    print("  " + "-"*30)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_b, y_b, mask_b in train_dl:
            X_b    = X_b.to(DEVICE)
            y_b    = y_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_b)
            loss  = masked_bce_loss(preds, y_b, mask_b)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_dl)

        if epoch % 5 == 4 or epoch == 0:
            val_auc, val_preds = get_dnn_aucs(model, val_dl)

            if epoch % 10 == 9 or epoch == 0:
                print(f"  {epoch+1:>6} {avg_loss:>10.4f} "
                      f"{val_auc:>10.4f}")

            if val_auc > best_auc:
                best_auc     = val_auc
                patience_ctr = 0
                torch.save(model.state_dict(),
                           f'{MODELS_DIR}dnn_best.pt')
            else:
                patience_ctr += 1
                if patience_ctr >= patience // 5:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

    model.load_state_dict(
        torch.load(f'{MODELS_DIR}dnn_best.pt',
                   map_location=DEVICE)
    )

    print(f"\n  Best Val AUC : {best_auc:.4f}")
    print(f"  Time         : {timer(start)}")
    print(f"  Saved        : models/dnn_best.pt")

    val_dl2 = DataLoader(ToxDataset(X_val, y_val),
                         batch_size=512, shuffle=False,
                         num_workers=0)
    model.eval()
    all_preds, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for X_b, y_b, mask_b in val_dl2:
            p = torch.sigmoid(
                model(X_b.to(DEVICE))
            ).cpu().numpy()
            all_preds.append(p)
            all_labels.append(y_b.numpy())
            all_masks.append(mask_b.numpy())

    preds_arr  = np.vstack(all_preds)
    labels_arr = np.vstack(all_labels)
    masks_arr  = np.vstack(all_masks)

    dnn_aucs = {}
    print(f"\n  Per-task AUC:")
    for i, col in enumerate(TARGET_COLS):
        m = masks_arr[:, i]
        if m.sum() > 10 and labels_arr[m, i].sum() > 0:
            auc = roc_auc_score(labels_arr[m, i], preds_arr[m, i])
            dnn_aucs[col] = round(auc, 4)
            print(f"    {col:<20} {auc:.4f}")

    json.dump(dnn_aucs,
              open(f'{MODELS_DIR}dnn_aucs.json', 'w'), indent=2)

    return model, dnn_aucs, preds_arr


# ════════════════════════════════════════════════════════════════
# MODEL 3 — RANDOM FOREST
# ════════════════════════════════════════════════════════════════
def train_gnn(epochs=50):
    print_header("MODEL 3 — Random Forest")
    start = time.time()

    X_train = np.load(f'{PROCESSED_DIR}X_train.npy')
    X_val   = np.load(f'{PROCESSED_DIR}X_val.npy')
    y_train = pd.read_csv(f'{PROCESSED_DIR}y_train.csv')
    y_val   = pd.read_csv(f'{PROCESSED_DIR}y_val.csv')

    models    = {}
    rf_aucs   = {}
    val_preds = np.zeros((len(X_val), len(TARGET_COLS)))

    for i, col in enumerate(TARGET_COLS):
        tr_mask  = y_train[col] != -1
        val_mask = y_val[col]   != -1

        X_tr = X_train[tr_mask]
        y_tr = y_train.loc[tr_mask, col].values
        X_v  = X_val[val_mask]
        y_v  = y_val.loc[val_mask, col].values

        clf = RandomForestClassifier(
            n_estimators = 300,
            max_depth    = 10,
            class_weight = 'balanced',
            n_jobs       = -1,
            random_state = 42
        )
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_v)[:, 1]
        val_preds[val_mask, i] = probs
        auc          = roc_auc_score(y_v, probs)
        rf_aucs[col] = round(auc, 4)
        models[col]  = clf
        print(f"  {col:<20} AUC: {auc:.4f}")

    mean_auc = round(np.mean(list(rf_aucs.values())), 4)
    print(f"\n  RF Mean AUC : {mean_auc}")
    print(f"  Time        : {timer(start)}")

    pickle.dump(models,
                open(f'{MODELS_DIR}rf_models.pkl', 'wb'))
    json.dump(rf_aucs,
              open(f'{MODELS_DIR}rf_aucs.json', 'w'), indent=2)
    print(f"  Saved       : models/rf_models.pkl")

    return models, rf_aucs, val_preds


# ════════════════════════════════════════════════════════════════
# MODEL 4 — STACKING ENSEMBLE
# ════════════════════════════════════════════════════════════════
def train_ensemble(X_val, y_val, xgb_models,
                   dnn_model, rf_val_preds=None):
    print_header("MODEL 4 — Stacking Ensemble")
    start = time.time()

    print("  Getting XGBoost predictions...")
    xgb_probs = np.zeros((len(X_val), len(TARGET_COLS)))
    for i, col in enumerate(TARGET_COLS):
        probs = xgb_models[col].predict_proba(X_val)[:, 1]
        xgb_probs[:, i] = probs

    print("  Getting DNN predictions...")
    val_dl = DataLoader(
        ToxDataset(X_val, y_val),
        batch_size=512, shuffle=False, num_workers=0
    )
    dnn_model.eval()
    dnn_list = []
    with torch.no_grad():
        for X_b, _, _ in val_dl:
            p = torch.sigmoid(
                dnn_model(X_b.to(DEVICE))
            ).cpu().numpy()
            dnn_list.append(p)
    dnn_probs = np.vstack(dnn_list)

    meta_models   = {}
    ensemble_aucs = {}

    print(f"\n  {'Task':<20} {'XGB':>8} {'DNN':>8} "
          f"{'RF':>8} {'ENS':>8}")
    print("  " + "-"*55)

    for i, col in enumerate(TARGET_COLS):
        mask   = y_val[col] != -1
        y_true = y_val.loc[mask, col].values

        meta_cols = [xgb_probs[mask, i], dnn_probs[mask, i]]
        if rf_val_preds is not None:
            meta_cols.append(rf_val_preds[mask, i])

        meta_X  = np.column_stack(meta_cols)
        xgb_auc = roc_auc_score(y_true, xgb_probs[mask, i])
        dnn_auc = roc_auc_score(y_true, dnn_probs[mask, i])

        rf_str = "       -"
        if rf_val_preds is not None:
            rf_auc = roc_auc_score(y_true, rf_val_preds[mask, i])
            rf_str = f"{rf_auc:>8.4f}"

        base = LogisticRegression(C=1.0, max_iter=1000)
        meta = CalibratedClassifierCV(
            base, cv=3, method='isotonic'
        )
        meta.fit(meta_X, y_true)
        ens_preds = meta.predict_proba(meta_X)[:, 1]
        ens_auc   = roc_auc_score(y_true, ens_preds)

        meta_models[col]   = meta
        ensemble_aucs[col] = round(ens_auc, 4)

        print(f"  {col:<20} {xgb_auc:>8.4f} {dnn_auc:>8.4f} "
              f"{rf_str} {ens_auc:>8.4f}")

    mean_auc = round(np.mean(list(ensemble_aucs.values())), 4)
    print(f"\n  Ensemble Mean AUC : {mean_auc}")
    print(f"  Time              : {timer(start)}")

    pickle.dump(meta_models,
                open(f'{MODELS_DIR}meta_models.pkl', 'wb'))
    json.dump(ensemble_aucs,
              open(f'{MODELS_DIR}ensemble_aucs.json', 'w'),
              indent=2)
    print(f"  Saved : models/meta_models.pkl")

    return meta_models, ensemble_aucs


# ════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════
def print_final_table(xgb_aucs, dnn_aucs, rf_aucs, ens_aucs):
    print_header("FINAL RESULTS — All Models")

    print(f"\n  {'Task':<20} {'XGBoost':>9} {'DNN':>9} "
          f"{'RF':>9} {'Ensemble':>10}")
    print("  " + "-"*60)

    for col in TARGET_COLS:
        xgb = f"{xgb_aucs.get(col, 0):.4f}"
        dnn = f"{dnn_aucs.get(col, 0):.4f}"
        rf  = f"{rf_aucs.get(col, 0):.4f}" \
              if col in rf_aucs else "      -"
        ens = f"{ens_aucs.get(col, 0):.4f}"
        print(f"  {col:<20} {xgb:>9} {dnn:>9} "
              f"{rf:>9} {ens:>10}")

    print("  " + "-"*60)

    xgb_mean = np.mean(list(xgb_aucs.values()))
    dnn_mean = np.mean(list(dnn_aucs.values()))
    rf_mean  = np.mean(list(rf_aucs.values())) if rf_aucs else 0
    ens_mean = np.mean(list(ens_aucs.values()))

    print(f"  {'MEAN':<20} {xgb_mean:>9.4f} {dnn_mean:>9.4f} "
          f"{rf_mean:>9.4f} {ens_mean:>10.4f}")

    comparison = {
        col: {
            'xgboost':  xgb_aucs.get(col, 0),
            'dnn':      dnn_aucs.get(col, 0),
            'rf':       rf_aucs.get(col, 0),
            'ensemble': ens_aucs.get(col, 0)
        }
        for col in TARGET_COLS
    }
    json.dump(comparison,
              open(f'{MODELS_DIR}all_aucs_comparison.json', 'w'),
              indent=2)
    print(f"\n  Saved : models/all_aucs_comparison.json")


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════
def run():
    total_start = time.time()

    print_header("TOX21 TRAINING PIPELINE")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights) = load_data()

    xgb_models, xgb_aucs = train_xgboost(
        X_train, y_train, X_val, y_val, class_weights
    )

    dnn_model, dnn_aucs, dnn_val_preds = train_dnn(
        X_train, y_train, X_val, y_val,
        epochs=80, batch_size=256, lr=3e-4
    )

    rf_models, rf_aucs, rf_val_preds = train_gnn(epochs=50)

    meta_models, ens_aucs = train_ensemble(
        X_val, y_val,
        xgb_models, dnn_model, rf_val_preds
    )

    print_final_table(xgb_aucs, dnn_aucs, rf_aucs, ens_aucs)

    print_header("TRAINING COMPLETE")
    print(f"""
  Total time : {timer(total_start)}

  Models saved:
    models/xgb_models.pkl
    models/dnn_best.pt
    models/rf_models.pkl
    models/meta_models.pkl

  Results saved:
    models/xgb_aucs.json
    models/dnn_aucs.json
    models/rf_aucs.json
    models/ensemble_aucs.json
    models/all_aucs_comparison.json

  Next step → python src/evaluate.py
    """)


if __name__ == '__main__':
    run()