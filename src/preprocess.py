

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, MACCSkeys,
    rdMolDescriptors, SaltRemover
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# Sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

# ── Config ────────────────────────────────────────────────────────
RAW_PATH      = 'Data/raw/tox21.csv'
PROCESSED_DIR = 'data/processed/'
MODELS_DIR    = 'models/'

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Create output folders
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs('reports/',    exist_ok=True)
os.makedirs('logs/',       exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ════════════════════════════════════════════════════════════════
def step1_load(path: str) -> pd.DataFrame:
    print("\n" + "="*55)
    print("STEP 1 — Loading data")
    print("="*55)

    df = pd.read_csv(path)

    print(f"  Shape          : {df.shape}")
    print(f"  Compounds      : {len(df)}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  SMILES null    : {df['smiles'].isna().sum()}")
    print(f"  SMILES dupes   : {df['smiles'].duplicated().sum()}")

    print(f"\n  Target missing % per task:")
    for col in TARGET_COLS:
        pct = df[col].isna().mean() * 100
        print(f"    {col:<20} {pct:.1f}%")

    return df


# ════════════════════════════════════════════════════════════════
# STEP 2 — Clean SMILES + Standardize Molecules
# ════════════════════════════════════════════════════════════════
def clean_mol(smiles: str):
    """
    Full molecule cleaning pipeline:
    1. Parse SMILES
    2. Remove salts
    3. Keep largest fragment
    4. Normalize + uncharge
    5. Sanitize
    """
    if pd.isna(smiles) or str(smiles).strip() == '':
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None

        # Remove salts
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        # Keep largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        # Normalize
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        # Uncharge
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Sanitize
        Chem.SanitizeMol(mol)

        # Remove tiny molecules
        if mol.GetNumHeavyAtoms() < 3:
            return None

        return mol

    except Exception:
        return None


def step2_clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*55)
    print("STEP 2 — Cleaning SMILES")
    print("="*55)

    df['mol'] = [
        clean_mol(s)
        for s in tqdm(df['smiles'], desc="  Cleaning")
    ]

    before = len(df)
    df = df[df['mol'].notna()].reset_index(drop=True)
    after  = len(df)

    print(f"  Before cleaning : {before}")
    print(f"  After cleaning  : {after}")
    print(f"  Removed         : {before - after}")

    # Canonical SMILES
    df['canonical_smiles'] = df['mol'].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    # Remove duplicates after canonicalization
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset='canonical_smiles'
    ).reset_index(drop=True)
    print(f"  Duplicates removed: {before_dedup - len(df)}")
    print(f"  Final count     : {len(df)}")

    return df


# ════════════════════════════════════════════════════════════════
# STEP 3 — Process Labels
# ════════════════════════════════════════════════════════════════
def step3_labels(df: pd.DataFrame):
    print("\n" + "="*55)
    print("STEP 3 — Processing labels")
    print("="*55)

    # NaN → -1 sentinel (critical for masked loss)
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(-1).astype(int)

    # Compute class weights for XGBoost
    class_weights = {}
    label_stats   = {}

    print(f"\n  {'Task':<20} {'Labeled':>8} {'Pos':>6} "
          f"{'Neg':>7} {'PosRate':>8} {'Weight':>8}")
    print("  " + "-"*60)

    for col in TARGET_COLS:
        mask  = df[col] != -1
        n_lab = int(mask.sum())
        n_pos = int((df.loc[mask, col] == 1).sum())
        n_neg = int((df.loc[mask, col] == 0).sum())
        ratio = round(n_neg / max(n_pos, 1), 1)
        pos_rate = round(n_pos / max(n_lab, 1) * 100, 1)

        class_weights[col] = ratio
        label_stats[col]   = {
            'labeled':      n_lab,
            'positive':     n_pos,
            'negative':     n_neg,
            'pos_rate_pct': pos_rate,
            'class_weight': ratio
        }
        print(f"  {col:<20} {n_lab:>8} {n_pos:>6} "
              f"{n_neg:>7} {pos_rate:>7}% {ratio:>8}x")

    return df, class_weights, label_stats


# ════════════════════════════════════════════════════════════════
# STEP 4 — Feature Extraction
# ════════════════════════════════════════════════════════════════

# RDKit descriptors we care about
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


def get_morgan_fp(mol, radius=2, nbits=2048):
    """ECFP4 — captures circular atomic environments"""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=nbits, useChirality=True
    )
    return np.array(fp, dtype=np.float32)


def get_maccs_keys(mol):
    """166-bit MACCS structural keys"""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def get_rdkit_descriptors(mol):
    """30 physicochemical descriptors"""
    vals = []
    for name, fn in RDKIT_DESCS:
        try:
            v = float(fn(mol))
            vals.append(0.0 if (np.isnan(v) or np.isinf(v)) else v)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


def extract_features_single(mol) -> np.ndarray:
    """All features for one molecule concatenated"""
    morgan = get_morgan_fp(mol)       # 2048
    maccs  = get_maccs_keys(mol)      # 167
    rdkit  = get_rdkit_descriptors(mol)  # 30
    return np.concatenate([morgan, maccs, rdkit])  # 2245 total


def step4_features(df: pd.DataFrame):
    print("\n" + "="*55)
    print("STEP 4 — Extracting molecular features")
    print("="*55)
    print("  Morgan FP  : 2048 bits (ECFP4, radius=2)")
    print("  MACCS keys : 167 bits")
    print("  RDKit desc : 30 physicochemical")
    print("  Total      : 2245 features per molecule")
    print()

    all_features = []
    for mol in tqdm(df['mol'], desc="  Extracting"):
        all_features.append(extract_features_single(mol))

    X = np.array(all_features, dtype=np.float32)
    print(f"\n  Feature matrix shape: {X.shape}")

    # Build feature names (needed for SHAP later)
    feature_names = (
        [f'morgan_{i}'    for i in range(2048)] +
        [f'maccs_{i}'     for i in range(167)]  +
        [name for name, _ in RDKIT_DESCS]
    )
    print(f"  Feature names count: {len(feature_names)}")

    return X, feature_names


# ════════════════════════════════════════════════════════════════
# STEP 5 — Clean Feature Matrix
# ════════════════════════════════════════════════════════════════
def step5_clean_features(X: np.ndarray, feature_names: list):
    print("\n" + "="*55)
    print("STEP 5 — Cleaning feature matrix")
    print("="*55)
    print(f"  Input shape : {X.shape}")

    # Fix NaN and Inf
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    print(f"  NaN found   : {nan_count}")
    print(f"  Inf found   : {inf_count}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip outliers on descriptor columns only
    # First 2215 columns are binary (Morgan + MACCS) — don't clip
    N_BINARY = 2048 + 167
    if X.shape[1] > N_BINARY:
        desc = X[:, N_BINARY:]
        mean = desc.mean(axis=0)
        std  = desc.std(axis=0) + 1e-8
        X[:, N_BINARY:] = np.clip(
            desc,
            mean - 10 * std,
            mean + 10 * std
        )
        print(f"  Outlier clip: applied to {desc.shape[1]} descriptor columns")

    # Remove near-zero variance features
    selector = VarianceThreshold(threshold=0.01)
    X_clean  = selector.fit_transform(X)
    kept_mask = selector.get_support()
    removed   = int((~kept_mask).sum())
    print(f"  Low variance removed: {removed}")
    print(f"  Output shape: {X_clean.shape}")

    kept_names = [n for n, k in zip(feature_names, kept_mask) if k]
    return X_clean, kept_names, selector


# ════════════════════════════════════════════════════════════════
# STEP 6 — Scaffold Split
# ════════════════════════════════════════════════════════════════
def step6_split(df: pd.DataFrame, X: np.ndarray,
                train_frac=0.70, val_frac=0.15, seed=42):
    print("\n" + "="*55)
    print("STEP 6 — Scaffold split (70 / 15 / 15)")
    print("="*55)
    print("  Using scaffold split — prevents data leakage")
    print("  Random split would leak similar molecules into test")

    from collections import defaultdict

    # Compute Murcko scaffold per molecule
    scaffolds = defaultdict(list)
    for idx, mol in enumerate(tqdm(df['mol'], desc="  Scaffolds")):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            scaffold = Chem.MolToSmiles(mol)
        scaffolds[scaffold].append(idx)

    # Sort by scaffold group size descending
    scaffold_sets = sorted(
        scaffolds.values(),
        key=lambda x: len(x),
        reverse=True
    )

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)

    train_idx, val_idx, test_idx = [], [], []

    for group in scaffold_sets:
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    print(f"\n  Unique scaffolds : {len(scaffold_sets)}")
    print(f"  Train            : {len(train_idx)} "
          f"({len(train_idx)/n_total:.1%})")
    print(f"  Val              : {len(val_idx)} "
          f"({len(val_idx)/n_total:.1%})")
    print(f"  Test             : {len(test_idx)} "
          f"({len(test_idx)/n_total:.1%})")

    # Split feature matrix
    X_train = X[train_idx]
    X_val   = X[val_idx]
    X_test  = X[test_idx]

    # Split dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    split_indices = {
        'train': train_idx,
        'val':   val_idx,
        'test':  test_idx
    }

    return X_train, X_val, X_test, train_df, val_df, test_df, split_indices


# ════════════════════════════════════════════════════════════════
# STEP 7 — Normalize Features
# ════════════════════════════════════════════════════════════════
def step7_normalize(X_train, X_val, X_test,
                    n_binary=2048 + 167):
    print("\n" + "="*55)
    print("STEP 7 — Normalizing features")
    print("="*55)
    print("  Binary FP columns  : NOT scaled (already 0/1)")
    print("  Descriptor columns : RobustScaler")
    print("  Fit on TRAIN only  : no data leakage")

    if X_train.shape[1] <= n_binary:
        print("  All features binary — no scaling needed")
        return X_train, X_val, X_test, None

    # Split binary vs continuous
    X_tr_bin  = X_train[:, :n_binary]
    X_tr_desc = X_train[:, n_binary:]
    X_v_bin   = X_val[:, :n_binary]
    X_v_desc  = X_val[:, n_binary:]
    X_te_bin  = X_test[:, :n_binary]
    X_te_desc = X_test[:, n_binary:]

    # Fit ONLY on train
    scaler = RobustScaler()
    X_tr_desc_sc = scaler.fit_transform(X_tr_desc)
    X_v_desc_sc  = scaler.transform(X_v_desc)
    X_te_desc_sc = scaler.transform(X_te_desc)

    X_train_final = np.hstack([X_tr_bin, X_tr_desc_sc])
    X_val_final   = np.hstack([X_v_bin,  X_v_desc_sc])
    X_test_final  = np.hstack([X_te_bin, X_te_desc_sc])

    print(f"  Binary features    : {n_binary}")
    print(f"  Descriptor features: {X_tr_desc.shape[1]}")
    print(f"  Final train shape  : {X_train_final.shape}")
    print(f"  Final val shape    : {X_val_final.shape}")
    print(f"  Final test shape   : {X_test_final.shape}")

    return X_train_final, X_val_final, X_test_final, scaler


# ════════════════════════════════════════════════════════════════
# STEP 8 — Save Everything
# ════════════════════════════════════════════════════════════════
def step8_save(X_train, X_val, X_test,
               train_df, val_df, test_df,
               feature_names, class_weights,
               label_stats, split_indices,
               scaler, var_selector):
    print("\n" + "="*55)
    print("STEP 8 — Saving processed data")
    print("="*55)

    # Feature matrices
    np.save(f'{PROCESSED_DIR}X_train.npy', X_train)
    np.save(f'{PROCESSED_DIR}X_val.npy',   X_val)
    np.save(f'{PROCESSED_DIR}X_test.npy',  X_test)
    print("  X_train.npy ✓")
    print("  X_val.npy   ✓")
    print("  X_test.npy  ✓")

    # Labels
    train_df[TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}y_train.csv', index=False)
    val_df[TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}y_val.csv', index=False)
    test_df[TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}y_test.csv', index=False)
    print("  y_train.csv ✓")
    print("  y_val.csv   ✓")
    print("  y_test.csv  ✓")

    # SMILES files for GNN
    train_df[['canonical_smiles'] + TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}train_smiles.csv', index=False)
    val_df[['canonical_smiles'] + TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}val_smiles.csv', index=False)
    test_df[['canonical_smiles'] + TARGET_COLS].to_csv(
        f'{PROCESSED_DIR}test_smiles.csv', index=False)
    print("  train_smiles.csv ✓")
    print("  val_smiles.csv   ✓")
    print("  test_smiles.csv  ✓")

    # Metadata JSON files
    json.dump(
        feature_names,
        open(f'{PROCESSED_DIR}feature_names.json', 'w'),
        indent=2
    )
    json.dump(
        class_weights,
        open(f'{PROCESSED_DIR}class_weights.json', 'w'),
        indent=2
    )
    json.dump(
        label_stats,
        open(f'{PROCESSED_DIR}label_stats.json', 'w'),
        indent=2
    )
    json.dump(
        split_indices,
        open(f'{PROCESSED_DIR}split_indices.json', 'w'),
        indent=2
    )
    print("  feature_names.json  ✓")
    print("  class_weights.json  ✓")
    print("  label_stats.json    ✓")
    print("  split_indices.json  ✓")

    # Sklearn objects
    if scaler is not None:
        pickle.dump(scaler,
            open(f'{MODELS_DIR}scaler.pkl', 'wb'))
        print("  scaler.pkl          ✓")

    pickle.dump(var_selector,
        open(f'{MODELS_DIR}var_selector.pkl', 'wb'))
    print("  var_selector.pkl    ✓")

    print(f"\n  All files saved to {PROCESSED_DIR}")


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════
def run():
    print("\n" + "█"*55)
    print("  TOX21 PREPROCESSING PIPELINE")
    print("█"*55)

    # Step 1
    df = step1_load(RAW_PATH)

    # Step 2
    df = step2_clean_smiles(df)

    # Step 3
    df, class_weights, label_stats = step3_labels(df)

    # Step 4
    X, feature_names = step4_features(df)

    # Step 5
    X, feature_names, var_selector = step5_clean_features(
        X, feature_names
    )

    # Step 6
    (X_train, X_val, X_test,
     train_df, val_df, test_df,
     split_indices) = step6_split(df, X)

    # Step 7
    X_train, X_val, X_test, scaler = step7_normalize(
        X_train, X_val, X_test
    )

    # Step 8
    step8_save(
        X_train, X_val, X_test,
        train_df, val_df, test_df,
        feature_names, class_weights,
        label_stats, split_indices,
        scaler, var_selector
    )

    # Final summary
    print("\n" + "█"*55)
    print("  PREPROCESSING COMPLETE")
    print("█"*55)
    print(f"""
  Train : {X_train.shape[0]} compounds, {X_train.shape[1]} features
  Val   : {X_val.shape[0]} compounds, {X_val.shape[1]} features
  Test  : {X_test.shape[0]} compounds, {X_test.shape[1]} features
  Tasks : {len(TARGET_COLS)}

  Next step → python src/train.py
    """)


if __name__ == '__main__':
    run()