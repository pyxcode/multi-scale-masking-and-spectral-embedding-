"""
Linear probe — évalue la qualité des représentations Brain-JEPA
Tâche : CONTROL vs ADHD (classification binaire)
Métrique : AUC-ROC

Usage : python linear_probe.py --checkpoint logs/xxx/brainjepa_ucla-latest.pth.tar
"""

import sys
import os
# Ajouter racine ET src/ au path
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'src'))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from helper import init_model
from datasets.ucla import UCLADataset

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--data_dir', type=str, default='data/timeseries')
parser.add_argument('--participants', type=str, default='data/participants.tsv')
parser.add_argument('--gradient_csv', type=str, default='gradient_mapping_450.csv')
parser.add_argument('--n_folds', type=int, default=5)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── 1. Charger les labels ─────────────────────────────────────────────────────
print("\n[1] Labels...")
df = pd.read_csv(args.participants, sep='\t')
df = df[df['gender'].isin(['M', 'F'])].copy()
df['label'] = (df['gender'] == 'M').astype(int)
df['subject_id'] = df['participant_id'].str.strip()
print(f"  F : {(df['label']==0).sum()}, M : {(df['label']==1).sum()}")

# ── 2. Charger l'encoder ──────────────────────────────────────────────────────
print("\n[2] Encoder...")
gradient = torch.zeros(1, 400 * 10, 32).to(DEVICE)
if os.path.exists(args.gradient_csv):
    import pandas as pd_g
    gdf = pd_g.read_csv(args.gradient_csv, header=None)
    gradient = torch.tensor(gdf.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

encoder, _ = init_model(
    device=DEVICE,
    patch_size=16,
    crop_size=[400, 160],
    pred_depth=6,
    pred_emb_dim=384,
    model_name='vit_small',
    gradient_pos_embed=gradient,
    attn_mode='normal',
    add_w='origin',
)

ckpt = torch.load(args.checkpoint, map_location=DEVICE)
# Gérer le cas DDP (module.) ou non
state_dict = ckpt.get('encoder', ckpt)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
encoder.load_state_dict(state_dict)
encoder.eval()
print(f"  ✅ Checkpoint chargé")

# ── 3. Extraire les représentations ───────────────────────────────────────────
print("\n[3] Extraction des représentations...")
representations = {}

with torch.no_grad():
    for npy_path in sorted(glob(os.path.join(args.data_dir, 'sub-*.npy'))):
        subject_id = os.path.basename(npy_path).replace('.npy', '')
        ts = np.load(npy_path)  # [400, T]

        # Clip centré de 160 TRs
        T = ts.shape[1]
        start = max(0, (T - 160) // 2)
        ts = ts[:, start:start + 160]
        if ts.shape[1] < 160:
            pad = np.zeros((400, 160))
            pad[:, :ts.shape[1]] = ts
            ts = pad

        ts = torch.tensor(ts, dtype=torch.float32)
        ts = (ts - ts.mean()) / (ts.std() + 1e-6)
        ts = ts.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, 400, 160]

        h = encoder(ts)                    # [1, N, D]
        h = F.layer_norm(h, (h.size(-1),))
        rep = h.mean(dim=1).squeeze(0)     # [D] — mean pooling
        representations[subject_id] = rep.cpu().numpy()

print(f"  ✅ {len(representations)} sujets encodés")

# ── 4. Aligner avec les labels ────────────────────────────────────────────────
print("\n[4] Alignement labels/représentations...")
X, y, kept = [], [], []
for _, row in df.iterrows():
    sid = row['subject_id']
    if sid in representations:
        X.append(representations[sid])
        y.append(row['label'])
        kept.append(sid)

X = np.array(X)
y = np.array(y)
print(f"  {len(kept)} sujets avec labels + représentations")
print(f"  F : {(y==0).sum()}, M : {(y==1).sum()}")

# ── 5. Linear probe — cross-validation ───────────────────────────────────────
print(f"\n[5] Linear probe ({args.n_folds}-fold CV)...")
skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
aucs, accs, f1s = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)

    aucs.append(auc)
    accs.append(acc)
    f1s.append(f1)
    print(f"  Fold {fold+1} — AUC: {auc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

# ── 6. Résultats ──────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RÉSULTATS FINAUX — F vs M (Gender)")
print("="*50)
print(f"AUC  : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"Acc  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"F1   : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")