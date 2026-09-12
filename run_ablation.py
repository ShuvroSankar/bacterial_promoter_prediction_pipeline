import pandas as pd, numpy as np, joblib
from itertools import product
from math import log2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap

# ---- Load data (same as app/original training) ----
df_ec = pd.read_excel('Data_1/E.coli.xlsx')
df_bs = pd.read_excel('Data_1/B.subtilis.xlsx')
df = pd.concat([df_ec[['seq','lable']], df_bs[['seq','lable']]], ignore_index=True)
print("Combined:", df.shape, df['lable'].value_counts().to_dict())

# ---- Feature extraction (exact copy from app.py) ----
DDS_TABLE = {
    'AA': 1.0, 'AT': 0.5, 'AG': 0.8, 'AC': 0.9,
    'TA': 0.6, 'TT': 1.0, 'TG': 0.7, 'TC': 0.8,
    'GA': 0.9, 'GT': 0.8, 'GG': 1.0, 'GC': 1.2,
    'CA': 0.8, 'CT': 0.7, 'CG': 1.1, 'CC': 1.0
}
def gc_content(seq):
    valid = [b for b in seq if b in "ACGT"]
    return (seq.count("G") + seq.count("C")) / len(valid) if valid else 0.0
def shannon_entropy(seq):
    freq = {b: seq.count(b) / len(seq) for b in "ACGT"}
    return -sum(f * log2(f + 1e-9) for f in freq.values())
def motif_score(seq, motif):
    best, L = 0, len(motif)
    for i in range(len(seq) - L + 1):
        m = sum(seq[i+j] == motif[j] for j in range(L)) / L
        best = max(best, m)
    return best
def kmer_freq(seq, k):
    kmers = [''.join(p) for p in product("ACGT", repeat=k)]
    total = max(len(seq) - k + 1, 1)
    return {f'k{k}_{kmer}': seq.count(kmer) / total for kmer in kmers}
def dds_summary(seq):
    vals = [DDS_TABLE.get(seq[i:i+2], 0.5) for i in range(len(seq)-1)]
    feats = {
        'dds_mean': np.mean(vals), 'dds_std': np.std(vals),
        'dds_min': np.min(vals),   'dds_max': np.max(vals),
        'dds_range': np.max(vals) - np.min(vals)
    }
    for i, v in enumerate(vals[:10]):
        feats[f'dds_pos{i}'] = v
    return feats
def extract_features(seq):
    seq = seq.upper().replace(" ", "")[:81]
    while len(seq) < 81:
        seq += seq[-1]
    stability_values = {
        'AA': -1.0,  'AT': -0.88, 'TA': -0.58, 'AG': -1.3,
        'GA': -1.3,  'TT': -1.0,  'AC': -1.45, 'CA': -1.45,
        'TG': -1.44, 'GT': -1.44, 'TC': -1.28, 'CT': -1.28,
        'CC': -1.84, 'CG': -2.24, 'GC': -2.27, 'GG': -1.84
    }
    feats = {}
    feats['gc_content']    = gc_content(seq)
    feats['at_content']    = 1 - feats['gc_content']
    feats['entropy']       = shannon_entropy(seq)
    feats['seq_length']    = len(seq)
    feats['minus10_score'] = motif_score(seq, "TATAAT")
    feats['minus35_score'] = motif_score(seq, "TTGACA")
    feats.update(kmer_freq(seq, 2))
    feats.update(kmer_freq(seq, 3))
    feats.update(dds_summary(seq))
    dds_vals = [stability_values.get(seq[i:i+2], -1.3) for i in range(80)]
    for i, v in enumerate(dds_vals):
        feats[f'dds_pos{i}'] = v
    return feats

print("Extracting features...")
feat_dicts = df['seq'].apply(extract_features)
df_feat = pd.DataFrame(list(feat_dicts))
feature_cols = df_feat.columns.tolist()
X = df_feat[feature_cols].values.astype(np.float32)
y = df['lable'].values
print("Feature matrix:", X.shape)

# sanity check against saved feature_names.pkl
saved_names = joblib.load('feature_names.pkl')
print("Saved feature count:", len(saved_names), "| Extracted:", len(feature_cols))
print("Names match:", list(saved_names) == feature_cols)

# ---- Try to reproduce the exact reported test split ----
stage1_loaded = joblib.load('stage1_xgb.pkl')

for seed in [42, 0, 1, 2, 7, 21, 123]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    proba = stage1_loaded.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    print(f"seed={seed}: acc={acc:.4f} auc={auc:.4f} cm={cm.tolist()}")
