import pandas as pd, numpy as np, joblib
from itertools import product
from math import log2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

exec(open('run_ablation.py').read().split("# ---- Try to reproduce")[0])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stage1_accs, stage1_aucs = [], []
twostage_accs, twostage_aucs = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    m1 = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                        subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
    m1.fit(X_tr, y_tr)
    proba1 = m1.predict_proba(X_val)[:, 1]
    pred1 = (proba1 >= 0.5).astype(int)
    acc1 = accuracy_score(y_val, pred1)
    auc1 = roc_auc_score(y_val, proba1)

    # SHAP top-30 on this fold's Stage-1 model (training set only, no leakage)
    explainer = shap.TreeExplainer(m1)
    sv = explainer.shap_values(X_tr)
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top30 = np.argsort(mean_abs_shap)[-30:]

    m2 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                        eval_metric='logloss', random_state=42, n_jobs=-1)
    m2.fit(X_tr[:, top30], y_tr)
    proba2 = m2.predict_proba(X_val[:, top30])[:, 1]

    uncertain = (proba1 >= 0.3) & (proba1 < 0.7)
    final = proba1.copy()
    final[uncertain] = (proba1[uncertain] + proba2[uncertain]) / 2
    pred_final = (final >= 0.5).astype(int)
    acc2 = accuracy_score(y_val, pred_final)
    auc2 = roc_auc_score(y_val, final)

    stage1_accs.append(acc1); stage1_aucs.append(auc1)
    twostage_accs.append(acc2); twostage_aucs.append(auc2)
    print(f"Fold {fold+1}: Stage1 acc={acc1:.4f} auc={auc1:.4f}  |  TwoStage acc={acc2:.4f} auc={auc2:.4f}")

print(f"\n=== 5-fold CV summary (171 features, 7,990 sequences) ===")
print(f"Stage 1 alone : Accuracy {np.mean(stage1_accs):.4f} ± {np.std(stage1_accs):.4f}   AUC {np.mean(stage1_aucs):.4f} ± {np.std(stage1_aucs):.4f}")
print(f"Two-Stage     : Accuracy {np.mean(twostage_accs):.4f} ± {np.std(twostage_accs):.4f}   AUC {np.mean(twostage_aucs):.4f} ± {np.std(twostage_aucs):.4f}")
