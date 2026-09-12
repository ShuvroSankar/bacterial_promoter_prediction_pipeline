import pandas as pd, numpy as np, joblib
from itertools import product
from math import log2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

exec(open('run_ablation.py').read().split("# ---- Try to reproduce")[0])  # reuse feature extraction + X,y

stage1 = joblib.load('stage1_xgb.pkl')
stage2 = joblib.load('stage2_xgb.pkl')
top_idx = joblib.load('top_idx.pkl')
print("top_idx len:", len(top_idx))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---- Stage 1 alone ----
proba1 = stage1.predict_proba(X_test)[:,1]
pred1 = (proba1 >= 0.5).astype(int)
acc1 = accuracy_score(y_test, pred1)
auc1 = roc_auc_score(y_test, proba1)
cm1 = confusion_matrix(y_test, pred1)
print(f"\n=== Stage 1 ALONE (deployed model) ===")
print(f"Accuracy: {acc1:.4f}  AUC: {auc1:.4f}")
print("Confusion matrix:", cm1.tolist())

# ---- Two-stage (as coded / as deployed) ----
X_test_top = X_test[:, top_idx]
proba2 = stage2.predict_proba(X_test_top)[:,1]

def predict_two_stage(prob1, prob2):
    uncertain = (prob1 >= 0.3) & (prob1 < 0.7)
    final = prob1.copy()
    final[uncertain] = (prob1[uncertain] + prob2[uncertain]) / 2
    return final, uncertain

proba_2stage, uncertain_mask = predict_two_stage(proba1, proba2)
pred_2stage = (proba_2stage >= 0.5).astype(int)
acc2 = accuracy_score(y_test, pred_2stage)
auc2 = roc_auc_score(y_test, proba_2stage)
cm2 = confusion_matrix(y_test, pred_2stage)
print(f"\n=== TWO-STAGE (deployed, as described in paper) ===")
print(f"Accuracy: {acc2:.4f}  AUC: {auc2:.4f}")
print("Confusion matrix:", cm2.tolist())
print(f"Uncertain-zone packets (Stage1 prob in [0.3,0.7)): {uncertain_mask.sum()} / {len(y_test)} ({uncertain_mask.mean()*100:.1f}%)")

# Accuracy restricted to just the uncertain zone -- does Stage2 help exactly where it's invoked?
if uncertain_mask.sum() > 0:
    acc1_unc = accuracy_score(y_test[uncertain_mask], pred1[uncertain_mask])
    acc2_unc = accuracy_score(y_test[uncertain_mask], pred_2stage[uncertain_mask])
    print(f"\nWithin uncertain zone only:")
    print(f"  Stage 1 alone accuracy:  {acc1_unc:.4f}")
    print(f"  Two-Stage accuracy:      {acc2_unc:.4f}")

print(f"\n=== DELTA ===")
print(f"Accuracy: {acc2-acc1:+.4f}   AUC: {auc2-auc1:+.4f}")
