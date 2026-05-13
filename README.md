# 🧬 HybProm — Bacterial Promoter Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybprom-bacterial-promoter-predictor.streamlit.app/)

## Overview
HybProm is a hybrid two-stage XGBoost classifier for predicting bacterial promoter sequences. It combines DNA duplex stability (DDS), k-mer frequencies, and −10/−35 motif scores into a 171-feature hybrid representation.

## 🚀 Live App
👉 **[https://hybprom-bacterial-promoter-predictor.streamlit.app/](https://hybprom-bacterial-promoter-predictor.streamlit.app/)**

## Features
- **171 hybrid features**: DDS positional values + dinucleotide/trinucleotide k-mers + motif scores
- **Two-stage XGBoost**: Stage 1 (full features) → SHAP selection → Stage 2 (top 30 features)
- **SHAP interpretability**: Per-prediction feature importance visualization
- **Cross-domain validation**: Tested on archaeal promoters (Halobacterium, Sulfolobus)
- **DDS profile plot**: Visualizes DNA duplex stability across the sequence

## Results

| Dataset | Accuracy | AUC-ROC |
|---|---|---|
| Bacteria (test set) | 81.4% | 0.895 |
| Archaea (cross-domain) | 53.5% | 0.556 |

## Input
Enter an **81 bp upstream sequence** (−80 to +1 relative to TSS) using only A, T, G, C characters.

### Example sequences
- **Promoter**: `TAACATTACTGTAAGGATATTGAAATAAAAAATAGCTGGTTGATCGTGTATAATCTTCCTAGATGTTAACAAACAGGGGGA`
- **Non-promoter**: `AAGAGAAGCCGACAATCATAGCTACGCCGATCATCGCCGCAAAACCACCAGCAAACGGCGCTTCGCCGATTGTCCAGTTGC`

## Model Architecture
Input (81bp sequence)
↓
Feature Extraction (171 features)
├── DDS positional values (80)
├── Trinucleotide k-mers (64)
├── Dinucleotide k-mers (16)
├── Basic stats: GC%, AT%, entropy, length (4)
├── Motif scores: −10, −35 box (2)
└── DDS summary stats (5)
↓
Stage 1: XGBoost (200 estimators, depth 7)
↓
SHAP Feature Selection (top 30)
↓
Stage 2: XGBoost (100 estimators, depth 5)
↓
Final Prediction + SHAP Explanatio
## Training Data
- **Positives**: Experimentally validated promoters from *E. coli* and *B. subtilis*
- **Negatives**: Biological non-promoter intergenic regions
- **Total**: 7,990 sequences (balanced)

## Cross-Domain Analysis
The model's near-chance performance on archaeal sequences (53.5%) confirms it learned bacterial-specific promoter signals (σ factor −10/−35 boxes) rather than generic DNA patterns. Archaeal transcription uses fundamentally different machinery (TBP/TFB proteins, TATA box).

## Files
| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `stage1_xgb.pkl` | Stage 1 XGBoost model |
| `stage2_xgb.pkl` | Stage 2 XGBoost model |
| `feature_names.pkl` | Feature name list |
| `top_idx.pkl` | SHAP-selected top 30 feature indices |
| `viz_data.pkl` | Precomputed visualization data |
| `HybProm.ipynb` | Full analysis notebook |
| `requirements.txt` | Python dependencies |

## References
- Ganzerla et al. (2024) — CDBProm: Comprehensive Directory of Bacterial Promoters
- RegulonDB v14.5 — *E. coli* promoter database
- Ganzerla et al. — Archaeal promoter dataset
