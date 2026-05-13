import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from itertools import product
from math import log2

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="HybProm — Bacterial Promoter Predictor",
                   layout="wide", page_icon="🧬")

# ── Feature extraction ────────────────────────────────────────
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
    """Extract 171 hybrid features using 81bp sequences."""
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
    # 80 positional DDS values (matching training)
    dds_vals = [stability_values.get(seq[i:i+2], -1.3) for i in range(80)]
    for i, v in enumerate(dds_vals):
        feats[f'dds_pos{i}'] = v
    return feats

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    m1       = joblib.load("stage1_xgb.pkl")
    m2       = joblib.load("stage2_xgb.pkl")
    feat_names = joblib.load("feature_names.pkl")
    top_idx  = joblib.load("top_idx.pkl")
    viz_data = joblib.load("viz_data.pkl")
    explainer = shap.TreeExplainer(m1)
    return m1, m2, feat_names, top_idx, explainer, viz_data

try:
    model1, model2, feature_names, top_idx, explainer, viz_data = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ── Prediction ────────────────────────────────────────────────
def predict(seq):
    feats = extract_features(seq)
    X = np.array([[feats[f] for f in feature_names]], dtype=np.float32)
    prob1 = model1.predict_proba(X)[0, 1]
    uncertain = 0.3 <= prob1 < 0.7
    if uncertain:
        prob2 = model2.predict_proba(X[:, top_idx])[0, 1]
        final_prob = (prob1 + prob2) / 2
        stage = "Two-stage (refined)"
    else:
        final_prob = prob1
        stage = "Stage 1 (confident)"
    shap_vals = explainer.shap_values(X)
    return final_prob, stage, shap_vals, X, feats

# ── UI ────────────────────────────────────────────────────────
st.title("🧬 HybProm — Bacterial Promoter Predictor")
st.markdown("""
Enter an **81 bp upstream sequence** (−80 to +1 relative to TSS). 
HybProm uses hybrid features — DNA duplex stability, k‑mer frequencies,  
and −10/−35 motif scores — with a two‑stage XGBoost classifier.
""")

# Sidebar examples
st.sidebar.header("Example sequences")
examples = {
    "Promoter (E. coli, 99.97% confidence)": "TAACATTACTGTAAGGATATTGAAATAAAAAATAGCTGGTTGATCGTGTATAATCTTCCTAGATGTTAACAAACAGGGGGA",
    "Non-promoter (E. coli, 0.04% confidence)": "AAGAGAAGCCGACAATCATAGCTACGCCGATCATCGCCGCAAAACCACCAGCAAACGGCGCTTCGCCGATTGTCCAGTTGC",
}
for label, eseq in examples.items():
    if st.sidebar.button(label):
        st.session_state["seq_input"] = eseq

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")
st.sidebar.markdown("- Stage 1: XGBoost (200 est., depth 7)")
st.sidebar.markdown("- Stage 2: XGBoost (100 est., depth 5)")
st.sidebar.markdown("- Features: 171 (DDS-81 + k-mers + motifs)")
st.sidebar.markdown("- Accuracy: 81.4% | AUC: 0.895")

# Input
default_seq = st.session_state.get("seq_input", "")
seq_input = st.text_area("DNA sequence (81 bp, A/T/G/C only)",
                          value=default_seq, height=80,
                          placeholder="Paste your 81 bp sequence here...")

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("🔬 Predict", type="primary", use_container_width=True)

if predict_btn:
    seq = seq_input.strip().upper().replace(" ", "").replace("\n", "")

    # Validation
    if not seq:
        st.error("Please enter a sequence.")
    elif len(seq) != 81:
        st.error(f"Sequence must be exactly 81 bp. Yours is {len(seq)} bp.")
    elif set(seq) - set("ACGT"):
        bad = set(seq) - set("ACGT")
        st.error(f"Invalid characters found: {bad}. Only A, T, G, C allowed.")
    elif not models_loaded:
        st.error("Model files not found. Make sure stage1_xgb.pkl, stage2_xgb.pkl, and feature_names.pkl are in the same directory.")
    else:
        with st.spinner("Analyzing sequence..."):
            prob, stage, shap_vals, X, feats = predict(seq)

        # Result
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            color = "green" if prob >= 0.5 else "red"
            label = "Promoter" if prob >= 0.5 else "Non-Promoter"
            st.metric("Prediction", label)
        with r2:
            st.metric("Promoter Probability", f"{prob:.4f}")
        with r3:
            st.metric("Decision Stage", stage)

        # Confidence bar
        st.progress(float(prob), text=f"Confidence: {prob*100:.1f}%")

        # Tabs for details
        tab1, tab2, tab3 = st.tabs(["SHAP Explanation", "Feature Values", "Sequence Analysis"])

        with tab1:
            st.subheader("Feature Importance (SHAP)")
            fig, ax = plt.subplots(figsize=(12, 4))
            # Bar plot of top 15 SHAP values
            shap_series = pd.Series(shap_vals[0], index=feature_names)
            top15 = shap_series.abs().nlargest(15).index
            top15_vals = shap_series[top15]
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in top15_vals]
            ax.barh(top15[::-1], top15_vals[::-1], color=colors[::-1])
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on promoter probability)")
            ax.set_title("Top 15 features driving this prediction\n(Red = pushes toward promoter, Blue = pushes away)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.subheader("Extracted Feature Values")
            feat_df = pd.DataFrame({
                'Feature': list(feats.keys()),
                'Value': list(feats.values()),
                'SHAP': [shap_vals[0][feature_names.index(f)] 
                         if f in feature_names else 0 for f in feats.keys()]
            }).sort_values('SHAP', key=abs, ascending=False)
            st.dataframe(feat_df.style.format({'Value': '{:.4f}', 'SHAP': '{:.4f}'}),
                        use_container_width=True)

        with tab3:
            st.subheader("Sequence Analysis")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GC Content", f"{feats['gc_content']*100:.1f}%")
            c2.metric("Shannon Entropy", f"{feats['entropy']:.3f}")
            c3.metric("−10 Box Score", f"{feats['minus10_score']:.3f}")
            c4.metric("−35 Box Score", f"{feats['minus35_score']:.3f}")

            st.markdown("**Sequence with position markers:**")
            positions = "".join([str(i % 10) for i in range(81)])
            st.code(f"Pos: {positions}\nSeq: {seq}\n      {'':.<10}−35 region{'':.>10}{'':.<10}−10 region{'':.>10}+1")

            # DDS profile plot
            dds_vals = [DDS_TABLE.get(seq[i:i+2], 0.5) for i in range(59)]
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(range(-59, 0), dds_vals, color='#2c7bb6', linewidth=1.5)
            ax2.fill_between(range(-59, 0), dds_vals, alpha=0.3, color='#2c7bb6')
            ax2.axvspan(-35, -30, alpha=0.2, color='orange', label='−35 box region')
            ax2.axvspan(-12, -7, alpha=0.2, color='green', label='−10 box region')
            ax2.set_xlabel("Position relative to TSS")
            ax2.set_ylabel("DNA Duplex Stability")
            ax2.set_title("DDS Profile")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
# Add to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Cross-Domain Results**")
st.sidebar.markdown("- Bacteria (test set): 81.4%")
st.sidebar.markdown("- Archaea (cross-domain): 53.5%")
st.sidebar.markdown("- Interpretation: Model learned bacterial-specific signals")

# Add a new section at the bottom of the app
st.markdown("---")
st.header("🔬 Cross-Domain Analysis: Bacteria vs Archaea")

col1, col2 = st.columns(2)

with col1:
    st.subheader("In-Domain (Bacteria)")
    st.metric("Accuracy", "81.4%")
    st.metric("AUC-ROC", "0.895")
    st.metric("Training species", "E. coli + B. subtilis")
    st.markdown("""
    Model trained on bacterial promoters using:
    - −10 box (TATAAT)  
    - −35 box (TTGACA)
    - σ factor recognition
    """)

with col2:
    st.subheader("Cross-Domain (Archaea)")
    st.metric("Accuracy", "53.5%", delta="-27.9%", delta_color="inverse")
    st.metric("AUC-ROC", "0.556", delta="-0.339", delta_color="inverse")
    st.metric("Test sequences", "7,494 (3,753 + 3,741)")
    st.markdown("""
    Performance drops to near-chance because:
    - Archaea use TATA box (TTTAAA)
    - BRE element replaces −35 box
    - Distinct transcription machinery (TBP/TFB vs σ)
    """)

# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bacteria probability distribution (from test set)
# axes[0].hist(y_te_proba[y_te==1], bins=30, alpha=0.6, color='#e74c3c', label='Promoter')
# axes[0].hist(y_te_proba[y_te==0], bins=30, alpha=0.6, color='#3498db', label='Non-Promoter')
# axes[0].set_title('Bacteria (In-Domain)\nClear separation')
# axes[0].set_xlabel('Predicted Promoter Probability')
# axes[0].set_ylabel('Count')
# axes[0].legend()

axes[0].hist(viz_data['bacteria_promoter_probs'],    bins=30, alpha=0.6, color='#e74c3c', label='Promoter')
axes[0].hist(viz_data['bacteria_nonpromoter_probs'], bins=30, alpha=0.6, color='#3498db', label='Non-Promoter')
axes[1].hist(viz_data['archaea_promoter_probs'],     bins=30, alpha=0.6, color='#e74c3c', label='Promoter')
axes[1].hist(viz_data['archaea_nonpromoter_probs'],  bins=30, alpha=0.6, color='#3498db', label='Non-Promoter')

# Archaea probability distribution
# Line 292-293 — replace with:
axes[1].hist(viz_data['archaea_promoter_probs'],    bins=30, alpha=0.6, color='#e74c3c', label='Promoter')
axes[1].hist(viz_data['archaea_nonpromoter_probs'], bins=30, alpha=0.6, color='#3498db', label='Non-Promoter')
axes[1].set_title('Archaea (Cross-Domain)\nDistributions overlap')
axes[1].set_xlabel('Predicted Promoter Probability')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info("""
**Biological Interpretation:** The model's near-chance performance on archaeal sequences 
confirms it learned *bacterial-specific* promoter signals rather than generic DNA patterns. 
Archaeal transcription uses fundamentally different machinery (TBP/TFB proteins instead of 
bacterial σ factors), explaining the lack of cross-domain generalization.
""")
