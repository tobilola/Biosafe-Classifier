"""
BioSafe Classifier — Streamlit Dashboard

Interactive interface for:
  1. Live query classification with risk breakdown
  2. Model performance metrics and confusion matrix
  3. Feature importance visualization
  4. Batch analysis of multiple queries
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.features import extract_features, get_feature_columns
from utils.taxonomy import RISK_LEVELS, TOPIC_CATEGORIES, ESCALATION_SIGNALS, DEESCALATION_SIGNALS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BioSafe Classifier",
    page_icon="BS",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = "models/classifier.joblib"
FEATURES_PATH = "models/feature_columns.json"
METRICS_PATH = "models/metrics.json"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    pipeline = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return pipeline, feature_cols, metrics


pipeline, feature_cols, metrics = load_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_query(query: str) -> dict:
    """Run the full pipeline on a single query and return results."""
    feats = extract_features(query)
    feat_vector = np.array([[feats[c] for c in feature_cols]])
    proba = pipeline.predict_proba(feat_vector)[0]
    predicted_class = int(np.argmax(proba))

    return {
        "query": query,
        "predicted_risk": predicted_class,
        "risk_label": RISK_LEVELS[predicted_class]["label"],
        "confidence": float(proba[predicted_class]),
        "probabilities": {RISK_LEVELS[i]["label"]: round(float(p), 4) for i, p in enumerate(proba)},
        "features": feats,
    }


def risk_color(level: int) -> str:
    return RISK_LEVELS.get(level, {}).get("color", "#888888")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("BioSafe Classifier")
st.sidebar.markdown(
    "An LLM query safety screening system for detecting potentially "
    "harmful biological queries directed at AI models."
)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Live Classification", "Batch Analysis", "Model Performance", "Taxonomy Explorer"])

# ---------------------------------------------------------------------------
# Page: Live Classification
# ---------------------------------------------------------------------------
if page == "Live Classification":
    st.title("Live Query Classification")
    st.markdown("Enter a biological query to classify its risk level.")

    if pipeline is None:
        st.error("Model not found. Run `python train.py` first.")
        st.stop()

    query = st.text_area(
        "Enter query:",
        height=100,
        placeholder="e.g., How does CRISPR-Cas9 create double-strand breaks?",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        classify_btn = st.button("Classify", type="primary", use_container_width=True)

    if classify_btn and query.strip():
        result = classify_query(query)
        level = result["predicted_risk"]
        color = risk_color(level)

        st.markdown("---")

        # Risk level header
        st.markdown(
            f"### Risk Level: <span style='color:{color}; font-size:1.4em;'>"
            f"{level} — {result['risk_label'].upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {result['confidence']:.1%}")
        st.markdown(f"**Description:** {RISK_LEVELS[level]['description']}")

        # Probability distribution
        st.markdown("#### Probability Distribution")
        prob_df = pd.DataFrame(
            [
                {
                    "Risk Level": f"{i} — {RISK_LEVELS[i]['label']}",
                    "Probability": result["probabilities"][RISK_LEVELS[i]["label"]],
                }
                for i in range(5)
            ]
        )
        st.bar_chart(prob_df.set_index("Risk Level"))

        # Feature breakdown
        with st.expander("Feature Breakdown"):
            feats = result["features"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Topic Match", feats["best_topic"])
                st.metric("Base Risk", feats["base_risk"])
                st.metric("Topic Match Score", f"{feats['topic_match_score']:.3f}")
            with col2:
                st.metric("Escalation Score", f"{feats['escalation_total']:.2f}")
                st.metric("De-escalation Score", f"{feats['deescalation_total']:.2f}")
                st.metric("Composite Risk", f"{feats['composite_risk_score']:.2f}")
            with col3:
                st.metric("Word Count", feats["word_count"])
                st.metric("Specificity Score", feats["specificity_score"])
                st.metric("Protocol Request", "Yes" if feats["has_protocol_request"] else "No")

            # Topic category hits
            st.markdown("**Topic Category Hits:**")
            topic_hits = {
                k.replace("topic_", ""): v
                for k, v in feats.items()
                if k.startswith("topic_") and v > 0
            }
            if topic_hits:
                st.json(topic_hits)
            else:
                st.caption("No specific topic keywords matched.")

            # Escalation signal hits
            esc_hits = {
                k.replace("esc_", "").replace("_hits", ""): v
                for k, v in feats.items()
                if k.startswith("esc_") and k.endswith("_hits") and v > 0
            }
            if esc_hits:
                st.markdown("**Escalation Signals Triggered:**")
                st.json(esc_hits)

            deesc_hits = {
                k.replace("deesc_", "").replace("_hits", ""): v
                for k, v in feats.items()
                if k.startswith("deesc_") and k.endswith("_hits") and v > 0
            }
            if deesc_hits:
                st.markdown("**De-escalation Signals Triggered:**")
                st.json(deesc_hits)

    # Example queries
    st.markdown("---")
    st.markdown("#### Try These Examples")
    examples = [
        "What is the purpose of gel electrophoresis?",
        "How does CRISPR-Cas9 work?",
        "What techniques are used to study gain-of-function mutations in influenza?",
        "What specific mutations increase the transmissibility of H5N1 in mammalian hosts?",
        "How do I aerosolize anthrax spores for maximum dispersal?",
    ]
    for ex in examples:
        st.code(ex, language=None)

# ---------------------------------------------------------------------------
# Page: Batch Analysis
# ---------------------------------------------------------------------------
elif page == "Batch Analysis":
    st.title("Batch Query Analysis")
    st.markdown("Paste multiple queries (one per line) to classify them all at once.")

    if pipeline is None:
        st.error("Model not found. Run `python train.py` first.")
        st.stop()

    batch_input = st.text_area(
        "Queries (one per line):",
        height=200,
        placeholder="How does PCR work?\nHow do I make anthrax resistant to all treatments?",
    )

    if st.button("Classify All", type="primary"):
        queries = [q.strip() for q in batch_input.strip().split("\n") if q.strip()]
        if queries:
            results = [classify_query(q) for q in queries]
            df = pd.DataFrame(
                [
                    {
                        "Query": r["query"][:80] + ("..." if len(r["query"]) > 80 else ""),
                        "Risk Level": r["predicted_risk"],
                        "Label": r["risk_label"],
                        "Confidence": f"{r['confidence']:.1%}",
                        "Topic": r["features"]["best_topic"],
                    }
                    for r in results
                ]
            )
            st.dataframe(df, use_container_width=True)

            # Summary stats
            st.markdown("#### Distribution")
            dist = df["Risk Level"].value_counts().sort_index()
            st.bar_chart(dist)

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance Metrics")

    if metrics is None:
        st.error("Metrics not found. Run `python train.py` first.")
        st.stop()

    # Overall metrics
    st.markdown("#### Overall (5-Fold Cross-Validation)")
    col1, col2, col3 = st.columns(3)
    acc = metrics.get("accuracy", 0)
    macro_f1 = metrics.get("macro avg", {}).get("f1-score", 0)
    weighted_f1 = metrics.get("weighted avg", {}).get("f1-score", 0)
    col1.metric("Accuracy", f"{acc:.1%}")
    col2.metric("Macro F1", f"{macro_f1:.3f}")
    col3.metric("Weighted F1", f"{weighted_f1:.3f}")

    # Per-class metrics
    st.markdown("#### Per-Class Performance")
    class_data = []
    for i in range(5):
        label = RISK_LEVELS[i]["label"]
        if label in metrics:
            m = metrics[label]
            class_data.append({
                "Risk Level": f"{i} — {label}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1-Score": f"{m['f1-score']:.3f}",
                "Support": int(m["support"]),
            })
    if class_data:
        st.dataframe(pd.DataFrame(class_data), use_container_width=True)

    # Feature importance (from RF component)
    st.markdown("#### Top Feature Importances")
    try:
        rf_model = pipeline.named_steps["clf"].estimators_[0]
        importances = rf_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
        imp_df = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
        st.bar_chart(imp_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")

# ---------------------------------------------------------------------------
# Page: Taxonomy Explorer
# ---------------------------------------------------------------------------
elif page == "Taxonomy Explorer":
    st.title("Risk Taxonomy Explorer")
    st.markdown("Browse the dual-use risk taxonomy powering the classifier.")

    # Risk levels
    st.markdown("#### Risk Levels")
    for level_id, info in RISK_LEVELS.items():
        color = info["color"]
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>"
            f"Level {level_id} — {info['label'].upper()}</span>: {info['description']}",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Topic categories
    st.markdown("#### Topic Categories")
    for name, info in TOPIC_CATEGORIES.items():
        with st.expander(f"{name} (base risk: {info['base_risk']})"):
            st.markdown(f"**{info['description']}**")
            st.markdown(f"Keywords: {', '.join(info['keywords'])}")

    st.markdown("---")

    # Escalation signals
    st.markdown("#### Escalation Signals")
    for name, info in ESCALATION_SIGNALS.items():
        with st.expander(f"{name} (weight: {info['weight']})"):
            st.markdown(f"**{info['description']}**")
            st.markdown(f"Keywords: {', '.join(info['keywords'])}")

    # De-escalation signals
    st.markdown("#### De-escalation Signals")
    for name, info in DEESCALATION_SIGNALS.items():
        with st.expander(f"{name} (weight: {info['weight']})"):
            st.markdown(f"**{info['description']}**")
            st.markdown(f"Keywords: {', '.join(info['keywords'])}")
