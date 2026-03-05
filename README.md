# BioSafe Classifier

A query safety screening system I built to classify biological prompts directed at large language models by dual-use risk level. The goal is to distinguish legitimate scientific questions from potential misuse attempts, scoring each query from benign (level 0) to critical (level 4).

## Motivation

As LLMs get better at biology, there's a real need to screen what people are asking them. A grad student asking "how does CRISPR work?" and someone asking "how do I engineer a pathogen to evade all current vaccines?" are fundamentally different queries, but both are just text to a model. I wanted to build a system that could tell the difference, using a risk taxonomy grounded in actual biosecurity policy frameworks.

This is designed as a safety layer that could sit in front of any LLM handling biological queries. It flags high-risk prompts for review or refusal while keeping false positives low so legitimate researchers aren't blocked.

## How It Works

```
User Query
    |
    v
+-------------------+
| Feature Extraction |  Keyword matching, intent signals,
| Pipeline           |  text statistics, topic classification
+-------------------+
    |
    v
+-------------------+
| Risk Taxonomy      |  10 topic categories, 4 escalation signals,
| (taxonomy.py)      |  3 de-escalation signals, 5 risk levels
+-------------------+
    |
    v
+-------------------+
| Ensemble Classifier|  Random Forest + Gradient Boosting
| (Voting, soft)     |  with StandardScaler preprocessing
+-------------------+
    |
    v
+-------------------+
| Risk Assessment    |  Risk level (0-4), confidence score,
| Output             |  probability distribution, feature breakdown
+-------------------+
```

## Risk Levels

| Level | Label    | Description                                      |
|-------|----------|--------------------------------------------------|
| 0     | Benign   | General biology education or common knowledge     |
| 1     | Low      | Legitimate research topic, no dual-use concern    |
| 2     | Moderate | Dual-use adjacent topic, context-dependent        |
| 3     | High     | Explicitly dual-use technique or knowledge        |
| 4     | Critical | Direct request for weaponization or harm          |

## Risk Taxonomy

The classification runs on a taxonomy I built from publicly available biosecurity policy frameworks.

**Topic Categories (10):** pathogen enhancement, toxin production, select agents, resistance/evasion engineering, synthetic biology risk, delivery/dispersal, genome editing, general molecular biology, bioinformatics, education

**Escalation Signals (4):** specificity requests (step-by-step protocols), evasion intent (avoiding detection), scale interest (mass production), targeting signals (specific populations or infrastructure)

**De-escalation Signals (3):** academic context (thesis, coursework), safety focus (biosafety, countermeasures), policy context (regulation, governance)

The idea is that risk isn't just about the topic. Someone asking about gain-of-function research for a literature review is very different from someone asking for a step-by-step protocol to do it undetected. The escalation and de-escalation signals capture that context.

## Setup

```bash
git clone https://github.com/tobilola/biosafe-classifier.git
cd biosafe-classifier
pip install -r requirements.txt
```

## Usage

### 1. Generate training data and train the classifier

```bash
python train.py
```

This generates 85 synthetic labeled examples across all five risk levels, extracts 36 features per query, trains a Random Forest + Gradient Boosting ensemble with 5-fold cross-validation, and saves the model to `models/`.

### 2. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard has four views:

- **Live Classification:** Type any biological query and get a risk assessment with confidence scores, probability distribution, and full feature breakdown showing exactly why the system flagged it the way it did
- **Batch Analysis:** Paste multiple queries to classify them all at once and see the risk distribution
- **Model Performance:** Cross-validation metrics, per-class precision/recall/F1, and feature importance rankings
- **Taxonomy Explorer:** Browse all risk categories, escalation signals, and keyword lists

### 3. Use it programmatically

```python
from utils.features import extract_features, get_feature_columns
import joblib, json, numpy as np

pipeline = joblib.load("models/classifier.joblib")
with open("models/feature_columns.json") as f:
    feature_cols = json.load(f)

query = "How does CRISPR-Cas9 work?"
feats = extract_features(query)
X = np.array([[feats[c] for c in feature_cols]])
proba = pipeline.predict_proba(X)[0]
predicted = int(np.argmax(proba))
print(f"Risk level: {predicted}, Confidence: {proba[predicted]:.1%}")
```

## Project Structure

```
biosafe-classifier/
├── app.py                  Streamlit dashboard
├── train.py                Training pipeline
├── requirements.txt
├── README.md
├── data/
│   └── training_data.json  Synthetic labeled dataset (generated)
├── models/
│   ├── classifier.joblib   Trained ensemble model (generated)
│   ├── feature_columns.json
│   └── metrics.json
└── utils/
    ├── __init__.py
    ├── taxonomy.py         Risk taxonomy and keyword definitions
    ├── features.py         Feature extraction pipeline
    └── generate_data.py    Synthetic data generator
```

## Technical Details

**Features extracted per query (36 total):**
- Topic category keyword matches across 10 categories
- Escalation signal detection (4 signal types, weighted)
- De-escalation signal detection (3 signal types, weighted)
- Composite risk score combining base risk with escalation and de-escalation modifiers
- Text statistics including word count, specificity indicators, and protocol request detection

**Model:** Soft-voting ensemble of Random Forest (200 trees, balanced class weights) and Gradient Boosting (200 estimators), with StandardScaler preprocessing. Evaluated with stratified 5-fold cross-validation.

**Training data:** 85 synthetic biological queries I wrote and labeled across all five risk levels. Balanced across categories to make sure the classifier learns to distinguish between education, legitimate research, dual-use adjacent work, explicitly dangerous requests, and weaponization attempts.

## Limitations and What's Next

I want to be upfront about where this stands and what it would take to make it production-ready.

- **Training data is synthetic and small.** I wrote all 85 examples myself. Real deployment would need a much larger, expert-labeled dataset with diverse phrasing and adversarial examples designed to trick the system.
- **Keyword-based features are brittle.** A sophisticated user can rephrase queries to dodge keyword detection. The next version should incorporate embedding-based semantic similarity using something like BioBERT so it catches meaning, not just words.
- **No adversarial robustness testing yet.** I haven't stress-tested this against jailbreak-style prompts, prompt injection, or obfuscation techniques. That's a priority for the next iteration.
- **Single-query context window.** Right now each query is classified independently. In reality, risk can accumulate across a multi-turn conversation where no single message is dangerous on its own.
- **LLM-in-the-loop evaluation.** A production version would benefit from using an LLM to assess intent and context beyond what keyword matching can capture.

## Biosecurity Frameworks Referenced

- U.S. Government Policy for Oversight of Dual Use Research of Concern (2014)
- Australia Group Common Control Lists
- Biological Weapons Convention (BWC)
- NSABB Recommendations for Evaluation of Dual Use Research
- CDC/USDA Select Agent and Toxin List

## License

MIT
