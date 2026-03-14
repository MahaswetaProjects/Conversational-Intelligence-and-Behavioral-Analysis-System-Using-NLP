# ConvoIQ — Conversational Intelligence & Behavioral Analytics System

> End-to-end NLP project: from raw WhatsApp exports to a production-grade analytics dashboard with ML-powered authorship prediction.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

ConvoIQ transforms unstructured WhatsApp group chat exports into a rich intelligence platform. It combines classical NLP, behavioral feature engineering, and Random Forest classification to extract deep insights from conversational data.

**Key Differentiators:**
- **Behavioral Fingerprinting** — Unique communication DNA per user (vocabulary richness, peak hours, response patterns)
- **Conversation Momentum** — Rolling-average trend to track group engagement over time
- **Sentiment Timeline** — VADER-powered daily sentiment tracking across the group
- **Response Chain Analysis** — Who replies to whom, revealing interaction graphs
- **Author Prediction** — TF-IDF + behavioral features → Random Forest classifier with confidence scores

---

## Project Structure

```
convoiq/
├── app.py                        # Streamlit dashboard (main entry point)
├── requirements.txt
│
├── src/
│   ├── preprocessor.py           # WhatsApp chat parser & feature engineering
│   └── analytics.py              # All analytics computations (modular)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_power_bi_export.ipynb
│
├── data/
│   ├── powerbi_chat_dataset.csv  # Enriched dataset
│   ├── user_summary.csv          # Per-user stats
│   └── daily_trend.csv           # Daily message counts
│
└── models/
    ├── model.pkl                 # Trained Random Forest
    ├── vectorizer.pkl            # TF-IDF vectorizer
    └── label_encoder.pkl         # User label encoder
```

---

## Quick Start

```bash
# 1. Clone / download the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (once)
# Open notebooks/03_modelling.ipynb and run all cells
# This saves model.pkl, vectorizer.pkl, label_encoder.pkl → models/

# 4. Launch the dashboard
streamlit run app.py
```

---

## Methodology

### 1. Data Preprocessing (`src/preprocessor.py`)
- Regex-based parsing of WhatsApp export format (`D/M/YY, HH:MM - User: message`)
- Removal of system messages, media placeholders, deleted messages
- User anonymisation (Name/Phone → User1, User2, …)
- Temporal feature extraction: hour, day, month, week
- Text feature extraction: message length, word count, emoji count, question detection, link detection
- Response time computation and conversation-starter flagging

### 2. Exploratory Analysis (`notebooks/02_exploratory_analysis.ipynb`)
- Message frequency by user, hour, day, and month
- Activity heatmap (hour × day matrix)
- Response time analysis — fastest responders
- VADER sentiment scoring per message
- Keyword frequency and vocabulary analysis

### 3. Authorship Prediction Model (`notebooks/03_modelling.ipynb`)

| Component | Detail |
|-----------|--------|
| Text Features | TF-IDF (unigrams + bigrams, top 5,000 features, sublinear_tf) |
| Behavioral Features | Hour, day-of-week, message length, word count |
| Classifier | Random Forest (300 estimators, stratified split) |
| Evaluation | Accuracy, Precision, Recall, F1, 5-Fold Cross-Validation |

### 4. Dashboard (`app.py`)

Five interactive sections:

| Tab | Description |
|-----|-------------|
| Overview | KPIs, volume trend, sentiment distribution, auto-insights |
| User Analytics | Leaderboard, behavioral fingerprint radar, response times |
| Time Analysis | Hourly bars, day-of-week, heatmap, monthly breakdown |
| NLP Insights | Sentiment timeline, word clouds, keyword frequency, response chains |
| Author Prediction | Real-time authorship prediction with confidence scores |

---

## Behavioral Fingerprint

Each user is profiled across six dimensions visualised on a radar chart:

- **Volume** — Relative message count
- **Vocab Richness** — Unique words / total words ratio
- **Emoji Usage** — Emojis per message
- **Question Rate** — Proportion of messages containing `?`
- **Conv Starter** — How often the user initiates new conversations
- **Msg Length** — Average characters per message

Badges are auto-assigned: *Night Owl*, *Question Asker*, *Emoji Lover*, *Conversation Starter*, *Rich Vocabulary*.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Core | Python 3.11, Pandas, NumPy |
| NLP & ML | Scikit-learn, VADER Sentiment, TF-IDF |
| Visualisation | Plotly, Matplotlib, WordCloud |
| Deployment | Streamlit |
| Serialisation | Joblib, SciPy (sparse matrices) |

---

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to `app.py`
4. Add `models/*.pkl` files to the repo (or use `st.secrets` + cloud storage)
5. Deploy 

---

## Future Improvements

- [ ] BERT/sentence-transformers for richer text embeddings
- [ ] Topic modelling with BERTopic
- [ ] NetworkX interaction graph visualisation
- [ ] REST API with FastAPI for scalable serving
- [ ] Real-time file upload → instant re-analysis pipeline
- [ ] Multi-language support (Hinglish NLP)

---

## Privacy Note

All personal identifiers (names, phone numbers) are anonymised to `User1, User2, …` during preprocessing. No raw personal data is stored or transmitted.

---

*Built as a placement-ready NLP portfolio project demonstrating end-to-end machine learning engineering.*
