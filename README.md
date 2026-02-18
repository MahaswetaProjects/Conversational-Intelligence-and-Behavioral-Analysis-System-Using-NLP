# Conversational Intelligence and Behavioral Analytics System Using NLP

## Overview

This project is an end-to-end Natural Language Processing (NLP) system designed to analyze conversational data, extract behavioral insights, and predict the likely author of a message. It demonstrates the complete machine learning lifecycle — from raw data ingestion and preprocessing to modeling, analytics, and deployment.

The system combines text analytics, behavioral feature engineering, machine learning classification, and interactive visualization to showcase how unstructured communication data can be transformed into actionable insights.

---

## Key Objectives

* Build a structured dataset from raw chat exports
* Perform exploratory and behavioral analysis of conversations
* Develop an NLP model to predict message authorship
* Create an interactive analytics dashboard
* Deploy a real-time prediction interface

---

## Features

* End-to-end data pipeline for chat processing
* Text feature extraction using TF-IDF
* Behavioral metrics such as activity patterns and message length
* Machine learning classification model for author prediction
* Interactive Gradio application for real-time inference
* Business intelligence dashboard for insights

---

## Tech Stack

### Programming & Core

* Python

### Data Processing

* Pandas
* NumPy

### NLP & Machine Learning

* NLTK
* spaCy
* Scikit-learn

### Visualization & Analytics

* Matplotlib
* Seaborn
* Plotly
* Power BI

### Deployment

* Gradio

---

## Dataset

The dataset consists of exported chat conversations containing the following fields:

* `date` — message date
* `time` — message timestamp
* `user` — sender name
* `message` — text content

All data used in this project is anonymized to preserve privacy.

---

## Methodology

### 1. Data Preprocessing

* Parsing chat exports
* Removing system messages
* Text cleaning and normalization
* Feature creation (time features, message length)

### 2. Exploratory Analysis

* User activity distribution
* Temporal messaging trends
* Behavioral communication patterns

### 3. Feature Engineering

* TF-IDF vectorization of message text
* Behavioral attributes

### 4. Modeling

* Random Forest classifier
* Train-test split evaluation
* Performance metrics (Accuracy, Precision, Recall, F1)

### 5. Deployment

* Model and vectorizer serialization
* Real-time prediction interface using Gradio

### 6. Analytics Dashboard

* Message volume trends
* User activity comparison
* Time-based communication insights

---

## Results

The system successfully demonstrates that conversational text contains distinguishable linguistic and behavioral patterns that can be leveraged for predictive modeling. The deployed interface enables real-time predictions, while the dashboard provides interpretable insights for stakeholders.

---

## Applications

* Collaboration and communication analytics
* Digital behavior research
* Customer interaction monitoring
* Authorship attribution systems

---

## Future Improvements

* Incorporate contextual embeddings (BERT or similar models)
* Add sentiment and emotion analysis
* Build a REST API for scalable deployment
* Enable file upload for real-time chat analysis

