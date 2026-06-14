# Aerospace NLP Document Classifier

NLP pipeline for classifying and searching aerospace and defense technical documents. Built with TF-IDF feature extraction, Naive Bayes and SVM classifiers, cosine similarity search, and an interactive browser-based search interface.

---

## What it does

Technical document management is a real problem in aerospace organizations — engineers spend significant time hunting for the right manual, report, or specification across thousands of documents. This project builds a system that automatically categorizes documents by type and makes them searchable by content similarity.

**Document classification** — TF-IDF with bigram features extracts the most discriminating terms from each document. A Naive Bayes classifier and a linear SVM are trained and compared. The SVM achieves ~80% accuracy on held-out documents across aerospace document categories including propulsion, avionics, structures, guidance systems, and maintenance procedures.

**Semantic search** — cosine similarity between TF-IDF vectors lets you find documents related to a query or to another document, even when exact keywords don't match. The similarity network visualization shows which documents cluster together by content.

**Interactive search interface** — `document_search.html` is a standalone browser tool that loads the search index from `search_data.json` and lets you query the corpus without running any Python.

---

## Files

```
Untitled.ipynb          — full NLP pipeline: preprocessing, training, evaluation
classifier_analysis.png — confusion matrix, classification report, model comparison
similarity_network.png  — document similarity graph showing content clusters
wordclouds.png          — per-category term frequency visualizations
document_search.html    — interactive browser search interface
search_data.json        — precomputed search index for the HTML interface
```

---

## Pipeline

**Text preprocessing** — lowercasing, punctuation removal, stop word filtering, stemming. Aerospace documents contain a lot of domain-specific terminology that standard stop word lists don't handle well, so a custom aerospace stop word list was added.

**Feature extraction** — TF-IDF with unigrams and bigrams. Bigrams capture phrases like "fuel injection", "thrust vector", and "guidance system" that unigrams would split apart and lose meaning.

**Classification** — two models trained and compared:
- Multinomial Naive Bayes — fast, interpretable, strong baseline
- Linear SVM — better on high-dimensional sparse text features, higher accuracy

**Similarity search** — cosine similarity between document TF-IDF vectors. Documents pointing in the same direction in feature space are semantically similar even if they don't share exact words.

**Visualization** — word clouds per category show the most distinctive terms, the similarity network shows document clustering, and the classifier analysis shows per-class precision, recall, and F1.

---

## Running it

```bash
git clone https://github.com/cymmerman/aerospace-nlp-document-classifier.git
cd aerospace-nlp-document-classifier
pip install numpy pandas scikit-learn nltk matplotlib wordcloud jupyter
jupyter notebook
```

Open `Untitled.ipynb` and run all cells.

To use the search interface without Python, just open `document_search.html` in a browser.

---

## Stack

- Python — scikit-learn, NLTK, pandas, numpy
- Visualization — matplotlib, wordcloud
- Search interface — vanilla JavaScript, HTML

---

## Related projects

- [`turbofan-predictive-maintenance`](https://github.com/cymmerman/turbofan-predictive-maintenance) — anomaly detection and RUL prediction on NASA sensor data
- [`mcmc-bayesian-flight-estimation`](https://github.com/cymmerman/mcmc-bayesian-flight-estimation) — Metropolis-Hastings MCMC from scratch for Bayesian flight parameter estimation
- [`ml-pipeline-deployment`](https://github.com/cymmerman/ml-pipeline-deployment) — production FastAPI deployment with SHAP, drift detection, and live dashboard
