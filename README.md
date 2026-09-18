# B198c7-AI-Applications-for-Digital-Business-
# SteamSense: NLP Sentiment Analysis & Topic Modeling on Steam Reviews

SteamSense is an end-to-end Natural Language Processing (NLP) pipeline designed to systematically analyze and classify player feedback on the Steam platform. Processing a massive dataset of over 9 million records, the system extracts binary sentiment classes and uncovers latent text themes using unsupervised topic modeling, converting raw text data into actionable business intelligence for game developers.

## 🚀 Key Features
* **Large-Scale Data Pipeline:** Efficient chunk-based data parsing optimized to bypass strict hardware memory limitations.
* **Stratified Sampling & Cleaning:** Text normalization via regex clearing, NLTK word tokenization, and WordNet lemmatization.
* **Multi-Model Sentiment Benchmarking:** Comparative validation across rule-based lexicons (VADER), statistical models (Naive Bayes, Logistic Regression), and deep learning transformers (BERT).
* **Latent Dirichlet Allocation (LDA):** Automated topic extraction utilizing the `gensim` library to pinpoint top positive praise and negative complaints.

## 📊 Quantitative Performance & Results

The models were evaluated using a stratified 80/20 test split on a balanced 100,000-review English subset. The quantitative metrics recorded are as follows:

| Model Approach | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| VADER (Lexicon Baseline) | 0.7000 | 0.9800 | 0.7100 | 0.8206 |
| Multinomial Naive Bayes (TF-IDF) | 0.8931 | 0.9800 | 0.9100 | 0.9416 |
| Logistic Regression (TF-IDF) | 0.9012 | 0.9800 | 0.9100 | 0.9452 |
| **Fine-tuned BERT (Transformer)** | **0.9600** | **0.9800** | **1.0000** | **0.9900** |

### Core Analytics Visualizations

* **Exploratory Data Analysis Overview:** Displays severe class imbalance (89.2% positive reviews) alongside review word length trends and top-performing game distributions.
* **Topic Modeling Frequency Distribution:** Highlights clear semantic clusters showing gameplay engagement patterns vs. performance optimization crashes.
* **Confusion Matrices Metrics Validation:** Side-by-side behavioral charts mapping predictive accuracy vs. true baseline user tags.

## 🛠️ Project Structure & Technologies
* **Environment:** Google Colab (T4 GPU Accelerated Session Architecture)
* **Core Packages:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `transformers`, `torch`, `gensim`, `matplotlib`, `seaborn`

## 📋 Data Resource Reference
* **Dataset Used:** Steam Reviews 2021 Dataset (9.6M+ user annotations)
* **Source Archive:** https://www.kaggle.com/datasets/najzeko/steam-reviews-2021
