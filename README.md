
# Sentiment Modeling on Financial News

This project explores the use of Natural Language Processing (NLP) and financial language models (FinBERT) to classify financial news headlines into sentiment categories and evaluate their potential for predicting market behavior.

---

## Project Overview

Financial news plays a key role in shaping investor sentiment and influencing market movements.
In this notebook, we:

1. Load and preprocess a Kaggle financial news dataset (2025).
2. Explore dataset features (headline text, sentiment labels, sector, impact level, market events, trading volume, index changes).
3. Experiment with multiple NLP pipelines:

   * TF-IDF + Logistic Regression
   * Fine-tuned FinBERT (sequence classification)
   * FinBERT embeddings + Logistic Regression
   * FinBERT embeddings + XGBoost
4. Visualize embeddings with PCA and t-SNE to assess class separability.
5. Evaluate model performance using precision, recall, F1-score, and confusion matrices.

---

## Key Findings

* Negative vs Neutral sentiment classification is not easily separable in this dataset.
* All tested models (Logistic Regression, XGBoost, FinBERT fine-tuning) achieved around 50% accuracy, close to random guessing.
* PCA and t-SNE visualizations confirmed that embeddings of Negative and Neutral overlap significantly.
* This suggests that labels are noisy/ambiguous or that headline text alone is insufficient for clear sentiment separation.

---

## Dataset

* Source: [Kaggle – Financial News Market Events Dataset 2025](https://www.kaggle.com/datasets/pratyushpuri/financial-news-market-events-dataset-2025)
* Features include: `Headline`, `Sentiment`, `Impact_Level`, `Sector`, `Market_Event`, `Index_Change_Percent`, and more.

---

## Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## Results Summary

| Model                              | Accuracy | Notes                           |
| ---------------------------------- | -------- | ------------------------------- |
| TF-IDF + Logistic Regression       | ~0.52    | Baseline                        |
| FinBERT Fine-tuning (2 classes)    | ~0.51    | Overfits, no gain               |
| FinBERT Embeddings + Logistic Reg. | ~0.52    | Linear separation not possible  |
| FinBERT Embeddings + XGBoost       | ~0.51    | Non-linear, still no separation |

---

## Conclusion

This dataset is not well-suited for distinguishing Negative vs Neutral sentiment.
The experiments highlight the challenges of financial sentiment classification, the importance of embedding visualization, and the need to combine text with structured financial features for stronger predictive performance.

