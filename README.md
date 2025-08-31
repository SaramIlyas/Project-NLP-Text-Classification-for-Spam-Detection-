# NLP Text Classification for Spam Detection

A compact, reproducible project for **spam vs. ham** SMS classification. It includes classic ML baselines (TF–IDF + Logistic Regression / Linear SVM / Multinomial Naive Bayes), a lightweight neural model (Keras/TensorFlow), 
threshold tuning for class-imbalance, and **explainability** with SHAP.

> **Course/Topic:** DLBAIPNLP01 – Topic 2 (Spam text classification)  
> **Report:** see `report/` (PDF/DOCX)  
> **Notebook:** see `notebooks/`



##  Methods Overview

- **Preprocessing / Features**
  - Text normalization (lowercasing, basic cleaning)
  - Train/validation/test split with **stratification**
  - **TF–IDF** (fit on train only; transform val/test)
- **Models**
  - Logistic Regression (class-weight aware), Linear SVM, Multinomial Naive Bayes
  - Lightweight Neural model (Keras/TensorFlow) with tokenization + padding
  - **EarlyStopping on PR-AUC**, class weights for imbalance
- **Threshold tuning**
  - Operating threshold tuned on validation set (optimize F1 / F\_{β} or desired trade‑off), then applied to test
- **Explainability**
  - **SHAP** (on the TF–IDF pipeline) for global & local feature attributions

## Evaluation

We report and discuss on the **held-out test set**:

- Accuracy (for context), **Precision**, **Recall**, **F1**
- **ROC-AUC**, **PR-AUC**
- Confusion matrix
- Calibrated vs. tuned-threshold performance (briefly)

## 🔁 Reproducibility

- Fixed seeds (Python/NumPy/TensorFlow) set at the notebook top.
- Train/val/test split with a constant `random_state`.
- Exact versions recorded below and in `requirements.txt`.
- Paths are **relative**; no absolute system paths are required.

### `requirements.txt`

```txt
python-dateutil==2.9.0.post0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.13.1
matplotlib==3.9.0
seaborn==0.13.2
tensorflow==2.16.1
keras==3.3.3
shap==0.45.1
jupyterlab==4.2.3
ipykernel==6.29.5
```

##  Data

**Dataset:** SMS Spam Collection (5,572 messages; “ham”/“spam”).  
You may use source below mirrored on Kaggle).

Kaggle mirror:  
  `https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset`
