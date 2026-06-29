# Insurance Claim Risk Prediction 🚗💰
## 🔹 Executive Summary
This repository contains a production-ready data science framework designed to predict the likelihood of an insurance policyholder filing a claim. Built using Python and Scikit-Learn, the core engine processes over 595,000 corporate records containing high-dimensional, anonymized features. The project strictly focuses on resolving real-world data challenges, specifically hidden missing data architectures and extreme target class imbalances

* **Domain:** Finance / Insurance
* **Dataset:** 595,212 observations with 59 anonymized features. Missing values are natively encoded as `-1`.

---

## 🛠️ Challenges Faced & Solutions

### 1. Hidden Missing Values (`-1` Encoding)
* **Challenge:** The dataset initially showed zero missing entries because NaNs were implicitly encoded as `-1`. 
* **Solution:** Replaced all `-1` flags with proper `np.nan` values. This exposed 13 columns with missing values. The feature `ps_car_03_cat` had >70% missing data and was dropped. Categorical features were imputed using the **Mode**, and continuous features using the **Median** to protect data integrity.

### 2. Severe Class Imbalance (3.64% Minor Class)
* **Challenge:** Only 3.64% of drivers filed a claim. Standard machine learning algorithms trained on this data would optimize for accuracy by simply predicting `0` (No Claim) for everything.
* **Solution:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the *training split* after performing a stratified 80/20 train-test split. This balanced the training target ratio to a clean 50/50 without causing data leakage into our validation set.

---

## 📊 Model Comparison Report

Two benchmark models were trained on the SMOTE-balanced training subset and evaluated on an unseen stratified test set:

| Evaluation Metric | Logistic Regression | Random Forest Classifier |
| :--- | :--- | :--- |
| **Overall Accuracy** | 89% | 93% |
| **Class 1 (Claim) Recall**| 10% | 06% |
| **Class 1 Precision** | 04% | 05% |
| **ROC-AUC Score** | 0.5270 | 0.5551 |

##  Data set Download link 
https://drive.google.com/file/d/12ivQiTKYXQH2LYH9kWoufEhawTEP3qNt/view?usp=sharing
## Google colab link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolipakaAbhiram/Insurance_Claim_Prediction/blob/main/InsClaimPred.ipynb)
