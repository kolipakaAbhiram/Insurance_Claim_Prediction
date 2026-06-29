# Insurance Claim Risk Prediction 🚗💰

## 📋 Project Overview
This project focuses on building a predictive machine learning pipeline to determine the likelihood of an insurance policyholder filing a claim. Deep insight into user risk profiles allows insurance providers to optimize policy pricing, manage financial risk, and target the right audiences with tailored marketing strategies.

* **Domain:** Finance / Insurance
* **Dataset:** 595,212 observations with 59 anonymized features. Missing values are natively encoded as `-1`[cite: 2].

---

## 🛠️ Challenges Faced & Solutions

### 1. Hidden Missing Values (`-1` Encoding)
* **Challenge:** The dataset initially showed zero missing entries because NaNs were implicitly encoded as `-1`[cite: 2]. 
* **Solution:** Replaced all `-1` flags with proper `np.nan` values[cite: 2]. This exposed 13 columns with missing values[cite: 2]. The feature `ps_car_03_cat` had >70% missing data and was dropped[cite: 2]. Categorical features were imputed using the **Mode**, and continuous features using the **Median** to protect data integrity[cite: 2].

### 2. Severe Class Imbalance (3.64% Minor Class)
* **Challenge:** Only 3.64% of drivers filed a claim[cite: 2]. Standard machine learning algorithms trained on this data would optimize for accuracy by simply predicting `0` (No Claim) for everything.
* **Solution:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the *training split* after performing a stratified 80/20 train-test split[cite: 2]. This balanced the training target ratio to a clean 50/50 without causing data leakage into our validation set[cite: 2].

---

## 📊 Model Comparison Report

Two benchmark models were trained on the SMOTE-balanced training subset and evaluated on an unseen stratified test set[cite: 2]:

| Evaluation Metric | Logistic Regression | Random Forest Classifier |
| :--- | :--- | :--- |
| **Overall Accuracy** | 89%[cite: 2] | 93%[cite: 2] |
| **Class 1 (Claim) Recall**| 10%[cite: 2] | 06%[cite: 2] |
| **Class 1 Precision** | 04%[cite: 2] | 05%[cite: 2] |
| **ROC-AUC Score** | 0.5270[cite: 2] | 0.5551[cite: 2] |

### Production Recommendation
While the **Random Forest** model delivers a higher baseline accuracy (93%) and a slightly superior ROC-AUC score (0.5551)[cite: 2], both models show exceptionally low precision and recall on the actual claim event (`Class 1`)[cite: 2]. This implies that anonymized features require deeper feature engineering (e.g., target encoding categorical variables, threshold moving) before immediate production deployment.

---

## 🎯 Strategic Suggestions for Marketing & Underwriting Teams

1. **Risk-Adjusted Premium Bundling:** Use the model's predicted risk probabilities rather than binary outputs. Customers flagged with an extremely low probability of making a claim can be targeted by marketing for discounted premium rates or safe-driver reward renewals.
2. **De-incentivize High-Risk Profiles:** For users showing a higher propensity for claims, suggest policy variants featuring higher deductibles. This retains them as buying customers while safeguarding the business from massive claim liabilities.
3. **Feature Enrichment Campaigns:** Since the raw data shows weak linear correlations, the marketing and onboarding teams should aim to collect explicit behavioral indicators (e.g., annual mileage, vehicle usage type) to significantly boost predictive accuracy.

##  Data set Download link 
https://drive.google.com/file/d/12ivQiTKYXQH2LYH9kWoufEhawTEP3qNt/view?usp=sharing
