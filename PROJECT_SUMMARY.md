# OncoSense AI: Project Executive Summary
> **Early Detection. Smarter Decisions.**

OncoSense AI is an AI-powered diagnostic assistant that analyzes fine needle aspirate (FNA) measurements to predict whether a breast tumor is benign or malignant. This project was developed to demonstrate a complete machine learning workflow with high-fidelity visualization, explainable AI (XAI) risk analysis, and robust evaluation.

---

## 📊 Quick Performance Summary

The project trained and compared three classification models on the Wisconsin Breast Cancer Diagnostic dataset (80% Train, 20% Test):

| Metric | Logistic Regression | Decision Tree | Random Forest (Selected Model) |
| :--- | :---: | :---: | :---: |
| **Train Accuracy** | $98.68\%$ | $100.00\%$ | **$100.00\%$** |
| **Test Accuracy** | $96.49\%$ | $92.98\%$ | **$97.37\%$** |
| **Precision** | $97.50\%$ | $90.48\%$ | **$100.00\%$** |
| **Recall (Sensitivity)**| $92.86\%$ | $90.48\%$ | **$92.86\%$** |
| **F1 Score** | $95.12\%$ | $90.48\%$ | **$96.30\%$** |
| **ROC-AUC Score** | **$99.60\%$** | $92.46\%$ | **$99.29\%$** |
| **Generalization Status**| Highly Stable | Overfitting | **Highly Stable & Robust** |

*Note: **Random Forest** was automatically selected as the final production model due to its superior F1-Score ($0.9630$), zero False Positive Rate (100% Precision), and excellent generalization control.*

---

## 🔑 Key Achievements & Highlights

1. **Explainable AI (XAI) Risk Meter**: Developed a dynamic, console-based risk meter colored using ANSI escape codes to display low, moderate, or high risk based on classification probabilities.
2. **Local Feature Contribution**: Coded a patient-level explanation engine that details the top 3 nuclear characteristics driving the risk, calculating feature deviation from the benign cohort baseline.
3. **Overfitting Mitigation**: Analyzed the generalization gap across linear, tree, and ensemble models, highlighting why tree models overfit and how bagging (Random Forest) resolves it.
4. **Professional Visual Assets**: Generated 9 publication-grade charts covering class distributions, correlations, outliers, ROC curves, and confusion matrices saved in the `visualizations/` folder.

---

## 💼 Portfolio & Interview Talking Points

- **Metric Selection**: *"In medical diagnostics, a False Negative can prevent life-saving treatment. Therefore, I designed OncoSense AI to select models based on the **F1-Score** (which balances precision and recall) rather than raw accuracy."*
- **Explainability**: *"Instead of delivering a black-box binary prediction, the system acts as a clinical assistant by generating a patient risk report containing confidence percentages and listing the top nuclear characteristics driving the risk."*
- **Multicollinearity**: *"During EDA, I identified strong multicollinearity among size-based features. While this poses stability issues for Logistic Regression coefficients, Random Forest handles it natively through random feature splitting."*

---

## 🚀 Ready-to-Post LinkedIn Template

You can copy and paste this template to share your work on LinkedIn:

```text
🚀 Excited to share my latest machine learning project: OncoSense AI! 🧬

Early detection saves lives. I built a Breast Cancer Risk Predictor using the Wisconsin Diagnostic dataset to classify breast tumor mass measurements as Benign or Malignant. 

Key Highlights of the Project:
✔ Developed a complete pipeline including data inspection, StandardScaler scaling, and stratified splitting.
✔ Trained and compared Logistic Regression, Decision Trees, and Random Forest.
✔ Random Forest achieved a Test Accuracy of 97.37% and an F1 Score of 96.30%.
✔ Integrated a visual AI Risk Meter that dynamically scales with prediction probability.
✔ Built a patient-level explanation engine identifying the top clinical indicators contributing to high-risk diagnoses (Explainable AI).

Check out the full repository and visualizations here! 
👉 [Insert Github Link]

#MachineLearning #DataScience #HealthcareAI #Python #ArtificialIntelligence #PortfolioProject
```
