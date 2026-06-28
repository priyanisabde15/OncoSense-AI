# OncoSense AI: Breast Cancer Risk Predictor
> **Early Detection. Smarter Decisions.**

OncoSense AI is a professional-grade, educational machine learning project that predicts whether a breast tumor is **Benign** or **Malignant** using the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. 

Rather than simply outputting a binary classification, OncoSense AI calculates a clinical risk score, displays a dynamic, color-coded **AI Risk Meter**, lists the top biological features contributing to the risk, and compares multiple models to select the most reliable predictor.

---

## 🌟 Key Features

- **End-to-End ML Pipeline**: Covers data inspection, automated preprocessing, exploratory data analysis (EDA), model training, evaluation, and overfitting analysis.
- **Model Comparison**: Automatically trains and evaluates **Logistic Regression**, **Decision Tree**, and **Random Forest**.
- **Automated Model Selection**: Dynamically chooses the best model based on the **F1-Score** (crucial in medical settings where False Negatives must be minimized).
- **Interactive Risk Prediction Engine**: Evaluates patient data through a beautiful CLI loop, displaying:
  - **Prediction & Confidence** (e.g., Malignant @ 98.4%)
  - **Visual AI Risk Meter** (🟢 Low, 🟡 Moderate, 🔴 High)
  - **Local Feature Contribution** explaining *why* a patient is classified as high-risk.
- **High-Quality Visualizations**: Saves 9 publication-grade charts to the `visualizations/` folder, including ROC curves and side-by-side Confusion Matrices.

---

## 📁 Repository Structure

Click the links below to explore the project files directly:

* 📄 [requirements.txt](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/requirements.txt) — Project dependencies.
* 🐍 [breast_cancer_risk_predictor.py](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/breast_cancer_risk_predictor.py) — Complete executable python pipeline & interactive engine.
* 📓 [breast_cancer_risk_predictor.ipynb](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/breast_cancer_risk_predictor.ipynb) — Interactive Jupyter notebook with markdown notes and LaTeX equations.
* 📊 [Breast_Cancer.csv](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/Breast_Cancer.csv) — Wisconsin Breast Cancer Diagnostic dataset.
* 📘 [PROJECT_DOCUMENTATION.md](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/PROJECT_DOCUMENTATION.md) — Comprehensive technical documentation, mathematical models, and feature interpretations.
* 📒 [PROJECT_SUMMARY.md](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/PROJECT_SUMMARY.md) — Executive summary tailored for job portfolios and recruiters.
* 📂 [visualizations/](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/visualizations) — Output directory containing generated plots.

---

## ⚙️ Setup and Installation

### 1. Install Dependencies
Make sure Python 3.8+ is installed. Clone/open the workspace directory and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Main Script
You can execute the entire pipeline and launch the interactive engine by running:
```bash
python breast_cancer_risk_predictor.py
```

### 3. Run in Non-Interactive Mode (For Automated Validation)
If you want to run the pipeline, generate all visualizations, and execute a quick test case without launching the interactive CLI loop:
```bash
python breast_cancer_risk_predictor.py --non-interactive
```

---

## 🖥️ Interactive CLI Preview

When you run the script, you are presented with an interactive terminal interface:

```text
============================================================
       ONCOSENSE AI: INTERACTIVE RISK PREDICTION ENGINE     
============================================================
Active Model: Random Forest
Select an option:
  1. Pick a random patient case from the Test Set
  2. Select patient case by ID (index 0 - 113)
  3. Manual measurement entry (Enter top indicators)
  4. Return to main script / Exit

Enter choice (1-4): 1

--------------------------------------------------
              ONCOSENSE RISK REPORT             
--------------------------------------------------
Patient Case Ref: Test Index #32
Actual Clinical Diagnosis: Malignant
OncoSense AI Prediction:   Malignant
Prediction Confidence:     97.00%

Visual Risk Meter:
[■■■■■■■■■■] 97.0% - 🔴 HIGH RISK

Top Contributing Risk Features:
• Worst Concave Points (Elevated compared to typical benign baseline)
• Worst Area (Elevated compared to typical benign baseline)
• Worst Perimeter (Elevated compared to typical benign baseline)
--------------------------------------------------
```

---

## 📈 Visualizations Generated
The script automatically creates and saves 9 plots under `visualizations/`:
1. `diagnosis_distribution.png` — Class balance visualization.
2. `correlation_heatmap.png` — Heatmap highlighting multicollinear and target-correlated features.
3. `feature_distribution.png` — Kernel Density Estimation (KDE) comparison of benign vs. malignant features.
4. `pair_plot.png` — Pairwise joint scatter distributions of key features.
5. `feature_correlation_with_diagnosis.png` — Horizontal bar chart showing target Pearson correlations.
6. `boxplots_outliers.png` — Boxplots highlighting outlier points in clinical sizes.
7. `roc_curves.png` — Receiver Operating Characteristic curve for Logistic Regression, Decision Tree, and Random Forest.
8. `confusion_matrices.png` — Confusion Matrices showing True Positives, False Positives, True Negatives, and False Negatives.
9. `feature_importance.png` — Importance ranking showing top features driving the model.

---

## 👩‍🔬 Educational Background & Clinical Context

OncoSense AI uses features derived from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

For an extensive technical breakdown of the mathematics of Logistic Regression, Decision Tree Gini calculations, and clinical explanations, refer to [PROJECT_DOCUMENTATION.md](file:///c:/PROFFESIONAL%20DOCS/InternPE%20Internship/Task%204%20Breast%20Cancer%20Prediction/PROJECT_DOCUMENTATION.md).
