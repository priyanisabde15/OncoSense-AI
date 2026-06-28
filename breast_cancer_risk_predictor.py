import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set styling for plots to look professional and modern
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Color Palette for visuals (Hex colors for premium design)
PALETTE = {
    'Malignant': '#e74c3c', # Vibrant red
    'Benign': '#2ecc71',    # Emerald green
    'Primary': '#2c3e50',   # Dark Slate Blue
    'Secondary': '#34495e', # Lighter Slate
    'Accent': '#f1c40f'      # Bright Yellow
}

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Prints a styled header for the console CLI."""
    print("\n" + "=" * 60)
    print(f" {title.center(58)} ")
    print("=" * 60)

# ==========================================
# PHASE 1: DATA INSPECTION & PREPROCESSING
# ==========================================

def load_and_inspect_data(file_path):
    """Loads dataset and performs initial inspection."""
    print_header("PHASE 1: DATASET INSPECTION")
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    print(f"✔ Dataset loaded successfully.")
    print(f"✔ Shape: {df.shape[0]} rows (patients), {df.shape[1]} columns (features).")
    
    print("\n[Data Types Summary]")
    print(df.dtypes.value_counts())
    
    # Missing Value Analysis
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"\n[Missing Value Analysis]")
    if total_missing == 0:
        print("✔ No missing values found in the dataset.")
    else:
        print(f"⚠ Found {total_missing} missing values:")
        print(missing_values[missing_values > 0])
        
    # Duplicate Analysis
    duplicates = df.duplicated().sum()
    print(f"\n[Duplicate Check]")
    if duplicates == 0:
        print("✔ No duplicate records found.")
    else:
        print(f"⚠ Found {duplicates} duplicate records.")
        
    # Class Balance Analysis
    class_counts = df['diagnosis'].value_counts()
    print(f"\n[Class Balance Analysis - Target Column 'diagnosis']")
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        diagnosis_name = "Malignant" if label == 'M' else "Benign"
        print(f"  - {diagnosis_name} ({label}): {count} patients ({percentage:.2f}%)")
        
    return df

def preprocess_data(df):
    """Preprocesses the breast cancer dataset."""
    print_header("PHASE 2: DATA PREPROCESSING")
    
    # Create a copy to prevent SettingWithCopyWarning
    data = df.copy()
    
    # 1. Drop identifier column (id) - it has no predictive power and can lead to overfitting
    print("1. Removing identifier column 'id'...")
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
        print("   ✔ 'id' column dropped.")
    
    # Remove any completely empty columns (often generated as 'Unnamed: 32')
    empty_cols = [col for col in data.columns if 'Unnamed' in col]
    if empty_cols:
        data = data.drop(columns=empty_cols)
        print(f"   ✔ Dropped empty column(s): {empty_cols}")

    # 2. Handling Missing Values (Just in case, though usually WBCD is clean)
    print("2. Handling missing values...")
    null_cols = data.columns[data.isnull().any()].tolist()
    if null_cols:
        for col in null_cols:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"   ✔ Filled missing values in '{col}' with median: {median_val}")
    else:
        print("   ✔ No missing values to impute.")

    # 3. Duplicate removal
    print("3. Checking and removing duplicate records...")
    initial_shape = data.shape[0]
    data = data.drop_duplicates()
    final_shape = data.shape[0]
    if initial_shape > final_shape:
        print(f"   ✔ Removed {initial_shape - final_shape} duplicate rows.")
    else:
        print("   ✔ No duplicate rows found.")

    # 4. Label Encoding of diagnosis (M -> 1, B -> 0)
    # Necessary because ML models work with numbers, not text labels.
    print("4. Label encoding target column 'diagnosis' (M = 1, B = 0)...")
    le = LabelEncoder()
    data['diagnosis'] = le.fit_transform(data['diagnosis'])
    print(f"   ✔ Class mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Separate features and target
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    feature_names = X.columns.tolist()

    # 5. Train-Test Split (80% Train, 20% Test)
    # Necessary to evaluate the model on unseen data and check for overfitting.
    print("5. Performing Train-Test Split (80% Training, 20% Testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✔ Training Set size: {X_train.shape[0]} samples")
    print(f"   ✔ Testing Set size: {X_test.shape[0]} samples")

    # 6. Feature Scaling using StandardScaler
    # Necessary because measurements are in different scales (e.g. area is ~1000s, smoothness is ~0.1s).
    # StandardScaler scales features to have mean=0 and variance=1.
    print("6. Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✔ Feature scaling complete.")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

# ==========================================
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ==========================================

def perform_eda(df, feature_names, output_dir="visualizations"):
    """Generates and saves professional visualizations for EDA."""
    print_header("PHASE 3: EXPLORATORY DATA ANALYSIS & PLOTTING")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Plot 1: Diagnosis Distribution
    plt.figure(figsize=(8, 6))
    colors = [PALETTE['Benign'], PALETTE['Malignant']]
    ax = sns.countplot(x='diagnosis', hue='diagnosis', data=df, palette=colors, legend=False)
    plt.title('Tumor Diagnosis Distribution (Class Balance)', pad=15)
    plt.xlabel('Diagnosis (B = Benign, M = Malignant)')
    plt.ylabel('Count')
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnosis_distribution.png'), dpi=300)
    plt.close()
    print("✔ Saved: diagnosis_distribution.png")

    # Plot 2: Correlation Heatmap (Top 15 features for readability)
    plt.figure(figsize=(12, 10))
    numeric_df = df.copy()
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['id'])
    # Encode target to numeric for correlation if it's not already numeric
    if not pd.api.types.is_numeric_dtype(numeric_df['diagnosis']):
        numeric_df['diagnosis'] = LabelEncoder().fit_transform(numeric_df['diagnosis'].astype(str))
        
    # Get top 15 correlated features with diagnosis to draw heatmap
    correlations = numeric_df.corr()
    top_corr_features = correlations['diagnosis'].abs().sort_values(ascending=False).index[:15]
    sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap (Top 15 Correlated Columns)', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("✔ Saved: correlation_heatmap.png")

    # Plot 3: Feature Distribution (radius_mean & texture_mean)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Radius Mean
    sns.kdeplot(data=df, x='radius_mean', hue='diagnosis', fill=True, palette=colors, alpha=0.5, ax=axes[0], common_norm=False)
    axes[0].set_title('Distribution of Radius Mean by Diagnosis')
    axes[0].set_xlabel('Radius Mean (mm)')
    axes[0].set_ylabel('Density')
    
    # Texture Mean
    sns.kdeplot(data=df, x='texture_mean', hue='diagnosis', fill=True, palette=colors, alpha=0.5, ax=axes[1], common_norm=False)
    axes[1].set_title('Distribution of Texture Mean by Diagnosis')
    axes[1].set_xlabel('Texture Mean (Gray-scale value)')
    axes[1].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distribution.png'), dpi=300)
    plt.close()
    print("✔ Saved: feature_distribution.png")

    # Plot 4: Pair Plot (Top 4 Important Features)
    # We will pick 4 highly correlated features with diagnosis
    top_features = correlations['diagnosis'].abs().sort_values(ascending=False).index[1:5].tolist() # Exclude diagnosis itself
    pair_grid = sns.pairplot(df, vars=top_features, hue='diagnosis', palette=colors, diag_kind='kde', height=2.5)
    pair_grid.fig.suptitle('Pair Plot of Top 4 Distinctive Features', y=1.02, fontsize=16)
    pair_grid.savefig(os.path.join(output_dir, 'pair_plot.png'), dpi=300)
    plt.close()
    print("✔ Saved: pair_plot.png")

    # Plot 5: Feature Correlation with Diagnosis
    plt.figure(figsize=(12, 6))
    target_corrs = correlations['diagnosis'].drop('diagnosis').sort_values()
    target_corrs.plot(kind='barh', color=plt.cm.viridis(np.linspace(0, 1, len(target_corrs))))
    plt.title('Feature Correlation with Target Diagnosis', pad=15)
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_with_diagnosis.png'), dpi=300)
    plt.close()
    print("✔ Saved: feature_correlation_with_diagnosis.png")

    # Plot 6: Boxplots for Outlier Detection (Key features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    key_features = ['radius_mean', 'area_mean', 'perimeter_mean', 'concavity_mean']
    for i, feature in enumerate(key_features):
        row, col = i // 2, i % 2
        sns.boxplot(x='diagnosis', y=feature, hue='diagnosis', data=df, palette=colors, ax=axes[row, col], legend=False)
        axes[row, col].set_title(f'Outlier Detection - {feature}')
        axes[row, col].set_xlabel('Diagnosis')
        axes[row, col].set_ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots_outliers.png'), dpi=300)
    plt.close()
    print("✔ Saved: boxplots_outliers.png")
    
    print("\n✔ All EDA plots generated and saved in the 'visualizations/' directory.")

# ==========================================
# PHASE 4: MODEL TRAINING & EVALUATION
# ==========================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, output_dir="visualizations"):
    """Trains 3 models, evaluates them, plots ROC/confusion matrices, and performs overfitting analysis."""
    print_header("PHASE 4: MODEL TRAINING & COMPARATIVE EVALUATION")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    roc_curves = {}
    confusion_matrices = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Train accuracy for overfitting check
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Test metrics
        test_acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results[name] = {
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC Score': auc,
            'model_object': model
        }

        # Store ROC curve values
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves[name] = (fpr, tpr, auc)

        # Store confusion matrix
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

        print(f"\n[{name} Evaluation]")
        print(f"  - Train Accuracy: {train_acc:.4f}")
        print(f"  - Test Accuracy:  {test_acc:.4f}")
        print(f"  - Precision:      {precision:.4f}")
        print(f"  - Recall:         {recall:.4f}")
        print(f"  - F1 Score:       {f1:.4f}")
        print(f"  - ROC-AUC Score:  {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

    # Create Comparison Table
    comparison_df = pd.DataFrame(results).T.drop(columns=['model_object'])
    print("\n[Model Comparison Summary Table]")
    print(comparison_df.to_string())

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    print("\n✔ Saved: roc_curves.png")

    # Plot Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    print("✔ Saved: confusion_matrices.png")

    # Automated Model Selection (Best F1 Score is primary for medical contexts)
    best_model_name = max(results, key=lambda k: results[k]['F1 Score'])
    best_model_obj = results[best_model_name]['model_object']
    print(f"\n👑 AUTOMATED MODEL SELECTION:")
    print(f"   The selected best-performing model is: **{best_model_name}** (Highest F1 Score = {results[best_model_name]['F1 Score']:.4f})")

    # Plot Feature Importance for the best model
    plot_feature_importance(best_model_obj, best_model_name, feature_names, output_dir)

    return best_model_name, best_model_obj, results

def plot_feature_importance(model, model_name, feature_names, output_dir="visualizations"):
    """Extracts and plots feature importances or coefficients for the best model."""
    plt.figure(figsize=(12, 8))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Feature Importances (Gini Impurity) - {model_name}'
        label = 'Importance Score'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        title = f'Feature Importances (Absolute Coefficients) - {model_name}'
        label = 'Absolute Coefficient Magnitude'
    else:
        print("Model does not support coefficient or feature importance extraction.")
        return

    feat_importances = pd.Series(importances, index=feature_names)
    # Get top 15 features for clarity in graph
    top_15_feats = feat_importances.sort_values(ascending=True).tail(15)
    
    top_15_feats.plot(kind='barh', color=plt.cm.coolwarm(np.linspace(0.8, 0.2, len(top_15_feats))))
    plt.title(title, pad=15)
    plt.xlabel(label)
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    print("✔ Saved: feature_importance.png")

# ==========================================
# PHASE 5: OVERFITTING ANALYSIS
# ==========================================

def print_overfitting_analysis(results, best_model_name):
    """Outputs a detailed, beginner-friendly analysis of overfitting for all models."""
    print_header("PHASE 5: OVERFITTING ANALYSIS & EDUCATIONAL DISCUSSION")
    
    print("What is Overfitting?")
    print("Overfitting occurs when a Machine Learning model learns the 'noise' and specific details")
    print("of the training data too well, such that it performs exceptionally on the training set")
    print("but fails to generalize well to unseen test data. It is like memorizing the answers to")
    print("a specific exam rather than understanding the underlying concepts.")
    print("-" * 60)

    for name, metrics in results.items():
        train_acc = metrics['Train Accuracy'] * 100
        test_acc = metrics['Test Accuracy'] * 100
        diff = train_acc - test_acc
        
        print(f"\n👉 {name}:")
        print(f"   - Training Accuracy: {train_acc:.2f}%")
        print(f"   - Testing Accuracy:  {test_acc:.2f}%")
        print(f"   - Generalization Gap: {diff:.2f}%")
        
        if diff > 5.0:
            print("   🚨 STATUS: POTENTIAL OVERFITTING. The model is memorizing the training data.")
            if name == 'Decision Tree':
                print("      *Reason*: Decision Trees are prone to overfitting because they grow deep,")
                print("      creating specific split rules for minor training details. Pre-pruning")
                print("      (limiting depth) or using an ensemble (Random Forest) helps mitigate this.")
        else:
            print("   ✅ STATUS: GOOD GENERALIZATION. Training and testing scores are close.")

    print("\n[Educational Recommendation]")
    if best_model_name == 'Random Forest':
        print("Random Forest reduces overfitting by averaging multiple decision trees trained on")
        print("different subsets of the data (bagging). This makes it highly robust for breast cancer prediction.")
    elif best_model_name == 'Logistic Regression':
        print("Logistic Regression is a linear model that generalizes very well due to its simplicity,")
        print("especially when L2 regularization (ridge) is automatically applied by scikit-learn.")

# ==========================================
# PHASE 6: INTERACTIVE RISK PREDICTION ENGINE
# ==========================================

def get_risk_meter_str(probability):
    """Generates a colored, interactive visual risk meter based on prediction probability."""
    # Scale probability to 0-10 segments
    segments = int(round(probability * 10))
    meter = "■" * segments + "□" * (10 - segments)
    
    # ANSI color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if probability < 0.35:
        color = GREEN
        risk_level = "🟢 LOW RISK"
    elif probability < 0.70:
        color = YELLOW
        risk_level = "🟡 MODERATE RISK"
    else:
        color = RED
        risk_level = "🔴 HIGH RISK"

    return f"{color}[{meter}] {probability*100:.1f}% - {risk_level}{RESET}"

def explain_local_prediction(patient_vector, scaler, feature_names, best_model_obj, benign_mean_scaled):
    """Calculates and displays the top features contributing to high risk (malignancy)."""
    # Standardize the features if they aren't already scaled
    # We will compute feature deviation from typical benign average
    patient_scaled = patient_vector.flatten()
    
    # Calculate feature contribution: Importance * (patient_value - benign_average)
    # Features with highest positive value are pointing towards malignancy
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, 'coef_'):
        importances = np.abs(best_model_obj.coef_[0])
        # Normalize coefficients to sum to 1 to resemble importance weights
        importances = importances / np.sum(importances)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    deviations = patient_scaled - benign_mean_scaled
    contributions = importances * deviations
    
    # Create DataFrame for analysis
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Patient Scaled Value': patient_scaled,
        'Benign Mean Scaled': benign_mean_scaled,
        'Contribution Score': contributions
    })
    
    # Sort by contribution (positive scores suggest malignancy risk)
    top_risk_contribs = contrib_df.sort_values(by='Contribution Score', ascending=False).head(3)
    
    # We also format feature names to be more readable
    formatted_contribs = []
    for idx, row in top_risk_contribs.iterrows():
        feat_name = row['Feature'].replace('_', ' ').title()
        val_deviation = "Elevated" if row['Patient Scaled Value'] > row['Benign Mean Scaled'] else "Suppressed"
        formatted_contribs.append(f"• {feat_name} ({val_deviation} compared to typical benign baseline)")
        
    return formatted_contribs

def run_interactive_engine(X_test, X_test_scaled, y_test, feature_names, best_model_name, best_model_obj, scaler):
    """Interactive CLI menu for patient predictions and risk calculations."""
    # Precompute Benign means on scaled training data for local predictions
    # This acts as our 'baseline' benign patient representation
    benign_mean_scaled = X_test_scaled[y_test == 0].mean(axis=0)

    # Get a list of actual test indices to select from
    y_test_reset = y_test.reset_index(drop=True)
    malignant_indices = y_test_reset[y_test_reset == 1].index.tolist()
    benign_indices = y_test_reset[y_test_reset == 0].index.tolist()

    while True:
        print_header("ONCOSENSE AI: INTERACTIVE RISK PREDICTION ENGINE")
        print(f"Active Model: {best_model_name}")
        print("Select an option:")
        print("  1. Pick a random patient case from the Test Set")
        print("  2. Select patient case by ID (index 0 - {})".format(len(X_test) - 1))
        print("  3. Manual measurement entry (Enter top indicators)")
        print("  4. Return to main script / Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            idx = np.random.choice(len(X_test))
            patient_data = X_test_scaled[idx].reshape(1, -1)
            true_label = "Malignant" if y_test_reset.iloc[idx] == 1 else "Benign"
            display_prediction(patient_data, true_label, best_model_obj, scaler, feature_names, benign_mean_scaled, idx)
            input("\nPress Enter to return to menu...")
            
        elif choice == '2':
            try:
                idx_str = input(f"Enter patient index (0-{len(X_test)-1}): ").strip()
                idx = int(idx_str)
                if idx < 0 or idx >= len(X_test):
                    print("❌ Invalid index. Must be between 0 and {}.".format(len(X_test)-1))
                    input("Press Enter to continue...")
                    continue
                patient_data = X_test_scaled[idx].reshape(1, -1)
                true_label = "Malignant" if y_test_reset.iloc[idx] == 1 else "Benign"
                display_prediction(patient_data, true_label, best_model_obj, scaler, feature_names, benign_mean_scaled, idx)
                input("\nPress Enter to return to menu...")
            except ValueError:
                print("❌ Invalid input. Please enter a valid integer index.")
                input("Press Enter to continue...")
                
        elif choice == '3':
            run_manual_entry(best_model_obj, scaler, feature_names, benign_mean_scaled)
            input("\nPress Enter to return to menu...")
            
        elif choice == '4':
            print("\nExiting Interactive Engine. Thank you for using OncoSense AI!")
            break
        else:
            print("❌ Invalid selection. Please enter a number between 1 and 4.")
            input("Press Enter to continue...")

def display_prediction(patient_scaled, true_label, model, scaler, feature_names, benign_mean_scaled, test_idx=None):
    """Helper function to format and show the predicted diagnosis details."""
    prob = model.predict_proba(patient_scaled)[0, 1]
    pred_class = model.predict(patient_scaled)[0]
    pred_label = "Malignant" if pred_class == 1 else "Benign"
    
    # Calculate confidence based on predicted class probability
    confidence = prob if pred_class == 1 else (1 - prob)

    print("\n" + "-" * 50)
    print("              ONCOSENSE RISK REPORT             ")
    print("-" * 50)
    if test_idx is not None:
        print(f"Patient Case Ref: Test Index #{test_idx}")
    print(f"Actual Clinical Diagnosis: {true_label}")
    print(f"OncoSense AI Prediction:   {pred_label}")
    print(f"Prediction Confidence:     {confidence * 100:.2f}%")
    
    print("\nVisual Risk Meter:")
    print(get_risk_meter_str(prob))
    
    # Generate Top Contributing Features
    print("\nTop Contributing Risk Features:")
    contribs = explain_local_prediction(patient_scaled, scaler, feature_names, model, benign_mean_scaled)
    for c in contribs:
        print(c)
    print("-" * 50)

def run_manual_entry(model, scaler, feature_names, benign_mean_scaled):
    """Enables manual entry of the top predictive features while defaulting other metrics."""
    print_header("MANUAL MEASUREMENT ENTRY ASSISTANT")
    print("To make manual testing accessible, you will enter measurements for the")
    print("top 5 key features identified as the most critical predictors of malignancy.")
    print("All other features will be filled with standard benchmark values.")
    print("-" * 60)
    
    # Features of interest (mapping user friendly labels to column names)
    key_features = {
        'concave points_worst': ('Worst Concave Points (0.0 to 0.3)', 0.12),
        'perimeter_worst': ('Worst Perimeter (50.0 to 250.0 mm)', 115.0),
        'radius_worst': ('Worst Radius (7.0 to 36.0 mm)', 16.0),
        'area_worst': ('Worst Area (180.0 to 4200.0 mm²)', 880.0),
        'concavity_worst': ('Worst Concavity (0.0 to 1.2)', 0.4)
    }

    # Initialize a baseline data vector (using mean of all test features)
    baseline_patient = np.zeros(len(feature_names))
    
    # Ask user for values
    user_inputs = {}
    for feature_col, (label, default) in key_features.items():
        while True:
            val_str = input(f"Enter {label} [Default={default}]: ").strip()
            if val_str == "":
                val = default
                break
            try:
                val = float(val_str)
                if val < 0:
                    print("❌ Value cannot be negative. Try again.")
                    continue
                break
            except ValueError:
                print("❌ Invalid input. Please enter a valid float number.")
        user_inputs[feature_col] = val

    # Assemble complete patient vector in original scale order
    complete_patient = []
    for col in feature_names:
        if col in user_inputs:
            complete_patient.append(user_inputs[col])
        else:
            # Use benign average as reference baseline for other features to not introduce bias
            # Map index to get original scale
            col_idx = feature_names.index(col)
            complete_patient.append(scaler.mean_[col_idx]) # Using scaler mean (overall dataset average)
            
    complete_patient = np.array(complete_patient).reshape(1, -1)
    
    # Scale patient data using same scaler
    patient_scaled = scaler.transform(complete_patient)
    
    # Display prediction
    display_prediction(patient_scaled, "Unknown (User Created)", model, scaler, feature_names, benign_mean_scaled)

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================

def main():
    # Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    dataset_path = os.path.join(script_dir, "Breast_Cancer.csv")
    
    # Execute Pipeline
    df = load_and_inspect_data(dataset_path)
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = preprocess_data(df)
    
    perform_eda(df, feature_names)
    
    best_model_name, best_model_obj, results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    print_overfitting_analysis(results, best_model_name)
    
    # Check for non-interactive flag for testing purposes
    if len(sys.argv) > 1 and sys.argv[1] == '--non-interactive':
        print("\n[Testing Mode] Running sample prediction to verify Interactive Engine...")
        # Simulating pick a random patient (option 1)
        y_test_reset = y_test.reset_index(drop=True)
        benign_mean_scaled = X_test_scaled[y_test == 0].mean(axis=0)
        idx = 0 # First patient
        patient_data = X_test_scaled[idx].reshape(1, -1)
        true_label = "Malignant" if y_test_reset.iloc[idx] == 1 else "Benign"
        display_prediction(patient_data, true_label, best_model_obj, scaler, feature_names, benign_mean_scaled, idx)
        print("\n✔ Interactive prediction testing successful. Exiting due to --non-interactive flag.")
        return
        
    # Start the interactive UI
    run_interactive_engine(X_test, X_test_scaled, y_test, feature_names, best_model_name, best_model_obj, scaler)

if __name__ == "__main__":
    main()
