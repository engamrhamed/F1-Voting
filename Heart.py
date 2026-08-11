from tkinter import Y

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
)
from scipy.stats import mode
from tabulate import tabulate

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load and preprocess the dataset
# ---------------------------------------------------------------------

# # Fetch the dataset
# from ucimlrepo import fetch_ucirepo
# heart_disease = fetch_ucirepo(id=45)
# # Extract features and target
# X = heart_disease.data.features
# y = heart_disease.data.targets
# # metadata 
# print(heart_disease.metadata) 
# y.name = "target"
# dset = pd.concat([X, y], axis=1)


dset = pd.read_csv("heart_disease.csv")
X = dset.drop("target", axis=1)
y = dset["target"]
y = y.squeeze()

print("Target variable values:\n", y.value_counts())

if y.nunique() > 2:
    y = y.apply(lambda x: 1 if x > 0 else 0)

print("Missing values in features:\n", X.isnull().sum())
X = X.fillna(X.median())
X = pd.get_dummies(X, drop_first=True)

num_classes = len(np.unique(y))

# Visualize the dataset
from sklearn.decomposition import PCA
# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the features to mean=0, std=1

# Step 2: Apply PCA to reduce the dimensions to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Create a scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=10)
plt.title("PCA of Heart Disease Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Target (0: No Disease, 1: Disease)")  # Color bar for target
plt.show()

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Heart Disease Dataset Feature Importance')
plt.show()

# Calculate the correlation matrix
import seaborn as sns
correlation_matrix = dset.corr()
# Create the heatmap plot
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
# Add title
plt.title('Correlation Matrix Heatmap')
# Show the plot
plt.show()
# Plot the correlation matrix as bars
plt.figure(figsize=(10, 6))

# Iterate through columns and plot correlation with other columns
for i in range(len(correlation_matrix.columns)):
    plt.bar(correlation_matrix.columns, correlation_matrix.iloc[:, i], label=f'Correlation with {correlation_matrix.columns[i]}')
# Add labels and title
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.title('Correlation of Features as Bars')
# Show the plot
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Plot the correlation using a horizontal bar plot
correlation = dset.corr()['target'].drop('target')
plt.figure(figsize=(10, 6))
correlation.sort_values().plot(kind='barh', color='skyblue')
plt.title('Correlation between Heart Disease and Numeric Features')
plt.xlabel('Correlation')
plt.ylabel('Numerical Features')
plt.show()

# ---------------------------------------------------------------------
# 3. Data split: stratified 80/20 train/test, then a further
#    stratified 20% validation split carved out of the training set.
#    -> Overall proportions: 64% train / 16% validation / 20% test.
# ---------------------------------------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y_train_full,
)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", X_test.shape)

# ---------------------------------------------------------------------
# 4. Standardize features (fit on train only, applied to val and test)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 5. Initialize classifiers.
# ---------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------
# 6. Train on X_train; compute per-class F1-scores and accuracy on the
#    VALIDATION split (X_val/y_val).
# ---------------------------------------------------------------------
f1_scores = {}
val_accuracies = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    f1_scores[name] = f1_score(y_val, y_val_pred, average=None, labels=range(num_classes))
    val_accuracies[name] = accuracy_score(y_val, y_val_pred)

print("\nValidation-derived per-class F1-scores (used for voting weights):")
for name, scores in f1_scores.items():
    print(f"  {name}: {np.round(scores, 4)}")

print("\nValidation accuracy for each classifier (used for WV weights):")
for name, acc in val_accuracies.items():
    print(f"  {name}: {acc:.4f}")

# ---------------------------------------------------------------------
# 7. Predict on the held-out TEST set for final evaluation.
# ---------------------------------------------------------------------
predictions = {}
probabilities = {}

for name, clf in classifiers.items():
    predictions[name] = clf.predict(X_test)
    probabilities[name] = clf.predict_proba(X_test)


def evaluate_classifier(y_true, y_pred, y_prob, name, n_classes):
    print(f"\nConfusion Matrix for {name}:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_true, y_pred))

    roc_auc = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr, tpr)
    print(f"\nAUC for each class in {name}:")
    for i in range(n_classes):
        print(f"Class {i}: {roc_auc[i]:.4f}")


for name in classifiers:
    evaluate_classifier(y_test, predictions[name], probabilities[name], name, num_classes)

# ---------------------------------------------------------------------
# 8. Voting strategies -- 
# ---------------------------------------------------------------------

# --- Cumulative Class F1-score-based voting (CCF1V) ---
final_predictions_cumulative_f1 = []
final_probabilities_cumulative_f1 = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    class_f1_scores = np.zeros(num_classes)
    for name in classifiers:
        predicted_class = predictions[name][i]
        class_f1_scores[predicted_class] += f1_scores[name][predicted_class]
    final_predictions_cumulative_f1.append(np.argmax(class_f1_scores))
    final_probabilities_cumulative_f1[i] = class_f1_scores / np.sum(class_f1_scores)

# --- Enhanced Class F1-score-based voting (ECF1V) ---
normalized_f1_scores = {
    name: scores / np.sum(scores) for name, scores in f1_scores.items()
}

final_predictions_f1_enhanced = []
final_probabilities_f1_enhanced = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    class_f1_scores = np.zeros(num_classes)
    for name in classifiers:
        predicted_class = predictions[name][i]
        confidence = probabilities[name][i][predicted_class]
        class_f1_scores[predicted_class] += (
            normalized_f1_scores[name][predicted_class] * confidence
        )
    final_predictions_f1_enhanced.append(np.argmax(class_f1_scores))
    final_probabilities_f1_enhanced[i] = class_f1_scores / np.sum(class_f1_scores)

# --- Highest Class F1-score-based voting (HCF1V) ---
final_predictions_highest_f1 = []
final_probabilities_highest_f1 = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    best_f1_score = -1
    best_prediction = -1
    for name in classifiers:
        predicted_class = predictions[name][i]
        clf_f1 = f1_scores[name][predicted_class]
        if clf_f1 > best_f1_score:
            best_f1_score = clf_f1
            best_prediction = predicted_class
    final_predictions_highest_f1.append(best_prediction)
    final_probabilities_highest_f1[i, best_prediction] = 1

# --- Majority voting (MV) ---
all_predictions = np.array([predictions[name] for name in classifiers])
final_predictions_majority, _ = mode(all_predictions, axis=0)
final_predictions_majority = final_predictions_majority.flatten()
final_probabilities_majority = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    final_probabilities_majority[i, final_predictions_majority[i]] = 1

# --- Soft voting (SV) ---
final_predictions_soft = []
final_probabilities_soft = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    avg_prob = np.zeros(num_classes)
    for name in classifiers:
        avg_prob += probabilities[name][i]
    final_predictions_soft.append(np.argmax(avg_prob))
    final_probabilities_soft[i] = avg_prob / np.sum(avg_prob)

# --- Weighted voting (WV) -- 
final_predictions_weighted = []
final_probabilities_weighted = np.zeros((len(X_test), num_classes))
for i in range(len(X_test)):
    class_weighted_scores = np.zeros(num_classes)
    for name in classifiers:
        predicted_class = predictions[name][i]
        class_weighted_scores[predicted_class] += val_accuracies[name]
    final_predictions_weighted.append(np.argmax(class_weighted_scores))
    final_probabilities_weighted[i] = class_weighted_scores / np.sum(class_weighted_scores)

# ---------------------------------------------------------------------
# 9. Final metrics 
# ---------------------------------------------------------------------
methods = {
    "Enhanced F1-score-based voting (ECF1V)": (final_predictions_f1_enhanced, final_probabilities_f1_enhanced),
    "Cumulative F1-score-based voting (CCF1V)": (final_predictions_cumulative_f1, final_probabilities_cumulative_f1),
    "Highest F1-score-based voting (HCF1V)": (final_predictions_highest_f1, final_probabilities_highest_f1),
    "Majority voting (MV)": (final_predictions_majority, final_probabilities_majority),
    "Soft voting (SV)": (final_predictions_soft, final_probabilities_soft),
    "Weighted voting (WV)": (final_predictions_weighted, final_probabilities_weighted),
}

comparison_rows = []
for method_name, (y_pred, y_prob) in methods.items():
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    auc_roc = roc_auc_score(y_test, y_pred)
    comparison_rows.append({
        "Voting Method": method_name,
        "Accuracy": f"{acc:.4f}",
        "F1 Score": f"{f1w:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "AUC-ROC": f"{auc_roc:.4f}",
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n" + "=" * 70)
print("Results (validation-derived weights, test-set evaluation):")
print(tabulate(comparison_df, headers="keys", tablefmt="pretty"))
print("=" * 70)

comparison_df.to_csv("uhdd_comparison.csv", index=False)
print("\nSaved to: uhdd_comparison.csv")