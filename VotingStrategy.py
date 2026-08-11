
from tkinter import Y

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import cycle
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
)
from scipy.stats import mode
from tabulate import tabulate

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Generate Gaussian Mixture dataset with 4 nonlinear clusters
# ---------------------------------------------------------------------

X, y = make_blobs(n_samples=3000, centers=4, cluster_std=3.5, random_state=42)
num_classes = len(np.unique(y))

# Plot the dataset
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=10)
plt.title("Gaussian Mixture Dataset (Nonlinear)")
plt.show()


# ---------------------------------------------------------------------
# 2. Data split: stratified 80/20 train/test, then a further
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
# 3. Initialize classifiers.
# ---------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),

    # "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    # "KNN": KNeighborsClassifier(),
    # "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=3),
}

# ---------------------------------------------------------------------
# 4. Train on X_train; compute per-class F1-scores and accuracy on the
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
# 5. Predict on the held-out TEST set for final evaluation.
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
# 6. Voting strategies -- 
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
# 7. Confusion Matrix, Classification Report, and ROC-AUC for all voting methods
# ---------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, y_prob, method_name, n_classes):
    print(f"\nConfusion Matrix for {method_name}:")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"\nClassification Report for {method_name}:")
    print(classification_report(y_true, y_pred))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {method_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Print AUC for each class
    print(f"\nAUC for each class in {method_name}:")
    for i in range(n_classes):
        print(f"Class {i}: {roc_auc[i]:.4f}")

# Evaluate voting methods
evaluate_predictions(y_test, final_predictions_f1_enhanced, final_probabilities_f1_enhanced, "Enhanced Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_cumulative_f1, final_probabilities_cumulative_f1, "Cumulative Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_highest_f1, final_probabilities_highest_f1, "Highest Class F1-score-based Voting", num_classes)
evaluate_predictions(y_test, final_predictions_majority, final_probabilities_majority, "Majority voting", num_classes)
evaluate_predictions(y_test, final_predictions_soft, final_probabilities_soft, "Soft voting", num_classes)
evaluate_predictions(y_test, final_predictions_weighted, final_probabilities_weighted, "Weighted voting", num_classes)

# ---------------------------------------------------------------------
# 8. Final metrics 
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
    auc_roc = roc_auc_score(y_test,  y_prob, multi_class="ovr", average="weighted")
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

comparison_df.to_csv("GM_comparison.csv", index=False)
print("\nSaved to: GM_comparison.csv")




# ---------------------------------------------------------------------
# 1. Generate a spiral dataset with four moons dataset (nonlinear with 4 classes)
# ---------------------------------------------------------------------

from sklearn.datasets import make_moons
def generate_four_moons(n_samples_per_class, noise=0.3):
    # Generate two sets of moons and shift them to create 4 classes
    X1, y1 = make_moons(n_samples=n_samples_per_class * 2, noise=noise, random_state=42)
    X2, y2 = make_moons(n_samples=n_samples_per_class * 2, noise=noise, random_state=43)
    
    # Shift the second set of moons to create distinct clusters
    X2[:, 0] += 2.5  # Shift x-coordinate
    X2[:, 1] += 2.5  # Shift y-coordinate
    
    # Assign new class labels
    y1 = y1 * 2      # Classes 0 and 2
    y2 = y2 * 2 + 1  # Classes 1 and 3
    
    # Combine the datasets
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    
    return X, y

# Generate four moons dataset
n_samples_per_class = 500  # 500 samples per class, total 2000
X, y = generate_four_moons(n_samples_per_class, noise=0.1)

# Visualize the dataset
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=10)
plt.title("Four Moons Dataset (Nonlinear)")
plt.show()


# ---------------------------------------------------------------------
# 2. Data split: stratified 80/20 train/test, then a further
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
# 3. Initialize classifiers.
# ---------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),

    # "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    # "KNN": KNeighborsClassifier(),
    # "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=3),
}

# ---------------------------------------------------------------------
# 4. Train on X_train; compute per-class F1-scores and accuracy on the
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
# 5. Predict on the held-out TEST set for final evaluation.
# ---------------------------------------------------------------------
predictions = {}
probabilities = {}

for name, clf in classifiers.items():
    predictions[name] = clf.predict(X_test)
    probabilities[name] = clf.predict_proba(X_test)

for name in classifiers:
    evaluate_classifier(y_test, predictions[name], probabilities[name], name, num_classes)

# ---------------------------------------------------------------------
# 6. Voting strategies -- 
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
# 7. Confusion Matrix, Classification Report, and ROC-AUC for all voting methods
# ---------------------------------------------------------------------

# Evaluate voting methods
evaluate_predictions(y_test, final_predictions_f1_enhanced, final_probabilities_f1_enhanced, "Enhanced Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_cumulative_f1, final_probabilities_cumulative_f1, "Cumulative Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_highest_f1, final_probabilities_highest_f1, "Highest Class F1-score-based Voting", num_classes)
evaluate_predictions(y_test, final_predictions_majority, final_probabilities_majority, "Majority voting", num_classes)
evaluate_predictions(y_test, final_predictions_soft, final_probabilities_soft, "Soft voting", num_classes)
evaluate_predictions(y_test, final_predictions_weighted, final_probabilities_weighted, "Weighted voting", num_classes)

# ---------------------------------------------------------------------
# 8. Final metrics 
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
    auc_roc = roc_auc_score(y_test,  y_prob, multi_class="ovr", average="weighted")
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

comparison_df.to_csv("Spiral_comparison.csv", index=False)
print("\nSaved to: Spiral_comparison.csv")


# ---------------------------------------------------------------------
# 1. Generate a dataset with 4 classes, but different levels of difficulty
# ---------------------------------------------------------------------

centers = [(0, 0), (5, 5), (3, -3), (8, 1)]  # Class centroids
cluster_std = [0.5, 1.5, 2.5, 0.8]  # Different spread per class

X, y = make_blobs(n_samples=[1000, 500, 300, 1200], centers=centers, cluster_std=cluster_std, random_state=42)
# Visualize the dataset
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=10)
plt.title("Complex Dataset (Nonlinear)")
plt.show()


# ---------------------------------------------------------------------
# 2. Data split: stratified 80/20 train/test, then a further
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
# 3. Initialize classifiers.
# ---------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),

    # "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    # "KNN": KNeighborsClassifier(),
    # "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=3),
}

# ---------------------------------------------------------------------
# 4. Train on X_train; compute per-class F1-scores and accuracy on the
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
# 5. Predict on the held-out TEST set for final evaluation.
# ---------------------------------------------------------------------
predictions = {}
probabilities = {}

for name, clf in classifiers.items():
    predictions[name] = clf.predict(X_test)
    probabilities[name] = clf.predict_proba(X_test)

for name in classifiers:
    evaluate_classifier(y_test, predictions[name], probabilities[name], name, num_classes)

# ---------------------------------------------------------------------
# 6. Voting strategies -- 
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
# 7. Confusion Matrix, Classification Report, and ROC-AUC for all voting methods
# ---------------------------------------------------------------------

# Evaluate voting methods
evaluate_predictions(y_test, final_predictions_f1_enhanced, final_probabilities_f1_enhanced, "Enhanced Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_cumulative_f1, final_probabilities_cumulative_f1, "Cumulative Class F1-score-based voting", num_classes)
evaluate_predictions(y_test, final_predictions_highest_f1, final_probabilities_highest_f1, "Highest Class F1-score-based Voting", num_classes)
evaluate_predictions(y_test, final_predictions_majority, final_probabilities_majority, "Majority voting", num_classes)
evaluate_predictions(y_test, final_predictions_soft, final_probabilities_soft, "Soft voting", num_classes)
evaluate_predictions(y_test, final_predictions_weighted, final_probabilities_weighted, "Weighted voting", num_classes)

# ---------------------------------------------------------------------
# 8. Final metrics 
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
    auc_roc = roc_auc_score(y_test,  y_prob, multi_class="ovr", average="weighted")
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

comparison_df.to_csv("Complex_comparison.csv", index=False)
print("\nSaved to: Complex_comparison.csv")