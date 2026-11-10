"""
Practical Engineering Case Study: Supervised Machine Learning Classification
Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 NEW RECORDS, KEEPING 0 AND 1 LABELS)
# ============================================================

data = [
    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
    # 15 Normal Records (Class 0)
    [1,  46, 1.1, 1492,  850, 0],
    [2,  49, 1.3, 1485, 1150, 0],
    [3,  52, 1.6, 1490, 1300, 0],
    [4,  57, 1.9, 1470, 1750, 0],
    [5,  59, 2.1, 1502, 1950, 0],
    [6,  47, 1.2, 1515,  950, 0],
    [7,  63, 2.5, 1445, 2400, 0],
    [8,  54, 1.7, 1488, 1550, 0],
    [9,  65, 2.7, 1438, 2600, 0],
    [10, 50, 1.4, 1500, 1050, 0],
    [11, 62, 2.3, 1455, 2250, 0],
    [12, 55, 1.8, 1478, 1650, 0],
    [13, 60, 2.2, 1462, 2050, 0],
    [14, 51, 1.5, 1508, 1250, 0],
    [15, 58, 2.0, 1468, 1850, 0],

    # 15 Failure Risk Records (Class 1)
    [16, 79, 5.4, 1385, 4900, 1],
    [17, 86, 6.3, 1345, 5900, 1],
    [18, 92, 7.3, 1315, 6900, 1],
    [19, 75, 5.0, 1395, 4600, 1],
    [20, 96, 7.9, 1285, 7500, 1],
    [21, 83, 5.9, 1365, 5600, 1],
    [22, 101, 8.6, 1250, 8300, 1],
    [23, 77, 5.1, 1390, 4700, 1],
    [24, 89, 6.9, 1325, 6700, 1],
    [25, 94, 7.6, 1295, 7300, 1],
    [26, 104, 9.0, 1230, 8800, 1],
    [27, 81, 5.6, 1375, 5300, 1],
    [28, 90, 7.2, 1305, 7100, 1],
    [29, 87, 6.4, 1335, 6200, 1],
    [30, 98, 8.3, 1265, 8000, 1],
]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 MACHINE RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

# Separate predictors (X) and target (y)
feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", s=100
)
plt.title("Machine temperature vs. vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================
results = []
predictions = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure risk"],
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================
results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================
tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================
# New observation with high temperature, vibration, and hours
new_machine = pd.DataFrame([{
    "Temperature_C": 88,
    "Vibration_mm_s": 6.6,
    "Speed_RPM": 1330,
    "Operating_Hours": 6300
}])

print("\nNEW MACHINE OBSERVATION")
print(new_machine)

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])
    label = "Failure risk" if predicted_class == 1 else "Normal"
    print(f"{name:15s} prediction: {label}")

print("\nEND OF PRACTICAL CASE STUDY")
