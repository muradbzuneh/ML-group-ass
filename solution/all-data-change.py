"""
WOLLO UNIVERSITY - FUNDAMENTALS OF MACHINE LEARNING (CHAPTER 3)
Section 3.12 Classwork: Practical Engineering Case Study
Title: Pump Condition Classification (Normal, Warning, Fault)

Evaluated Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian NB)
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
    classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. DATASET: 30 INDUSTRIAL PUMP RECORDS (3 CLASSES)
# ============================================================
# Features: Vibration (mm/s), RPM, Temperature (°C), Pressure (bar), Operating Hours
# Target: 0 = Normal, 1 = Warning, 2 = Fault

data = [
    # ID, Vibration_mm_s, Speed_RPM, Temperature_C, Pressure_bar, Operating_Hours, Status
    # Class 0: Normal (10 records)
    [1,  1.4, 1490, 52, 4.2,  950, 0],
    [2,  1.6, 1500, 55, 4.1, 1200, 0],
    [3,  1.3, 1485, 50, 4.3,  800, 0],
    [4,  1.8, 1510, 58, 4.0, 1500, 0],
    [5,  1.5, 1495, 54, 4.2, 1100, 0],
    [6,  1.7, 1475, 56, 4.1, 1350, 0],
    [7,  1.2, 1505, 49, 4.4,  700, 0],
    [8,  1.9, 1480, 59, 4.0, 1600, 0],
    [9,  1.4, 1515, 53, 4.3, 1050, 0],
    [10, 1.6, 1490, 55, 4.2, 1250, 0],

    # Class 1: Warning (10 records)
    [11, 3.4, 1440, 68, 3.6, 3200, 1],
    [12, 3.8, 1430, 71, 3.5, 3600, 1],
    [13, 3.2, 1450, 66, 3.7, 3000, 1],
    [14, 4.1, 1420, 73, 3.4, 3900, 1],
    [15, 3.6, 1435, 69, 3.6, 3400, 1],
    [16, 3.5, 1445, 67, 3.7, 3300, 1],
    [17, 4.0, 1425, 72, 3.4, 3800, 1],
    [18, 3.7, 1430, 70, 3.5, 3500, 1],
    [19, 4.2, 1415, 74, 3.3, 4100, 1],
    [20, 3.5, 1440, 69, 3.6, 3350, 1],

    # Class 2: Fault (10 records)
    [21, 6.8, 1340, 88, 2.7, 6500, 2],
    [22, 7.5, 1310, 92, 2.4, 7200, 2],
    [23, 6.2, 1360, 85, 2.9, 6100, 2],
    [24, 8.2, 1280, 97, 2.1, 7900, 2],
    [25, 7.1, 1330, 90, 2.5, 6800, 2],
    [26, 7.4, 1320, 91, 2.5, 7000, 2],
    [27, 8.0, 1290, 95, 2.2, 7600, 2],
    [28, 6.5, 1350, 87, 2.8, 6300, 2],
    [29, 8.6, 1260, 99, 1.9, 8200, 2],
    [30, 7.2, 1325, 89, 2.6, 6900, 2],
]

columns = [
    "Pump_ID", "Vibration_mm_s", "Speed_RPM",
    "Temperature_C", "Pressure_bar", "Operating_Hours", "Status"
]
df = pd.DataFrame(data, columns=columns)

class_names = ["Normal", "Warning", "Fault"]
feature_cols = ["Vibration_mm_s", "Speed_RPM", "Temperature_C", "Pressure_bar", "Operating_Hours"]

X = df[feature_cols]
y = df["Status"]

# ============================================================
# PART 2. STRATIFIED TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ============================================================
# PART 3. DEFINE THE 5 REQUIRED CLASSIFIERS
# ============================================================
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr"))
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
}

# ============================================================
# PART 4. TRAIN & EVALUATE USING SECTION 3.12 PRIORITIES
# ============================================================
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Class 2 is 'Fault'
    fault_recall = recall_score(y_test, y_pred, labels=[2], average=None, zero_division=0)[0]
    fault_precision = precision_score(y_test, y_pred, labels=[2], average=None, zero_division=0)[0]

    results.append({
        "Algorithm": model_name,
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "Recall (Fault)": fault_recall,
        "Precision (Fault)": fault_precision
    })

    # Confusion Matrix display
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. EVALUATION PRIORITY COMPARISON TABLE
# ============================================================
eval_table = pd.DataFrame(results).set_index("Algorithm")
print("\n" + "=" * 70)
print("SECTION 3.12 EVALUATION PRIORITIES REPORT")
print("=" * 70)
print(eval_table.round(3))

# ============================================================
# PART 6. UNSEEN SAMPLE INFERENCE WITH OPERATIONAL CAUTION
# ============================================================
new_observation = pd.DataFrame([{
    "Vibration_mm_s": 6.9,
    "Speed_RPM": 1335,
    "Temperature_C": 88,
    "Pressure_bar": 2.6,
    "Operating_Hours": 6400
}])

print("\n" + "=" * 70)
print("TESTING UNSEEN PUMP SENSOR OBSERVATION")
print("=" * 70)
print(new_observation)

print("\nPredicted Condition:")
for name, model in models.items():
    pred_idx = int(model.predict(new_observation)[0])
    print(f"{name:15s} -> {class_names[pred_idx]}")

print("\n" + ">" * 5 + " OPERATIONAL CAUTION: " + "<" * 5)
print("Predictions should support — not replace — engineering inspection and safety procedures.")