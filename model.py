import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("placement.csv")

print("\n========== DATASET PREVIEW ==========\n")
print(df.head())

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

# ==========================================
# REMOVE NULL VALUES
# ==========================================

df.dropna(inplace=True)

# ==========================================
# LABEL ENCODING
# ==========================================

le = LabelEncoder()

# Convert text columns into numeric values

df['PlacementStatus'] = le.fit_transform(
    df['PlacementStatus']
)

df['ExtracurricularActivities'] = le.fit_transform(
    df['ExtracurricularActivities']
)

df['PlacementTraining'] = le.fit_transform(
    df['PlacementTraining']
)

# ==========================================
# FEATURES
# ==========================================

X = df[[
    'CGPA',
    'Internships',
    'Projects',
    'Workshops/Certifications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'ExtracurricularActivities',
    'PlacementTraining',
    'SSC_Marks',
    'HSC_Marks'
]]

# ==========================================
# TARGET
# ==========================================

y = df['PlacementStatus']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODELS
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": SVC(
        probability=True
    )
}

# ==========================================
# STORE RESULTS
# ==========================================

results = []

# ==========================================
# TRAIN ALL MODELS
# ==========================================

for name, model in models.items():

    # Train Model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability
    y_prob = model.predict_proba(X_test)[:,1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    # Store Results
    results.append([
        name,
        accuracy,
        f1,
        precision,
        recall,
        auc
    ])

# ==========================================
# CREATE MODEL COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        'Model',
        'Accuracy',
        'F1',
        'Precision',
        'Recall',
        'AUC'
    ]
)

# Add Rank Column

results_df['Rank'] = results_df['Accuracy'].rank(
    ascending=False,
    method='dense'
).astype(int)

# Arrange Columns

results_df = results_df[[
    'Rank',
    'Model',
    'Accuracy',
    'F1',
    'Precision',
    'Recall',
    'AUC'
]]

# Sort by Accuracy

results_df = results_df.sort_values(
    by='Accuracy',
    ascending=False
)

# Round Values

results_df = results_df.round(3)

# ==========================================
# SAVE MODEL COMPARISON CSV
# ==========================================

results_df.to_csv(
    "model_comparison_report.csv",
    index=False
)

print("\n========== MODEL COMPARISON REPORT ==========\n")

print(results_df)

print("\nmodel_comparison_report.csv created successfully")

# ==========================================
# BEST MODEL
# ==========================================

best_model = RandomForestClassifier(
    random_state=42
)

best_model.fit(X_train, y_train)

# ==========================================
# SAVE MODEL
# ==========================================

pickle.dump(
    best_model,
    open("model.pkl", "wb")
)

print("\nmodel.pkl created successfully")

# ==========================================
# CONFUSION MATRIX
# ==========================================

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred_best
)

# ==========================================
# CREATE CONFUSION MATRIX TABLE
# ==========================================

cm_df = pd.DataFrame({

    'Model': ['Random Forest', '', ''],

    ' ': [
        'True C0',
        'True C1',
        ''
    ],

    'Pred C0': [
        cm[0][0],
        cm[1][0],
        ''
    ],

    'Pred C1': [
        cm[0][1],
        cm[1][1],
        ''
    ]
})

# ==========================================
# SAVE CONFUSION MATRIX CSV
# ==========================================

cm_df.to_csv(
    "confusion_matrix.csv",
    index=False
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm_df)

print("\nconfusion_matrix.csv created successfully")

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n========== ALL FILES CREATED ==========\n")

print("1. model.pkl")
print("2. model_comparison_report.csv")
print("3. confusion_matrix.csv")