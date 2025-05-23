# 🛠️ Setup
# pip install pandas numpy scikit-learn fairlearn matplotlib seaborn

# 📦 Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

# 🧹 Step 1: Load and Prepare the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

data = pd.read_csv(url, names=column_names, sep=r",\s*", engine="python")
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object'):
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data.drop("income", axis=1)
y = data["income"]  # already encoded as 0/1 by LabelEncoder

# 🧠 Step 2: Train a Basic Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 📊 Step 3: Fairness Audit
sensitive_features = X_test["sex"]
metrics = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "selection_rate": selection_rate,
}
frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)

print("\n📊 Fairness Metrics by Group:")
print(frame.by_group)

print("\n📏 Demographic Parity Difference:", demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features))
print("📏 Equalized Odds Difference:", equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features))

# 🔧 Step 4: Apply Bias Mitigation (Post-processing)
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

thresh_opt = ThresholdOptimizer(
    estimator=clf,
    constraints="equalized_odds",
    prefit=True,
    predict_method='predict_proba'
)
thresh_opt.fit(X_train, y_train, sensitive_features=X_train["sex"])
y_pred_fair = thresh_opt.predict(X_test, sensitive_features=X_test["sex"])

fair_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred_fair, sensitive_features=sensitive_features)

print("\n✅ After Mitigation — Fairness Metrics by Group:")
print(fair_frame.by_group)

print("\n📏 New Demographic Parity Difference:", demographic_parity_difference(y_test, y_pred_fair, sensitive_features=sensitive_features))
print("📏 New Equalized Odds Difference:", equalized_odds_difference(y_test, y_pred_fair, sensitive_features=sensitive_features))

# 🔍 Step 5: Compare and Reflect
# You can visually or numerically compare the results above to reflect on fairness trade-offs.
