# -----------------------------
# Loading the Iris Dataset
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the built-in Iris dataset from scikit-learn
iris = datasets.load_iris()

# Extract feature matrix (x) and target labels (y)
x = iris['data']
y = iris['target']

# Inspect dataset shapes and classes
print("Feature shape (x):", x.shape)
print("Target shape (y):", y.shape)
print("Target classes:", iris.target_names)

# -----------------------------
# 1. Multiclass Classification using OneVsRestClassifier
# -----------------------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize a base logistic regression model and wrap it in OneVsRestClassifier
log_reg = LogisticRegression(random_state=1)
ovr_clf = OneVsRestClassifier(log_reg)

# Fit the OvR classifier to the Iris dataset (x, y)
ovr_clf.fit(x, y)

# Evaluate the accuracy score
print("OneVsRestClassifier Score:", ovr_clf.score(x, y))


# -----------------------------
# 2. Making Predictions and Generating a Confusion Matrix
# -----------------------------
from sklearn.metrics import confusion_matrix

# Predict target classes across the entire feature matrix
y_pred = ovr_clf.predict(x)
print("Predictions:\n", y_pred)

# Compute and display the confusion matrix comparing true labels (y) to predictions (y_pred)
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# -----------------------------
# 3. Native Multiclass Logistic Regression (multi_class='ovr')
# -----------------------------
# Alternatively, LogisticRegression can natively perform OvR classification via the multi_class parameter
ovr_log = OneVsRestClassifier(LogisticRegression(random_state=1))
ovr_log.fit(x, y)

# Evaluate accuracy score
print("LogisticRegression (OVR) Score:", ovr_log.score(x, y))