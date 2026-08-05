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
# 1. Multiclass Classification using OneVsOneClassifier (OvO)
# -----------------------------
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

# Initialize a base Support Vector Classifier and wrap it in OneVsOneClassifier
svc_clf = SVC(random_state=1)
ovo_clf = OneVsOneClassifier(svc_clf)

# Fit the OvO classifier to the Iris dataset (x, y)
ovo_clf.fit(x, y)

# Evaluate the accuracy score
print("OneVsOneClassifier Score:", ovo_clf.score(x, y))


# -----------------------------
# 2. Making Predictions and Generating a Confusion Matrix (OvO wrapper)
# -----------------------------
from sklearn.metrics import confusion_matrix

# Predict target classes across the entire feature matrix
y_pred = ovo_clf.predict(x)
print("OvO Wrapper Predictions:\n", y_pred)

# Compute and display the confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("OvO Wrapper Confusion Matrix:\n", conf_matrix)


# -----------------------------
# 3. Native Multiclass SVC (decision_function_shape='ovo')
# -----------------------------
# The SVC class in scikit-learn natively supports OvO classification internally.
ovo_svc = SVC(decision_function_shape='ovo', random_state=1)

# Fit the native OvO SVC model
ovo_svc.fit(x, y)

# Evaluate the accuracy score
print("Native SVC (OvO) Score:", ovo_svc.score(x, y))


# -----------------------------
# 4. Making Predictions and Generating a Confusion Matrix (Native SVC)
# -----------------------------
# Predict target classes using the native SVC model
y_pred_native = ovo_svc.predict(x)
print("Native SVC Predictions:\n", y_pred_native)

# Compute and display the confusion matrix
conf_matrix_native = confusion_matrix(y, y_pred_native)
print("Native SVC Confusion Matrix:\n", conf_matrix_native)