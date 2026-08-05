#ROC curve(Receiver Operating Characteristic curve) is a graphical representation of the performance of a binary classifier system as its discrimination threshold is varied. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The area under the ROC curve (AUC) provides a single measure of overall performance across all thresholds.
# -----------------------------
# 1. Import Data and Libraries
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Extract 'sepal length' (index 0) as the feature x, reshaped to a 2D column vector [-1, 1]
x = iris["data"][:, 0].reshape(-1, 1)
# Create binary targets (y): 1 if the target is Setosa (index 0), 0 otherwise
y = (iris["target"] == 0).astype(np.int_)


# -----------------------------
# 2. Logistic Regression Model
# -----------------------------
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with solver="lbfgs" and C=0.1
log_reg = LogisticRegression(solver="lbfgs", C=0.1, random_state=1)

# Fit the model to the training data
log_reg.fit(x, y)

# Predict class labels using the default threshold (0.5 probability / 0 decision function score)
y_pred = log_reg.predict(x)


# -----------------------------
# 3. Model Evaluation Metrics (Default Threshold)
# -----------------------------
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute precision, recall, and F1 scores
print("Precision Score:", precision_score(y, y_pred))
print("Recall Score:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))


# -----------------------------
# 4. Custom Threshold Evaluation (Decision Function)
# -----------------------------
# Compute the decision function scores (confidence scores: h = b + w1*x1)
h = log_reg.decision_function(x)

# Set a custom threshold (e.g., threshold = 0.1)
threshold = 0.1

# Generate new predictions based on the custom threshold
y_pred_mod = (h > threshold).astype(int)

# Evaluate metrics with the custom threshold
print("Custom Threshold Confusion Matrix:\n", confusion_matrix(y, y_pred_mod))
print("Custom Threshold Precision Score:", precision_score(y, y_pred_mod))
print("Custom Threshold Recall Score:", recall_score(y, y_pred_mod))
print("Custom Threshold F1 Score:", f1_score(y, y_pred_mod))


# -----------------------------
# 5. Precision-Recall Curve Calculation and Plotting
# -----------------------------
from sklearn.metrics import precision_recall_curve

# Calculate precisions, recalls, and thresholds across various decision thresholds
precisions, recalls, thresholds = precision_recall_curve(y, h)

# Plot precision and recall against thresholds (excluding the last precision point to match array dimensions)
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], "bo--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "go-", label="Recall", linewidth=2)
plt.xlabel("Threshold")
plt.legend()
plt.show()

# Note: In newer versions of scikit-learn, 'plot_precision_recall_curve' is deprecated 
# in favor of 'RocCurveDisplay.from_estimator' or 'PrecisionRecallDisplay.from_estimator'.
from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(log_reg, x, y)
plt.show()


# -----------------------------
# 1. ROC Curve Calculation
# -----------------------------
from sklearn.metrics import roc_curve

# Compute False Positive Rate (fpr), True Positive Rate (tpr), and thresholds 
# using true labels (y) and decision function scores (h)
fpr, tpr, thresholds = roc_curve(y, h)

# Inspect the computed metrics
print("False Positive Rate (fpr):", fpr)
print("True Positive Rate (tpr):", tpr)
print("Thresholds:", thresholds)


# -----------------------------
# 2. Plotting the ROC Curve
# -----------------------------
import matplotlib.pyplot as plt

# Plot the ROC curve (fpr vs tpr) with blue markers and lines
plt.plot(fpr, tpr, 'bo-', linewidth=2, label="roc_curve_plot")

# Plot a diagonal dashed reference line representing a random guess (FPR = TPR)
plt.plot([0, 1], [0, 1], 'k--')

# Set axis labels and display the plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Note: In newer versions of scikit-learn, 'plot_roc_curve' is deprecated 
# in favor of 'RocCurveDisplay.from_estimator'.
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(log_reg, x, y)
plt.show()


# -----------------------------
# 3. Area Under the ROC Curve (AUC)
# -----------------------------
from sklearn.metrics import roc_auc_score

# Compute the Area Under the ROC Curve using true labels (y) and decision scores (h)
auc_score = roc_auc_score(y, h)
print("ROC AUC Score:", auc_score)