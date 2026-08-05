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
#precision is the ratio of true positives to the sum of true positives and false positives
#recall is the ratio of true positives to the sum of true positives and false negatives
#f1 score is the harmonic mean of precision and recall, providing a single metric that balances both
print("Precision Score:", precision_score(y, y_pred))
print("Recall Score:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))

# -----------------------------
# 4. Custom Threshold Evaluation (Decision Function or boundary)
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
print("thresholds:\n", thresholds)

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


