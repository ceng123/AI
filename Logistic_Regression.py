# -----------------------------
# 1. Import Data and Libraries
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Keys: ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
print("Dataset Keys:", iris.keys())

# Inspect the first 5 rows of feature data
print("First 5 data rows:\n", iris['data'][:5])

# Inspect feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print("Feature Names:", iris["feature_names"])


# -----------------------------
# 2. Prepare Binary Classification Data (Ex1)
# -----------------------------
# Extract 'sepal length' (index 0) as the feature x, reshaped to a 2D column vector [-1, 1]
x = iris["data"][:, 0].reshape(-1, 1)

# Create binary targets (y): 1 if the target is Setosa (index 0), 0 otherwise
y = (iris["target"] == 0).astype(np.int_)

# Inspect the first 3 samples of x and the complete target array y
print("x[:3]:\t", x[:3])
print("y:", y)


# -----------------------------
# 3. Logistic Regression Model
# -----------------------------
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with solver='lbfgs' and C=1 (Note: C = 1/alpha, larger C means less regularization)
log_reg = LogisticRegression(solver="lbfgs", C=1, random_state=1)

# Fit the model to the training data
log_reg.fit(x, y)

# Inspect the model's intercept and coefficients
print("Logistic Regression Intercept:", log_reg.intercept_)
print("Logistic Regression Coefficients:", log_reg.coef_)

# Evaluate model accuracy score on the dataset
print("Logistic Regression Score:", log_reg.score(x, y))

# Define new data points to predict
x_new = [[5], [6.5]]

# Predict class labels for new data
print("Predictions (predict):", log_reg.predict(x_new))

# Predict class probabilities for new data
print("Prediction Probabilities (predict_proba):\n", log_reg.predict_proba(x_new))


# -----------------------------
# 4. SGDClassifier Model
# -----------------------------
from sklearn.linear_model import SGDClassifier

# Initialize SGDClassifier with loss='log' (or 'log_loss'), penalty='l2', and alpha=0.0001 (Default for SVM is loss='hinge')
sgd_clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, random_state=1)

# Fit the SGD classifier to the data
sgd_clf.fit(x, y)

# Inspect the intercept and coefficients of the SGD classifier
print("SGDClassifier Intercept:", sgd_clf.intercept_)
print("SGDClassifier Coefficients:", sgd_clf.coef_)

# Evaluate accuracy score of the SGD classifier
print("SGDClassifier Score:", sgd_clf.score(x, y))

# Predict class labels for new data points
print("SGD Predictions (predict):", sgd_clf.predict(x_new))

# Predict class probabilities for new data points
print("SGD Prediction Probabilities (predict_proba):\n", sgd_clf.predict_proba(x_new))


# -----------------------------
# 1. Calculate Decision Boundary
# -----------------------------
# The decision boundary for a 1D logistic regression model occurs where h = b + w1*x1 = 0.
# Rearranging the equation yields x1 = -b / w1.
# Here, log_reg.intercept_ corresponds to b, and log_reg.coef_ corresponds to w1.

decision_boundary = - log_reg.intercept_ / log_reg.coef_
print("Decision Boundary Value:", decision_boundary)


# -----------------------------
# 2. Plotting the Decision Boundary and Probabilities
# -----------------------------
import matplotlib.pyplot as plt

# Generate 100 evenly spaced points from 3 to 9 for continuous curve plotting
xs = np.linspace(3, 9, 100).reshape(-1, 1)

# Predict the probabilities for the generated points (column 0: probability of class 0, column 1: probability of class 1)
y_proba = log_reg.predict_proba(xs)

# Set up the figure size
plt.figure(figsize=(8, 3))

# Plot actual data points: Setosa (y == 1) as blue dots, and Not Setosa (y == 0) as red crosses
plt.plot(x[y == 1], y[y == 1], "b.", label="Setosa")
plt.plot(x[y == 0], y[y == 0], "rx", label="Not Setosa")

# Plot the predicted probability curves for both classes
plt.plot(xs, y_proba[:, 0], "r:", label="Probability Class 0")
plt.plot(xs, y_proba[:, 1], "b:", label="Probability Class 1")

# Plot a vertical dashed black line ("k--") at the calculated decision boundary value
plt.plot([decision_boundary[0], decision_boundary[0]], [0, 1], "k--", label='Boundary')

# Set axis labels and formatting
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$\hat p$", fontsize=14)

# Display the legend on the right side
plt.legend(loc="center right", fontsize=14)

# Show the plot
plt.show()