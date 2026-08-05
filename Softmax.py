# -----------------------------
# 1. Data Preparation (Multi-class Classification)
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Extract the first 2 features (sepal length and sepal width) for a 2D feature space
x = iris["data"][:, :2]

# Use the full target labels (0: Setosa, 1: Versicolor, 2: Virginica) for multi-class classification
y = iris["target"]
print("y",y)

# -----------------------------
# 2. Softmax Regression (Multinomial Logistic Regression) Model
# -----------------------------
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with multi_class='multinomial' for Softmax regression
softmax_reg = LogisticRegression(C=10, random_state=1)

# Fit the model to the training data
softmax_reg.fit(x, y)

# Inspect the intercepts and coefficients (note: there is a set of parameters for each class)
print("Intercepts:", softmax_reg.intercept_)
print("Coefficients:\n", softmax_reg.coef_)

# Evaluate model accuracy score on the dataset
print("Score:", softmax_reg.score(x, y))

# Define new multi-feature data points to predict
x_new = [[6.7, 4.25], [4.9, 3.3], [5.2, 2.5]]

# Predict class labels for new data
print("Predictions (predict):", softmax_reg.predict(x_new))

# Predict class probabilities across all classes for new data
print("Prediction Probabilities (predict_proba):\n", softmax_reg.predict_proba(x_new))


# -----------------------------
# 3. SGDClassifier Model (Linear Classifier with Gradient Descent)
# -----------------------------
from sklearn.linear_model import SGDClassifier

# Initialize SGDClassifier with loss='log' (or 'log_loss'), penalty='l2', and alpha=0.0001
sgd_clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, random_state=1)

# Fit the SGD classifier to the data
sgd_clf.fit(x, y)

# Inspect the intercepts and coefficients of the SGD classifier
print("SGD Intercepts:", sgd_clf.intercept_)
print("SGD Coefficients:\n", sgd_clf.coef_)

# Evaluate accuracy score of the SGD classifier
print("SGD Score:", sgd_clf.score(x, y))

# Predict class labels for new data points
print("SGD Predictions (predict):", sgd_clf.predict(x_new))

# Predict class probabilities for new data points
print("SGD Prediction Probabilities (predict_proba):\n", sgd_clf.predict_proba(x_new))


# -----------------------------
# 4. Plotting Multi-class Decision Boundaries (Softmax Regression)
# -----------------------------
# Generate grid points spanning the feature spaces to create background prediction regions
x1, x2 = np.meshgrid(
    np.linspace(4.2, 8, 1000).reshape(-1, 1),
    np.linspace(1.8, 4.5, 1000).reshape(-1, 1)
)
x_new_grid = np.c_[x1.ravel(), x2.ravel()]
y_pred = softmax_reg.predict(x_new_grid)

# Set up the figure size
plt.figure(figsize=(10, 4))

# Plot actual data points for each class (0: Setosa, 1: Versicolor, 2: Virginica)
plt.plot(x[y==0, 0], x[y==0, 1], "bo", label="Setosa")
plt.plot(x[y==1, 0], x[y==1, 1], "rx", label="Versicolor")
plt.plot(x[y==2, 0], x[y==2, 1], "g^", label="Virginica")

# Reshape predictions and plot background contour zones for multi-class classification
zz = y_pred.reshape(x1.shape)
plt.contourf(x1, x2, zz, alpha=0.2)

# Set axis labels, limits, and customized legend positioning
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14)
plt.legend(loc="upper left", bbox_to_anchor=(0.69, 1), fontsize=12)
plt.axis([4.2, 8, 1.8, 4.5])

# Show the final plot
plt.show()