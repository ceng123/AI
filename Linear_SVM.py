# -----------------------------
# 1. Import Libraries and Load Data
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Extract features at index 2 and 3 (petal length and petal width) as matrix x
x = iris["data"][:, (2, 3)]

# Create binary targets (y): 1.0 if the target is Virginica (index 2), 0.0 otherwise
y = (iris["target"] == 2).astype(np.float64)

# Inspect the shapes and first few entries
# print("Feature shape:", x.shape)
# print("Target shape:", y.shape)
# print("First 3 rows of features:\n", x[:3])




# -----------------------------
# LinearSVC Classifier Pipeline
# -----------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Construct a Pipeline that performs:
# 1. Feature standardization (mean=0, variance=1)
# 2. Linear Support Vector Classification with hinge loss and C=1
svm_clf = Pipeline([
    ('scal', StandardScaler()),
    ('lin_svc', LinearSVC(C=1, loss='hinge', random_state=1))
])

# Fit the pipeline to the feature data (x) and binary targets (y)
svm_clf.fit(x, y)

# Inspect the intercept and coefficients of the underlying LinearSVC model
print("Intercept:", svm_clf['lin_svc'].intercept_)
print("Coefficients:", svm_clf['lin_svc'].coef_)

# Evaluate accuracy score on the dataset
print("Score:", svm_clf.score(x, y))

# Make a prediction for a new data point
x_new = [[5.5, 1.7]]
print("Prediction:", svm_clf.predict(x_new))


def plot_predictions(model, x, y):
    # 1. Create a dense 2D grid spanning the feature space of x
    x1s = np.linspace(x.min(axis=0)[0] - 0.05, x.max(axis=0)[0] + 0.05, 1000)
    x2s = np.linspace(x.min(axis=0)[1] - 0.05, x.max(axis=0)[1] + 0.05, 1000)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]

    # 2. Predict class labels across the grid and plot filled contours for background regions
    y_pred = model.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, cmap=plt.cm.brg, alpha=0.2)

    # 3. Compute decision function scores and plot boundary lines at levels [-1, 0, 1] (ideal for SVM margins)
    y_decision = model.decision_function(X_new).reshape(x1.shape)
    plt.contour(x1, x2, y_decision, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'], alpha=0.8)

    # 4. Plot actual training data points for both classes (y=0 and y=1)
    plt.plot(x[y == 0, 0], x[y == 0, 1], 'bo', markersize=3, label='$y=0$')
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'r^', markersize=3, label='$y=1$')

    # 5. Format labels, legend, and axes
    plt.legend(loc='center left')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)

plt.figure(figsize=(5, 3))

# Call the custom plot_predictions function using our trained SVM classifier, features, and targets
plot_predictions(svm_clf, x, y)

# Render and display the plot
plt.show()



#method 2: Using SVC with hinge loss


# -----------------------------
# Support Vector Classification using SVC(kernel='linear')
# -----------------------------
from sklearn.svm import SVC

# Construct a Pipeline combining standard scaling and an SVC model with a linear kernel
svc_clf = Pipeline([
    ('scal', StandardScaler()),
    ('lin_svc', SVC(kernel='linear', C=1, random_state=1))
])

# Fit the pipeline to the feature data (x) and binary targets (y)
svc_clf.fit(x, y)

# Inspect the intercept and coefficients of the fitted linear SVC model
print("Intercept:", svc_clf['lin_svc'].intercept_)
print("Coefficients:", svc_clf['lin_svc'].coef_)

# Evaluate accuracy score on the dataset
print("Score:", svc_clf.score(x, y))

# Make a prediction for a new data point
x_new = [[5.5, 1.7]]
print("Prediction:", svc_clf.predict(x_new))

# Set the figure size to 5 inches wide by 3 inches high
plt.figure(figsize=(5, 3))

# Call the custom plot_predictions function using our trained SVC classifier, features, and targets
plot_predictions(svc_clf, x, y)

# Render and display the plot
plt.show()

#method 3: Using SGDClassifier with hinge loss

# -----------------------------
# SGDClassifier with Hinge Loss Pipeline
# -----------------------------
from sklearn.linear_model import SGDClassifier

# Construct a Pipeline combining standard scaling and an SGDClassifier with hinge loss
sgd_clf = Pipeline([
    ('scal', StandardScaler()),
    ('lin_svc', SGDClassifier(loss='hinge', alpha=1 / (1 * len(y)), random_state=1))
])

# Fit the pipeline to the feature data (x) and binary targets (y)
sgd_clf.fit(x, y)

# Inspect the intercept and coefficients of the underlying SGD classifier model
print("Intercept:", sgd_clf['lin_svc'].intercept_)
print("Coefficients:", sgd_clf['lin_svc'].coef_)

# Evaluate accuracy score on the dataset
print("Score:", sgd_clf.score(x, y))

# Make a prediction for a new data point
x_new = [[5.5, 1.7]]
print("Prediction:", sgd_clf.predict(x_new))


# -----------------------------
# Visualizing the SGDClassifier Predictions and Margins
# -----------------------------
import matplotlib.pyplot as plt

# Set the figure size to 5 inches wide by 3 inches high
plt.figure(figsize=(5, 3))

# Call the custom plot_predictions function using our trained SGD classifier, features, and targets
plot_predictions(sgd_clf, x, y)

# Render and display the plot
plt.show()