# -----------------------------
# 1. Generate Synthetic Linear Data
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
m = 50
x = 2 * np.random.rand(m, 1)
y = (4 + 3 * x + np.random.randn(m, 1)).ravel()


# -----------------------------
# 2. Train LinearSVR Model
# -----------------------------
from sklearn.svm import LinearSVR

# Initialize and fit LinearSVR with an epsilon margin of 0.5
svm_reg = LinearSVR(epsilon=0.5, random_state=1)
svm_reg.fit(x, y)

print("Intercept:", svm_reg.intercept_)
print("Coefficients:", svm_reg.coef_)
print("Score (R^2):", svm_reg.score(x, y))


# -----------------------------
# 3. Helper Function to Find Support Vectors
# -----------------------------
def find_support_vectors(svm_reg, x, y, epsilon):
    y_pred = svm_reg.predict(x)
    off_margin = (np.abs(y - y_pred) >= epsilon)
    return np.argwhere(off_margin)


# -----------------------------
# 4. Helper Function to Plot SVM Regression
# -----------------------------
def plot_svm_regression(model, x, y, epsilon, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = model.predict(x1s)

    # Plot regression line and the epsilon-insensitive tube margins
    plt.plot(x1s, y_pred, 'r-', linewidth=2, label=r'$\hat{y}$')
    plt.plot(x1s, y_pred + epsilon, 'k--')
    plt.plot(x1s, y_pred - epsilon, 'k--')

    # Identify and highlight support vectors lying outside or on the margin boundary
    model.support_ = find_support_vectors(model, x, y, epsilon)
    plt.scatter(x[model.support_], y[model.support_], c='red', s=50, alpha=0.3)

    # Plot original data points and configure labels
    plt.plot(x, y, 'bo', markersize=3)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$y$', fontsize=14, rotation=0)
    plt.legend(loc='lower right')


# -----------------------------
# 5. Visualizing the SVR Results
# -----------------------------
plt.figure(figsize=(5, 3))
plot_svm_regression(svm_reg, x, y, 0.5, [0, 2, 3, 11])
plt.show()