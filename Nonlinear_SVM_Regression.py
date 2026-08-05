# -----------------------------
# 1. Generate Synthetic Nonlinear Data
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
m = 100
x = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * x + 0.5 * x**2 + np.random.randn(m, 1) / 10).ravel()


# -----------------------------
# 2. Polynomial Feature Expansion with LinearSVR
# -----------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVR
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
poly_svm_reg = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('poly_svm', LinearSVR(max_iter=10000, epsilon=0.1, random_state=1))
])

poly_svm_reg.fit(x, y)
print("Poly LinearSVR Intercept:", poly_svm_reg['poly_svm'].intercept_)
print("Poly LinearSVR Coefficients:", poly_svm_reg['poly_svm'].coef_)
print("Poly LinearSVR Score (R^2):", poly_svm_reg.score(x, y))

# Visualize Poly LinearSVR
plt.figure(figsize=(5, 3))
plot_svm_regression(poly_svm_reg, x, y, 0.1, [-1, 1, 0, 1])
plt.show()


# -----------------------------
# 3. Polynomial Feature Expansion with SGDRegressor
# -----------------------------
from sklearn.linear_model import SGDRegressor

poly_sgd_reg = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('poly_sgd', SGDRegressor(loss='epsilon_insensitive', epsilon=0.1, random_state=1))
])

poly_sgd_reg.fit(x, y)
print("Poly SGD Intercept:", poly_sgd_reg['poly_sgd'].intercept_)
print("Poly SGD Coefficients:", poly_sgd_reg['poly_sgd'].coef_)
print("Poly SGD Score (R^2):", poly_sgd_reg.score(x, y))

# Visualize Poly SGDRegressor
plt.figure(figsize=(5, 3))
plot_svm_regression(poly_sgd_reg, x, y, 0.1, [-1, 1, 0, 1])
plt.show()


# -----------------------------
# 4. Support Vector Regression with Polynomial Kernel SVR(kernel='poly')
# -----------------------------
from sklearn.svm import SVR

poly_svr_reg = SVR(kernel='poly', degree=2, epsilon=0.1, gamma=1)
poly_svr_reg.fit(x, y)
print("Poly Kernel SVR Score (R^2):", poly_svr_reg.score(x, y))

# Visualize Poly Kernel SVR
plt.figure(figsize=(5, 3))
plot_svm_regression(poly_svr_reg, x, y, 0.1, [-1, 1, 0, 1])
plt.show()


# -----------------------------
# 5. Support Vector Regression with RBF Kernel SVR(kernel='rbf')
# -----------------------------
rbf_svr_reg = SVR(kernel='rbf', degree=2, epsilon=0.1, gamma=1)
rbf_svr_reg.fit(x, y)
print("RBF Kernel SVR Score (R^2):", rbf_svr_reg.score(x, y))

# Visualize RBF Kernel SVR
plt.figure(figsize=(5, 3))
plot_svm_regression(rbf_svr_reg, x, y, 0.1, [-1, 1, 0, 1])
plt.show()