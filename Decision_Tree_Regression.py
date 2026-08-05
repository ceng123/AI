# -----------------------------
# 1. Generate Synthetic Nonlinear Data
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
m = 200
x = np.random.rand(m, 1)
y = 4 * (x - 0.5) ** 2 + np.random.randn(m, 1) / 10


# -----------------------------
# 2. Train DecisionTreeRegressor
# -----------------------------
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=1)
tree_reg.fit(x, y)

print("Score (R^2):", tree_reg.score(x, y))


# -----------------------------
# 3. Visualize and Export the Tree Structure
# -----------------------------
from sklearn import tree

plt.figure(figsize=(5, 3))
tree.plot_tree(
    tree_reg, feature_names=["x1"], rounded=True, filled=True
)
plt.savefig("tree_reg.pdf")
plt.show()


# -----------------------------
# 4. Making Predictions and Feature Importances
# -----------------------------
print("Prediction for [[0.4]]:", tree_reg.predict([[0.4]]))
print("Feature Importances:", tree_reg.feature_importances_)


# -----------------------------
# 5. Helper Function to Plot Regression Predictions
# -----------------------------
def plot_regression_predictions(
    tree_reg, x, y, axes=[0, 1, -0.2, 1]
):
  x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
  y_pred = tree_reg.predict(x1)

  plt.plot(x, y, "b.")
  plt.plot(x1, y_pred, "r-", linewidth=2, label=r"$\hat{y}$")

  plt.axis(axes)
  plt.xlabel("$x_1$", fontsize=18)
  plt.ylabel("$y$", fontsize=18, rotation=0)
  plt.legend(loc="upper center", fontsize=14)


# -----------------------------
# 6. Visualizing the Stepwise Regression Curve
# -----------------------------
plt.figure(figsize=(5, 3))
plot_regression_predictions(tree_reg, x, y)
plt.show()