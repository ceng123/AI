# -----------------------------
# 1. Loading and Slicing the Iris Dataset
# -----------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[:, 2:]  # Petal length and petal width
y = iris.target


# -----------------------------
# 2. Training a Decision Tree Classifier
# -----------------------------
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=2)
tree_clf.fit(x, y)

print("Accuracy Score:", tree_clf.score(x, y))


# -----------------------------
# 3. Making Predictions and Inspecting Probabilities
# -----------------------------
sample_pred = tree_clf.predict([[5, 1.5]])
print("Prediction for [[5, 1.5]]:", sample_pred)

sample_proba = tree_clf.predict_proba([[5, 1.5]])
print("Class Probabilities:", sample_proba)

print("Feature Importances:", tree_clf.feature_importances_)


# -----------------------------
# 4. Defining the Decision Boundary Plot Function
# -----------------------------
def plot_decision_boundary(clf, x, y, axes=[0, 7.5, 0, 3]):
  x1s = np.linspace(axes[0], axes[1], 100)
  x2s = np.linspace(axes[2], axes[3], 100)
  x1, x2 = np.meshgrid(x1s, x2s)
  xnew = np.c_[x1.ravel(), x2.ravel()]
  y_pred = clf.predict(xnew).reshape(x1.shape)

  custom_cmap = ListedColormap(['red', 'green', 'purple'])
  plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.2)

  plt.plot(x[y == 0, 0], x[y == 0, 1], 'rs', markersize=3, label='Setosa')
  plt.plot(x[y == 1, 0], x[y == 1, 1], 'g^', markersize=3, label='Versicolor')
  plt.plot(x[y == 2, 0], x[y == 2, 1], 'bo', markersize=3, label='Virginica')

  plt.legend(loc='upper left')
  plt.xlabel('$x_1$', fontsize=14)
  plt.ylabel('$x_2$', fontsize=14)


# -----------------------------
# 5. Visualizing the Decision Boundary
# -----------------------------
plt.figure(figsize=(5, 3))
plot_decision_boundary(tree_clf, x, y)
plt.show()