# -----------------------------
# 1. Data Preparation (Ex2: Using 2 Features)
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Extract the first 2 features (sepal length and sepal width) for a 2D feature space
x = iris["data"][:, :2]

# Create binary targets (y): 1 if the target is Setosa (index 0), 0 otherwise
y = (iris["target"] == 0).astype(np.int_)

# Inspect the first 3 samples of x and the complete target array y
print("x[:3]:\n", x[:3])
print("y:\n", y)


# -----------------------------
# 2. Logistic Regression Model (2D Feature Space)
# -----------------------------
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with solver="lbfgs", C=1, and a fixed random state
log_reg = LogisticRegression(solver="lbfgs", C=1, random_state=42)

# Fit the model to the training data
log_reg.fit(x, y)

# Inspect the intercept and coefficients (now we have two coefficients: w1 and w2)
print("Intercept:", log_reg.intercept_)
print("Coefficients:", log_reg.coef_)

# Evaluate model accuracy score on the dataset
print("Score:", log_reg.score(x, y))

# Define new multi-feature data points to predict
x_new = [[5.5, 2.8], [4.5, 3.25]]

# Predict class labels for new data
print("Predictions (predict):", log_reg.predict(x_new))

# Predict class probabilities for new data
print("Prediction Probabilities (predict_proba):\n", log_reg.predict_proba(x_new))


# -----------------------------
# 3. Calculating the Decision Boundary Line
# -----------------------------
# The decision boundary equation for two features is: h = b + w1*x1 + w2*x2 = 0
# Rearranging to solve for x2 gives: x2 = -(b + w1*x1) / w2

# Generate 5 evenly spaced points for x1 spanning the minimum and maximum of the first feature
x1_boundary = np.linspace(x.min(axis=0)[0]-0.05, x.max(axis=0)[0]+0.05, 5)

# Calculate the corresponding x2 values for the decision boundary line
x2_boundary = - (log_reg.intercept_[0] + log_reg.coef_[0][0] * x1_boundary) / log_reg.coef_[0][1]

# Combine x1 and x2 boundary coordinates into a single array
boundary = np.c_[x1_boundary, x2_boundary]
print("Decision Boundary Coordinates:\n", boundary)


# -----------------------------
# 4. Plotting the 2D Decision Boundary and Contour Map
# -----------------------------
# Generate grid points to create a background prediction contour map
x1s = np.linspace(4.2, 8, 100).reshape(-1, 1)
x2s = np.linspace(1.5, 6, 100).reshape(-1, 1)
x1, x2 = np.meshgrid(x1s, x2s)
x_new_grid = np.c_[x1.ravel(), x2.ravel()]
y_pred = log_reg.predict(x_new_grid)

# Set up the figure size
plt.figure(figsize=(8, 3))

# Plot actual data points: Setosa (y == 1) as blue dots, and Not Setosa (y == 0) as red crosses
plt.plot(x[y == 1, 0], x[y == 1, 1], "b.", label="Setosa")
plt.plot(x[y == 0, 0], x[y == 0, 1], "rx", label="Not Setosa")

# Reshape predictions and plot contour background zones
zz = y_pred.reshape(x1.shape)
plt.contourf(x1, x2, zz, alpha=0.2)

# Plot the decision boundary line as a black dashed line ("k--")
plt.plot(boundary[:, 0], boundary[:, 1], "k--", label='Boundary')

# Set axis labels and legend
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14)
plt.legend(loc="upper left", fontsize=12)

# Show the final plot
plt.show()