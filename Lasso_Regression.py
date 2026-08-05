
#lasso regression is a linear model that uses L1 regularization 
# to prevent overfitting and perform feature selection. 
# It is particularly useful when dealing with high-dimensional datasets 
# where some features may be irrelevant or redundant.

# -----------------------------
# Data Generation and Plotting
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# Set random seed for reproducibility
np.random.seed(1)
m = 100

# Generate synthetic feature data x1 and target data y based on a quadratic equation with noise
x1 = 10 * np.random.rand(m, 1) - 6
y = 10 + 6 * x1 + 5 * x1**2 + 30 * np.random.randn(m, 1)

def plot_model(model, x1, y):
    # Generate 1000 evenly spaced points from min to max of x1 for a smooth curve, 
    # linspace is used to create a range of values
    x1s = np.linspace(x1.min(), x1.max(), 1000).reshape(-1, 1)
    
    # Predict target values using the trained model/pipeline
    y_pred = model.predict(x1s)
    
    # Plot training data as blue dots and test data as green crosses
    plt.plot(x1_train, y_train, 'b.', markersize=3)
    plt.plot(x1_test, y_test, 'gx', markersize=3)
    
    # Plot the model's predicted curve as a red solid line
    plt.plot(x1s, y_pred, "r-", linewidth=2, label=r"$\hat{y}$")
    
    # Format axis labels, legends, and display limits
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("y", rotation=0, fontsize=18)
    plt.legend(loc="upper center", fontsize=10)
    plt.axis([x1.min()-0.1, x1.max()+0.1, y.min()-5, y.max()+5])


# -----------------------------
# 2. Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

# Split the dataset into training sets (80%) and testing sets (20%)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)

# -----------------------------
# 1. Lasso Regression via Pipeline (Closed-form/Coordinate Descent)
# -----------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

# Construct a Pipeline that performs:
# 1. Polynomial feature expansion (degree=2, no bias)
# 2. Feature standardization (mean=0, variance=1)
# 3. Lasso regression using L1 regularization penalty

lasso_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('lasso', Lasso(alpha=1, random_state=1))
])

# Train the pipeline using the training features and targets
lasso_reg.fit(x1_train, y_train)

# Evaluate model performance (R-squared score) on training and testing sets
train_score = lasso_reg.score(x1_train, y_train)
test_score = lasso_reg.score(x1_test, y_test)
print(f"Lasso R^2 Scores -> Train: {train_score}, Test: {test_score}")

# Make predictions using the first 3 samples of the test set
predictions_lasso = lasso_reg.predict(x1_test[:3])
print("Lasso Predictions for x1_test[:3]:\n", predictions_lasso)


# -----------------------------
# 2. Lasso Regression via SGDRegressor (Gradient Descent)
# -----------------------------
from sklearn.linear_model import SGDRegressor

# Construct a Pipeline using SGDRegressor with L1 regularization penalty (penalty='l1')
lasso_sgd = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('lasso', SGDRegressor(penalty='l1', alpha=1, random_state=1))
])

# Train the pipeline using the training features and flattened training targets
lasso_sgd.fit(x1_train, y_train.ravel())

# Evaluate model performance on training and testing sets
train_score_sgd = lasso_sgd.score(x1_train, y_train)
test_score_sgd = lasso_sgd.score(x1_test, y_test)
print(f"SGD Lasso R^2 Scores -> Train: {train_score_sgd}, Test: {test_score_sgd}")

# Make predictions using the first 3 samples of the test set
predictions_sgd = lasso_sgd.predict(x1_test[:3])
print("SGD Lasso Predictions for x1_test[:3]:", predictions_sgd)
# -----------------------------
# 3. Plotting the Lasso Model
# -----------------------------
# Create a figure canvas and use the previously defined plot_model function to visualize the fit

plt.figure(figsize=(4, 3))
plot_model(lasso_reg, x1, y)
plt.show()

