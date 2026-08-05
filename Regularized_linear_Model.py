# -----------------------------
# 1. Data Generation and Plotting
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

# Plot and save the generated data
# plt.figure(figsize=(4, 3))
# plt.plot(x1, y, "b.", markersize=3)
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("y", rotation=0, fontsize=18)
# plt.savefig("plot_1.pdf", dpi=300, bbox_inches='tight')
# plt.show()


# -----------------------------
# 2. Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

# Split the dataset into training sets (80%) and testing sets (20%)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)


# -----------------------------
# 3. Ridge Regression with Pipeline (Closed-form)
# -----------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# Build a Pipeline combining Polynomial Features, Standardization, and Ridge Regression
ridge_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('ridge', Ridge(alpha=m/2, random_state=1))
])

# Train the pipeline using the training data
ridge_reg.fit(x1_train, y_train)

# Evaluate model performance (R-squared score) on training and testing sets
train_score = ridge_reg.score(x1_train, y_train)
test_score = ridge_reg.score(x1_test, y_test)
print(f"R^2 Scores -> Train: {train_score}, Test: {test_score}")

# Make predictions using the first 3 samples of the test set
predictions = ridge_reg.predict(x1_test[:3])
print("Predictions for x1_test[:3]:\n", predictions)


# -----------------------------
# 1. Ridge Regression via Stochastic Gradient Descent (SGD) Pipeline
# -----------------------------
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Construct a Pipeline that performs:
# 1. Polynomial feature expansion (degree=2, no bias)
# 2. Feature standardization (mean=0, variance=1)
# 3. Ridge regression optimization using Stochastic Gradient Descent (penalty='l2')
ridge_sgd = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('ridge', SGDRegressor(penalty='l2', alpha=1, random_state=1))
])


# -----------------------------
# 2. Model Training (Fit)
# -----------------------------
# Fit the pipeline using the training features (x1_train) and the flattened training targets (y_train.ravel())
ridge_sgd.fit(x1_train, y_train.ravel())


# -----------------------------
# 3. Model Evaluation and Prediction
# -----------------------------
# Evaluate the R-squared score on both the training and testing sets
train_score = ridge_sgd.score(x1_train, y_train)
test_score = ridge_sgd.score(x1_test, y_test)
print(f"R^2 Scores -> Train: {train_score}, Test: {test_score}")

# Predict the target values for the first 3 samples of the test set
predictions = ridge_sgd.predict(x1_test[:3])
print("Predictions for x1_test[:3]:", predictions)

def plot_model(model, x1, y):
    # Generate 1000 evenly spaced points from min to max of x1 for a smooth curve
    x1s = np.linspace(x1.min(), x1.max(), 1000).reshape(-1, 1)
    
    # Predict target values using the trained model/pipeline
    y_pred = model.predict(x1s)
    
    # Plot training data as blue dots and test data as green crosses
    plt.plot(x1_train, y_train, 'b.', markersize=3)
    plt.plot(x1_test, y_test, 'gx', markersize=3)
    
    # Plot the model's predicted curve as a red solid line
    plt.plot(x1s, y_pred, "r-", linewidth=2, label="$\hat y$")
    
    # Format axis labels, legends, and display limits
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("y", rotation=0, fontsize=18)
    plt.legend(loc="upper center", fontsize=10)
    plt.axis([x1.min()-0.1, x1.max()+0.1, y.min()-5, y.max()+5])


# -----------------------------
# 2. Render the Plot
# -----------------------------
# Create a figure canvas
plt.figure(figsize=(4, 3))

# Call the function to plot the trained ridge regression model against data
plot_model(ridge_reg, x1, y)

# Show the final plot
plt.show()