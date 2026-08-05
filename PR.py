# -----------------------------
# 1. Data Generation
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1)
m = 100

# Generate synthetic feature data x1 and target data y based on a quadratic equation with noise
x1 = 10 * np.random.rand(m, 1) - 6
y = 10 + 6 * x1 + 5 * x1**2 + 30 * np.random.randn(m, 1)

# (Optional) Plot the data
# plt.figure(figsize=(3, 2))
# plt.plot(x1, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("y", rotation=0, fontsize=18)
# plt.show()


# -----------------------------
# 2. Data Preprocessing (Polynomial Transformation & Standardization)
# -----------------------------
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Step A: Generate polynomial features up to degree 2 (excluding bias term)
poly = PolynomialFeatures(degree=2, include_bias=False)
x1_p = poly.fit_transform(x1)
print("Polynomial Features (x1_p):\n", x1_p[:5])  # Display the first 5 rows of polynomial features
# Step B: Standardize the polynomial features so they have mean=0 and variance=1
scal = StandardScaler()
x1_ps = scal.fit_transform(x1_p)


# -----------------------------
# 3. Model Training (Linear Regression)
# -----------------------------
from sklearn.linear_model import LinearRegression

# Initialize and fit the standard Linear Regression model using polynomial features
poly_reg = LinearRegression()
poly_reg.fit(x1_ps, y)

# Inspect the intercept and coefficients of the Linear Regression model
print("LinearRegression Intercept:", poly_reg.intercept_)
print("LinearRegression Coefficients:", poly_reg.coef_)


# -----------------------------
# 4. Model Training (SGD Regressor)
# -----------------------------
from sklearn.linear_model import SGDRegressor

# Initialize and fit the SGD Regressor model using the processed features and flattened target array
sgd_reg = SGDRegressor()
sgd_reg.fit(x1_ps, y.ravel())

# Inspect the intercept and coefficients of the SGD Regressor model
print("SGDRegressor Intercept:", sgd_reg.intercept_)
print("SGDRegressor Coefficients:", sgd_reg.coef_)

#predict
x1_new = np.array([[-5], [1]])
x1_new_p =poly.transform(x1_new)
x1_new_ps = scal.transform(x1_new_p)
poly_pred = poly_reg.predict(x1_new_ps)
print("LinearRegression Predictions:", poly_pred)
sgd_pred = sgd_reg.predict(x1_new_ps)
print("SGDRegressor Predictions:", sgd_pred)


#traditional VS pipeline

# x-> polynomial features -> standardization ->model training(LinearRegression/SGDRegressor)

# -----------------------------
# 1. Build Pipeline
# -----------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# Construct a Pipeline that sequences multiple processing and modeling steps:
# Step 1 ('poly'): Generate polynomial features up to degree 2 without bias
# Step 2 ('scal'): Standardize the features (mean=0, variance=1)
# Step 3 ('poly_reg'): Fit a Linear Regression model on the transformed data
pip_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('poly_reg', LinearRegression())
])


# -----------------------------
# 2. Model Training (Fit)
# -----------------------------
# Fit the entire pipeline using the raw feature data x1 and target data y.
# The pipeline automatically runs fit_transform through 'poly' and 'scal' before fitting 'poly_reg'.
pip_reg.fit(x1, y)

# Inspect the intercept and coefficients of the final linear regression step within the pipeline
print("Pipeline Intercept:", pip_reg['poly_reg'].intercept_)
print("Pipeline Coefficients:", pip_reg['poly_reg'].coef_)


# -----------------------------
# 3. Making Predictions (Predict)
# -----------------------------
# Prepare new raw data points to predict
x_new = [[-5], [1]]

# Pass raw new data directly into the pipeline. 
# It will automatically apply the same polynomial transformation and scaling steps sequentially.
predictions = pip_reg.predict(x_new)
print("Pipeline Predictions:", predictions)

# Set up the figure size
plt.figure(figsize=(3, 2))

# Create 100 evenly spaced points spanning from the minimum to the maximum value of x1, 
# and reshape it into a column vector (-1, 1)
x1s = np.linspace(x1.min(), x1.max(), 100).reshape(-1, 1)

# Predict the corresponding y values (y_pred) using our trained pipeline (pip_reg)
y_pred = pip_reg.predict(x1s)


# -----------------------------
# 2. Plotting the Results
# -----------------------------
# Plot the original data points as blue dots ("b.")
plt.plot(x1, y, "b.")

# Plot the model's predicted curve as a red solid line ("r-") with a width of 2 and a label
plt.plot(x1s, y_pred, "r-", linewidth=2, label="$\hat y$")

# Set axis labels and formatting
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)

# Display the legend at the upper right corner
plt.legend(loc="upper right", fontsize=12)

# Show the final plot
plt.show()