import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 使用矩陣計算：Normal Equation
# Fit
X = np.c_[np.ones((x1.shape[0], 1)), x1]
X[:5]

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
theta

# Predict
x1_new = np.array([[50], [80]])
X_new = np.c_[np.ones((2, 1)), x1_new]

y_pred = X_new.dot(theta)
y_pred