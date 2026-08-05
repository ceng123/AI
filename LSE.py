# Linear regression using the least squares method
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
np.random.seed(1)#Sureing reproducible results

m = 100

x1 = 50 + 30 * np.random.rand(m, 1)#Suring random values for x1 between 50 and 80
y = 135 + 0.5 * x1 + 3 * np.random.randn(m, 1)

#print(x1[:5])
#print(y[:5])
#plt.figure(figsize=(5,4))

#plt.plot(x1, y, "b.")

#plt.xlabel("$x_1$", fontsize=18)
#plt.ylabel("$y$", rotation=0, fontsize=18)

#plt.savefig("plot_ex1.pdf", dpi=300, bbox_inches='tight')

#plt.show()
lin_reg = LinearRegression()
lin_reg.fit(x1, y)#training the model using the least squares method

lin_reg.intercept_, lin_reg.coef_#intercept and coefficient of the linear regression model

# Predict
x1_new = np.array([[50], [80]])
y_pred = lin_reg.predict(x1_new)

# Plot the regression model
# plt.figure(figsize=(5,4))

# x1s = np.linspace(x1.min(), x1.max(), 10).reshape(-1,1)
# y_pred = lin_reg.predict(x1s)

# plt.plot(x1, y, "b.")
# plt.plot(x1s, y_pred, "r-", linewidth=2, label="$\hat y$")

# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("y", rotation=0, fontsize=18)
# plt.legend(loc="upper left", fontsize=14)

# plt.show()
# 使用矩陣計算：Normal Equation
# Fit
X = np.c_[np.ones((x1.shape[0], 1)), x1]
print(X[:5])

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(theta)

# Predict
x1_new = np.array([[50], [80]])
X_new = np.c_[np.ones((2, 1)), x1_new]

y_pred = X_new.dot(theta)
print(y_pred)