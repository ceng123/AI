import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 2. Model Training (Fit)
# -----------------------------
from sklearn.linear_model import SGDRegressor

# Data
np.random.seed(1)#Sureing reproducible results

m = 100

x1 = 50 + 30 * np.random.rand(m, 1)#Suring random values for x1 between 50 and 80
y = 135 + 0.5 * x1 + 3 * np.random.randn(m, 1)

# 建立標準化物件
scal = StandardScaler()
x1_scal = scal.fit_transform(x1)
# 查看標準化後的前 5 筆資料
print(x1_scal[:5])

# Initialize the Stochastic Gradient Descent Regressor with hyperparameters
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)

# Fit the model using the standardized features (x1_scal) and target data (y)
# Note: y.ravel() flattens y into a 1D array to meet scikit-learn requirements
sgd_reg.fit(x1_scal, y.ravel())

# Inspect the trained intercept and coefficients (weights)
print("Intercept:", sgd_reg.intercept_)
print("Coefficients:", sgd_reg.coef_)

# Prepare new, unseen data for prediction
x1_new = [[50], [80]]

# Crucial: Transform new data using the same scaler fitted on the training data (use transform only)
x1_new_scal = scal.transform(x1_new)

# Feed the standardized new data into the model to get predictions
predictions = sgd_reg.predict(x1_new_scal)
print("Predictions:", predictions)


# Add a column of ones to the standardized feature data (x1_scal) to account for the intercept term
X = np.c_[np.ones((m, 1)), x1_scal]

# Define the number of iterations and the learning rate (eta)
n_iterations = 1000
eta = 0.1

# Initialize random weights (theta) for the model
theta = np.random.randn(2, 1)

# Perform gradient descent iteratively
for t in range(n_iterations):
    # Compute the gradient (gd) of the Mean Squared Error cost function
    gd = 2 / m * X.T.dot(X.dot(theta) - y)
    
    # Update the parameters (theta) by taking a step in the direction of the negative gradient
    theta = theta - eta * gd
print("Theta after Gradient Descent:", theta)

# -----------------------------
# 1. Learning Schedule Function 
# -----------------------------
# Defines a function to dynamically calculate the learning rate (eta) over time
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)


# -----------------------------
# 2. Stochastic Gradient Descent (SGD) Implementation
# -----------------------------
# Set the number of epochs (complete passes through the training set)
n_epochs = 50

# Initialize random weights (theta) for the model
theta = np.random.randn(2, 1)
print("Initial Theta:", theta)
# Outer loop: iterates through each epoch
for epoch in range(n_epochs):
    # Inner loop: iterates through each sample in the dataset (m samples total)
    for i in range(m):

        # Randomly pick a sample index
        ind = np.random.randint(m)
        
        # Extract the selected single feature row and transpose it into a column vector
        xi = X[ind:ind+1].T
        print("xi shape:", xi)
        yi = y[ind:ind+1]
        
        # Dynamically calculate the learning rate based on total iteration step
        eta = learning_schedule(epoch * m + i, 5, 50)
        
        # Compute the gradient using only the single randomly chosen sample (xi, yi)
        gd = 2 * xi.dot(xi.T.dot(theta) - yi)
        
        # Update the model parameters (theta) using the calculated gradient and learning rate
        theta = theta - eta * gd
        # -----------------------------

# 1. Mini-batch Gradient Descent Setup
# -----------------------------
# Set the number of epochs and the size of each mini-batch
n_epochs = 50
minibatch_size = 20

# Initialize random weights (theta) for the model
theta = np.random.randn(2, 1)

# Initialize the total iteration step counter
t = 0

# -----------------------------
# 2. Training Loop with Learning Schedule
# -----------------------------
# Outer loop: iterates through each epoch
for epoch in range(n_epochs):
    
    # Shuffle the dataset indices randomly at the start of each epoch
    ind = np.random.permutation(m)
    X_shuffled = X[ind]
    y_shuffled = y[ind]
    
    # Inner loop: iterates through the dataset in chunks of 'minibatch_size'
    for i in range(0, m, minibatch_size):
        
        # Extract a mini-batch of features and targets
        Xi = X_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        
        # Increment the total iteration counter
        t += 1
        
        # Dynamically calculate the learning rate (eta) based on the iteration step
        eta = learning_schedule(t, 5, 50)
        
        # Compute the gradient using the current mini-batch (Xi, yi)
        gd = 2 / minibatch_size * Xi.T.dot(Xi.dot(theta) - yi)
        
        # Update the model parameters (theta) using the calculated gradient and learning rate
        theta = theta - eta * gd