#GD with and without momentum
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

np.random.seed(89)

# Formula for MSE
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

# the number of datapoints
n = 200
x = 2*np.random.rand(n,1)
y = 4 + 3*x + 2*x**2 + np.random.randn(n,1) #f(x)=a_0+a_1x+a_2x^2

X = np.c_[np.ones((n,1)), x, x**2] #need a nWhatew column with the X values squared


beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

beta = np.random.randn(3,1)


eta =0.25 #learning rate
Niterations = 1000 #number of iterations
momentum = 0.05

#To store predicted values
ypredict = []
ypredict_with_momentum = []

# GD without momentum
for iter in range(Niterations):
    gradient = (2.0/n) * X.T @ (X @ beta - y)
    beta -= eta * gradient

    # Calculate and store the predicted values
    ypredict_current = X @ beta
    ypredict.append(ypredict_current)

# GD with Momentum
beta_with_momentum = np.random.randn(3, 1)
for iter in range(Niterations):
    gradient = (2.0/n) * X.T @ (X @ beta_with_momentum - y)
    beta_with_momentum -= eta * gradient + momentum * gradient

    # Calculate and store the predicted values
    ypredict_momentum = X @ beta_with_momentum
    ypredict_with_momentum.append(ypredict_momentum)


# Calculate the MSE for GD without momentum
mse_no_momentum = MSE(y, ypredict[-1])  # Use the last predicted values

# Calculate the MSE for GD with momentum
mse_with_momentum = MSE(y, ypredict_with_momentum[-1])  # Use the last predicted values

# Print the MSE for both cases
print(f"MSE (Without Momentum): {mse_no_momentum}")
print(f"MSE (With Momentum): {mse_with_momentum}")

# Plot data points into plot
plt.plot(x, y, 'ro', label='Data Points')

# Plot GD without momentum
plt.plot(x, ypredict[-1], "b-", label='GD without Momentum')  # Use the last predicted values

# Plot GD with Momentum
plt.plot(x, ypredict_with_momentum[-1], "g-", label='GD with Momentum')  # Use the last predicted values

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient Descent')
plt.legend()
plt.axis([0, 2.0, 0, 15.0])
plt.show()