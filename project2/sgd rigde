#SGD ridge

from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(89)

# Formula for MSE
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

n = 400
x = 2 * np.random.rand(n, 1)
y = 4 + 3 * x + 2 * x**2 + np.random.randn(n, 1)  # f(x) = a_0 + a_1x + a_2x^2

X = np.c_[np.ones((n, 1)), x, x**2]

# Ridge parameter lambda
lmbda = 0.001
Id = n * lmbda * np.eye(X.shape[1])
# Number of epochs and minibatches
n_epochs = 50
minibatch_size = 20

t0, t1 = 45, 150  # Learning schedule parameters

# Learning schedule function
def learning_schedule(t):
    return t0 / (t + t1)

# Initialization of Theta for SGD
theta_sgd = np.random.randn(3, 1)

# SGD
for epoch in range(n_epochs):
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index + minibatch_size]
        yi = y[random_index:random_index + minibatch_size]
        t = epoch * (n // minibatch_size) + i  # Total iteration count
        eta = learning_schedule(t)  # Compute the learning rate using the schedule
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta_sgd - yi)
        gradients += 2 * lmbda * theta_sgd #add the regularization term
        theta_sgd = theta_sgd - eta * gradients

# Plot data points
plt.plot(x, y, 'ro', label='Data Points')

# New data
xnew = np.array([[0], [2]])
Xnew = np.c_[np.ones((2, 1)), xnew, xnew**2]

ypredict_sgd = Xnew @ theta_sgd

# Calculate MSE for the method
mse_sgd = MSE(y[-2:], ypredict_sgd)

plt.plot(xnew, ypredict_sgd, "b-", label=f"SGD Ridge Regression (MSE: {mse_sgd:.2f}")

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'SGD Ridge Regression')
plt.legend()
plt.show()