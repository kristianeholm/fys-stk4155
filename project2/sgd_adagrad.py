#SGD with and without momentum
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(89)

# Formula for MSE
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

n = 400
x = 2*np.random.rand(n,1)
y = 4 + 3*x + 2*x**2 + np.random.randn(n,1) #f(x)=a_0+a_1x+a_2x^2

X = np.c_[np.ones((n,1)), x, x**2]

theta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

#sgd parameters
# number of epochs and minibatches
n_epochs = 50
minibatch_size = 20

t0, t1 = 5, 50  # learning schedule parameters

# learning schedule function
def learning_schedule(t):
    return t0 / (t + t1)


# Initialization of Theta for SGD with and without momentum
theta_sgd = np.random.randn(3,1)
theta_momentum = np.random.randn(3,1)

#adagrad parameters
delta = 1e-8

G= np.zeros(theta_sgd.shape)
G_momentum = np.zeros(theta_momentum.shape)

# SGD without momentum + adagrad
for epoch in range(n_epochs):
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index+minibatch_size]
        yi = y[random_index:random_index+minibatch_size]
        gradients = 2/minibatch_size * xi.T @ (xi @ theta_sgd - yi)

        G += gradients**2
        eta = learning_schedule(epoch * n//minibatch_size + i)
        eta_adjusted = eta / (delta + np.sqrt(G)) #update eta with adagrad

        theta_sgd = theta_sgd - eta * gradients


#velocity for SGD with momentum
velocity = np.zeros((3,1))
mu = 0.3  # momentum

# SGD with momentum
for epoch in range(n_epochs):
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index+minibatch_size]
        yi = y[random_index:random_index+minibatch_size]
        gradients = 2/minibatch_size * xi.T @ (xi @ theta_momentum - yi)

        G_momentum += gradients**2
        eta = learning_schedule(epoch * n//minibatch_size + i)
        eta_adjusted = eta / (delta + np.sqrt(G_momentum)) #update eta with adagrad
        velocity = mu * velocity + eta_adjusted * gradients
        theta_momentum = theta_momentum - velocity



# New data
xnew = np.array([[0],[2]])
Xnew = np.c_[np.ones((2,1)), xnew, xnew**2]

#ypredict_linreg = Xnew @ theta_linreg
ypredict_sgd = Xnew @ theta_sgd
ypredict_momentum = Xnew @ theta_momentum

# Calculate MSE for each method
mse_sgd = MSE(y[-2:], ypredict_sgd)
mse_momentum = MSE(y[-2:], ypredict_momentum)

#plotting
plt.plot(xnew, ypredict_sgd, "b-", label=f"SGD Without Momentum (MSE: {mse_sgd:.2f})")
plt.plot(xnew, ypredict_momentum, "g-", label=f"SGD With Momentum (MSE: {mse_momentum:.2f})")

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'SGD with adagrad')
plt.legend()
plt.show()